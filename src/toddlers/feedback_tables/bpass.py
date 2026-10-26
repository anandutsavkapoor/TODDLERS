"""BPASS v2.2.1 adapter for the unified feedback table (:func:`toddlers.feedback_tables.write_feedback_table`).

Reads the BPASS release files of one IMF and star type (``yields-``, ``ionizing-``,
``spectra-<star>-imf_<imf>.z<Z>.dat``, plain or gzipped) into the writer's arrays, with the
recipe of the original TODDLERS BPASS generator, so the package can rebuild the tables it
ships (``BPASS_<imf>_<star>_feedback.h5``):

* source nodes: 51 ages, log10(age/yr) = 6.0 ... 11.0 in steps of 0.1, and 13 Z;
* L_mech = E_SN + E_wind with E_wind = 2e30 kg x (yields column 5, Msun m^2 s^-2 / yr), as the
  BPASS v2.2.1 manual prescribes ("multiply by 2x10^30 kg and get J/yr"), and E_SN in J/yr
  (column 6), converted with 1 J/yr = 0.317098 erg/s;
* F_ram components = sqrt(2 E Mdot) for winds and supernovae, Mdot = H + He + Z ejecta
  [Msun/yr], Msun = 1.989e33 g, yr = 3.1536e7 s;
* L_bol = Lsun x sum of the spectrum (1 A bins), L_ion = the same below 912 A, L_LyW = Lsun x
  mean over 995-1005 A [erg/s/A], with Lsun = 3.848e33 erg/s (BPASS manual value);
  Q_HI = 10**(ionizing column 2);
* a zero total at a source node is set to 1e-16 (the original generator's floor) before log;
* every log quantity is interpolated linearly in log age onto the shipped 15000-node time
  grid (1e-6 ... 50 Myr and 51 ... 2000 Myr, log spaced), held constant below 1 Myr;
* the components passed to the writer partition the resampled total by the source-node
  component fraction, interpolated linearly in log age (a zero-total node counts as wind);
* Q_HeI, Q_HeII = photon integrals of the spectrum up to and including the 504 and 228 A
  bins (Q = sum L_lambda dlambda lambda / (h c), 1 A bins); stored only if the same integral
  up to and including the 912 A bin reproduces the ionizing file's Q_HI within
  QHI_CROSSCHECK_TOL at every source node (the release's Q_HI
  counts the 912 A bin), else listed as missing with the worst ratio; the per-node ratios are
  recorded either way. L_ion keeps the released convention (bins below 912 A);
* spectral hardness is not stored: it is derived on load as Q_HeI / Q_HI and
  Q_HeII / Q_HI.

No hoki dependency: the files are read with numpy in the same way hoki reads them.
"""
import hashlib
import os

import numpy as np

from .table import write_feedback_table

JYR2ERGS = 0.317098          # erg/s per J/yr (1e7 / 3.1536e7)
MSUN_G = 1.989e33            # g
YR_S = 365 * 24 * 60 * 60    # s
LSUN = 3.848e33              # erg/s, BPASS manual
WIND_KG = 2e30               # kg, BPASS manual factor for the wind-energy column
SOURCE_ZERO_FLOOR = 1e-16    # original generator
H_PLANCK = 6.62607015e-27    # erg s (CODATA 2018, exact)
C_LIGHT = 2.99792458e10      # cm/s (exact)
QHI_CROSSCHECK_TOL = 0.02    # relative tolerance of the Q_HI cross-check

Z_NODES = np.array([1e-5, 1e-4, .001, .002, .003, .004, .006, .008, .010, .014, .020, .030, .040])
ZFILE = ['em5', 'em4', '001', '002', '003', '004', '006', '008', '010', '014', '020', '030', '040']
LOG_AGE_MYR = np.arange(6, 11.1, .1) - 6
T_MYR = np.concatenate([np.logspace(-6, np.log10(50), 10000, endpoint=True),
                        np.logspace(np.log10(51), np.log10(2e3), 5000, endpoint=True)])

YIELD_COLS = ['log_age', 'H_wind', 'He_wind', 'Z_wind', 'E_wind', 'E_sn', 'H_sn', 'He_sn', 'Z_sn']


def _md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def _path(d, kind, star, imf, zs):
    p = os.path.join(d, f"{kind}-{star}-imf_{imf}.z{zs}.dat")
    return p if os.path.exists(p) else p + '.gz'


def _floor0(a):
    a = np.array(a, dtype=float)
    a[a == 0] = SOURCE_ZERO_FLOOR
    return a


def _resample(log_src):
    """(n_Z, 51) log values on the source ages -> (n_Z, len(T_MYR)), linear in log age."""
    log_t = np.log10(T_MYR)
    return np.array([np.interp(log_t, LOG_AGE_MYR, row) for row in log_src])


def read_bpass(release_dir, imf, star_type):
    """Source-node arrays (n_Z, 51) and the md5 of every file read."""
    md5 = {}
    yields, ion, spec, waves = [], [], [], None
    for zs in ZFILE:
        py, pi, ps = (_path(release_dir, k, star_type, imf, zs) for k in ('yields', 'ionizing', 'spectra'))
        for p in (py, pi, ps):
            md5[os.path.basename(p)] = _md5(p)
        y = np.loadtxt(py)
        yields.append({k: y[:, i] for i, k in enumerate(YIELD_COLS)})
        ion.append(np.loadtxt(pi)[:, 1])
        s = np.loadtxt(ps)
        if waves is None:
            waves = s[:, 0]
        elif not np.array_equal(waves, s[:, 0]):
            raise ValueError(f"{ps}: wavelength grid differs")
        spec.append(s.T[1:, :])
    return yields, np.array(ion), np.array(spec), waves, md5


def bpass_source_nodes(yields, ion, spec, waves):
    """Log quantities on the source nodes, the component fractions, and the extras."""
    n = {}
    E_w, E_sn, F_w, F_sn = [], [], [], []
    for d in yields:
        M_wind = d['H_wind'] + d['He_wind'] + d['Z_wind']
        M_SN = d['H_sn'] + d['He_sn'] + d['Z_sn']
        e_wind = WIND_KG * d['E_wind']
        e_sn = d['E_sn']
        E_w.append(JYR2ERGS * e_wind)
        E_sn.append(JYR2ERGS * e_sn)
        F_w.append(np.sqrt(2 * JYR2ERGS * e_wind * (MSUN_G / YR_S) * M_wind))
        F_sn.append(np.sqrt(2 * JYR2ERGS * e_sn * (MSUN_G / YR_S) * M_SN))
    E_w, E_sn, F_w, F_sn = map(np.array, (E_w, E_sn, F_w, F_sn))
    # the same operation order as the original generator: E = 0.317098 * (E_SN + E_wind)
    E_tot = np.array([JYR2ERGS * (d['E_sn'] + WIND_KG * d['E_wind']) for d in yields])
    F_tot = F_w + F_sn
    with np.errstate(divide='ignore', invalid='ignore'):
        n['f_L_wind'] = np.where(E_tot > 0, E_w / E_tot, 1.0)
        n['f_F_wind'] = np.where(F_tot > 0, F_w / F_tot, 1.0)
        n['zero_total_nodes'] = {'L_mech': int(np.sum(E_tot == 0)), 'F_ram': int(np.sum(F_tot == 0))}
        n['log_L_mech'] = np.log10(_floor0(E_tot))
        n['log_F_ram'] = np.log10(_floor0(F_tot))
        n['log_L_bol'] = np.log10(np.sum(spec, axis=2) * LSUN)
        L_ion = LSUN * np.sum(spec[:, :, waves < 912], axis=2)
        n['zero_total_nodes']['L_ion'] = int(np.sum(L_ion == 0))
        n['log_L_ion'] = np.log10(_floor0(L_ion))
        n['log_Q_HI'] = np.asarray(ion, dtype=float)
        band = np.logical_and(waves >= 995., waves <= 1005.)
        L_lyw = LSUN * np.mean(spec[:, :, band], axis=2)
        n['zero_total_nodes']['L_LyW'] = int(np.sum(L_lyw == 0))
        n['log_L_LyW'] = np.log10(_floor0(L_lyw))
        # photon rates below the edges from the spectrum (1 A bins, Lsun / A per 1e6 Msun)
        ph = LSUN * spec * (waves * 1e-8) / (H_PLANCK * C_LIGHT)
        # edge bins included (lambda <= edge): the convention of the release's ionizing files
        qs = {k: np.sum(ph[:, :, waves <= edge], axis=2) for k, edge in (('HI', 912.0), ('HeI', 504.0), ('HeII', 228.0))}
        n['Q_HI_spectrum_over_file'] = qs['HI'] / 10 ** np.asarray(ion, dtype=float)
        n['zero_total_nodes']['Q_HeI'] = int(np.sum(qs['HeI'] == 0))
        n['zero_total_nodes']['Q_HeII'] = int(np.sum(qs['HeII'] == 0))
        n['log_Q_HeI'] = np.log10(_floor0(qs['HeI']))
        n['log_Q_HeII'] = np.log10(_floor0(qs['HeII']))
    return n


def qhi_crosscheck(nodes):
    """Q_HI(spectrum) / Q_HI(ionizing file) per source node: worst ``|ratio - 1|``, its node, and pass."""
    r = nodes['Q_HI_spectrum_over_file']
    dev = np.abs(r - 1.0)
    dev = np.where(np.isfinite(dev), dev, np.inf)
    i, j = np.unravel_index(int(np.argmax(dev)), dev.shape)
    return {'worst_abs_ratio_minus_1': float(dev[i, j]), 'worst_Z': float(Z_NODES[i]),
            'worst_log_age_yr': float(LOG_AGE_MYR[j] + 6), 'n_nodes': int(dev.size),
            'n_nodes_over_tol': int(np.sum(dev > QHI_CROSSCHECK_TOL)), 'tolerance': QHI_CROSSCHECK_TOL,
            'pass': bool(np.all(dev <= QHI_CROSSCHECK_TOL))}


def bpass_quantities(nodes):
    """Writer arrays on T_MYR from the source-node arrays, and the missing list."""
    keys = ['log_L_mech', 'log_F_ram', 'log_L_bol', 'log_L_ion', 'log_Q_HI', 'log_L_LyW']
    cc = qhi_crosscheck(nodes)
    if cc['pass']:
        keys += ['log_Q_HeI', 'log_Q_HeII']
    r = {k: _resample(nodes[k]) for k in keys}
    fL, fF = _resample(nodes['f_L_wind']), _resample(nodes['f_F_wind'])
    L, F = 10 ** r['log_L_mech'], 10 ** r['log_F_ram']
    q = {'L_bol': 10 ** r['log_L_bol'], 'L_ion': 10 ** r['log_L_ion'], 'Q_HI': 10 ** r['log_Q_HI'],
         'L_LyW': 10 ** r['log_L_LyW'],
         'L_mech_wind': L * fL, 'L_mech_SN': L * (1.0 - fL),
         'F_ram_wind': F * fF, 'F_ram_SN': F * (1.0 - fF)}
    missing = {}
    if cc['pass']:
        q['Q_HeI'], q['Q_HeII'] = 10 ** r['log_Q_HeI'], 10 ** r['log_Q_HeII']
    else:
        why = (f"spectrum photon integral below 912 A differs from the ionizing file's Q_HI by up to "
               f"{cc['worst_abs_ratio_minus_1']:.3g} (> {QHI_CROSSCHECK_TOL}) at Z={cc['worst_Z']}, "
               f"log age {cc['worst_log_age_yr']:.1f}")
        missing = {'Q_HeI': why, 'Q_HeII': why}
    return q, missing, cc


def build_bpass_table(release_dir, imf, star_type, out_dir, template='BPASS', version='v2.2.1'):
    """
    Build a BPASS feedback table from the release files.

    Args:
        release_dir (str): directory with the yields-, ionizing- and spectra- files.
        imf (str): BPASS IMF label without the ``imf_`` prefix, e.g. 'chab100'.
        star_type (str): 'bin' or 'sin'.
        out_dir (str): output directory.

    Returns:
        dict: what :func:`toddlers.feedback_tables.write_feedback_table` returns.
    """
    yields, ion, spec, waves, md5 = read_bpass(release_dir, imf, star_type)
    nodes = bpass_source_nodes(yields, ion, spec, waves)
    q, missing, cc = bpass_quantities(nodes)
    upper = ''.join(ch for ch in imf.split('chab')[-1] if ch.isdigit()) if 'chab' in imf else ''
    imf_def = {'kind': 'read_from_source', 'name': f"imf_{imf}",
               'form': 'BPASS release IMF named in the file names (imf_<label>)',
               'm_up_msun': float(upper) if upper else None,
               'm_up_basis': 'digits of the release file label' if upper else None,
               'source_files_md5': md5, 'release_version': version}
    if 'chab' in imf:
        imf_def.update({
            'form': 'Chabrier (2003): exponential cut-off below 1 Msun, high-mass slope -2.3',
            'm_low_msun': 0.1, 'break_msun': 1.0, 'high_mass_slope': -2.3,
            'literature': [
                {'ref': 'Stanway & Eldridge 2018, MNRAS 479, 75 (arXiv:1805.08784v1, md5 69fe11c02a9f5193a1027eb8986d6229)',
                 'section_2.4': 'These are broken power law functions, with a lower IMF slope, alpha1, applied '
                                'between 0.1 Msun and M1 and an upper slope, alpha2, applied between M1 and Mmax',
                 'section_2.4_chabrier': 'For BPASS v2.2 we also calculate models for two additional IMFs which, '
                                         'rather than acting as a broken power law instead introduce an exponential '
                                         'cut-off at masses below 1 Msun and a high mass slope of -2.3, following the '
                                         'prescription of Chabrier (2003). We calculate this model with upper mass '
                                         'cut-offs of 100 and 300 Msun.',
                 'table_1': 'Model | alpha1 (0.1 Msun - M1) | alpha2 (M1 - Mmax) | M1 | Mmax; '
                            'chab100 | exp cutoff | -2.3 | 1.0 | 100 Msun; chab300 | exp cutoff | -2.3 | 1.0 | 300 Msun'}],
            'm_low_basis': 'Stanway & Eldridge 2018 Table 1: the alpha1 column is headed (0.1 Msun - M1); the text '
                           'states the 0.1 Msun lower limit for the broken power laws and does not restate it for '
                           'the Chabrier rows (read from the column heading)',
        })
    else:
        imf_def.update({'m_low_msun': None, 'm_low_basis': 'not looked up for this IMF label (UNVERIFIED)'})
    norm_basis = ('Eldridge et al. 2017, PASA 34, e058 (arXiv:1710.02154v1, md5 4cf438b645e3fc1aa34b0ea54870055e), '
                  'text introducing its Figures 5 and 6: "assuming a co-eval stellar population of total mass 10^6 Msun"; '
                  'Stanway & Eldridge 2018 appendix figure axes: "Luminosity in Lsun per Angstrom per 10^6 Msun '
                  'initially formed"')
    meta = {
        'template': template, 'imf': imf, 'star_type': star_type,
        'source': {'code': 'BPASS', 'version': version, 'input_files_md5': md5,
                   'adapter': 'toddlers.feedback_tables.bpass'},
        'imf_definition': imf_def,
        'normalization': {'rule': 'BPASS: 1e6 Msun formed at t = 0 (co-eval population)',
                          'per_mass_msun': 1e6, 'population_mass_msun': 1e6, 'm_low_msun': imf_def['m_low_msun'],
                          'm_up_msun': imf_def['m_up_msun'], 'basis': norm_basis},
        'conventions': {
            'L_mech': 'E_SN + E_wind (yields columns 6 and 5)',
            'wind_energy_factor_kg': WIND_KG,
            'wind_energy_factor_basis': 'BPASS v2.2.1 manual, README: Yields, column 5',
            'E_SN': '1e51 erg per supernova (BPASS manual)',
            'F_ram': 'sqrt(2 E_wind Mdot_wind) + sqrt(2 E_SN Mdot_SN)',
            'components_rule': 'resampled total apportioned by the source-node component fraction, '
                               'interpolated linearly in log age',
            'resampling': 'log quantities linear in log age onto the 15000-node grid; held below 1 Myr',
            'L_ion': 'Lsun x sum of the spectrum below 912 A',
            'Q_HeI_Q_HeII': 'photon integral of the spectrum up to and including the 504 / 228 A bins, if the '
                            'Q_HI cross-check (up to and including the 912 A bin) passes',
            'L_ion_edge': 'bins below 912 A (released convention; the 912 A bin is not included)',
            'Q_HI_crosscheck': cc,
            'L_LyW': 'Lsun x mean of the spectrum over its 1 A nodes in 995-1005 A',
            'source_zero_floor': SOURCE_ZERO_FLOOR,
            'source_zero_nodes': nodes['zero_total_nodes'],
            'constants': {'Jyr2ergs': JYR2ERGS, 'Msun_g': MSUN_G, 'yr_s': YR_S, 'Lsun_erg_s': LSUN,
                          'h_erg_s': H_PLANCK, 'c_cm_s': C_LIGHT},
            'time_grid_source_resolution': '0.1 dex in log age (51 ages, 1 Myr to 100 Gyr)',
        },
    }
    return write_feedback_table(T_MYR, Z_NODES, q, meta, out_dir, missing=missing, legacy=True)
