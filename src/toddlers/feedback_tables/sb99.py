"""Starburst99 adapter for the unified feedback table (:func:`toddlers.feedback_tables.write_feedback_table`).

Reads the Starburst99 output files of one run per metallicity (``<model>.power1``,
``.snr1``, ``.yield1``, ``.quanta1``, ``.spectrum1``, ``.input1``) and turns them into the
writer's arrays, with the recipe of the original TODDLERS SB99 generator, so the package can
rebuild the table it ships (``SB99_kroupa100_feedback.h5``):

* ``L_bol`` = 10**LOG L (quanta); ``L_ion`` = 10**(H I % of L) x L_bol; ``Q_HI`` = 10**H I rate;
  ``Q_HeI``, ``Q_HeII`` = 10**He I rate, 10**He II rate (quanta; nested bands below 504 and
  228 A, galaxy.f);
* ``L_LyW`` = mean of the stellar spectrum over 995-1005 A [erg/s/A];
* winds: power P_w = 10**POWER ALL (power1), mass-loss rate Mdot_w = 10**ONLY WINDS (yield1);
  supernovae: P_SN = 10**ALL SUPERNOVAE POWER (snr1), Mdot_SN = 10**ONLY SNE (yield1);
* F_ram components = sqrt(2 P Mdot) for each, with Msun = 1.989e33 g and yr = 3.1536e7 s;
* L_mech = 10**STARS + SUPERNOVAE POWER (snr1), the total Starburst99 prints. Its component
  columns are rounded to 0.001 dex and do not add up to it exactly (up to ~0.001 dex), so
  the components passed to the writer are that total apportioned by P_w / (P_w + P_SN);
  the largest mismatch is recorded in the metadata;
* the IMF definition and normalization are read from the run's ``.input1`` (XPONENT,
  XMASLIM, TOMA), with its path and md5.

Starburst99 writes log10(0) as -30, so no zero reaches the writer's floor.
"""
import hashlib
import os
import re

import numpy as np

from .table import write_feedback_table, broken_power_law

MSUN_G = 1.989e33          # g, value of the original generator
YR_S = 365 * 24 * 60 * 60  # s, value of the original generator

FDICT = {
    'power1': ['Time', 'Power_All', 'Power_OB', 'Power_RSG', 'Power_LBV', 'Power_WR', 'Energy',
               'Momentum_All', 'Momentum_OB', 'Momentum_RSG', 'Momentum_LBV', 'Momentum_WR'],
    'snr1': ['Time', 'All_Rate', 'All_Power', 'All_Energy', 'Type_IB_Rate', 'Type_IB_Power',
             'Type_IB_Energy', 'Typical_Mass', 'Lowest_Mass', 'Stars_SN_Power', 'Stars_SN_Energy'],
    'quanta1': ['Time', 'HI_rate', 'HI%', 'HeI_rate', 'HeI%', 'HeII_rate', 'HeII%', 'Log_Luminosity'],
    'spectrum1': ['Time', 'Wavelength', 'Total', 'Stellar', 'Nebular'],
    'yield1': ['Time', 'H', 'He', 'C', 'N', 'O', 'Mg', 'Si', 'S', 'Fe', 'All_winds', 'All_SN',
               'All_winds_SN', 'Total_mass'],
}


def _md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_sb99_file(path, keys):
    """Parse one Starburst99 output file: header lines up to the one starting with TIME
    (and a units line starting with [s^-1]) are skipped; columns are named by ``keys``.
    A file with repeated times (the spectrum) is reshaped to (n_time, n_wavelength)."""
    with open(path) as f:
        lines = f.readlines()
    h = 0
    while True:
        tok = lines[h].split()
        h += 1
        if tok and tok[0] == 'TIME':
            break
    if lines[h].split()[0] == '[s^-1]':
        h += 1
    rows = np.array([[float(x) for x in ln.split()] for ln in lines[h:] if ln.strip()])
    data = {k: rows[:, i] for i, k in enumerate(keys)}
    if data['Time'][0] == data['Time'][1]:
        nx = 1
        while data['Time'][nx] == data['Time'][nx - 1]:
            nx += 1
        ny = rows.shape[0] // nx
        data = {k: v.reshape(ny, nx) for k, v in data.items()}
        data['Time'] = data['Time'][:, 0]
        data['Wavelength'] = data['Wavelength'][0, :]
    return data


def parse_sb99_input(path):
    """The run settings of a Starburst99 ``.input1`` file (each value follows its [KEY] line)."""
    with open(path) as f:
        lines = [ln.rstrip('\n') for ln in f]
    out = {}
    for i, ln in enumerate(lines):
        m = re.search(r'\[([A-Z0-9_,.]+)\]\s*$', ln)
        if m:
            j = i + 1
            while j < len(lines) and re.match(r'^\s*(GENEVA|PADOVA|\(|$)', lines[j]):
                j += 1
            if j < len(lines):
                out[m.group(1)] = lines[j].strip()
    return out


def read_sb99_run(run_dir, model_name):
    """All the files of one Starburst99 run, parsed (dict suffix -> columns) plus md5s."""
    data, md5 = {}, {}
    for suffix, keys in FDICT.items():
        p = os.path.join(run_dir, f"{model_name}.{suffix}")
        data[suffix] = parse_sb99_file(p, keys)
        md5[os.path.basename(p)] = _md5(p)
    p = os.path.join(run_dir, f"{model_name}.input1")
    settings = parse_sb99_input(p) if os.path.exists(p) else {}
    if settings:
        md5[os.path.basename(p)] = _md5(p)
    return data, settings, md5


def sb99_quantities(runs):
    """Writer arrays from parsed Starburst99 runs (list ordered by Z).

    Returns:
        tuple: (t_myr, quantities, diagnostics)
    """
    t = runs[0]['power1']['Time'] / 1e6
    q = {k: [] for k in ('L_bol', 'Q_HI', 'Q_HeI', 'Q_HeII', 'L_ion', 'L_LyW', 'L_mech_wind',
                         'L_mech_SN', 'F_ram_wind', 'F_ram_SN')}
    max_mismatch = 0.0
    for d in runs:
        # The files print the time with different precision (snr1: 3 significant digits,
        # power1: 5), so the grids are compared to that precision; power1 sets the nodes.
        for suffix in ('snr1', 'quanta1', 'yield1', 'spectrum1'):
            tt = d[suffix]['Time']
            if tt.shape != d['power1']['Time'].shape or not np.allclose(tt, d['power1']['Time'], rtol=5e-3, atol=0):
                raise ValueError(f"{suffix} time grid differs from power1")
        if not np.array_equal(d['power1']['Time'] / 1e6, t):
            raise ValueError("time grids differ between metallicities")
        mdot_w = 10 ** d['yield1']['All_winds'] * (MSUN_G / YR_S)
        mdot_sn = 10 ** d['yield1']['All_SN'] * (MSUN_G / YR_S)
        p_w = 10 ** d['power1']['Power_All']
        p_sn = 10 ** d['snr1']['All_Power']
        p_tot = 10 ** d['snr1']['Stars_SN_Power']
        f_w = p_w / (p_w + p_sn)
        max_mismatch = max(max_mismatch, float(np.max(np.abs(np.log10((p_w + p_sn) / p_tot)))))
        L_bol = 10 ** d['quanta1']['Log_Luminosity']
        q['L_bol'].append(L_bol)
        q['L_ion'].append(10 ** d['quanta1']['HI%'] * L_bol)
        q['Q_HI'].append(10 ** d['quanta1']['HI_rate'])
        q['Q_HeI'].append(10 ** d['quanta1']['HeI_rate'])
        q['Q_HeII'].append(10 ** d['quanta1']['HeII_rate'])
        lam = d['spectrum1']['Wavelength']
        band = (lam >= 995.0) & (lam <= 1005.0)
        q['L_LyW'].append(np.mean((10 ** d['spectrum1']['Stellar'])[:, band], axis=1))
        q['L_mech_wind'].append(p_tot * f_w)
        q['L_mech_SN'].append(p_tot * (1.0 - f_w))
        q['F_ram_wind'].append(np.sqrt(2 * p_w * mdot_w))
        q['F_ram_SN'].append(np.sqrt(2 * p_sn * mdot_sn))
    q = {k: np.array(v) for k, v in q.items()}
    return t, q, {'max_abs_log10_components_over_published_total': max_mismatch}


def sb99_imf_definition(settings, input_md5):
    """IMF definition and normalization from the Starburst99 run settings (``.input1``), the
    same for every metallicity (else ValueError)."""
    keys = ('XPONENT', 'XMASLIM', 'TOMA', 'ISF')
    first = None
    for z, s in settings.items():
        got = tuple(s.get(k) for k in keys)
        if None in got:
            raise ValueError(f"Z={z}: the .input1 lacks one of {keys}")
        if first is None:
            first = got
        elif got != first:
            raise ValueError(f"IMF settings differ between metallicities: {first} vs {got}")
    xp, xm, toma, isf = first
    if float(isf) > 0:
        raise ValueError("Starburst99 run is continuous star formation (ISF > 0); tables are bursts")
    exps = [float(x) for x in xp.split(',')]
    lims = [float(x) for x in xm.split(',')]
    mass = float(toma) * 1e6
    d = broken_power_law(exps, lims, mass)
    d['source'] = {'files': 'Starburst99 .input1 [XPONENT], [XMASLIM], [TOMA] (10^6 Msun)',
                   'input_files_md5': input_md5}
    norm = {'rule': 'per 1e6 Msun of the full IMF (Starburst99 fixed-mass burst)', 'per_mass_msun': 1e6,
            'population_mass_msun': mass, 'm_low_msun': lims[0], 'm_up_msun': lims[-1]}
    return d, norm


def build_sb99_table(run_dirs, imf, out_dir, Z=None, template='SB99'):
    """
    Build an SB99 feedback table from Starburst99 runs, one per metallicity.

    Args:
        run_dirs (dict): Z (mass fraction) -> (run directory, model name), e.g.
            ``{0.02: ('.../Z_020', 'lim_100_Z_.020'), ...}``.
        imf (str): IMF label of the table (file ``SB99_<imf>_feedback.h5``).
        out_dir (str): output directory.
        Z (list, optional): metallicity order (default sorted keys of run_dirs).
        template (str): template name recorded in the table.

    Returns:
        FeedbackTable (written; ``.path``, ``.md5``).
    """
    Zs = sorted(run_dirs) if Z is None else list(Z)
    runs, settings, md5 = [], {}, {}
    for z in Zs:
        rd, model = run_dirs[z]
        d, s, m = read_sb99_run(rd, model)
        runs.append(d)
        settings[str(z)] = s
        md5.update({os.path.join(os.path.basename(os.path.normpath(rd)), k): v for k, v in m.items()})
    t, q, diag = sb99_quantities(runs)
    imf_def, norm = sb99_imf_definition(settings, {k: v for k, v in md5.items() if k.endswith('.input1')})
    s0 = settings[str(Zs[0])]
    meta = {
        'template': template, 'imf': imf, 'star_type': None,
        'source': {'code': 'Starburst99', 'input_files_md5': md5, 'run_settings': settings,
                   'adapter': 'toddlers.feedback_tables.sb99'},
        'imf_definition': imf_def,
        'normalization': norm,
        'conventions': {
            'L_mech': 'Starburst99 STARS + SUPERNOVAE power (snr1)',
            'components_rule': 'published total apportioned by P_wind / (P_wind + P_SN)',
            'F_ram': 'sqrt(2 P_wind Mdot_wind) + sqrt(2 P_SN Mdot_SN)',
            'L_ion': '10**(H I % of L) x L_bol (quanta1)',
            'Q_HeI_Q_HeII': 'quanta1 He I and He II rates (nested bands, galaxy.f)',
            'L_LyW': 'mean stellar spectrum over its nodes in 995-1005 A',
            'constants': {'Msun_g': MSUN_G, 'yr_s': YR_S},
            'wind_v_inf': 'Starburst99 internal (IWIND setting in run_settings)',
            'E_SN': 'Starburst99 internal',
            'sn_cut_msun': s0.get('SNCUT'), 'bh_cut_msun': s0.get('BHCUT'),
            'time_grid_source_resolution': 'Starburst99 linear time step TBIV (run_settings)',
            **diag,
        },
    }
    return write_feedback_table(t, np.array(Zs, dtype=float), q, meta, out_dir, legacy=True)
