"""Consistency checks between the feedback interpolants, StellarFeedback and the Cloudy star tables.

Cloudy takes the spectral SHAPE from the star table and the normalization from the
interpolant (``luminosity total = L_bol``), so the ionizing photon rate Cloudy sees is
L_bol,interp * (Q/L)_table. These tests check that chain.
"""
import io
import contextlib
import math
import os
import pickle

import numpy as np
import pytest

from toddlers.constants import M_SUN, MYR_TO_SEC, PC_TO_CM, CLOUDY_DATA_DIR
from toddlers._paths import get_data_dir
from toddlers.stellar_feedback import StellarFeedback

pytestmark = pytest.mark.data

H_CGS = 6.62607015e-27
C_ANGSTROM = 2.99792458e18
LYMAN_EDGE = 911.75


def _read_star_table(path):
    with open(path) as f:
        lines = f.readlines()
    n_mod, n_wl = int(lines[5]), int(lines[6])
    toks = np.array(' '.join(lines[11:]).split(), dtype=float)
    age_logz = toks[:2 * n_mod].reshape(n_mod, 2)
    i = 2 * n_mod
    lam = toks[i:i + n_wl] * float(lines[8])
    flux = toks[i + n_wl:i + n_wl + n_mod * n_wl].reshape(n_mod, n_wl) * float(lines[10])
    return age_logz[:, 0], age_logz[:, 1], lam, flux


def _l_and_q(lam, f):
    m = lam < LYMAN_EDGE
    return np.trapz(f, lam), np.trapz(lam[m] * f[m] / (H_CGS * C_ANGSTROM), lam[m])


def _quiet_feedback(**kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return StellarFeedback(**kw)


@pytest.mark.parametrize("template,imf,star_type,Z", [
    ("SB99", "kroupa100", None, 0.02),
    ("BPASS", "chab100", "bin", 0.014),
    ("pySB99", "kroupa_100", None, 0.014),
])
@pytest.mark.parametrize("eta_sf", [0.05, 1.0])
def test_feedback_units_chain(template, imf, star_type, Z, eta_sf):
    """A 1e6 Msun burst returns exactly 10**interpolant on the precalc grid, through the
    getters and through get_feedback_data_per_generation (the path the Cloudy input uses)."""
    sf = _quiet_feedback(template=template, Z=Z, M_cl_init=1e6 / eta_sf * M_SUN, eta_sf=eta_sf,
                         t_list_collapse=[], imf=imf, star_type=star_type)
    assert sf.M_stellar == pytest.approx(1e6, rel=1e-12)
    for t_myr in (1.0, 3.0, 10.0):
        t = t_myr * MYR_TO_SEC
        ref = {k: 10 ** float(getattr(sf, k + '_interpolant')(t_myr, sf.Z)[0])
               for k in ('L_bolo', 'Q_ion', 'L_mech', 'F_ram', 'L_ion')}
        assert sf.get_bolometric_luminosity(t) == pytest.approx(ref['L_bolo'], rel=1e-6)
        assert sf.get_ionizing_photon_rate(t) == pytest.approx(ref['Q_ion'], rel=1e-6)
        assert sf.get_mechanical_luminosity(t) == pytest.approx(ref['L_mech'], rel=1e-6)
        assert sf.get_ram_force(t) == pytest.approx(ref['F_ram'], rel=1e-6)
        pg = sf.get_feedback_data_per_generation(t)
        assert pg['Q_i'][0] == pytest.approx(ref['Q_ion'], rel=1e-6)
        assert pg['L_i'][0] + pg['L_n'][0] == pytest.approx(ref['L_bolo'], rel=1e-6)


def test_sb99_star_table_matches_interpolant():
    """SB99 kroupa100 star table (if present in CLOUDY_DATA_DIR): (Q/L)_table x L_interp
    within 0.01 dex of Q_interp, and the table L (x 4 pi (10 pc)^2) within 1% of L_interp,
    at table nodes near 1, 3, 10 Myr and Z = 0.004, 0.020."""
    path = os.path.join(CLOUDY_DATA_DIR, 'SB99_kroupa100_sin_burst.ascii')
    if not os.path.exists(path):
        pytest.skip(f"star table not present: {path}")
    ages, logz, lam, flux = _read_star_table(path)
    from toddlers.feedback_tables import load_table
    tab = load_table('SB99', 'kroupa100')
    obj = {2: tab.log_interpolant('L_bol'), 4: tab.log_interpolant('Q_HI')}
    u = 4 * np.pi * (10 * PC_TO_CM) ** 2
    for Z in (0.004, 0.020):
        idx = np.where(np.isclose(logz, np.log10(Z), atol=1e-4))[0]
        for a in (1.0, 3.0, 10.0):
            j = idx[np.argmin(np.abs(np.log10(ages[idx] / (a * 1e6))))]
            L, Q = _l_and_q(lam, flux[j])
            Li = 10 ** float(obj[2](ages[j] / 1e6, Z)[0])
            Qi = 10 ** float(obj[4](ages[j] / 1e6, Z)[0])
            assert abs(math.log10(Li * Q / L / Qi)) <= 0.01
            assert abs(L * u / Li - 1) <= 0.01


@pytest.mark.parametrize("Z", [0.0004, 0.01, 0.03])
def test_pysb99_off_grid_metallicity_raises(Z):
    """pySB99 follows the SB99/BPASS rule: Z must be a node of the interpolant's Z axis
    (below, between and above the kroupa_100 nodes 0.006, 0.014, 0.020)."""
    with pytest.raises(ValueError):
        _quiet_feedback(template='pySB99', Z=Z, M_cl_init=2e7 * M_SUN, eta_sf=0.05,
                        t_list_collapse=[], imf='kroupa_100')


def test_pysb99_grid_metallicity_accepted():
    sf = _quiet_feedback(template='pySB99', Z=0.014, M_cl_init=2e7 * M_SUN, eta_sf=0.05,
                         t_list_collapse=[], imf='kroupa_100')
    assert sf.Z == 0.014


@pytest.mark.slow
@pytest.mark.parametrize("imf", ["kroupa_100", "topheavy_100", "example_massive"])
def test_pysb99_star_table_matches_interpolant(tmp_path, monkeypatch, imf):
    """A pySB99 table written now by SpectralTableGenerator: (Q/L)_table x L_interp within
    0.01 dex of Q_interp, and the table L (x 1e40, the pySB99 unit chain) within 1% of
    L_interp, at table nodes near 1, 3, 10 Myr, Z = 0.006 and 0.014. Covers an IMF
    population, a top-heavy one and a custom one (built with auto_extend_imf=True)."""
    import toddlers.cloudy_stellar_spectra_generator as csg
    monkeypatch.setattr(csg, 'CLOUDY_DATA_DIR', str(tmp_path))
    time_points = np.logspace(4, np.log10(5e7), 50)
    with contextlib.redirect_stdout(io.StringIO()):
        gen = csg.SpectralTableGenerator(template='pySB99', imf=imf, star_type='sin',
                                         max_age=time_points[-1])
        gen.generate_spectral_table(time_points=time_points, Z_values=[0.006, 0.014, 0.020])
    path = str(tmp_path / f'pySB99_{imf}_sin_burst.ascii')
    ages, logz, lam, flux = _read_star_table(path)
    from toddlers.feedback_tables import load_table
    tab = load_table('pySB99', imf)
    obj = {2: tab.log_interpolant('L_bol'), 4: tab.log_interpolant('Q_HI')}
    # the sidecar pairs the star table with this feedback table
    import json
    side = json.load(open(path + '.meta.json'))
    assert side['paired_feedback_md5'] == tab.md5
    assert side['sidecar_schema_version'] == 1 and side['units']['F_lambda'] == '1e40 erg s-1 Angstrom-1'
    worst, worst_l = 0.0, 0.0
    for Z in (0.006, 0.014):
        idx = np.where(np.isclose(logz, np.log10(Z), atol=1e-4))[0]
        for a in (1.0, 3.0, 10.0):
            j = idx[np.argmin(np.abs(np.log10(ages[idx] / (a * 1e6))))]
            L, Q = _l_and_q(lam, flux[j])
            Li = 10 ** float(obj[2](ages[j] / 1e6, Z)[0])
            Qi = 10 ** float(obj[4](ages[j] / 1e6, Z)[0])
            worst = max(worst, abs(math.log10(Li * Q / L / Qi)))
            worst_l = max(worst_l, abs(L * 1e40 / Li - 1))
    assert worst <= 0.01, f"max |dlog Q| = {worst:.4f} dex"
    assert worst_l <= 0.01, f"max |L_table/L_interp - 1| = {worst_l:.4f}"


def test_pysb99_table_requires_auto_extend_in_metadata(tmp_path, monkeypatch):
    """A generator configuration without auto_extend_imf (IMF population) cannot define the
    table's IMF upper limit, so the star-table generator refuses it instead of guessing."""
    import json, shutil, h5py
    import toddlers.cloudy_stellar_spectra_generator as csg
    shutil.copy(os.path.join(get_data_dir(), 'pySB99_kroupa_100_feedback.h5'), tmp_path)
    p = tmp_path / 'pySB99_kroupa_100_feedback.h5'
    with h5py.File(p, 'a') as f:
        src = json.loads(f.attrs['source'])
        src['generator_config'].pop('auto_extend_imf')
        f.attrs['source'] = json.dumps(src, sort_keys=True)
        del f.attrs['content_md5']
    monkeypatch.setenv('TODDLERS_DATA', str(tmp_path))
    with pytest.raises(ValueError, match='auto_extend_imf'):
        with contextlib.redirect_stdout(io.StringIO()):
            csg.SpectralTableGenerator(template='pySB99', imf='kroupa_100', star_type='sin')


def test_custom_population_config_records_auto_extend_true():
    """Custom-population tables are generated with auto_extend_imf=True (the IMF upper
    limit still selects the track set), so their configuration must say so."""
    from toddlers.pysb99.generate_pysb99_interpolants import _population_config
    from toddlers.feedback_tables import load_table

    class _Gen:
        metallicities, Z_values, spectral_library, rotation = ['MW'], [0.014], 'FW', False
        run_speed_mode = 'DEFAULT'
    cfg = _population_config(_Gen(), 'c', population_type='custom', custom_star_numbers={50.0: 10})
    assert cfg['auto_extend_imf'] is True and 'generation_date' not in cfg
    for name in ('example_massive', 'example_intermediate', 'example_lowmass'):
        tab = load_table('pySB99', name)
        assert tab.metadata['source']['generator_config']['auto_extend_imf'] is True, name
        assert tab.imf['definition']['kind'] == 'custom_population'


def test_pysb99_config_records_population_settings():
    from toddlers.pysb99.generate_pysb99_interpolants import _population_config
    from toddlers.feedback_tables import load_table

    class _Gen:
        metallicities, Z_values, spectral_library, rotation = ['MW'], [0.014], 'FW', False
        run_speed_mode = 'DEFAULT'
    cfg = _population_config(_Gen(), 'x', population_type='imf', imf_exponents=[1.3, 2.3],
                             imf_mass_limits=(0.1, 0.5, 100.0), auto_extend_imf=False)
    assert cfg['auto_extend_imf'] is False and cfg['run_speed_mode'] == 'DEFAULT'
    for name in ('kroupa_100', 'topheavy_100'):
        g = load_table('pySB99', name).metadata['source']['generator_config']
        assert g['auto_extend_imf'] is False and g['run_speed_mode'] == 'DEFAULT', name
