"""Supernova point events reach the evolution with their full energy and momentum.

Custom star lists (and stochastic draws) deposit each supernova in one centred time bin.
Interpolated in log10 between that spike and its neighbours, such a bin would deliver only
part of its energy, so a point-event table reads its SN channel linearly in t, which
conserves each bin's energy through the precalculation at any generation start, also off
the 0.025 Myr precalculation grid. IMF tables (spread deposition), SB99 and BPASS keep the
log10 rule. Bars: delivered SN energy and momentum equal the table's within 1e-9 in every
generation; the shipped custom table delivers 4 SNe x 1e51 erg per population.
"""
import copy

import numpy as np
import pytest

import toddlers.feedback_tables as ftmod
from toddlers.constants import MYR_TO_SEC, M_SUN
from toddlers.exceptions import ConfigurationError
from toddlers.feedback_tables.table import _PointEventTotal
from toddlers.stellar_feedback import StellarFeedback

AGE_MAX = 30.0
M_CL, ETA = 1e5, 0.1
_read = ftmod.read_table_file


def _no_sn(path):
    t = copy.copy(_read(path))
    t.values = {k: (np.zeros_like(v) if k in ('L_mech_SN', 'F_ram_SN') else v) for k, v in t.values.items()}
    t._log_cache = {}
    return t


def _feedback(monkeypatch, imf, t_starts, no_sn=False, template='pySB99', Z=0.014):
    monkeypatch.setattr(ftmod, 'read_table_file', _no_sn if no_sn else _read)
    sf = StellarFeedback(template, Z, M_CL * M_SUN, ETA, [t * MYR_TO_SEC for t in t_starts[1:]], imf=imf)
    monkeypatch.setattr(ftmod, 'read_table_file', _read)
    return sf


def _exact(x, y, a, b):
    xs = np.union1d(x[(x > a) & (x < b)], [a, b])
    return float(np.trapz(np.interp(xs, x, y), xs))


def _delivered(sf, handle, i):
    """Integral over ages [0, AGE_MAX] of generation i as the precalculation samples it."""
    tk, t0 = np.asarray(sf.time_range), sf.t_start_list[i]
    first = int(np.searchsorted(tk, t0))
    ages = [(tk[first - 1] - t0) / MYR_TO_SEC] if first > 0 else []
    vals = [0.0] if first > 0 else []
    for t in tk[first:]:
        ages.append((t - t0) / MYR_TO_SEC)
        vals.append(float(np.squeeze(sf._get_quantity_burst(t - t0, getattr(sf, handle), i))))
    return _exact(np.array(ages), np.array(vals), 0.0, AGE_MAX) * MYR_TO_SEC


@pytest.mark.parametrize('t_starts', [[0.0, 1.0125, 4.0123], [0.0, 1.0, 4.0]], ids=['off_grid', 'on_grid'])
@pytest.mark.parametrize('imf', ['example_massive', 'example_intermediate'])
def test_sn_energy_and_momentum_reach_every_generation(monkeypatch, imf, t_starts):
    full = _feedback(monkeypatch, imf, t_starts)
    bare = _feedback(monkeypatch, imf, t_starts, no_sn=True)
    tab = full.feedback_table
    zi = tab.z_index(0.014)
    per = tab.metadata['normalization']['per_mass_msun']
    for i in range(len(t_starts)):
        for handle, q in (('L_mech_interpolant', 'L_mech_SN'), ('F_ram_interpolant', 'F_ram_SN')):
            got = _delivered(full, handle, i) - _delivered(bare, handle, i)
            ref = _exact(tab.t, tab.values[q][zi] * full.M_stellar_list[i] / per, 0.0, AGE_MAX) * MYR_TO_SEC
            assert got == pytest.approx(ref, rel=1e-9), (i, q)


def test_custom_population_delivers_its_supernovae(monkeypatch):
    sf = _feedback(monkeypatch, 'example_massive', [0.0])
    bare = _feedback(monkeypatch, 'example_massive', [0.0], no_sn=True)
    m_pop = sf.feedback_table.metadata['normalization']['population_mass_msun_per_Z']['MW']
    e = _delivered(sf, 'L_mech_interpolant', 0) - _delivered(bare, 'L_mech_interpolant', 0)
    # 1 x 100, 1 x 30 and 2 x 15 Msun: four SNe of 1e51 erg, all before 30 Myr
    assert e == pytest.approx(4e51 * M_CL * ETA / m_pop, rel=1e-9)


@pytest.mark.parametrize('imf', ['kroupa_100', 'topheavy_100'])
def test_spread_tables_keep_the_log_rule(imf):
    t = ftmod.read_table_file(ftmod.find_table('pySB99', imf, None))
    assert t.metadata['conventions']['sn_time_deposition_mode'] == 'spread'
    assert not isinstance(t.feedback_interpolant('L_mech'), _PointEventTotal)


def test_custom_tables_are_point_event_tables():
    for imf in ('example_massive', 'example_intermediate', 'example_lowmass'):
        t = ftmod.read_table_file(ftmod.find_table('pySB99', imf, None))
        assert t.sn_point_events
        assert isinstance(t.feedback_interpolant('F_ram'), _PointEventTotal)


def _checker():
    sf = StellarFeedback.__new__(StellarFeedback)
    sf.derivative_dt = 0.025 * MYR_TO_SEC
    return sf


def test_point_event_table_off_the_precalc_step_is_refused():
    t = copy.copy(ftmod.read_table_file(ftmod.find_table('pySB99', 'example_massive', None)))
    t.t = np.arange(len(t.t)) * 0.03
    with pytest.raises(ConfigurationError, match='whole multiples'):
        _checker()._check_point_event_grid(t)


def test_table_without_a_recorded_mode_warns():
    t = copy.copy(ftmod.read_table_file(ftmod.find_table('pySB99', 'example_massive', None)))
    t.metadata = copy.deepcopy(t.metadata)
    del t.metadata['conventions']['sn_time_deposition_mode']
    assert not t.sn_point_events
    with pytest.warns(UserWarning, match='sn_time_deposition_mode'):
        _checker()._check_point_event_grid(t)


@pytest.mark.parametrize('t_starts', [[0.0], [0.0, 1.0125]], ids=['gen0', 'off_grid_gen1'])
@pytest.mark.parametrize('frac', [0.0, 0.5, 0.9])
def test_force_derivative_is_the_derivative_of_the_precalculated_force(monkeypatch, t_starts, frac):
    """At nodes AND between them (mid-segment), the derivative the
    phase-1 pressure equation uses equals the slope of the F_ram (and L_mech) the solver sees."""
    sf = _feedback(monkeypatch, 'example_massive', t_starts)
    dt = sf.derivative_dt
    for t in sf.time_range[100:1300:7] + frac * dt:
        tk = np.floor(t / dt + 1e-9) * dt
        for f, df in ((sf.get_ram_force, sf.get_ram_force_derivative),
                      (sf.get_mechanical_luminosity, sf.get_mechanical_luminosity_derivative)):
            slope = (f(tk + dt) - f(tk)) / dt
            assert df(t) == pytest.approx(slope, rel=1e-6, abs=1e-30)


def test_force_derivative_integrates_to_the_force(monkeypatch):
    sf = _feedback(monkeypatch, 'example_massive', [0.0, 1.0125])
    t = sf.time_range[80:1400]
    d = np.array([sf.get_ram_force_derivative(x) for x in t[:-1]])
    assert np.sum(d * np.diff(t)) == pytest.approx(sf.get_ram_force(t[-1]) - sf.get_ram_force(t[0]), rel=1e-9)
