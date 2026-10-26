"""pySB99 core-collapse supernovae: energy and momentum in the mechanical feedback.

Checks, as unit tests where they fit the suite:

- G-b   SNe off: the mechanical outputs are the winds alone (bit-identical arrays).
- G-c   energy and momentum bookkeeping: sum_i L_SN(t_i) dt_i = E_SN N_SN and
        sum_i F_SN(t_i) dt_i = sum n p_SN, to 1e-10 relative, for an IMF population
        (spread deposition) and a custom one (point events).
- G-d   one SN per exploding star at its death time: the single-star database
        attributes, and the stochastic deposition (one star; N stars at uniform ages).
- G-f   a star above the SN window (150 Msun) collapses directly: no SN.
- G-a   the shipped kroupa_100 interpolant at Z = 0.020 matches the shipped SB99 one
        after 4 Myr: |median| <= 0.15 dex and every point <= 0.35 dex in each window.
- metadata: products record the SN treatment; files without the keys read as winds only.
- PCHIP: death time and final mass by monotone PCHIP in log-log through the Geneva track
  nodes (exact at the nodes); check R: rms second difference of log L_mech of the shipped
  kroupa_100 at Z = 0.020 (0.05 Myr bin centres, 5-30 Myr) <= 0.003 dex.

Physics: E_SN = 1e51 erg (Leitherer et al. 1999, Sect. 10); SN window 8-120 Msun (SB99
SNCUT/BHCUT defaults); ejecta M_final - 1.4 Msun; momentum sqrt(2 E_SN M_ej) per SN.
"""
import json
import os
import pickle
import types
import warnings

import h5py
import numpy as np
import pytest

pytestmark = pytest.mark.data

E_SN = 1e51
YEAR_S = 3.15576e7
MSUN_G = 1.989e33


def _synth(**kw):
    from toddlers.pysb99.pysb99_core import StellarPopulationConfig, StellarPopulationSynthesis
    cfg = StellarPopulationConfig(**kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return StellarPopulationSynthesis(cfg)


def _sn_only(syn):
    """Run the SN step on a winds-only stub (the SN term does not depend on spectra)."""
    ind = syn._get_timestep_0_ind(syn.times_steps_SB99[0])
    p0 = syn._get_timestep_params(syn.times_fine[0], ind)
    if syn.config.custom_star_numbers is not None:
        n = syn._create_custom_star_numbers(p0[3], syn.config.custom_star_numbers)
        edges = None
    else:
        n, _, _, xmhigh, xmlow = syn.calc_Nostars(p0[3], syn.config.imf_exponents,
                                                 syn.config.imf_mass_limits)
        edges = (np.asarray(xmlow, float), np.asarray(xmhigh, float))
    times = syn.times_fine
    res = types.SimpleNamespace(times=times, wind_power=np.full(len(times), 30.0),
                                wind_momentum=np.full(len(times), 25.0))
    syn._attach_supernovae(res, ind, n, edges)
    return res


@pytest.fixture(scope="module")
def kroupa_mwc_60myr():
    syn = _synth(total_mass=1e6, metallicity='MWC', imf_exponents=[1.3, 2.3],
                 imf_mass_limits=(0.1, 0.5, 100.0), auto_extend_imf=False,
                 time_end=60e6, time_step=0.1e6)
    return syn, _sn_only(syn)


def _bin_widths_s(times):
    from toddlers.pysb99.pysb99_core import StellarPopulationSynthesis
    return np.diff(StellarPopulationSynthesis.centred_bin_edges(times)) * YEAR_S


def test_imf_energy_and_momentum_bookkeeping(kroupa_mwc_60myr):
    """G-c, IMF rows (spread): every exploding row dies inside 0-60 Myr at Z = 0.020."""
    syn, res = kroupa_mwc_60myr
    ev = res.supernova_events
    assert ev['time_deposition'] == 'spread'
    n_exp = ev['n_stars'][ev['explodes']].sum()
    assert ev['t_death_yr'][ev['explodes']].max() < 59e6
    dt = _bin_widths_s(res.times)
    assert np.sum(res.sn_power * dt) == pytest.approx(E_SN * n_exp, rel=1e-10)
    p_tot = np.sum((ev['n_stars'] * ev['momentum_per_sn'])[ev['explodes']])
    assert np.sum(res.sn_momentum * dt) == pytest.approx(p_tot, rel=1e-10)
    # SN window: exactly the rows with 8 <= M_ini <= 120 explode; ejecta = M_final - 1.4
    m = ev['initial_mass']
    assert np.array_equal(ev['explodes'], (m >= 8.0 * (1 - 1e-3)) & (m <= 120.0 * (1 + 1e-3)))
    k = ev['explodes']
    assert np.allclose(ev['ejecta_mass'][k], ev['final_mass'][k] - 1.4)
    assert np.allclose(ev['momentum_per_sn'][k], np.sqrt(2 * E_SN * ev['ejecta_mass'][k] * MSUN_G))
    # Kroupa 0.1-100 Msun, 1e6 Msun: about one SN per 90 Msun (analytic 11132 for 8-100 Msun)
    assert n_exp == pytest.approx(11132.2, rel=1e-3)
    # no SN before the most massive star's death
    first = res.times[np.argmax(res.sn_number > 0)]
    assert first >= ev['t_death_yr'][k].min() - 0.05e6


def test_custom_population_point_events():
    """G-c and G-d for custom stars: point events in the bins holding each death time."""
    syn = _synth(total_mass=1e6, metallicity='MW', custom_star_numbers={20.0: 3, 9.0: 2},
                 time_end=40e6, time_step=0.1e6)
    res = _sn_only(syn)
    ev = res.supernova_events
    assert ev['time_deposition'] == 'point'
    pop = ev['n_stars'] > 0
    assert ev['n_stars'][pop].sum() == 5 and ev['explodes'][pop].all()
    dt = _bin_widths_s(res.times)
    assert np.sum(res.sn_power * dt) == pytest.approx(5 * E_SN, rel=1e-10)
    nz = np.nonzero(res.sn_number)[0]
    assert len(nz) == 2
    from toddlers.pysb99.pysb99_core import StellarPopulationSynthesis
    edges = StellarPopulationSynthesis.centred_bin_edges(res.times)
    for t_d, n in zip(ev['t_death_yr'][pop], ev['n_stars'][pop]):
        j = np.searchsorted(edges, t_d, side='right') - 1
        assert res.sn_number[j] == n


def test_nominal_8msun_track_explodes():
    """The MW "8 Msun" Geneva track starts at 7.999992 Msun: it is in the SN window."""
    syn = _synth(total_mass=80.0, metallicity='MW', custom_star_numbers={8.0: 10},
                 time_end=45e6, time_step=0.1e6)
    res = _sn_only(syn)
    ev = res.supernova_events
    k = ev['n_stars'] > 0
    assert ev['initial_mass'][k][0] < 8.0 and ev['explodes'][k][0]
    dt = _bin_widths_s(res.times)
    assert np.sum(res.sn_power * dt) == pytest.approx(10 * E_SN, rel=1e-10)


def test_direct_collapse_above_window():
    """G-f: a 150 Msun star (MW) is above the 120 Msun window: no SN energy or momentum."""
    syn = _synth(total_mass=150.0, metallicity='MW', custom_star_numbers={150.0: 1},
                 time_end=10e6, time_step=0.1e6)
    res = _sn_only(syn)
    ev = res.supernova_events
    k = ev['n_stars'] > 0
    assert ev['initial_mass'][k][0] > 120.0 and not ev['explodes'][k][0]
    assert np.all(res.sn_power == 0) and np.all(res.sn_momentum == 0)
    assert np.array_equal(res.mechanical_luminosity, res.wind_power)


def test_supernovae_off_is_winds_only():
    """G-b (in the suite): include_supernovae=False leaves the winds-only arrays."""
    syn = _synth(total_mass=1e6, metallicity='MWC', imf_exponents=[1.3, 2.3],
                 imf_mass_limits=(0.1, 0.5, 100.0), auto_extend_imf=False,
                 include_supernovae=False)
    res = _sn_only(syn)
    assert np.all(res.sn_power == 0) and np.all(res.sn_momentum == 0)
    assert np.array_equal(res.mechanical_luminosity, res.wind_power)
    assert np.array_equal(res.ram_force, res.wind_momentum)


def test_single_star_database_attributes():
    """G-d, database side: a 20 Msun MW star gets one SN at the end of its track."""
    from toddlers.pysb99.stochastic.database import _run_single_star, _supernova_attrs
    grid = np.linspace(1e-6, 50.0, 501)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _run_single_star(mass=20.0, metallicity='MW', time_grid_myr=grid)
    a = _supernova_attrs(res, 50.0)
    assert a['sn_fate'] == 'ccSN'
    assert a['t_death_myr'] == pytest.approx(8.731, abs=1e-3)   # Z014 20 Msun track end
    assert a['sn_energy_erg'] == E_SN
    assert a['sn_ejecta_mass_msun'] == pytest.approx(a['M_final_msun'] - 1.4)
    dt = _bin_widths_s(res.times)
    assert np.sum(res.sn_power * dt) == pytest.approx(E_SN, rel=1e-10)
    assert np.count_nonzero(res.sn_number) == 1


def _toy_database(path, with_sn=True):
    """A minimal h5 with the SN attributes of three MW masses (no feedback datasets)."""
    with h5py.File(path, 'w') as f:
        meta = f.create_group('metadata')
        if with_sn:
            meta.attrs.update({'sn_energy_erg': E_SN, 'sn_mass_min_msun': 8.0,
                               'sn_mass_max_msun': 120.0, 'sn_remnant_mass_msun': 1.4})
        z = f.create_group('Z_0.0140')
        z.attrs['Z_value'] = 0.014
        z.create_dataset('mass_grid', data=np.array([150.0, 20.0, 15.0]))   # as the real database
        for m, t_d, mf in ((15.0, 12.524, 13.25), (20.0, 8.731, 8.63), (150.0, 2.813, 41.16)):
            g = z.create_group(f'mass_{m:.1f}')
            g.attrs.update({'mass_Msun': m, 't_death_myr': t_d, 'M_final_msun': mf})


def test_stochastic_one_star_one_event(tmp_path):
    """G-d, stochastic side: one SN, in the bin holding t_d, carrying exactly E_SN."""
    from toddlers.pysb99.stochastic.interpolants import supernova_feedback
    db = str(tmp_path / 'toy.h5'); _toy_database(db)
    grid = np.arange(0.0, 30.05, 0.1)
    sn = supernova_feedback(np.array([20.0]), np.array([0.0]), grid, db, 'MW')
    dt = _bin_widths_s(grid * 1e6)
    assert np.count_nonzero(sn['sn_number']) == 1
    j = int(np.nonzero(sn['sn_number'])[0][0])
    assert grid[j] - 0.05 <= 8.731 < grid[j] + 0.05
    assert np.sum(sn['sn_power'] * dt) == pytest.approx(E_SN, rel=1e-10)
    p = np.sqrt(2 * E_SN * (8.63 - 1.4) * MSUN_G)
    assert np.sum(sn['sn_momentum'] * dt) == pytest.approx(p, rel=1e-10)
    # mass interpolation between database masses: PCHIP in log-log
    from scipy.interpolate import PchipInterpolator
    t_175 = 10 ** PchipInterpolator(np.log10([15.0, 20.0, 150.0]),
                                    np.log10([12.524, 8.731, 2.813]))(np.log10(17.5))
    sn2 = supernova_feedback(np.array([17.5]), np.array([0.0]), grid, db, 'MW')
    j2 = int(np.nonzero(sn2['sn_number'])[0][0])
    assert grid[j2] - 0.05 <= t_175 < grid[j2] + 0.05
    # N stars of one mass at uniform ages: N x E_SN in the grid; 150 Msun: no SN
    rng = np.random.default_rng(1)
    tau = rng.uniform(0.0, 10.0, 200)
    sn3 = supernova_feedback(np.full(200, 15.0), tau, grid, db, 'MW')
    assert np.sum(sn3['sn_power'] * dt) == pytest.approx(200 * E_SN, rel=1e-10)
    sn4 = supernova_feedback(np.array([150.0]), np.array([0.0]), grid, db, 'MW')
    assert sn4['n_exploding'] == 0 and np.all(sn4['sn_power'] == 0)


def test_database_of_another_version_is_loud(tmp_path):
    from toddlers.pysb99.stochastic.database import get_supernova_table
    db = str(tmp_path / 'legacy.h5'); _toy_database(db, with_sn=False)
    with pytest.raises(ValueError, match='not the single-star database of this version'):
        get_supernova_table(db, 'MW')


def test_metadata_records_supernovae():
    from toddlers._paths import get_data_dir
    from toddlers.pysb99.generate_pysb99_interpolants import product_includes_supernovae
    from toddlers.feedback_tables import load_table
    md = load_table('pySB99', 'kroupa_100').metadata['source']['generator_config']
    assert product_includes_supernovae(md)
    assert md['sn_energy_erg'] == E_SN and md['sn_mass_max_msun'] == 120.0
    assert not product_includes_supernovae({'population_name': 'legacy'})


def test_kroupa_mechanical_feedback_matches_sb99_after_4myr():
    """G-a on the shipped interpolants (Z = 0.020)."""
    from toddlers._paths import get_data_dir
    from toddlers.feedback_tables import load_table
    tabs = [load_table('SB99', 'kroupa100'), load_table('pySB99', 'kroupa_100')]
    sb, py = [[t_.log_interpolant('L_mech'), t_.log_interpolant('F_ram')] for t_ in tabs]
    t = np.arange(0.1, 31.0, 0.1)
    for q in (0, 1):
        d_ = np.array([float(py[q](x, 0.020)[0]) - float(sb[q](x, 0.020)[0]) for x in t])
        for lo, hi in ((4.0, 5.0), (5.0, 10.0), (10.0, 30.01)):
            s = (t >= lo - 1e-9) & (t < hi)
            assert abs(np.median(d_[s])) <= 0.15, (q, lo, np.median(d_[s]))
            assert np.max(np.abs(d_[s])) <= 0.35, (q, lo, np.max(np.abs(d_[s])))


def test_death_time_pchip_loglog_through_track_nodes():
    """PCHIP: t_d and M_final are PCHIP in (log m, log y) through the track nodes, exact at nodes."""
    from scipy.interpolate import PchipInterpolator
    syn = _synth(total_mass=1e6, metallicity='MWC', imf_exponents=[1.3, 2.3],
                 imf_mass_limits=(0.1, 0.5, 100.0), auto_extend_imf=False)
    m0, t_end, m_fin = syn._sn_track_nodes()
    f_t, f_m = syn._sn_interpolators()
    assert np.array_equal(f_t(m0), t_end) and np.array_equal(f_m(m0), m_fin)
    ref = 10 ** PchipInterpolator(np.log10(m0), np.log10(t_end))(np.log10(17.5))
    assert float(f_t(np.array([17.5]))[0]) == pytest.approx(ref, rel=1e-12)
    # the 20 Msun Z = 0.020 track ends at 8.577 Myr
    k = np.argmin(abs(m0 - 20.0))
    assert t_end[k] / 1e6 == pytest.approx(8.577, abs=1e-3)


def test_kroupa_mechanical_feedback_is_smooth():
    """Check R on the shipped kroupa_100 (Z = 0.020): no sawtooth from the lifetime interpolation."""
    from toddlers._paths import get_data_dir
    from toddlers.feedback_tables import load_table
    ob = [load_table('pySB99', 'kroupa_100').log_interpolant('L_mech')]
    e = np.arange(0.0, 31.0 + 1e-9, 0.05)
    tc = 0.5 * (e[1:] + e[:-1])
    t = tc[(tc > 5) & (tc < 30)]
    x = np.array([float(ob[0](tt, 0.020)[0]) for tt in t])
    assert np.sqrt(np.mean(np.diff(x, 2) ** 2)) <= 0.003
