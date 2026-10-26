"""Run-time stochastic feedback build against the single-star database (data tests).

Pass rules:

B1  The grouped in-memory build (population.build_generation_table, snap) equals the
    per-star reference compute_time_delayed_feedback(interpolation_mode='snap') on the same
    stars (>= 0.8 Msun, burst and spread births), once both are put on the same mass
    normalization: every StellarFeedback quantity within rel 1e-12 at every time point.
B2  Stars below the track minimum add mass but no light: appending 0.5 Msun stars to a
    population changes the per-1e6 table by exactly log10(M_old / M_new) (abs 1e-12 dex).
S4  Same base seed and inputs, built twice from scratch (caches cleared): identical
    tables (array_equal) and identical StellarFeedback quantities.
S3b In a two-generation stochastic StellarFeedback, the generations have different star
    lists, generation 1 is the fresh draw of (base seed, 1), and a StellarFeedback rebuilt
    from the saved populations returns identical feedback.

The generation tables go through the unified table writer, so (i) L_ion is the < 912 A
luminosity: B1 compares it with the reference's L_ion_HI, not with the nested sum
L_ion; (ii) a zero node is floored at 1e-50 per 1e6 Msun (the schema floor): B2 and B3
skip floored nodes (B3 checks they sit at the floor).
"""
import numpy as np
import pytest

from toddlers.constants import M_SUN, MYR_TO_SEC
from toddlers.pysb99.stochastic import population as P
from toddlers.pysb99.stochastic.interpolants import compute_time_delayed_feedback
from toddlers.stellar_feedback import StellarFeedback

pytestmark = pytest.mark.data

KROUPA = P.IMFSpec('kroupa', 0.1, 100.0)
Z = 0.014
MAP = {'L_mech': 'wind_power', 'F_ram': 'wind_momentum', 'L_bolo': 'L_bol',
       'L_ion': 'L_ion_HI', 'Q_ion': 'Q_ion', 'L_LyW': 'L_LyW'}
FLOOR_LOG = -50.0     # log10 of the unified-table floor (toddlers.feedback_tables.FLOOR)


def _pop(spec, masses, births, sf_mode='burst', tau=None):
    return P.GenerationPopulation(
        masses_msun=np.asarray(masses, float), birth_times_myr=np.asarray(births, float),
        imf_name='kroupa', imf_m_low_msun=0.1, imf_m_up_msun=100.0, base_seed=0,
        generation_index=0, target_mass_msun=float(np.sum(masses)), sf_mode=sf_mode,
        tau_sf_myr=tau, metallicity_Z=Z, track_min_mass_msun=0.8,
        interpolation_mode='snap', database_md5=spec.database_md5(), seed_spawn_key=[0])


@pytest.mark.slow
@pytest.mark.parametrize("spread", [False, True])
def test_B1_grouped_build_equals_per_star_reference(spread):
    spec = P.StochasticSpec(KROUPA, 0, time_step_myr=0.1)
    m = P.draw_stop_after(KROUPA, 300.0, P.generation_rng(4, 0))
    m = m[m >= 0.8][:40]
    b = (np.random.default_rng(1).uniform(0, 2.0, len(m)) if spread else np.zeros(len(m)))
    pop = _pop(spec, m, b, 'constant_sfr' if spread else 'burst', 2.0 if spread else None)
    P._TABLE_CACHE.clear()
    tab = P.build_generation_table(pop, spec)
    ref = compute_time_delayed_feedback(m, b, tab.time_grid_myr, spec.database_path, 'MW',
                                        n_processes=1, verbose=False, interpolation_mode='snap')
    for k, rk in MAP.items():
        a = 10 ** tab.log[k]          # same normalization: all stars are >= 0.8 Msun here
        r = 10 ** ref[rk]
        lit = r > 1e-90
        assert np.allclose(a[lit], r[lit], rtol=1e-12, atol=0), (k, np.max(np.abs(a[lit] / r[lit] - 1)))


def test_B2_sub_track_stars_add_mass_not_light():
    spec = P.StochasticSpec(KROUPA, 0)
    m = P.draw_stop_after(KROUPA, 500.0, P.generation_rng(9, 0))
    lit = m[m >= 0.8]
    a = P.build_generation_table(_pop(spec, lit, np.zeros(len(lit))), spec)
    extra = np.concatenate([lit, np.full(200, 0.5)])
    b = P.build_generation_table(_pop(spec, extra, np.zeros(len(extra))), spec)
    shift = np.log10(lit.sum() / extra.sum())
    for k in a.log:
        ok = (a.log[k] > FLOOR_LOG) & (b.log[k] > FLOOR_LOG)
        assert np.allclose(b.log[k][ok] - a.log[k][ok], shift, rtol=0, atol=1e-12), k


def test_S4_build_reproducible():
    spec = P.StochasticSpec(KROUPA, 23)
    out = []
    for _ in range(2):
        P._TABLE_CACHE.clear()
        P._INTERP_CACHE.clear()
        sf = StellarFeedback('pySB99', Z, 2e4 * M_SUN, 0.05, [], stochastic=spec)
        out.append((sf.generation_tables[0].log,
                    [sf.get_ionizing_photon_rate(t * MYR_TO_SEC) for t in (0.5, 3.0, 12.0)]))
    for k in out[0][0]:
        assert np.array_equal(out[0][0][k], out[1][0][k]), k
    assert out[0][1] == out[1][1]


def test_S3b_generations_and_rebuild_from_saved():
    spec = P.StochasticSpec(KROUPA, 11)
    sf = StellarFeedback('pySB99', Z, 4e4 * M_SUN, 0.05, [4 * MYR_TO_SEC], stochastic=spec)
    p0, p1 = sf.populations
    assert not (len(p0.masses_msun) == len(p1.masses_msun)
                and np.array_equal(p0.masses_msun, p1.masses_msun))
    assert np.array_equal(p1.masses_msun,
                          P.draw_stop_after(KROUPA, p1.target_mass_msun, P.generation_rng(11, 1)))
    recs = P.population_records(sf.populations)
    pops = P.populations_from_records(recs)
    sf2 = StellarFeedback('pySB99', Z, 4e4 * M_SUN, 0.05, [4 * MYR_TO_SEC], stochastic=spec,
                          populations=pops)
    for t in (0.5, 3.0, 4.5, 9.0):
        assert sf2.get_ionizing_photon_rate(t * MYR_TO_SEC) == sf.get_ionizing_photon_rate(t * MYR_TO_SEC)
    # a tampered list is refused
    pops[1].masses_msun = pops[1].masses_msun[::-1].copy()
    with pytest.raises(ValueError):
        StellarFeedback('pySB99', Z, 4e4 * M_SUN, 0.05, [4 * MYR_TO_SEC], stochastic=spec,
                        populations=pops)


def test_B3_q_ion_is_the_hydrogen_ionizing_rate():
    """pySB99's ionizing bands are nested (hi_ionizing_flux integrates every lambda <= 912 A,
    hei <= 504 A, heii <= 228 A; pysb99_core.py:1730-1745), and the deterministic pySB99
    tables store Q_ion = Q_HI (generate_pysb99_interpolants.py:497-498). The stochastic
    Q_ion must be Q_HI too (summing the three bands counts He-ionizing photons twice or
    three times). Rule: table Q_ion equals the Q_HI component to rel 1e-12, in the run-time
    build and in compute_time_delayed_feedback."""
    spec = P.StochasticSpec(KROUPA, 0, time_step_myr=0.1)
    m = P.draw_stop_after(KROUPA, 300.0, P.generation_rng(4, 0))
    m = m[m >= 0.8]
    P._TABLE_CACHE.clear()
    tab = P.build_generation_table(_pop(spec, m, np.zeros(len(m))), spec)
    drawn = float(np.sum(m))
    lit = tab.linear_components['Q_HI'] * 1e6 / drawn > 1e-50
    assert np.allclose(10 ** tab.log['Q_ion'][lit] * drawn / 1e6, tab.linear_components['Q_HI'][lit],
                       rtol=1e-12, atol=0)
    assert np.all(tab.log['Q_ion'][~lit] == FLOOR_LOG)
    few = m[:5]
    ref = compute_time_delayed_feedback(few, np.zeros(len(few)), np.array([0.5, 1.0, 2.0]),
                                        spec.database_path, 'MW', n_processes=1, verbose=False)
    assert np.allclose(ref['Q_ion'], ref['Q_HI'], rtol=0, atol=1e-12)


def test_L_ion_is_the_HI_band_luminosity():
    """Stochastic build: a generation table's L_ion is the sum over its stars of the
    < 912 A luminosity L_ion_HI (rel 1e-12 per node above the floor)."""
    spec = P.StochasticSpec(KROUPA, 17)
    pop = P.draw_generation(spec, 0, 300.0, 'burst', Z)
    P._TABLE_CACHE.clear()
    tab = P.build_generation_table(pop, spec)
    L_HI = tab.linear_components['L_ion_HI'] * 1e6 / pop.drawn_mass_msun
    ok = L_HI > 1e-50
    r = tab.feedback_table.values['L_ion'][0][ok] / L_HI[ok] - 1
    assert ok.sum() > 1000 and np.max(np.abs(r)) <= 1e-12


@pytest.mark.slow
def test_M1_star_table_keeps_its_sidecar(tmp_path):
    """M1: the generation's star table is written in place with its sidecar
    <name>.meta.json, which pairs it with the generation's feedback table (content md5); no
    .part file and no <name>.part.meta.json is left."""
    import json, os
    spec = P.StochasticSpec(KROUPA, 17)
    pop = P.draw_generation(spec, 0, 60.0, 'burst', Z)
    name = P.ensure_star_table(pop, spec, output_dir=str(tmp_path))
    files = sorted(os.listdir(tmp_path))
    assert files == [name, name + '.meta.json'], files
    side = json.load(open(tmp_path / (name + '.meta.json')))
    assert side['filename'] == name and side['sidecar_schema_version'] == 1
    assert side['paired_feedback_content_md5'] == P.build_generation_table(pop, spec).feedback_table.content_md5()
