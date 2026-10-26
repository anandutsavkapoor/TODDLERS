"""Run-time stochastic populations (toddlers.pysb99.stochastic.population): checks S1, S3,
S4 (star lists), S6a, D5 (no re-convolution) and the
writer's population-record checks. No database needed. Stochastic runs draw at run time
only; the refusal of pre-drawn input tables is tested in test_stochastic_runtime_only.py.

Pass rules:

S1  200 draws of 1e4 Msun, Kroupa 0.1-100 Msun, streams generation_rng(1000 + k, 0).
    PASS if the mean number of stars per 1e3 Msun of target AND the mean mass fraction
    in stars >= 0.8 Msun each lie within 3 sigma (sigma = sample sd / sqrt(200)) of the
    analytic values, computed here by quadrature of the IMF (expected 1568.1 and 0.6121).
    Stop-after adds ~0.6 star per 1e3 Msun at 1e4 Msun (0.2 sigma); not corrected.
    Known-bad pre-check: the same rule on 0.8-100 Msun draws (a truncated range) must FAIL.
S3  Two generations of one run have different star lists, and each equals the draw from
    generation_rng(base_seed, i).
S4  Same base seed and inputs twice: identical star lists (array_equal).
S6a Constant SFR: at every birth, |mass formed by t - SFR t| <= m_k / 2 (midpoint rule),
    and 0 <= drawn - target < last star mass.

D5  (run-time) A two-generation constant-SFR
    stochastic run whose generation tables include the birth spread (ages included) returns,
    for each generation, the table value times drawn mass / 1e6 at rel 1e-12: no second
    convolution. The table builder is replaced by the synthetic one (no database).

Writer record checks:
U-IMF-b  A recorded drawn mass that is not the star-list sum is refused by the writer
    (rel 1e-12); a normalization m_low set to the track minimum is refused.
"""
import numpy as np
import pytest
from scipy.integrate import quad

from toddlers.constants import M_SUN, MYR_TO_SEC
from toddlers.pysb99.stochastic.population import (
    IMFSpec, GenerationPopulation, birth_times, draw_stop_after, generation_rng,
    StochasticSpec, generation_table_from_quantities)
from toddlers.pysb99.stochastic import population as P
from toddlers.stellar_feedback import StellarFeedback

KROUPA = IMFSpec('kroupa', 0.1, 100.0)


def _analytic(imf, m_split=0.8):
    xi = lambda m: float(imf.pdf(np.array([m]))[0])
    pts = [0.5] if imf.m_low < 0.5 < imf.m_up else None
    n = quad(xi, imf.m_low, imf.m_up, points=pts, limit=200)[0]
    mass = quad(lambda m: m * xi(m), imf.m_low, imf.m_up, points=pts, limit=200)[0]
    lo = max(imf.m_low, m_split)
    high = quad(lambda m: m * xi(m), lo, imf.m_up, limit=200)[0]
    return 1e3 * n / mass, high / mass


def _s1_rule(imf, analytic_imf=KROUPA, n_draws=200, target=1e4):
    n_per_1e3, frac = [], []
    for k in range(n_draws):
        m = draw_stop_after(imf, target, generation_rng(1000 + k, 0))
        n_per_1e3.append(len(m) * 1e3 / target)
        frac.append(m[m >= 0.8].sum() / m.sum())
    n_an, f_an = _analytic(analytic_imf)
    zn = (np.mean(n_per_1e3) - n_an) / (np.std(n_per_1e3, ddof=1) / np.sqrt(n_draws))
    zf = (np.mean(frac) - f_an) / (np.std(frac, ddof=1) / np.sqrt(n_draws))
    return abs(zn) <= 3 and abs(zf) <= 3, (np.mean(n_per_1e3), n_an, zn, np.mean(frac), f_an, zf)


def test_S1_analytic_values():
    n_an, f_an = _analytic(KROUPA)
    assert abs(n_an - 1568.1) < 0.1 and abs(f_an - 0.6121) < 1e-4


def test_S1_imf_counts():
    ok, info = _s1_rule(KROUPA)
    print("S1 full range: mean N/1e3 %.2f (analytic %.2f, z %.2f); f>=0.8 %.5f (analytic %.5f, z %.2f)" % info)
    assert ok, info


def test_S1_known_bad_truncated_range_fails():
    ok, info = _s1_rule(IMFSpec('kroupa', 0.8, 100.0))
    print("S1 known-bad 0.8-100: mean N/1e3 %.2f (analytic %.2f, z %.2f); f>=0.8 %.5f (analytic %.5f, z %.2f)" % info)
    assert not ok


def test_imf_range_required_and_inside_support():
    with pytest.raises(TypeError):
        IMFSpec('kroupa')
    with pytest.raises(ValueError):
        IMFSpec('salpeter', 0.05, 100.0)
    with pytest.raises(ValueError):
        IMFSpec('kroupa', 1.0, 0.5)


def test_stop_after_tolerance():
    for s in range(20):
        m = draw_stop_after(KROUPA, 300.0, generation_rng(s, 3))
        over = m.sum() - 300.0
        assert 0.0 <= over < m[-1] <= 100.0


def test_S3_generations_independent_and_reproducible():
    a = draw_stop_after(KROUPA, 1e3, generation_rng(17, 0))
    b = draw_stop_after(KROUPA, 1e3, generation_rng(17, 1))
    assert not (len(a) == len(b) and np.array_equal(a, b))
    assert np.array_equal(a, draw_stop_after(KROUPA, 1e3, generation_rng(17, 0)))
    assert np.array_equal(b, draw_stop_after(KROUPA, 1e3, generation_rng(17, 1)))


def test_S4_reproducible_and_global_state_untouched():
    np.random.seed(5)
    before = np.random.random()
    np.random.seed(5)
    a = draw_stop_after(KROUPA, 2e3, generation_rng(31, 2))
    after = np.random.random()
    b = draw_stop_after(KROUPA, 2e3, generation_rng(31, 2))
    assert np.array_equal(a, b)
    assert before == after


def test_S6a_constant_sfr_mass_history():
    for s in range(10):
        target, tau = 5e3, 2.0
        m = draw_stop_after(KROUPA, target, generation_rng(s, 0))
        t = birth_times(m, 'constant_sfr', target, tau)
        sfr = target / tau
        assert np.all(np.diff(t) > 0)
        formed_after = np.cumsum(m)               # mass formed just after each birth
        formed_before = formed_after - m
        dev = np.maximum(np.abs(formed_after - sfr * t), np.abs(formed_before - sfr * t))
        assert np.all(dev <= m / 2 * (1 + 1e-12) + 1e-9)
        assert 0.0 <= m.sum() - target < m[-1]


# ---------------------------------------------------------------------------
# Synthetic generation tables (no database)
# ---------------------------------------------------------------------------
Z = 0.014


def _synthetic(drawn_target, sf_mode='burst', tau=None, seed=7, imf=KROUPA):
    m = draw_stop_after(imf, drawn_target, generation_rng(seed, 0))
    pop = GenerationPopulation(
        masses_msun=m, birth_times_myr=birth_times(m, sf_mode, drawn_target, tau),
        imf_name='kroupa', imf_m_low_msun=imf.m_low, imf_m_up_msun=100.0, base_seed=seed,
        generation_index=0, target_mass_msun=drawn_target, sf_mode=sf_mode, tau_sf_myr=tau,
        metallicity_Z=Z, track_min_mass_msun=0.8, interpolation_mode='snap',
        database_md5='synthetic', seed_spawn_key=[0])
    return pop, _synthetic_table(pop)


def _synthetic_table(pop, spec=None):
    t = np.arange(0, 1205) * 0.025                               # 0 .. 30.1 Myr
    shape = np.where(t < 4.0, 0.0, -2.0 * (t - 4.0) / 26.0)    # flat, then falling
    q = {'L_mech_wind': 10 ** (40 + shape), 'L_mech_SN': np.zeros_like(t), 'F_ram_wind': 10 ** (31 + shape),
         'F_ram_SN': np.zeros_like(t), 'L_bol': 10 ** (42 + shape), 'L_ion': 10 ** (41.5 + shape),
         'Q_HI': 10 ** (52 + shape), 'Q_HeI': 10 ** (51 + shape), 'Q_HeII': 10 ** (48 + shape),
         'L_LyW': 10 ** (38 + shape)}
    q = {k: v[None, :] for k, v in q.items()}
    return generation_table_from_quantities(pop, t, q)


def test_D5_ages_included_generation_tables_not_convolved_again(monkeypatch):
    monkeypatch.setattr(P, 'build_generation_table', _synthetic_table)
    monkeypatch.setattr(P, 'draw_generation', _draw_synthetic(P.draw_generation))
    eta, M_cl, tau = 0.1, 1e4, 2.0
    sf = StellarFeedback('pySB99', Z, M_cl * M_SUN, eta, [5 * MYR_TO_SEC], mode='constant_sfr',
                         formation_timescale=tau * MYR_TO_SEC, stochastic=StochasticSpec(KROUPA, 7))
    assert len(sf.populations) == 2 and all(sf._ages_included)
    starts = (0.0, 5.0)
    for t in (0.5, 3.0, 6.0, 8.0, 20.0):              # generation 1 forms from 5 Myr
        expect = sum(10 ** tab.interpolant('Q_ion')(t - t0, Z) * pop.drawn_mass_msun / 1e6
                     for tab, pop, t0 in zip(sf.generation_tables, sf.populations, starts) if t > t0)
        got = sf.get_feedback_quantity(t * MYR_TO_SEC, sf.Q_ion_interpolant)
        assert got == pytest.approx(expect, rel=1e-12), t


def _draw_synthetic(draw):
    """draw_generation with the database lookups replaced (synthetic md5, track minimum 0.8)."""
    def wrapped(spec, i, target, sf_mode, Z_, tau_sf_myr=None, **kw):
        m = draw_stop_after(spec.imf, target, generation_rng(spec.base_seed, i))
        return GenerationPopulation(
            masses_msun=m, birth_times_myr=birth_times(m, sf_mode, target, tau_sf_myr),
            imf_name=spec.imf.name, imf_m_low_msun=spec.imf.m_low, imf_m_up_msun=spec.imf.m_up,
            base_seed=spec.base_seed, generation_index=i, target_mass_msun=target, sf_mode=sf_mode,
            tau_sf_myr=tau_sf_myr, metallicity_Z=Z_, track_min_mass_msun=0.8,
            interpolation_mode='snap', database_md5='synthetic', seed_spawn_key=[i])
    return wrapped


def test_UIMF_b_drawn_mass_and_m_low_checked():
    from toddlers.feedback_tables import build_feedback_table
    from toddlers.pysb99.stochastic.population import realization_metadata, population_record
    pop, table = _synthetic(1e3)
    ft = table.feedback_table
    q = dict(ft.values)
    rec = population_record(pop)
    bad = dict(rec, scalars=dict(rec['scalars'], drawn_mass_msun=pop.drawn_mass_msun * (1 + 1e-9)))
    meta = realization_metadata(pop)
    with pytest.raises(ValueError, match="star list sum"):
        build_feedback_table(ft.t, ft.Z, q, meta, population=bad)
    meta['normalization'] = dict(meta['normalization'], m_low_msun=0.8)
    with pytest.raises(ValueError, match="track-grid minimum"):
        build_feedback_table(ft.t, ft.Z, q, meta, population=rec)


def test_deterministic_tables_unaffected():
    sf = StellarFeedback('pySB99', Z, 1e6 * M_SUN, 0.05, [], imf='kroupa_100')
    assert sf.stochastic is None and sf.M_stellar_list[0] == pytest.approx(5e4)


def test_sample_imf_discrete_removed():
    from toddlers.pysb99.stochastic.sampling import sample_imf_discrete
    with pytest.raises(RuntimeError, match="population"):
        sample_imf_discrete(1e3, None, 'MW', m_upper=100.0)
