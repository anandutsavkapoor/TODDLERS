"""IMF sampling invariants (continuous sampler; no stellar database required)."""
import numpy as np

from toddlers.pysb99.stochastic.sampling import sample_imf, AVAILABLE_IMFS


def test_kroupa_is_available():
    assert "kroupa" in AVAILABLE_IMFS


def test_sampled_mass_close_to_target():
    # Stop-after sampling overshoots by at most one star (<= m_max),
    # so at 1e4 Msun the total is within a few per cent of the target.
    m = sample_imf(total_mass=1e4, imf_name="kroupa", m_min=0.08, m_max=120.0, seed=0)
    assert np.isclose(m.sum(), 1e4, rtol=0.05)


def test_sampled_masses_within_bounds():
    m = sample_imf(total_mass=1e4, imf_name="kroupa", m_min=0.08, m_max=120.0, seed=1)
    assert m.min() >= 0.08 - 1e-9      # allow float rounding at the boundary
    assert m.max() <= 120.0 + 1e-9


def test_sampling_is_reproducible_with_seed():
    a = sample_imf(total_mass=5e3, seed=42)
    b = sample_imf(total_mass=5e3, seed=42)
    assert np.array_equal(a, b)


def test_different_seeds_differ():
    a = sample_imf(total_mass=5e3, seed=1)
    b = sample_imf(total_mass=5e3, seed=2)
    assert not (len(a) == len(b) and np.array_equal(a, b))
