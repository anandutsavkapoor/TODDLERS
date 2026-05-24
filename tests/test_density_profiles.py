"""Invariants of the cloud density profiles (no data required)."""
import numpy as np
import pytest

from toddlers.cloud_density_profiles import (
    UniformDensity, BonnorEbertSphere, ModifiedBonnorEbertProfile,
)
from toddlers.constants import M_SUN

M_CL = 1e6 * M_SUN
N_AVG = 160.0


def _profile(name):
    if name == "uniform":
        return UniformDensity(M_CL, N_AVG)
    if name == "BE":
        return BonnorEbertSphere(M_CL, N_AVG)
    if name == "modBE":
        return ModifiedBonnorEbertProfile(M_CL, N_AVG, alpha=1.0)
    raise ValueError(name)


ALL = ["uniform", "BE", "modBE"]


@pytest.mark.parametrize("name", ALL)
def test_total_mass_recovered_at_R_cl(name):
    p = _profile(name)
    R_cl = p.get_parameters()["R_cl"]
    assert np.isclose(p.mass_enclosed(R_cl), M_CL, rtol=1e-3)


@pytest.mark.parametrize("name", ALL)
def test_mass_monotonic_and_zero_at_center(name):
    p = _profile(name)
    R_cl = p.get_parameters()["R_cl"]
    r = np.linspace(0.0, R_cl, 25)
    m = np.array([p.mass_enclosed(ri) for ri in r])
    assert m[0] < 1e-3 * M_CL                      # ~0 enclosed at center
    assert np.all(np.diff(m) >= -1e-6 * M_CL)      # non-decreasing


@pytest.mark.parametrize("name", ALL)
def test_density_positive(name):
    p = _profile(name)
    R_cl = p.get_parameters()["R_cl"]
    for f in (0.1, 0.5, 0.9):
        assert p.density(f * R_cl) > 0


def test_uniform_density_is_constant():
    p = UniformDensity(M_CL, N_AVG)
    R_cl = p.get_parameters()["R_cl"]
    assert np.isclose(p.density(0.1 * R_cl), p.density(0.9 * R_cl), rtol=1e-6)


@pytest.mark.parametrize("name", ["BE", "modBE"])
def test_centrally_concentrated(name):
    p = _profile(name)
    R_cl = p.get_parameters()["R_cl"]
    assert p.density(0.1 * R_cl) > p.density(0.9 * R_cl)
