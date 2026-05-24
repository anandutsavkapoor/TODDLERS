"""Sanity checks on physical constants and unit conversions."""
import numpy as np

from toddlers.constants import (
    M_SUN, MU_N, M_P, PC_TO_CM, MYR_TO_SEC, Z_SOLAR, G, C, K_BOLTZMANN,
)


def test_positive_constants():
    for name, val in [("M_SUN", M_SUN), ("M_P", M_P), ("PC_TO_CM", PC_TO_CM),
                      ("MYR_TO_SEC", MYR_TO_SEC), ("G", G), ("C", C),
                      ("K_BOLTZMANN", K_BOLTZMANN)]:
        assert val > 0, f"{name} should be positive"


def test_mean_mass_per_nucleus():
    # Fiducial Z-independent value MU_N = (14/11) m_p
    assert np.isclose(MU_N, (14.0 / 11.0) * M_P, rtol=1e-6)


def test_solar_metallicity_in_range():
    assert 0.0 < Z_SOLAR < 1.0


def test_unit_conversions():
    assert np.isclose(PC_TO_CM, 3.0857e18, rtol=1e-3)
    assert np.isclose(MYR_TO_SEC, 1e6 * 365.25 * 24 * 3600, rtol=1e-2)
    assert np.isclose(C, 2.998e10, rtol=1e-3)        # cm/s
    assert np.isclose(M_SUN, 1.989e33, rtol=1e-3)    # g
