"""StellarFeedback sanity (requires the feedback template data)."""
import numpy as np
import pytest

from toddlers.stellar_feedback import StellarFeedback
from toddlers.constants import M_SUN, MYR_TO_SEC

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def sb99():
    return StellarFeedback(template="SB99", star_type="sin", Z=0.02, M_cl_init=1e6 * M_SUN,
                           eta_sf=0.05, t_list_collapse=[])


@pytest.mark.parametrize("t_myr", [0.1, 1.0, 3.0, 10.0])
def test_feedback_positive_and_finite(sb99, t_myr):
    t = t_myr * MYR_TO_SEC
    for getter in (sb99.get_ionizing_photon_rate,
                   sb99.get_bolometric_luminosity,
                   sb99.get_mechanical_luminosity):
        val = getter(t)
        assert np.isfinite(val) and val > 0


def test_ionizing_output_declines_after_early_peak(sb99):
    q_early = sb99.get_ionizing_photon_rate(1.0 * MYR_TO_SEC)
    q_late = sb99.get_ionizing_photon_rate(10.0 * MYR_TO_SEC)
    assert q_early > q_late
