"""End-to-end full shell evolution invariants (requires feedback data).

Runs the integrator to completion and checks the trajectory is physically sane.
Marked `data` (needs the feedback database) and `slow` (a full run is ~30 s).
"""
import os

import numpy as np
import pytest

from toddlers.evolution import Evolution
from toddlers.constants import M_SUN

pytestmark = [pytest.mark.data, pytest.mark.slow]


def _fresh(**kw):
    """Build an Evolution after clearing any stale cached .dat (the constructor
    early-returns a half-initialised object if the output file already exists)."""
    ev = Evolution(**kw)
    if os.path.exists(ev.output_path):
        os.remove(ev.output_path)
        ev = Evolution(**kw)
    return ev


@pytest.fixture(scope="module")
def run():
    ev = _fresh(Z=0.02, eta_sf=0.05, n_cl=160.0, M_cl_init=1e6 * M_SUN,
                template="SB99", star_type="sin", profile_type="uniform",
                dynamic_cloud_density=False)
    results = ev.run_simulation()
    assert results, "run_simulation returned no results"
    return results[0]


def test_time_strictly_increasing(run):
    t = run["time"]
    assert np.all(np.diff(t) > 0)


def test_radius_positive_and_expanding_early(run):
    R = run["radius"]
    assert np.all(R > 0)
    assert R[len(R) // 4] > R[0]          # shell has expanded


def test_mass_and_values_finite_positive(run):
    assert np.all(run["mass"] > 0)
    for key in ("radius", "velocity", "mass", "time"):
        assert np.all(np.isfinite(run[key]))


def test_reaches_fragmentation(run):
    transitions = run.get("phase_transitions", [])
    assert any(tr[0] == "phase1_to_fragmentation" for tr in transitions)
