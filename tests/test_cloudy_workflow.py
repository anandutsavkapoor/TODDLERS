"""End-to-end Cloudy post-processing integration test.

Runs a short SB99 shell evolution, then a single Cloudy *shell* model on it, and
asserts Cloudy completed successfully. This exercises the full second stage of the
pipeline (evolution output -> Cloudy input generation -> Cloudy run -> parseable
output).

Marked ``cloudy`` (needs ``cloudy.exe`` on PATH and a valid ``CLOUDY_DATA_DIR``) and
``slow`` (a full evolution plus a Cloudy run). Deselected by default.

Uses SB99 (the original template set; single-star). BPASS works equally through this
path; pySB99 Cloudy coverage is a follow-up. (Note: use the canonical template strings
'SB99'/'BPASS'/'pySB99'; the evolution layer also accepts aliases such as 'SB99_100'
via ``startswith``, but the Cloudy spectral-table lookup matches the template exactly.)
"""
import os
import pytest

from toddlers.evolution import Evolution
from toddlers.cloudy_simulation_manager import CloudySimulationManager
from toddlers.cloudy_output_handler import CloudyOutputHandler
from toddlers.constants import M_SUN

pytestmark = [pytest.mark.cloudy, pytest.mark.slow]


@pytest.fixture(scope="module")
def evolution_dat():
    kw = dict(Z=0.02, eta_sf=0.05, n_cl=80.0, M_cl_init=1e6 * M_SUN,
              template="SB99", imf="kroupa100", star_type="sin",
              profile_type="uniform")
    ev = Evolution(**kw)
    if os.path.exists(ev.output_path):           # clear stale cached .dat
        os.remove(ev.output_path)
        ev = Evolution(**kw)
    ev.run_simulation()
    dat, _ = ev.get_output_paths()
    assert os.path.exists(dat), "evolution did not produce an output .dat"
    return dat


def test_single_shell_cloudy_model(evolution_dat):
    mgr = CloudySimulationManager(evolution_dat, method="adaptive",
                                  add_DIG=False, add_logger=False)
    tps = mgr.get_time_points()
    assert len(tps) > 1
    t = tps[1]                                  # an early timepoint

    gen = mgr.get_model_generator("shell")
    with mgr.change_dir(mgr.cloudy_run_dir):
        gen.write_input_file(t)
        mgr.run_simulation(t, "shell")
        assert CloudyOutputHandler("shell", t).check_cloudy_success()
