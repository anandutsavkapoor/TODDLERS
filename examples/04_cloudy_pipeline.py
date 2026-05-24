"""Full pipeline: shell evolution -> Cloudy post-processing (needs data + Cloudy).

Runs a deterministic SB99 evolution, then post-processes one shell timepoint with
Cloudy to produce an emergent model. Demonstrates the second stage of TODDLERS.

Requires: `python scripts/download_data.py`, plus `cloudy.exe` on PATH and a valid
CLOUDY_DATA_DIR. A full grid would loop over all timepoints/models (run_full_simulation);
here we run a single shell model to keep it quick.
"""
import os

from toddlers.evolution import Evolution
from toddlers.cloudy_simulation_manager import CloudySimulationManager
from toddlers.cloudy_output_handler import CloudyOutputHandler
from toddlers.constants import M_SUN, MYR_TO_SEC

# 1) shell evolution
ev = Evolution(Z=0.02, eta_sf=0.05, n_cl=160.0, M_cl_init=1e6 * M_SUN,
               template="SB99", imf="kroupa100", star_type="sin",
               profile_type="uniform")
ev.run_simulation()
dat, _ = ev.get_output_paths()
print("evolution output:", dat)

# 2) Cloudy post-processing of one shell timepoint
mgr = CloudySimulationManager(dat, method="adaptive", add_DIG=False, add_logger=False)
t = mgr.get_time_points()[1]
print(f"running Cloudy 'shell' model at t = {t / MYR_TO_SEC:.2f} Myr "
      f"(of {len(mgr.get_time_points())} timepoints)")
gen = mgr.get_model_generator("shell")
with mgr.change_dir(mgr.cloudy_run_dir):
    gen.write_input_file(t)
    mgr.run_simulation(t, "shell")
    ok = CloudyOutputHandler("shell", t).check_cloudy_success()
print("Cloudy run succeeded:", ok)
print("(loop over mgr.get_time_points() and all models, or mgr.run_full_simulation(), "
      "to build the full time-resolved SED set.)")
