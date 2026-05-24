"""Enumerate Cloudy work items from evolution outputs.

This is the physics half of the legacy ``WorkloadAnalyzer`` / ``CloudyWorkerManager``
(the cluster-specific load balancing and PBS/worker-module plumbing are dropped).
Given a set of evolution ``.dat`` files it decides, per timepoint, which Cloudy
model phases must run (``shell``, ``unified``, ``dig``, ``dissolved``) and yields one
task dict per (simulation, timepoint, phase) work item.

Phase ordering note: a ``dig`` model consumes the transmitted continuum of its inner
(``shell`` or ``unified``) model, so dig tasks must run *after* their inner model.
The task generator writes one file per phase and the submit templates run them as
dependent SLURM jobs, preserving this ordering. Within a phase the work items are
independent and safe to spread across the worker pool.
"""
from pathlib import Path

from ..constants import MYR_TO_SEC


def _models_to_run(sim_manager, t, has_dissolved, dissolution_time, add_dig,
                   continue_after_dissolution):
    """Decide which Cloudy phases run at time ``t`` (ported verbatim in logic)."""
    models = []
    if not has_dissolved or (dissolution_time and t < dissolution_time):
        if sim_manager.is_within_cloud_interp(t) > 0:
            models.extend(["shell", "unified"])
        else:
            models.append("shell")
        if add_dig:
            models.append("dig")
    else:
        if add_dig:
            models.append("dig")
        elif continue_after_dissolution:
            models.append("dissolved")
    return models


def enumerate_cloudy_tasks(evolution_files, method="adaptive", n_points=None,
                           add_dig=False, logU_background=None,
                           continue_after_dissolution=False, dust_to_metal=None,
                           verbose=True):
    """Yield Cloudy task dicts for the given evolution files.

    ``dust_to_metal``: ``None`` reads the DTM from each evolution file's
    simulation_params (self-consistent); a scalar or list sweeps explicit value(s).

    Each yielded dict carries ``stage='cloudy'`` plus the keys
    :func:`toddlers.hpc.runner.run_cloudy_task` consumes.
    """
    from ..cloudy_simulation_manager import CloudySimulationManager

    if dust_to_metal is None:
        dtm_values = None
    elif isinstance(dust_to_metal, (list, tuple)):
        dtm_values = list(dust_to_metal)
    else:
        dtm_values = [dust_to_metal]

    n_files = len(evolution_files)
    for i, sim_file in enumerate(evolution_files, 1):
        sim_file = str(Path(sim_file).resolve())
        if verbose:
            print(f"[enumerate] {i}/{n_files}: {Path(sim_file).name}")

        try:
            sim_manager = CloudySimulationManager(
                sim_file, method=method, n_points=n_points, add_DIG=add_dig,
                logU_background=logU_background,
                continue_after_dissolution=continue_after_dissolution,
                complete_init=True,
            )
        except Exception as exc:  # noqa: BLE001 - keep enumerating other files
            print(f"[enumerate] ERROR on {sim_file}: {exc}")
            continue

        # Locate dissolution transition, if any.
        dissolution_time = None
        for gen_data in sim_manager.all_results:
            for transition in gen_data.get("phase_transitions", []):
                if transition[0] == "phase2_to_dissolution":
                    dissolution_time = transition[1]
                    break
            if dissolution_time is not None:
                break
        has_dissolved = dissolution_time is not None

        file_dtm = sim_manager.simulation_params.get("dust_to_metal", 1.0)
        dtm_list = [file_dtm] if dtm_values is None else dtm_values

        for t in sim_manager.get_time_points():
            models = _models_to_run(sim_manager, t, has_dissolved, dissolution_time,
                                     add_dig, continue_after_dissolution)
            if not models:
                continue

            inner_prefix = None
            if "dig" in models:
                inner_prefix = ("unified" if sim_manager.is_within_cloud_interp(t) > 0
                                else "shell")

            for dtm in dtm_list:
                for phase in models:
                    task = {
                        "stage": "cloudy",
                        "sim_file": sim_file,
                        "time": float(t),
                        "time_Myr": float(t) / MYR_TO_SEC,
                        "phase": phase,
                        "dust_to_metal": float(dtm),
                        "add_dig": bool(add_dig),
                        "continue_after_dissolution": bool(continue_after_dissolution),
                    }
                    if logU_background is not None:
                        task["logU_background"] = float(logU_background)
                    if phase == "dig":
                        task["inner_prefix"] = inner_prefix
                    yield task
