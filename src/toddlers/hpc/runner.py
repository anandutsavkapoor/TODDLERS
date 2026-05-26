"""Per-task execution callables for the TODDLERS HPC worker pool.

These are the inline ``process_one`` functions the worker loop dispatches to. Each
takes a single task as a plain ``dict`` (one decoded JSON line from the task file)
and runs it to completion, raising on failure so the worker can record it.

Keeping these as importable functions (rather than argparse scripts, as the legacy
``hpc/evolution_runner.py`` / ``hpc/cloudy_runner.py`` were) is what lets one
long-lived worker process handle thousands of tasks without paying Python startup
per task.
"""
import os
from contextlib import contextmanager

from ..constants import M_SUN, MYR_TO_SEC


# ---------------------------------------------------------------------------
# Evolution work unit
# ---------------------------------------------------------------------------

# Keys the evolution task may carry. ``M_cl_init`` is given in solar masses in the
# task file (human-readable) and converted to cgs here, matching the convention of
# the legacy evolution_runner.
_EVOLUTION_PASSTHROUGH = (
    "Z", "eta_sf", "n_cl", "template", "imf", "star_type", "profile_type",
    "cluster_formation_mode", "formation_timescale", "dust_to_metal",
    "dynamic_cloud_density", "add_cover_frac", "post_sweep_covering_fraction",
)


def run_evolution_task(row: dict):
    """Run one 1D shell-evolution simulation from a task dict.

    ``M_cl_init`` is expected in solar masses. A ``covering_fraction`` shortcut is
    honoured for backward compatibility (covering_fraction < 1 -> add_cover_frac).
    Any other recognised Evolution keyword present in ``row`` is passed through.
    """
    from ..evolution import Evolution

    kwargs = {k: row[k] for k in _EVOLUTION_PASSTHROUGH if k in row}

    if "M_cl_init" in row:
        kwargs["M_cl_init"] = float(row["M_cl_init"]) * M_SUN

    # Backward-compatible covering_fraction shortcut (legacy CSV sweeps used this).
    if "covering_fraction" in row and "add_cover_frac" not in row:
        cf = float(row["covering_fraction"])
        if cf < 1.0:
            kwargs["add_cover_frac"] = True
            kwargs["post_sweep_covering_fraction"] = cf

    sim = Evolution(**kwargs)
    return sim.run_simulation()


# ---------------------------------------------------------------------------
# Cloudy work unit
# ---------------------------------------------------------------------------

# Set once per worker process, the first time a Cloudy model that requests the
# unattenuated nebular continuum completes, so the worker log states up front whether
# the binary carries the cloudy_patches/ patch (see the [patch-check] block below).
_PATCH_CHECK_LOGGED = False


@contextmanager
def _in_dir(path):
    """chdir into ``path`` for the duration, always restoring the old cwd.

    A worker processes many Cloudy tasks in one process; Cloudy input generation
    is run-directory relative, so each task must leave the cwd as it found it.
    """
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def run_cloudy_task(row: dict, cloudy_exe: str = None):
    """Run one Cloudy model (one simulation x timepoint x phase) from a task dict.

    Expected keys: ``sim_file``, ``time`` (seconds), ``phase`` in
    {shell, unified, dig, dissolved}. Optional: ``inner_prefix`` (required for the
    ``dig`` phase), ``add_dig``, ``logU_background``, ``continue_after_dissolution``,
    ``dust_to_metal``, ``force_regenerate``.

    ``cloudy_exe`` overrides the executable path (the cluster's cloudy.exe); when
    omitted the package default (``cloudy.exe`` on PATH) is used.

    Auto-repair: if Cloudy fails with a recognised, fixable signature (convergence,
    temperature/pressure floor, zone limit), the input is tweaked accordingly and
    the model rerun, up to ``max_repair_attempts`` times (default 2; disable with
    ``auto_repair=False``). Unknown failures propagate for manual review.
    """
    from subprocess import CalledProcessError

    from ..cloudy_simulation_manager import CloudySimulationManager
    from ..cloudy_output_handler import CloudyOutputHandler
    from .error_recovery import CloudyErrorClassifier, CloudyInputModifier

    sim_file = row["sim_file"]
    if not os.path.exists(sim_file):
        raise FileNotFoundError(f"Simulation file not found: {sim_file}")

    time = float(row["time"])
    phase = row["phase"]
    if phase not in ("shell", "unified", "dig", "dissolved"):
        raise ValueError(f"Unknown phase: {phase}")

    inner_prefix = row.get("inner_prefix") or None
    if phase == "dig" and inner_prefix not in ("shell", "unified"):
        raise ValueError("inner_prefix must be 'shell' or 'unified' for dig models")

    sim_manager = CloudySimulationManager(
        sim_file,
        add_DIG=bool(row.get("add_dig", True)),
        logU_background=row.get("logU_background"),
        continue_after_dissolution=bool(row.get("continue_after_dissolution", False)),
        complete_init=True,
        add_logger=False,
        dust_to_metal=float(row.get("dust_to_metal", 1.0)),
    )
    if cloudy_exe:
        sim_manager.cloudy_exec = cloudy_exe

    model_generator = sim_manager.get_model_generator(phase)
    force_regenerate = bool(row.get("force_regenerate", False))

    with _in_dir(sim_manager.cloudy_run_dir):
        output_handler = CloudyOutputHandler(phase, time, parse_data=False)
        input_exists = output_handler.check_input_exists()
        in_path = output_handler.get_file_path("in")
        out_path = output_handler.get_file_path("out")

        # Operational diagnostics: note when we are self-healing a previously broken
        # on-disk state (truncated input / invalid prior output, e.g. from a disk-full
        # mid-write) so a run's logs reveal infrastructure problems instead of hiding them
        # behind a silent re-run. No physics change (regenerated input is deterministic) --
        # distinct from "[repair]", which alters the model. Greppable as "[self-heal]".
        # The prior-output check only does I/O when a .out already exists (i.e. a re-run /
        # resume), so a clean first pass pays nothing.
        heal = []
        if not input_exists and os.path.exists(in_path):
            heal.append("empty/truncated input")
        if os.path.exists(out_path) and not output_handler.check_cloudy_success():
            heal.append("invalid prior output")
        if heal:
            print(f"[self-heal] {phase}@{time / MYR_TO_SEC:.2f}Myr: {', '.join(heal)} "
                  f"-> regenerating/re-running")

        if not input_exists or force_regenerate:
            if phase in ("shell", "unified", "dissolved"):
                model_generator.write_input_file(time)
            elif phase == "dig":
                run_flag = model_generator.write_input_file(time, inner_prefix)
                if not run_flag:
                    # No DIG input warranted at this timepoint; not a failure.
                    return None

        # Run, with bounded retry-and-repair for recognised Cloudy failures.
        auto_repair = bool(row.get("auto_repair", True))
        max_attempts = int(row.get("max_repair_attempts", 2))
        classifier, modifier = CloudyErrorClassifier(), CloudyInputModifier()

        attempt = 0
        repairs = []
        while True:
            try:
                sim_manager.run_simulation(time, phase)
                break
            except CalledProcessError:
                if not auto_repair or attempt >= max_attempts or not os.path.exists(out_path):
                    raise
                with open(out_path, errors="ignore") as f:
                    out_tail = "".join(f.readlines()[-120:])
                err = classifier.classify_error(out_tail)
                if err is None or not err.solution:
                    raise  # unknown / no auto-fix -> manual review
                with open(in_path, errors="ignore") as f:
                    in_text = f.read()
                new_text, desc = modifier.modify_input(in_text, err)
                if new_text == in_text:
                    raise  # modification was a no-op -> give up
                with open(in_path, "w") as f:
                    f.write(new_text)
                attempt += 1
                repairs.append(err.name)
                print(f"[repair] {phase}@{time / MYR_TO_SEC:.2f}Myr: {desc} "
                      f"(attempt {attempt}/{max_attempts})")

        # Post-run integrity: a model can "exit OK" yet write truncated or empty save
        # files (e.g. the disk filling mid-write). check_cloudy_success() validates the
        # essential outputs, so a False here means the run is unusable; raise so the
        # worker records FAIL and the resume gate re-runs it, rather than letting the
        # build consume empty continuum / line-less HR SEDs.
        if not output_handler.check_cloudy_success():
            raise RuntimeError(
                f"{phase}@{time / MYR_TO_SEC:.3f}Myr exited OK but produced invalid or "
                f"empty output (truncated save / disk full?)")

        # One-time per worker: warn if this cloudy.exe lacks the unattenuated-continuum
        # patch (cloudy_patches/). The shell/unified/dissolved inputs all request
        # `save diffuse continuum unattenuated`; a patched binary writes a non-empty
        # .diffContUnatt, an unpatched one writes nothing there and does NOT error.
        # _outputs_valid() above does not require that file, so an unpatched binary
        # "exits OK" yet produces no unattenuated nebular continuum. The patch is
        # optional, so this is a one-time note (greppable as "[patch-check]"), not a
        # failure: without it the noDust / variable-DTM SEDs fall back to the standard
        # attenuated diffuse continuum.
        global _PATCH_CHECK_LOGGED
        if not _PATCH_CHECK_LOGGED and phase in ("shell", "unified", "dissolved"):
            _PATCH_CHECK_LOGGED = True
            neb_path = output_handler.get_file_path("diffContUnatt")
            if not (os.path.exists(neb_path) and os.path.getsize(neb_path) > 0):
                exe = cloudy_exe or "cloudy.exe"
                print(f"[patch-check] {exe} has no unattenuated-continuum patch "
                      f"(cloudy_patches/): no .diffContUnatt written, so noDust / "
                      f"variable-DTM SEDs fall back to the attenuated diffuse continuum.")

    info = f"{phase}@{time / MYR_TO_SEC:.3f}Myr"
    if repairs:
        info += f" [repaired: {', '.join(repairs)}]"
    return info


# Dispatch table keyed on the task's ``stage`` field.
def dispatch(row: dict, cloudy_exe: str = None):
    """Run a single task, dispatching on its ``stage`` field."""
    stage = row.get("stage")
    if stage == "evolution":
        return run_evolution_task(row)
    if stage == "cloudy":
        return run_cloudy_task(row, cloudy_exe=cloudy_exe)
    raise ValueError(f"Unknown task stage: {stage!r}")
