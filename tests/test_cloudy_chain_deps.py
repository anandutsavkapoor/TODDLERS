#!/usr/bin/env python3
"""The first-round Cloudy phase chain must encode the real data dependencies.

  shell      -- no Cloudy-phase dependency (only the evolution job, `after`).
  unified    -- reads shell's .phy density structure -> afterok:shell.
  dissolved  -- reads only the evolution .dat (post-dissolution gas), NOT shell output, so it
                is a PEER of shell (shares `after`), running in parallel rather than waiting on
                shell. Validated by the 2026-05 Tier-2 pilot (dissolved finished clean while
                racing shell-resume). Overlapping the largest phase with shell shortens the
                per-DTM critical path.
  dig        -- consumes unified's transmitted continuum -> afterok:unified (only with --add-dig).

The dependency is passed to sbatch (printed in dry-run), not baked into the template, so this
test captures stdout and inspects the per-phase sbatch command lines.
"""
import sys
import types
from pathlib import Path

project_root = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(project_root))

from toddlers.hpc import campaign


def _run_chain(tmp_path, capsys, add_dig=False):
    taskdir = tmp_path / "tasks"; taskdir.mkdir()
    phases = ["shell", "unified", "dissolved"] + (["dig"] if add_dig else [])
    for ph in phases:
        (taskdir / f"cloudy_{ph}.tasks").write_text("dummytaskline\n")
    args = types.SimpleNamespace(
        account="acc", partition="part", ntasks=128, walltime="03:00:00",
        python_module="SciPy", activate_env="", toddlers_src="/src", toddlers_data="/data",
        work_dir=str(tmp_path), cloudy_exe="/cl.exe", small_to_large_ratio=None,
        add_dig=add_dig, max_nodes=8, output_root="", dry_run=True)
    campaign._submit_cloudy_chain(args, taskdir)
    out = capsys.readouterr().out
    # one printed sbatch command line per phase (the line carrying --export=ALL,PHASE=<ph>)
    return {ph: next(l for l in out.splitlines() if f"PHASE={ph}" in l) for ph in phases}


def test_shell_has_no_phase_dependency(tmp_path, capsys):
    lines = _run_chain(tmp_path, capsys)
    assert "--dependency" not in lines["shell"]


def test_unified_waits_on_shell(tmp_path, capsys):
    lines = _run_chain(tmp_path, capsys)
    assert "--dependency=afterok:DRYRUN" in lines["unified"]


def test_dissolved_is_peer_of_shell_no_dependency(tmp_path, capsys):
    # the fix: dissolved must NOT depend on shell (it shares shell's `after`, here None)
    lines = _run_chain(tmp_path, capsys)
    assert "--dependency" not in lines["dissolved"]


def test_dig_waits_on_unified(tmp_path, capsys):
    lines = _run_chain(tmp_path, capsys, add_dig=True)
    assert "--dependency=afterok:DRYRUN" in lines["dig"]
