"""Unit tests for the HPC task pipeline (no cluster, no data required).

These protect the orchestration logic that the worker pool depends on: grid
expansion, the modular work-slicing, flock-appended results, and the resume
accounting. The actual evolution/Cloudy execution is exercised by the data/cloudy
marked tests; here the per-task callable is stubbed out.
"""
import json

import pytest

from toddlers.hpc import generate_tasks, check_status, worker_loop, runner
from toddlers.hpc.error_recovery import CloudyErrorClassifier, CloudyInputModifier


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_evolution_grid_expansion(tmp_path):
    grid = {"Z": [0.01, 0.02], "n_cl": [10.0, 20.0], "template": "SB99"}
    (tmp_path / "grid.json").write_text(json.dumps(grid))
    generate_tasks.main(["evolution", "--grid", str(tmp_path / "grid.json"),
                         "-o", str(tmp_path / "tasks")])

    tasks = _read_jsonl(tmp_path / "tasks" / "evolution.tasks")
    # full Cartesian product: 2 x 2 x 1
    assert len(tasks) == 4
    assert all(t["stage"] == "evolution" for t in tasks)
    assert all(t["template"] == "SB99" for t in tasks)          # scalar broadcast
    assert {(t["Z"], t["n_cl"]) for t in tasks} == {
        (0.01, 10.0), (0.01, 20.0), (0.02, 10.0), (0.02, 20.0)}


def test_worker_processes_only_its_modular_slice(tmp_path, monkeypatch):
    # 7 tasks, 3 workers: worker 1 must handle exactly indices {1, 4}.
    task_file = tmp_path / "t.tasks"
    task_file.write_text("\n".join(
        json.dumps({"stage": "evolution", "i": i}) for i in range(7)))

    seen = []
    monkeypatch.setattr(runner, "dispatch",
                        lambda row, cloudy_exe=None: seen.append(row["i"]))
    # worker_loop imports dispatch via `from .runner import dispatch`, so patch there too
    monkeypatch.setattr(worker_loop, "dispatch", runner.dispatch, raising=False)

    base = tmp_path / "res" / "job.results"
    worker_loop.run(str(task_file), n_workers=3, worker_id=1, results_file=str(base))

    # worker 1 writes its own per-worker file (multi-node-safe; no shared lock)
    wfile = worker_loop.worker_results_path(str(base), 1)
    assert sorted(seen) == [1, 4]
    lines = open(wfile).read().splitlines()
    assert sorted(int(ln.split("\t")[0]) for ln in lines) == [1, 4]
    assert all(ln.split("\t")[1] == "OK" for ln in lines)


def test_worker_records_failure_without_dying(tmp_path, monkeypatch):
    task_file = tmp_path / "t.tasks"
    task_file.write_text("\n".join(
        json.dumps({"stage": "evolution", "i": i}) for i in range(2)))

    def boom(row, cloudy_exe=None):
        if row["i"] == 0:
            raise RuntimeError("kaboom")

    monkeypatch.setattr(runner, "dispatch", boom)
    base = tmp_path / "res.results"
    worker_loop.run(str(task_file), n_workers=1, worker_id=0, results_file=str(base))

    wfile = worker_loop.worker_results_path(str(base), 0)
    lines = dict(line.split("\t", 1) for line in open(wfile).read().splitlines())
    assert lines["0"].startswith("FAIL")
    assert lines["1"].startswith("OK")


def test_check_status_resume_accounting(tmp_path):
    task_file = tmp_path / "t.tasks"
    rows = [json.dumps({"stage": "evolution", "i": i}) for i in range(5)]
    task_file.write_text("\n".join(rows))

    # tasks 0,2 OK; task 3 FAIL; 1,4 never ran -> remaining = {1,3,4}
    results = tmp_path / "r.results"
    results.write_text("0\tOK\t\n2\tOK\t\n3\tFAIL\tboom\n")

    resume = tmp_path / "resume.tasks"
    with pytest.raises(SystemExit) as exc:
        check_status.main(["--task-file", str(task_file),
                           "--results", str(results), "-o", str(resume)])
    assert exc.value.code == 1                      # unfinished -> nonzero exit

    remaining = _read_jsonl(resume)
    assert {t["i"] for t in remaining} == {1, 3, 4}


def test_check_status_all_done_exit_zero(tmp_path):
    task_file = tmp_path / "t.tasks"
    task_file.write_text("\n".join(json.dumps({"i": i}) for i in range(2)))
    results = tmp_path / "r.results"
    results.write_text("0\tOK\t\n1\tOK\t\n")

    with pytest.raises(SystemExit) as exc:
        check_status.main(["--task-file", str(task_file), "--results", str(results)])
    assert exc.value.code == 0


def test_dispatch_rejects_unknown_stage():
    with pytest.raises(ValueError):
        runner.dispatch({"stage": "nonsense"})


# --- failed-run auto-repair (error_recovery) --------------------------------

def test_classifier_matches_known_failures():
    c = CloudyErrorClassifier()
    cases = {
        "ConvFail aborts since nTotalFailures=10 is >= LimFail=10": "Convergence Failure",
        "PROBLEM  ConvFail 1,  ionization not converged iteration 2 zone 1360": "Zone Non-Convergence",
        "DISASTER - A floating point exception occurred. Bailing out...": "Floating Point Exception",
        "PROBLEM DISASTER the kinetic temperature is below the lower limit": "Temperature Too Low",
        "ABORT DISASTER nPres2Ioniz exceeds limPres2Ioniz here": "Pressure Ionization Limit",
        "Calculation stopped because default number of zones reached": "Zone Limit Reached",
    }
    for text, name in cases.items():
        err = c.classify_error(text)
        assert err is not None and err.name == name


def test_zone_grind_outranks_zone_limit():
    # A grinder that hits the zone cap WITH per-zone non-convergence in the tail
    # must be routed to turbulence (root cause), not to "more zones" (symptom).
    c = CloudyErrorClassifier()
    tail = (
        "PROBLEM  ConvFail 5,  ionization not converged iteration 2 zone 2798\n"
        "Calculation stopped because default number of zones reached\n"
        "Cloudy ends: 2800 zones, 3 iterations, 2 warnings\n"
    )
    err = c.classify_error(tail)
    assert err is not None and err.name == "Zone Non-Convergence"
    assert err.modification_function == "add_turbulence_pressure"


def test_buried_zone_warning_needs_whole_out():
    # Regression: Cloudy prints the zone-cap warning, THEN dumps the final-iteration
    # per-zone element tables (thousands of lines), so the signature sits far from EOF
    # (observed ~3000 lines back in an 8026-line shell .out). The classifier must see the
    # whole .out -- runner.py reads f.read(), not a tail -- or the model fails unrepaired.
    c = CloudyErrorClassifier()
    out = ("Calculation stopped because default number of zones reached\n"
           + "Iron 1.80 2.44 3.76\n" * 3000
           + "Cloudy ends: 2800 zones, 3 iterations, 2 warnings\n"
           + "[Stop in cdMain at maincl.cpp:593, something went wrong]\n")
    assert c.classify_error(out).name == "Zone Limit Reached"      # whole-file: found
    tail = "".join(out.splitlines(keepends=True)[-200:])
    assert c.classify_error(tail) is None                          # 200-line tail: missed


def test_classifier_unknown_needs_manual_review():
    c = CloudyErrorClassifier()
    assert c.classify_error("some unrecognised cloudy chatter") is None
    assert c.requires_manual_review("some unrecognised cloudy chatter") is True
    assert c.requires_manual_review("ConvFail aborts since nTotalFailures=9 is >= LimFail=9") is False


def test_modifier_turbulence_floor_then_escalate():
    # Density-independent now: failing models sit at log n_H ~ 2.2-2.8. A tiny
    # input value is lifted to the floor on the first repair, then escalated;
    # the trailing "no pressure" is dropped so turbulent pressure is active.
    m = CloudyInputModifier()
    c = CloudyErrorClassifier()
    err = c.classify_error("ConvFail aborts since nTotalFailures=9 is >= LimFail=9")
    first, _ = m.modify_input("hden 2.8\nturbulence 0.103077 km/sec no pressure\n", err)
    assert "turbulence 1.500000 km/sec" in first
    assert "no pressure" not in first
    second, _ = m.modify_input(first, err)        # escalate x2 from the floor
    assert "turbulence 3.000000 km/sec" in second


def test_modifier_turbulence_clamped_then_noop():
    # Escalation is clamped at the ceiling, and once there the rewrite is a
    # no-op so the caller gives up cleanly instead of escalating forever.
    m = CloudyInputModifier()
    c = CloudyErrorClassifier()
    err = c.classify_error("ConvFail aborts since nTotalFailures=9 is >= LimFail=9")
    at_ceiling = f"turbulence {m.TURBULENCE_CEILING * 0.75:.6f} km/sec\n"  # x2 -> over ceiling
    clamped, _ = m.modify_input(at_ceiling, err)
    assert f"turbulence {m.TURBULENCE_CEILING:.6f} km/sec" in clamped
    again, _ = m.modify_input(clamped, err)       # already at ceiling -> identical text
    assert again == clamped


def test_modifier_increase_zones_bounded():
    m = CloudyInputModifier()
    assert "set nend 1200" in m.increase_zones("set nend 800\n")
    assert "set nend 6000" in m.increase_zones("set nend 4000\n")   # x1.5, below ceiling
    assert "set nend 12000" in m.increase_zones("set nend 9000\n")  # capped at 12000
