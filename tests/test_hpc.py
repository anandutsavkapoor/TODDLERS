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
        "PROBLEM DISASTER the kinetic temperature is below the lower limit": "Temperature Too Low",
        "ABORT DISASTER nPres2Ioniz exceeds limPres2Ioniz here": "Pressure Ionization Limit",
        "Calculation stopped because default number of zones reached": "Zone Limit Reached",
    }
    for text, name in cases.items():
        err = c.classify_error(text)
        assert err is not None and err.name == name


def test_classifier_unknown_needs_manual_review():
    c = CloudyErrorClassifier()
    assert c.classify_error("some unrecognised cloudy chatter") is None
    assert c.requires_manual_review("some unrecognised cloudy chatter") is True
    assert c.requires_manual_review("ConvFail aborts since nTotalFailures=9 is >= LimFail=9") is False


def test_modifier_turbulence_boosts_only_at_high_density():
    m = CloudyInputModifier()
    c = CloudyErrorClassifier()
    err = c.classify_error("ConvFail aborts since nTotalFailures=9 is >= LimFail=9")
    hi = "hden 4.0\nturbulence 3.0 km/sec no pressure\n"   # log nH 4.0 >= 3.5 -> boost x5
    lo = "hden 2.0\nturbulence 3.0 km/sec no pressure\n"   # below threshold -> unchanged velocity
    out_hi, _ = m.modify_input(hi, err)
    out_lo, _ = m.modify_input(lo, err)
    assert "turbulence 15.0" in out_hi
    assert "turbulence 3.0" in out_lo


def test_modifier_increase_zones_bounded():
    m = CloudyInputModifier()
    assert "set nend 1200" in m.increase_zones("set nend 800\n")
    assert "set nend 5000" in m.increase_zones("set nend 4000\n")  # capped at 5000
