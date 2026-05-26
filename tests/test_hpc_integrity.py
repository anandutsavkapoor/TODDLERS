"""Integrity tests for the HPC pipeline's failure modes (no cluster, no Cloudy, no data).

A worker pool on a shared cluster fails in messy, partial ways: jobs are killed at
walltime mid-write, disks/inode-quotas fill, and the resume gate re-runs the same task
across several rounds. These tests pin the defenses that turn those messes into
self-healing instead of silent corruption or infinite re-arm loops:

  * input-level   -- a .in truncated/blanked mid-write is treated as missing
                     (``CloudyOutputHandler.check_input_exists``)
  * output-level  -- a model that "exited OK" but wrote empty / header-clobbered save
                     files is rejected (``check_cloudy_success`` / ``_outputs_valid``)
  * runner glue   -- ``run_cloudy_task`` self-heals a broken on-disk state, fails loudly
                     on unusable output, and warns once if the binary lacks the patch
  * resume math   -- ``check_status`` keeps OK winning over a stale FAIL across rounds

The real Cloudy execution is stubbed; only the package's own logic is exercised, so this
runs in the default (fast) suite as a regression guard.
"""
import json
import os

import pytest

from toddlers.constants import MYR_TO_SEC
from toddlers.cloudy_output_handler import CloudyOutputHandler
from toddlers.hpc import check_status, runner

TIME = 0.10 * MYR_TO_SEC          # -> base filename "<phase>_0.10"


# --------------------------------------------------------------------------- helpers
def _base(prefix, time=TIME):
    return f"{prefix}_{time / MYR_TO_SEC:.2f}"


# A complete TODDLERS Cloudy input always carries the `save last continuum` command
# (the one save common to every phase). The truncated stub is what a worker killed
# mid-generation leaves: title + table star + luminosity, then nothing.
_COMPLETE_IN = (
    '# Shell model at t = 0.10 Myr\n'
    'table star "SB99_kroupa100_sin_burst.ascii" age=100000.00 logZ=-2.0\n'
    'luminosity total=41.2\n'
    'init "shell_density_law_0.10.ini"\n'
    'sphere\n'
    'Radius inner = 6.89 parsec linear\n'
    'save last continuum ".cont" units micron\n'
    'save last overview ".ovr"\n'
)
_TRUNCATED_IN = (
    '# Shell model at t = 0.10 Myr\n'
    'table star "SB99_kroupa100_sin_burst.ascii" age=100000.00 logZ=-2.0\n'
    'luminosity total=41.2\n'
)


def _write_input(run_dir, prefix, text, time=TIME):
    path = os.path.join(run_dir, f"{_base(prefix, time)}.in")
    with open(path, "w") as f:
        f.write(text)
    return path


def _write_outputs(run_dir, prefix, time=TIME, *, ok=True, diffcont=True,
                   cont="#wl flux\n0.1 1.0\n", missing=()):
    """Write a healthy model's save files into ``run_dir``; tweak one to corrupt it.

    Hashed files (cont/ovr/phy/cum/cumEmer) carry a '#' header; rad is a bare data row
    (Cloudy can begin .rad with data, so the validator only requires it be non-empty).
    """
    b = _base(prefix, time)
    p = lambda ext: os.path.join(run_dir, f"{b}.{ext}")
    with open(p("out"), "w") as f:
        f.write("... iterations ...\n Cloudy exited OK\n" if ok else "... stopped early ...\n")
    files = {
        "cont": cont,
        "ovr": "#depth Te\n1 2\n",
        "phy": "#r ne\n1 2\n",
        "rad": "1.0 2.0\n",                 # data row, no '#': allowed
        "cum": "#line lum\nH  1  4861A  1e40\n",
        "cumEmer": "#line lum\nH  1  4861A  1e40\n",
    }
    for ext, content in files.items():
        if ext in missing:
            continue
        with open(p(ext), "w") as f:
            f.write(content)
    if diffcont:
        with open(p("diffContUnatt"), "w") as f:
            f.write("#energy/um\tDiffContUnatt\tDiffContAtt\n1.0 2.0 3.0\n")


# =========================================================================== input-level
class TestInputIntegrity:
    """check_input_exists rejects everything Cloudy cannot run (campaign killed mid-write)."""

    def test_complete_input_accepted(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_input(tmp_path, "shell", _COMPLETE_IN)
        h = CloudyOutputHandler("shell", TIME, parse_data=False)
        assert h.check_input_exists() is True

    def test_truncated_stub_rejected(self, tmp_path, monkeypatch):
        # The exact failure that re-armed the campaign: 3-line stub, no density/save block.
        monkeypatch.chdir(tmp_path)
        _write_input(tmp_path, "shell", _TRUNCATED_IN)
        h = CloudyOutputHandler("shell", TIME, parse_data=False)
        assert h.check_input_exists() is False

    def test_zero_byte_input_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_input(tmp_path, "shell", "")
        h = CloudyOutputHandler("shell", TIME, parse_data=False)
        assert h.check_input_exists() is False

    def test_missing_input_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        h = CloudyOutputHandler("shell", TIME, parse_data=False)
        assert h.check_input_exists() is False

    def test_input_without_save_block_rejected(self, tmp_path, monkeypatch):
        # Truncated *after* the radius but before the save block: non-empty, has density,
        # but Cloudy would still produce no save files we depend on -> treat as incomplete.
        monkeypatch.chdir(tmp_path)
        _write_input(tmp_path, "shell",
                     '# Shell model\ntable star "x.ascii" age=1e5 logZ=-2.0\n'
                     'init "shell_density_law_0.10.ini"\nRadius inner = 6.89 parsec linear\n')
        h = CloudyOutputHandler("shell", TIME, parse_data=False)
        assert h.check_input_exists() is False


# =========================================================================== output-level
class TestOutputIntegrity:
    """check_cloudy_success / _outputs_valid reject 'exited OK' models with unusable saves."""

    def test_healthy_outputs_accepted(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_outputs(tmp_path, "shell")
        h = CloudyOutputHandler("shell", TIME, parse_data=False)
        assert h.check_cloudy_success() is True

    def test_missing_out_is_pending(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        h = CloudyOutputHandler("shell", TIME, parse_data=False)
        assert h.check_cloudy_success() is False

    def test_out_without_success_message_rejected(self, tmp_path, monkeypatch):
        # Killed mid-run: .out exists but never printed "Cloudy exited OK".
        monkeypatch.chdir(tmp_path)
        _write_outputs(tmp_path, "shell", ok=False)
        h = CloudyOutputHandler("shell", TIME, parse_data=False)
        assert h.check_cloudy_success() is False

    def test_zero_byte_essential_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_outputs(tmp_path, "shell", cont="")          # 0-byte .cont, disk full
        h = CloudyOutputHandler("shell", TIME, parse_data=False)
        assert h.check_cloudy_success() is False

    def test_nul_clobbered_header_rejected(self, tmp_path, monkeypatch):
        # A Lustre/NFS "file hole": the first line is NUL bytes, not the '#' header.
        monkeypatch.chdir(tmp_path)
        _write_outputs(tmp_path, "shell", cont="\x00\x00\x00\x00\n0.1 1.0\n")
        h = CloudyOutputHandler("shell", TIME, parse_data=False)
        assert h.check_cloudy_success() is False

    def test_whitespace_only_header_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_outputs(tmp_path, "shell", cont="    \n0.1 1.0\n")
        h = CloudyOutputHandler("shell", TIME, parse_data=False)
        assert h.check_cloudy_success() is False

    def test_missing_hash_header_rejected(self, tmp_path, monkeypatch):
        # .cont with a data first line (no '#'): the parsers need that header.
        monkeypatch.chdir(tmp_path)
        _write_outputs(tmp_path, "shell", cont="0.1 1.0\n0.2 2.0\n")
        h = CloudyOutputHandler("shell", TIME, parse_data=False)
        assert h.check_cloudy_success() is False

    def test_missing_essential_file_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_outputs(tmp_path, "shell", missing=("cum",))  # no line-luminosity save
        h = CloudyOutputHandler("shell", TIME, parse_data=False)
        assert h.check_cloudy_success() is False

    def test_rad_without_hash_is_allowed(self, tmp_path, monkeypatch):
        # .rad legitimately begins with a data row; it must NOT be rejected for that.
        monkeypatch.chdir(tmp_path)
        _write_outputs(tmp_path, "shell")                    # rad already "1.0 2.0\n"
        h = CloudyOutputHandler("shell", TIME, parse_data=False)
        assert h.check_cloudy_success() is True

    def test_dissolved_phase_needs_only_cont(self, tmp_path, monkeypatch):
        # The dissolved phase saves only .cont; absent .ovr/.cum must not fail it.
        monkeypatch.chdir(tmp_path)
        _write_outputs(tmp_path, "dissolved",
                       missing=("ovr", "phy", "rad", "cum", "cumEmer"))
        h = CloudyOutputHandler("dissolved", TIME, parse_data=False)
        assert h.check_cloudy_success() is True

    def test_validate_outputs_false_bypasses_integrity(self, tmp_path, monkeypatch):
        # The cheap ".out says OK" check, used where save-file integrity isn't needed.
        monkeypatch.chdir(tmp_path)
        _write_outputs(tmp_path, "shell", cont="")           # corrupt, but...
        h = CloudyOutputHandler("shell", TIME, parse_data=False)
        assert h.check_cloudy_success(validate_outputs=False) is True
        assert h.check_cloudy_success(validate_outputs=True) is False


# =========================================================================== runner glue
def _install_fake_simmanager(monkeypatch, run_dir, output_writer):
    """Replace CloudySimulationManager so run_cloudy_task runs without Cloudy.

    The fake's ``write_input_file`` regenerates a *complete* input and ``run_simulation``
    invokes ``output_writer(phase, time)`` (writing whatever save-file state the test
    wants) into the run dir. run_cloudy_task imports the class inside the function, so we
    patch it on its home module.
    """
    import toddlers.cloudy_simulation_manager as csm

    class _Gen:
        def __init__(self, phase):
            self.phase = phase

        def write_input_file(self, time, inner_prefix=None):
            _write_input(run_dir, self.phase, _COMPLETE_IN, time)
            return True

    class _SM:
        def __init__(self, sim_file, **kwargs):
            self.cloudy_run_dir = str(run_dir)
            self.cloudy_exec = None

        def get_model_generator(self, phase):
            return _Gen(phase)

        def run_simulation(self, time, phase):
            output_writer(phase, time)

    monkeypatch.setattr(csm, "CloudySimulationManager", _SM)


class TestRunnerSelfHeal:
    """run_cloudy_task turns broken on-disk state into recovery or a loud failure."""

    @pytest.fixture
    def sim_file(self, tmp_path):
        f = tmp_path / "sim.dat"
        f.write_text("# dummy evolution output\n")
        return str(f)

    def test_truncated_input_is_regenerated(self, tmp_path, monkeypatch, capsys, sim_file):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_input(str(run_dir), "shell", _TRUNCATED_IN)          # the killed-mid-write stub
        _install_fake_simmanager(monkeypatch, str(run_dir),
                                 lambda ph, t: _write_outputs(str(run_dir), ph, t))

        res = runner.run_cloudy_task(
            {"sim_file": sim_file, "time": TIME, "phase": "shell", "add_dig": False})

        assert res is not None
        out = capsys.readouterr().out
        assert "[self-heal]" in out and "truncated input" in out
        # the stub was replaced by a complete input
        regenerated = (run_dir / f"{_base('shell')}.in").read_text()
        assert "save last continuum" in regenerated

    def test_invalid_prior_output_triggers_rerun(self, tmp_path, monkeypatch, capsys, sim_file):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_input(str(run_dir), "shell", _COMPLETE_IN)           # input is fine...
        _write_outputs(str(run_dir), "shell", ok=False)            # ...but prior .out is broken
        _install_fake_simmanager(monkeypatch, str(run_dir),
                                 lambda ph, t: _write_outputs(str(run_dir), ph, t))  # rerun -> healthy

        res = runner.run_cloudy_task(
            {"sim_file": sim_file, "time": TIME, "phase": "shell", "add_dig": False})

        assert res is not None
        assert "invalid prior output" in capsys.readouterr().out

    def test_unusable_output_raises(self, tmp_path, monkeypatch, sim_file):
        # Model "exits OK" but writes a 0-byte .cont -> must FAIL (so the worker records it
        # and the resume gate re-runs), not pass silently into the STAB build.
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _install_fake_simmanager(monkeypatch, str(run_dir),
                                 lambda ph, t: _write_outputs(str(run_dir), ph, t, cont=""))

        with pytest.raises(RuntimeError):
            runner.run_cloudy_task(
                {"sim_file": sim_file, "time": TIME, "phase": "shell", "add_dig": False})


class TestPatchCheck:
    """The one-time [patch-check] warning when the binary lacks the unattenuated patch."""

    @pytest.fixture
    def sim_file(self, tmp_path):
        f = tmp_path / "sim.dat"
        f.write_text("# dummy\n")
        return str(f)

    def test_warns_once_when_unpatched(self, tmp_path, monkeypatch, capsys, sim_file):
        monkeypatch.setattr(runner, "_PATCH_CHECK_LOGGED", False)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # no .diffContUnatt written -> binary looks unpatched
        _install_fake_simmanager(monkeypatch, str(run_dir),
                                 lambda ph, t: _write_outputs(str(run_dir), ph, t, diffcont=False))

        runner.run_cloudy_task({"sim_file": sim_file, "time": TIME, "phase": "shell", "add_dig": False})
        first = capsys.readouterr().out
        assert "[patch-check]" in first and "unattenuated-continuum patch" in first

        # second task in the same worker: warning must NOT repeat
        runner.run_cloudy_task({"sim_file": sim_file, "time": 0.12 * MYR_TO_SEC,
                                "phase": "shell", "add_dig": False})
        assert "[patch-check]" not in capsys.readouterr().out

    def test_silent_when_patched(self, tmp_path, monkeypatch, capsys, sim_file):
        monkeypatch.setattr(runner, "_PATCH_CHECK_LOGGED", False)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _install_fake_simmanager(monkeypatch, str(run_dir),
                                 lambda ph, t: _write_outputs(str(run_dir), ph, t, diffcont=True))

        runner.run_cloudy_task({"sim_file": sim_file, "time": TIME, "phase": "shell", "add_dig": False})
        assert "[patch-check]" not in capsys.readouterr().out


# =========================================================================== resume math
class TestResumeAccounting:
    """check_status must converge, not loop, when the gate re-runs tasks across rounds."""

    def _run(self, tmp_path, task_lines, results_files):
        task_file = tmp_path / "t.tasks"
        task_file.write_text("\n".join(task_lines))
        rdir = tmp_path / "res"
        rdir.mkdir()
        for name, content in results_files.items():
            (rdir / name).write_text(content)
        resume = tmp_path / "resume.tasks"
        with pytest.raises(SystemExit) as exc:
            check_status.main(["--task-file", str(task_file),
                               "--results", str(rdir / "*.results"), "-o", str(resume)])
        remaining = [json.loads(ln) for ln in resume.read_text().splitlines() if ln.strip()]
        return exc.value.code, remaining

    def test_ok_in_a_later_round_beats_stale_fail(self, tmp_path):
        # Task 0 FAILED in round 1 then succeeded in round 2; it must NOT be re-run, while
        # the still-failed task 1 and the never-run task 2 remain. (Guards against the
        # gate re-arming a task that is already fixed -> the infinite-loop class of bug.)
        tasks = [json.dumps({"i": i}) for i in range(3)]
        code, remaining = self._run(tmp_path, tasks, {
            "cloudy_shell_1.results": "0\tFAIL\tboom\n1\tFAIL\tboom\n",
            "cloudy_shell_2.results": "0\tOK\t\n",
        })
        assert code == 1
        assert {t["i"] for t in remaining} == {1, 2}

    def test_malformed_results_lines_are_ignored(self, tmp_path):
        # Garbled/partial lines (a half-written results file) must not crash accounting.
        tasks = [json.dumps({"i": i}) for i in range(2)]
        code, remaining = self._run(tmp_path, tasks, {
            "a.results": "0\tOK\t\ngarbage-no-tabs\n\nnotanint\tOK\t\n",
        })
        assert code == 1
        assert {t["i"] for t in remaining} == {1}     # task 0 OK; task 1 still pending
