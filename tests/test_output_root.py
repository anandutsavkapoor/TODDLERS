#!/usr/bin/env python3
"""The Cloudy output location must be relocatable via $TODDLERS_OUTPUT_ROOT / --output-root.

Cloudy output is large (~0.5 TiB per DTM on a full grid). It was hardcoded to the package's
parent dir (dirname(dirname(__file__))/cloudy_output), which is wrong for a read-only / relocated
install and forced a symlink hack. These tests pin: (1) the shared resolver honors the env var
and otherwise preserves the historical default; (2) campaign exports TODDLERS_OUTPUT_ROOT into
both the Cloudy worker job and the build job (so the writer and the interpolant reader agree),
and points the archiver at the same root; (3) omitting it changes nothing (empty export -> default).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toddlers.utils import resolve_output_root
from toddlers.hpc import campaign


def test_resolver_uses_env_when_set(monkeypatch):
    monkeypatch.setenv("TODDLERS_OUTPUT_ROOT", "/scratch/run")
    assert resolve_output_root("/pkg/src") == "/scratch/run"


def test_resolver_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("TODDLERS_OUTPUT_ROOT", raising=False)
    assert resolve_output_root("/pkg/src") == "/pkg/src"


def _run_main(tmp_path, output_root):
    evo = (tmp_path / "evolution_output/template_BPASS/imf_chab100/star_type_bin/"
           "cluster_mode_burst/profile_type_uniform")
    evo.mkdir(parents=True)
    (evo / "sim_Z0.020_eta0.050_n160.0_logM6.00.dat").write_text("{}")
    work = tmp_path / "work"
    argv = [
        "--evolution-dir", str(evo), "--work-dir", str(work), "--stab-dir", str(tmp_path / "stab"),
        "--dust-to-metal", "1.0", "--stab", "none",
        "--account", "acc", "--partition", "doduo",
        "--toddlers-src", "/src", "--cloudy-exe", "/cl.exe", "--cloudy-data", "/d",
        "--python-module", "SciPy", "--dry-run",
    ]
    if output_root:
        argv += ["--output-root", output_root]
    campaign.main(argv)
    return ((work / "campaign_cloudy.sh").read_text(),
            (work / "campaign_stab.sh").read_text())


def test_output_root_exported_to_both_jobs(tmp_path):
    root = str(tmp_path / "scratch_run")
    cloudy, stab = _run_main(tmp_path, root)
    assert f"export TODDLERS_OUTPUT_ROOT={root}" in cloudy   # Cloudy workers write here
    assert f"export TODDLERS_OUTPUT_ROOT={root}" in stab     # interpolant build reads here
    assert f"{root}/cloudy_output" in stab                   # archiver targets the same root


def test_no_output_root_keeps_default(tmp_path):
    cloudy, stab = _run_main(tmp_path, "")
    # Cloudy template always carries the export; empty value -> runtime falls back to the default.
    assert "export TODDLERS_OUTPUT_ROOT=\n" in cloudy
    # No scratch root injected into the build job, and the archiver uses the package-src default.
    assert "export TODDLERS_OUTPUT_ROOT=/" not in stab
    assert "/src/cloudy_output" in stab
