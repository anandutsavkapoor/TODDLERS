#!/usr/bin/env python3
"""The Cloudy grain small-to-large ratio must be configurable for the v2-DTM grid.

The paper's v2-DTM templates use an ISM-like ratio of 0.40, vs the 0.10 (Orion-like) v2
default. The HPC campaign exposes ``--small-to-large-ratio`` which is threaded to the Cloudy
workers via the ``TODDLERS_SMALL_TO_LARGE_RATIO`` env var; the Cloudy input generator reads
that env when no explicit override is given. These tests pin both halves.
"""
import os
import sys
import types
from pathlib import Path

project_root = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(project_root))

from toddlers.cloudy_base_input_generator import resolve_small_to_large_ratio
from toddlers.constants import SMALL_TO_LARGE_MASS_RATIO
from toddlers.hpc import campaign

ENV = "TODDLERS_SMALL_TO_LARGE_RATIO"


def test_resolve_explicit_override_wins(monkeypatch):
    monkeypatch.setenv(ENV, "0.40")
    assert resolve_small_to_large_ratio(0.25) == 0.25            # override beats env


def test_resolve_env_used_when_no_override(monkeypatch):
    monkeypatch.setenv(ENV, "0.40")
    assert resolve_small_to_large_ratio(None) == 0.40


def test_resolve_empty_or_unset_falls_back_to_default(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert resolve_small_to_large_ratio(None) == SMALL_TO_LARGE_MASS_RATIO  # 0.10 (v2)
    monkeypatch.setenv(ENV, "")                                  # empty string is not a value
    assert resolve_small_to_large_ratio(None) == SMALL_TO_LARGE_MASS_RATIO


def _cloudy_args(tmp_path, ratio):
    taskdir = tmp_path / "tasks"; taskdir.mkdir()
    for ph in ("shell", "unified", "dissolved"):
        (taskdir / f"cloudy_{ph}.tasks").write_text("dummytaskline\n")
    args = types.SimpleNamespace(
        account="acc", partition="part", ntasks=128, walltime="03:00:00",
        python_module="SciPy", activate_env="", toddlers_src="/src", toddlers_data="/data",
        work_dir=str(tmp_path), cloudy_exe="/cl.exe", small_to_large_ratio=ratio,
        add_dig=False, max_nodes=8, output_root="", dry_run=True)
    campaign._submit_cloudy_chain(args, taskdir)
    return (tmp_path / "campaign_cloudy.sh").read_text()


def test_campaign_sets_ratio_when_given(tmp_path):
    txt = _cloudy_args(tmp_path, 0.40)
    assert "export TODDLERS_SMALL_TO_LARGE_RATIO=0.4" in txt


def test_campaign_leaves_ratio_empty_by_default(tmp_path):
    txt = _cloudy_args(tmp_path, None)
    # placeholder filled with empty string -> generator falls back to the 0.10 default
    assert "export TODDLERS_SMALL_TO_LARGE_RATIO=\n" in txt
