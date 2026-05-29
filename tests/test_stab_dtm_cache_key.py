#!/usr/bin/env python3
"""Regression test: the interpolant selective cache must key on DTM.

A variable-DTM build runs one ``toddlers.stab.interpolants`` invocation per DTM, all
sharing the same on-disk cache dir, each reading a different set of Cloudy output (the
``_dtmX.XX`` dirs). If ``DataManager._get_cache_path`` omits DTM, the first DTM's cache
entry is reused for every later DTM and all per-DTM interpolants come out byte-identical
-- a degenerate DTM axis (observed in the 2026-05 variable-DTM campaign: f_dust 0.02 and
1.0 SFR-scaled SEDs were identical). DTM=1.0 must carry no suffix so it matches the
no-suffix baseline Cloudy dirs.
"""
import sys
import types
from pathlib import Path

project_root = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(project_root))

from toddlers.stab.interpolants import DataManager


def _cache_path(tmp_path, dtm):
    """Build a DataManager without its heavy __init__ and ask for a cache path."""
    dm = DataManager.__new__(DataManager)
    dm.cache_dir = Path(tmp_path)
    dm.generator = types.SimpleNamespace(dust_to_metal=dtm)
    return dm._get_cache_path(0.008, 0.05, 80.0, 6.25)


def test_distinct_dtm_gives_distinct_cache_paths(tmp_path):
    paths = {dtm: _cache_path(tmp_path, dtm)
             for dtm in (0.001, 0.02, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0)}
    # every DTM must map to a unique cache file -> no cross-DTM reuse
    assert len({str(p) for p in paths.values()}) == len(paths)


def test_dtm_suffix_format_and_baseline(tmp_path):
    # non-baseline carries a _dtm<val> suffix (general format, no spurious trailing zeros);
    # baseline (1.0) carries none; small values stay distinct (1e-3 must NOT collapse to 0.00).
    assert _cache_path(tmp_path, 0.20).name == "Z0.008_eta0.05_n80.0_logM6.25_dtm0.2.pkl"
    assert _cache_path(tmp_path, 0.001).name == "Z0.008_eta0.05_n80.0_logM6.25_dtm0.001.pkl"
    assert _cache_path(tmp_path, 1.0).name == "Z0.008_eta0.05_n80.0_logM6.25.pkl"


def test_same_dtm_is_stable(tmp_path):
    assert _cache_path(tmp_path, 0.40) == _cache_path(tmp_path, 0.40)
