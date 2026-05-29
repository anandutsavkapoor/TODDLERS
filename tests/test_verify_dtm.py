#!/usr/bin/env python3
"""The per-DTM deletion gate must pass good products and fail broken/degenerate ones.

verify_dtm authorizes deleting a DTM's (truth) cloudy_output, so its deletion-critical checks
-- integrity, completeness, non-degeneracy -- are pinned here with synthetic interpolants.
(The physical and cloudy-truth checks need realistic SEDs / raw Cloudy output and are exercised
by the on-cluster self-test, not here.)
"""
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator as RGI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toddlers.hpc import verify_dtm as V

PREFIX = "BPASS_chab100_bin"
WL = np.array([0.1, 0.3, 1.0, 10.0, 50.0, 100.0, 500.0, 3000.0])  # micron, incl. FIR 30-300


def _rgi(values):
    grids = [np.log10(WL), np.linspace(0.1, 10, 4),
             np.array([1.0, 2.0]), np.array([1.0, 2.0]),
             np.array([1.0, 2.0]), np.array([1.0, 2.0])]
    return RGI(grids, values)


def _full_vals(fir_level=40.0):
    """6D log-L array (lam,t,Z,SFE,n,M); FIR points raised to fir_level for the bump."""
    shape = (len(WL), 4, 2, 2, 2, 2)
    v = np.full(shape, 38.0)
    fir = (WL >= 30) & (WL <= 300)
    v[fir] = fir_level                     # elevate FIR band -> controllable bump
    return v


def _write_dtm(interp_dir, dtm, values=None):
    """Write the four per-DTM SED interpolants for `dtm` into interp_dir."""
    values = _full_vals() if values is None else values
    for path in V._paths(str(interp_dir), PREFIX, dtm).values():
        with open(path, "wb") as fh:
            pickle.dump(_rgi(values), fh)


def test_integrity_pass(tmp_path):
    _write_dtm(tmp_path, 0.10)
    ok, _ = V.check_integrity(V._paths(str(tmp_path), PREFIX, 0.10))
    assert ok


def test_integrity_fail_when_a_product_missing(tmp_path):
    _write_dtm(tmp_path, 0.10)
    paths = V._paths(str(tmp_path), PREFIX, 0.10)
    Path(paths["tot_hr emergent (Dust hr)"]).unlink()
    ok, _ = V.check_integrity(paths)
    assert not ok


def test_integrity_fail_on_sentinel_shell(tmp_path):
    _write_dtm(tmp_path, 0.10, values=np.full((len(WL), 4, 2, 2, 2, 2), -99.0))
    ok, _ = V.check_integrity(V._paths(str(tmp_path), PREFIX, 0.10))
    assert not ok


def test_completeness_pass(tmp_path):
    _write_dtm(tmp_path, 0.10)
    ok, _ = V.check_completeness(V._paths(str(tmp_path), PREFIX, 0.10))
    assert ok


def test_completeness_fail_on_empty_cloud(tmp_path):
    v = _full_vals()
    v[:, :, 0, 0, 0, 0] = np.nan          # one cloud carries no data
    _write_dtm(tmp_path, 0.10, values=v)
    ok, _ = V.check_completeness(V._paths(str(tmp_path), PREFIX, 0.10))
    assert not ok


def test_nondegenerate_pass_when_distinct_and_monotonic(tmp_path):
    _write_dtm(tmp_path, 0.02, values=_full_vals(fir_level=39.0))   # less dust, smaller bump
    _write_dtm(tmp_path, 0.10, values=_full_vals(fir_level=40.0))   # more dust, bigger bump
    ok, _ = V.check_nondegenerate(str(tmp_path), PREFIX, 0.10, 0.02)
    assert ok


def test_nondegenerate_fail_when_byte_identical(tmp_path):
    _write_dtm(tmp_path, 0.02, values=_full_vals(fir_level=40.0))
    # copy 0.02's products onto 0.10's filenames -> byte-identical (the degenerate-axis bug)
    for k, src in V._paths(str(tmp_path), PREFIX, 0.02).items():
        shutil.copy(src, V._paths(str(tmp_path), PREFIX, 0.10)[k])
    ok, _ = V.check_nondegenerate(str(tmp_path), PREFIX, 0.10, 0.02)
    assert not ok


def test_nondegenerate_fail_on_wrong_direction(tmp_path):
    _write_dtm(tmp_path, 0.02, values=_full_vals(fir_level=41.0))   # smaller f_dust but BIGGER bump
    _write_dtm(tmp_path, 0.10, values=_full_vals(fir_level=39.0))   # bigger f_dust, smaller bump (wrong)
    ok, _ = V.check_nondegenerate(str(tmp_path), PREFIX, 0.10, 0.02)
    assert not ok
