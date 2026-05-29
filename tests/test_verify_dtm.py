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


# --- recollapse / case-B correlation (cell_logr + evolution_qabs) ---
def _q_history(jump=True):
    t = np.linspace(0.1, 25, 300)
    q = 1e52 * np.exp(-t / 4) + 1e50            # first generation declines
    if jump:
        late = t > 13
        q[late] += 3e52 * np.exp(-(t[late] - 13) / 4)   # recollapse: new generation
    return t, q


def test_cell_logr_tracking_high_r():
    t_ev, q = _q_history()
    t_cl = np.linspace(0.1, 22, 60)             # different (coarser) grid -> needs interpolation
    ha = 1.37e-12 * np.interp(t_cl, t_ev, q)    # intrinsic Halpha tracks Q_abs (case B)
    assert V.cell_logr(t_cl, ha, t_ev, q) > 0.99


def test_cell_logr_flat_is_zero():
    t_ev, q = _q_history()
    t_cl = np.linspace(0.1, 22, 60)
    ha = np.full(t_cl.size, 1e44)               # Halpha not varying -> not tracking
    assert V.cell_logr(t_cl, ha, t_ev, q) == 0.0


def test_cell_logr_missed_recollapse_drops_r():
    t_ev, q = _q_history(jump=True)
    _, q_nojump = _q_history(jump=False)
    t_cl = np.linspace(0.1, 22, 60)
    r_track = V.cell_logr(t_cl, 1.37e-12 * np.interp(t_cl, t_ev, q), t_ev, q)
    r_missed = V.cell_logr(t_cl, 1.37e-12 * np.interp(t_cl, t_ev, q_nojump), t_ev, q)
    assert r_missed < r_track and r_missed < 0.9   # missing the recollapse jump hurts correlation


def test_cell_logr_insufficient_overlap_returns_none():
    t_ev = np.linspace(20, 25, 100)             # no overlap with the early Cloudy grid
    t_cl = np.linspace(0.1, 5, 50)
    assert V.cell_logr(t_cl, np.ones(50), t_ev, np.ones(100)) is None


def test_evolution_qabs_sums_generations_and_components(monkeypatch):
    import toddlers.track_simulation as ts
    s = V.SEC_PER_MYR
    g1 = {"time": np.array([1, 2, 3.]) * s,
          "stellar_feedback": {"Q_i": np.array([[1e52], [5e51], [2e51]])},   # (N,1)
          "shell_properties": {"f_esc_i": np.array([0.0, 0.1, 0.2])}}
    g2 = {"time": np.array([10, 11.]) * s,
          "stellar_feedback": {"Q_i": np.array([[3e52, 1e52], [2e52, 5e51]])},  # (N,2): 2 active gens
          "shell_properties": {"f_esc_i": np.array([0.0, 0.05])}}
    monkeypatch.setattr(ts, "load_output_file", lambda p: ({}, [g1, g2]))
    t, q, ngen = V.evolution_qabs("dummy.dat")
    assert ngen == 2
    np.testing.assert_allclose(t, [1, 2, 3, 10, 11], rtol=1e-3)       # seconds -> Myr
    np.testing.assert_allclose(q[0], 1e52 * 1.0, rtol=1e-6)           # gen1 t=1
    np.testing.assert_allclose(q[3], (3e52 + 1e52) * 1.0, rtol=1e-6)  # gen2 t=10: components summed
    assert q[3] > q[2]                                                # recollapse jump preserved
