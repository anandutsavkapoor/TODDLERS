#!/usr/bin/env python3
"""Compare two cloud-family STAB libraries cell-by-cell.

Intended for validation: confirm that a STAB produced by this pipeline reproduces a
reference STAB (e.g. the shipped SKIRT ``TODDLERS_Cloud_*`` resource) at the parameter
cells they have in common. The "ours" library is typically a small subset grid and the
reference is the full production grid; we compare only at the overlapping nodes.

At a grid node a ``RegularGridInterpolator`` returns the stored node value, so the STAB
cell is the original Cloudy SED for that cloud, not an interpolation. If the input
evolution runs are the same verified runs used for the reference, and the Cloudy stage
matches, the overlapping cells should be byte-for-byte identical.

Usage::

    python compare_cloud_stabs.py OURS_DIR REF_DIR --prefix BPASS_chab100_bin

Compares all four variants (Dust/noDust x lr/hr) named
``ToddlersCloudSEDFamily_<prefix>{,_noDust}_{lr,hr}.stab`` in each directory.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from .stab_io import read_stab_file

VARIANTS = ["lr", "hr", "noDust_lr", "noDust_hr"]
PARAM_AXES = ["Z", "SFE", "n_cl", "M_cl"]


def ref_candidates(prefix, var):
    """Reference filenames to try for one of our variants.

    Our pipeline-native names are ``ToddlersCloudSEDFamily_<prefix>_<var>.stab`` with
    ``var`` in {lr, hr, noDust_lr, noDust_hr} (the Dust variant carries no token). The
    shipped SKIRT reference is the post-``rename.py`` convention
    ``ToddlersSEDFamily_Cloud_<prefix>_<Dust|noDust>_<lr|hr>.stab`` (Dust explicit). Try
    the renamed form first, then fall back to the native name (ref also un-renamed)."""
    res = "lr" if var.endswith("lr") else "hr"
    dust = "noDust" if "noDust" in var else "Dust"
    return [
        f"ToddlersSEDFamily_Cloud_{prefix}_{dust}_{res}.stab",  # shipped (renamed)
        f"ToddlersCloudSEDFamily_{prefix}_{var}.stab",          # pipeline-native
    ]


def axis(d, name):
    return np.asarray(d["axisGrids"][d["axisNames"].index(name)])


def match_idx(ref_vals, our_vals, rtol=1e-3):
    """Index, for each our_val, of the matching ref grid node (must agree within rtol)."""
    idx = []
    for ov in our_vals:
        j = int(np.argmin(np.abs(ref_vals - ov)))
        rel = abs(ref_vals[j] - ov) / max(abs(ov), 1e-30)
        if rel >= rtol:
            raise ValueError(f"no reference node matches {ov} (nearest {ref_vals[j]}, rel {rel:.2e})")
        idx.append(j)
    return np.array(idx)


def compare_variant(our_path, ref_path, time_tol):
    our = read_stab_file(str(our_path))
    ref = read_stab_file(str(ref_path))

    wlo, wlr = axis(our, "lambda"), axis(ref, "lambda")
    print(f"  lambda: ours N={wlo.size} [{wlo.min():.3e},{wlo.max():.3e}]  "
          f"ref N={wlr.size} [{wlr.min():.3e},{wlr.max():.3e}]")
    if wlo.shape != wlr.shape or not np.allclose(wlo, wlr, rtol=1e-6):
        print("  !! wavelength grids differ -> cannot compare values (line list / resolution mismatch)")
        return

    iax = {ax: match_idx(axis(ref, ax), axis(our, ax)) for ax in PARAM_AXES}

    to, tr = axis(our, "time"), axis(ref, "time")
    iT_o, iT_r = [], []
    for k, tv in enumerate(to):
        j = int(np.argmin(np.abs(tr - tv)))
        if abs(tr[j] - tv) < time_tol:
            iT_o.append(k); iT_r.append(j)
    iT_o, iT_r = np.array(iT_o), np.array(iT_r)
    exact_time = (to.size == tr.size and np.allclose(to, tr, atol=1e-6))
    print(f"  time: {'identical grids' if exact_time else f'matched {iT_o.size}/{to.size} within {time_tol} Myr'}")

    lam = np.arange(wlo.size)
    # ours is the subset (axes in our order -> identity); ref uses matched indices
    nax = [len(axis(our, ax)) for ax in PARAM_AXES]
    vo = np.asarray(our["values"])[0][np.ix_(lam, iT_o, *[np.arange(n) for n in nax])]
    vr = np.asarray(ref["values"])[0][np.ix_(lam, iT_r, iax["Z"], iax["SFE"], iax["n_cl"], iax["M_cl"])]
    print(f"  compared sub-array shape: {vo.shape}")

    if np.array_equal(vo, vr):
        print("  >> BYTE-IDENTICAL at the overlapping cells")
        return
    mask = np.isfinite(vr) & np.isfinite(vo) & (vr > vr.max() * 1e-10) & (vo > 0)
    rel = np.abs(vo[mask] - vr[mask]) / np.abs(vr[mask])
    print(f"  not bit-identical. non-trivial points: {mask.sum()}/{mask.size}")
    print(f"  rel diff: median={np.median(rel):.3e}  p95={np.percentile(rel, 95):.3e}  max={rel.max():.3e}")
    print(f"  ref med/max: {np.median(vr[mask]):.3e}/{vr[mask].max():.3e}  "
          f"ours med/max: {np.median(vo[mask]):.3e}/{vo[mask].max():.3e}")


def main():
    ap = argparse.ArgumentParser(description="Compare two cloud-family STAB libraries cell-by-cell.")
    ap.add_argument("ours_dir", help="dir with our ToddlersCloudSEDFamily_*.stab")
    ap.add_argument("ref_dir", help="dir with the reference ToddlersCloudSEDFamily_*.stab")
    ap.add_argument("--prefix", required=True, help="model prefix, e.g. BPASS_chab100_bin or SB99_kroupa100_sin")
    ap.add_argument("--time-tol", type=float, default=0.06,
                    help="Myr tolerance when time grids round differently (default 0.06)")
    args = ap.parse_args()

    ours_dir, ref_dir = Path(args.ours_dir), Path(args.ref_dir)
    for var in VARIANTS:
        fn = f"ToddlersCloudSEDFamily_{args.prefix}_{var}.stab"
        op = ours_dir / fn
        rp = next((ref_dir / c for c in ref_candidates(args.prefix, var)
                   if (ref_dir / c).exists()), None)
        print(f"\n===== {args.prefix} {var} =====")
        print(f"  ours: {op.name}")
        if not op.exists() or rp is None:
            print(f"  missing: ours={op.exists()} ref={rp is not None} "
                  f"(tried {ref_candidates(args.prefix, var)}) -> skip")
            continue
        print(f"  ref : {rp.name}")
        compare_variant(op, rp, args.time_tol)


if __name__ == "__main__":
    main()
