#!/usr/bin/env python3
"""Plot our cloud-family STAB cells against the shipped reference.

Diagnostic companion to ``toddlers.stab.compare``: instead of a numeric verdict it
draws (a) SED overlays ours-vs-reference at a representative cloud for a few times and
(b) the ours/reference ratio vs wavelength for every overlapping cell at a fixed time.
A flat ratio across wavelength indicates a uniform (e.g. Cloudy-version) scaling; ratio
structure at lines / ionization edges / the dust peak indicates a pipeline difference.

Usage::

    python compare_plot.py OURS_DIR REF_DIR --prefix BPASS_chab100_bin --out plots
"""
import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toddlers.stab.stab_io import read_stab_file
from toddlers.stab.compare import ref_candidates, axis, match_idx, PARAM_AXES


def load_pair(our_path, ref_path):
    our, ref = read_stab_file(str(our_path)), read_stab_file(str(ref_path))
    wlo, wlr = axis(our, "lambda"), axis(ref, "lambda")
    assert wlo.shape == wlr.shape and np.allclose(wlo, wlr, rtol=1e-6), "wavelength grids differ"
    iax = {ax: match_idx(axis(ref, ax), axis(our, ax)) for ax in PARAM_AXES}
    to, tr = axis(our, "time"), axis(ref, "time")
    iT_r = np.array([int(np.argmin(np.abs(tr - tv))) for tv in to])
    vo = np.asarray(our["values"])[0]                      # [lambda,time,Z,SFE,n,M]
    vr = np.asarray(ref["values"])[0][:, iT_r][:, :, iax["Z"]][:, :, :, iax["SFE"]]
    vr = vr[:, :, :, :, iax["n_cl"]][:, :, :, :, :, iax["M_cl"]]
    return wlo * 1e6, to, vo, vr  # lambda -> micron for plotting


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours_dir"); ap.add_argument("ref_dir")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", default="plots")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ours_dir, ref_dir = Path(args.ours_dir), Path(args.ref_dir)

    for var in ["hr", "noDust_hr"]:
        op = ours_dir / f"ToddlersCloudSEDFamily_{args.prefix}_{var}.stab"
        rp = next((ref_dir / c for c in ref_candidates(args.prefix, var)
                   if (ref_dir / c).exists()), None)
        if not op.exists() or rp is None:
            print(f"skip {var}: ours={op.exists()} ref={rp is not None}")
            continue
        lam, t, vo, vr = load_pair(op, rp)
        # representative cloud = last node on each axis (Z,SFE,n,M all index -1)
        cell = (slice(None), slice(None), -1, -1, -1, -1)
        so, sr = vo[cell], vr[cell]   # [lambda,time]
        # pick 3 times spread across the grid
        tidx = [int(0.1 * len(t)), int(0.5 * len(t)), int(0.9 * len(t))]

        # ---- Fig A: SED overlay + ratio at the representative cloud ----
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                     gridspec_kw=dict(height_ratios=[2, 1]))
        for k in tidx:
            ln, = a1.loglog(lam, sr[:, k], lw=2, alpha=0.5, label=f"ref t={t[k]:.1f} Myr")
            a1.loglog(lam, so[:, k], lw=1, ls="--", color=ln.get_color(),
                      label=f"ours t={t[k]:.1f} Myr")
            m = (sr[:, k] > sr[:, k].max() * 1e-10) & (so[:, k] > 0)
            a2.semilogx(lam[m], so[:, k][m] / sr[:, k][m], color=ln.get_color(), lw=1)
        a1.set_ylabel("L  [erg/s/micron]"); a1.legend(fontsize=7, ncol=2)
        a1.set_title(f"{args.prefix}  {var}  (Z,SFE,n,M = max node)")
        a2.axhline(1.0, color="k", lw=0.6)
        a2.set_ylabel("ours / ref"); a2.set_xlabel("wavelength [micron]")
        a2.set_ylim(0.9, 1.1)
        fig.tight_layout(); fp = f"{args.out}/sed_overlay_{args.prefix}_{var}.png"
        fig.savefig(fp, dpi=130); plt.close(fig); print("wrote", fp)

        # ---- Fig B: ratio vs wavelength, all 16 cells, at the mid time ----
        kmid = tidx[1]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        nZ, nS, nN, nM = vo.shape[2:]
        for iz in range(nZ):
            for js in range(nS):
                for kn in range(nN):
                    for lm in range(nM):
                        o, r = vo[:, kmid, iz, js, kn, lm], vr[:, kmid, iz, js, kn, lm]
                        m = (r > r.max() * 1e-10) & (o > 0)
                        ax.semilogx(lam[m], o[m] / r[m], lw=0.5, alpha=0.4)
        med = []
        for il in range(vo.shape[0]):
            o, r = vo[il, kmid].ravel(), vr[il, kmid].ravel()
            m = (r > 0) & (o > 0)
            med.append(np.median(o[m] / r[m]) if m.any() else np.nan)
        ax.semilogx(lam, med, "k", lw=1.6, label="median over 16 cells")
        ax.axhline(1.0, color="r", lw=0.7, ls=":")
        ax.set_ylim(0.85, 1.15); ax.set_ylabel("ours / ref")
        ax.set_xlabel("wavelength [micron]")
        ax.set_title(f"{args.prefix}  {var}  ratio @ t={t[kmid]:.1f} Myr (all 16 cells)")
        ax.legend(fontsize=8)
        fig.tight_layout(); fp = f"{args.out}/ratio_allcells_{args.prefix}_{var}.png"
        fig.savefig(fp, dpi=130); plt.close(fig); print("wrote", fp)


if __name__ == "__main__":
    main()
