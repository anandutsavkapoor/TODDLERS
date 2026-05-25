#!/usr/bin/env python3
"""Confirm the hr (line) and lr (continuum) SED tables behave as designed.

lr is a continuum-only sampling; hr adds resolved emission-line profiles on a denser
wavelength grid. This overlays the two for one cell/time and zooms onto the optical
nebular lines (Hbeta, [OIII], Halpha, [NII], [SII]) to show (i) hr carries the lines and
lr does not, and (ii) each line is sampled by several hr points (well-resolved profile).

Usage::  python hr_lr_check.py --interp-dir BPASS_chab100_bin_interp_tables --out plots
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toddlers.stab import config as cfg

LINES = {0.486133: r"H$\beta$", 0.495891: "[OIII]4959", 0.500684: "[OIII]5007",
         0.656280: r"H$\alpha$", 0.658345: "[NII]6584", 0.671644: "[SII]6716",
         0.673082: "[SII]6731"}


def load(D, tag):
    s = pickle.load(open(f"{D}/{tag}.pkl", "rb"))
    wl = 10.0**np.asarray(s.grid[0])               # axis0 = log10(lambda/micron)
    return wl, np.asarray(s.values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interp-dir", required=True)
    ap.add_argument("--out", default="plots")
    ap.add_argument("--time-index", type=int, default=8)   # ~2.1 Myr (young, strong lines)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    D = args.interp_dir
    P = cfg.MODEL_PREFIX

    wl_lr, v_lr = load(D, f"TODDLERS_totSED_lr_{P}")
    wl_hr, v_hr = load(D, f"TODDLERS_tot_hr_{P}_lines_emergent=True")
    k = args.time_index
    cell = (k, 1, 1, 1, 1)                          # Z,SFE,n,M = max node
    sed_lr = 10.0**v_lr[(slice(None),) + cell]
    sed_hr = 10.0**v_hr[(slice(None),) + cell]
    print(f"hr Nwl={wl_hr.size}, lr Nwl={wl_lr.size}, t-index {k}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    # ---- optical window: lines present in hr, absent in lr ----
    a = ax[0]
    o_hr = (wl_hr > 0.45) & (wl_hr < 0.70)
    o_lr = (wl_lr > 0.45) & (wl_lr < 0.70)
    a.semilogy(wl_hr[o_hr], sed_hr[o_hr], "C3", lw=0.8, label="hr (continuum + lines)")
    a.semilogy(wl_lr[o_lr], sed_lr[o_lr], "C0", lw=1.4, label="lr (continuum only)")
    for lam, name in LINES.items():
        a.axvline(lam, color="k", lw=0.4, ls=":", alpha=0.5)
    a.set_xlabel("wavelength [micron]"); a.set_ylabel(r"$L_\lambda$ [erg/s/micron]")
    a.set_title("(a) optical: hr resolves nebular lines, lr is continuum")
    a.legend(fontsize=9)

    # ---- tight zoom on Halpha+[NII]+[SII] showing sampling per line ----
    b = ax[1]
    z_hr = (wl_hr > 0.6535) & (wl_hr < 0.6755)
    z_lr = (wl_lr > 0.6535) & (wl_lr < 0.6755)
    b.plot(wl_hr[z_hr], sed_hr[z_hr], "C3.-", ms=4, lw=0.8, label="hr points")
    b.plot(wl_lr[z_lr], sed_lr[z_lr], "C0o-", ms=5, lw=1.4, label="lr points")
    for lam, name in LINES.items():
        if 0.6535 < lam < 0.6755:
            b.axvline(lam, color="k", lw=0.4, ls=":")
            b.text(lam, b.get_ylim()[1], name, rotation=90, va="top", fontsize=7)
    ascii_name = {0.656280: "Ha", 0.658345: "NII", 0.671644: "SII16", 0.673082: "SII31"}
    npts = {ascii_name[lam]: int(((wl_hr > lam - 5e-4) & (wl_hr < lam + 5e-4)).sum())
            for lam in LINES if 0.6535 < lam < 0.6755}
    print("hr points per line (+/-0.5nm):", npts)
    b.set_xlabel("wavelength [micron]"); b.set_ylabel(r"$L_\lambda$ [erg/s/micron]")
    b.set_title("(b) Halpha/[NII]/[SII] zoom: hr pts/line(0.5nm) " + str(npts))
    b.legend(fontsize=9)

    fig.tight_layout()
    fp = f"{args.out}/hr_lr_check_{P}.png"
    fig.savefig(fp, dpi=130); print("wrote", fp)


if __name__ == "__main__":
    main()
