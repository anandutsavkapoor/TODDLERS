#!/usr/bin/env python3
r"""Sanity check 1 - SFR-normalized SED versus dust-to-metal scaling f_dust.

For a single cloud cell (Z, SFE, n_cl) overplots lambda*L_lambda for every f_dust node
of the variable-DTM STAB, in the Dust and noDust variants. The physically expected
behaviour, and what this figure confirms:

  Dust    : UV/optical continuum suppressed and the FIR dust bump enhanced, monotonically
            in f_dust (energy absorbed in the UV/opt reappears in the FIR).
  noDust  : continuum is f_dust-invariant (no grains -> no dust emission/attenuation), a
            negative control; the emission lines still vary (nebular gas-state response).

Usage::

    python plot_sed_vs_dtm.py --stab-dir . --prefix BPASS_chab100_bin \
        --Z 0.02 --sfe 0.05 --ncl 160
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from _style import load_stab, cell_indices, cell_label

WL_LO, WL_HI = 0.1, 3000.0   # micron, plotting + y-limit window


def variant_curves(filepath, Z, sfe, ncl):
    names, grids, values, wl = load_stab(filepath)
    iZ, iS, iN = cell_indices(names, grids, Z, sfe, ncl)
    dtm = grids[names.index("DTM")]
    seds = values[0, :, iZ, iS, iN, :]               # (lambda, DTM)
    label, cell = cell_label(grids, names, iZ, iS, iN)
    return wl, dtm, seds, label, cell


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stab-dir", default=".")
    ap.add_argument("--prefix", default="BPASS_chab100_bin")
    ap.add_argument("--out", default="sed_vs_dtm.png")
    ap.add_argument("--Z", type=float, default=0.02)
    ap.add_argument("--sfe", type=float, default=0.05)
    ap.add_argument("--ncl", type=float, default=160.0)
    args = ap.parse_args()

    base = f"{args.stab_dir}/ToddlersSFRNormalizedSEDFamily_{args.prefix}"
    wl, dtm, sed_d, label, _ = variant_curves(f"{base}_DTM_hr.stab", args.Z, args.sfe, args.ncl)
    _, _, sed_n, _, _ = variant_curves(f"{base}_noDust_DTM_hr.stab", args.Z, args.sfe, args.ncl)

    # colour each f_dust by its value (log), continuous colourbar shows the gradient
    cnorm = mcolors.LogNorm(vmin=dtm.min(), vmax=dtm.max())
    cmap = plt.cm.viridis
    win = (wl >= WL_LO) & (wl <= WL_HI)

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.1), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, sed, title in ((axes[0], sed_d, "Dust (grains on)"),
                           (axes[1], sed_n, r"noDust (control)")):
        lo, hi = np.inf, 0.0
        for k, fd in enumerate(dtm):
            lL = wl * sed[:, k]
            ax.loglog(wl, lL, color=cmap(cnorm(fd)), lw=1.0)
            pos = lL[win][lL[win] > 0]
            if pos.size:
                lo, hi = min(lo, pos.min()), max(hi, pos.max())
        ax.set_xlim(WL_LO, WL_HI)
        ax.set_ylim(lo * 0.4, hi * 3)
        ax.axvspan(0.1, 0.4, color="0.85", zorder=0)
        ax.axvspan(10, 1000, color="navajowhite", alpha=0.45, zorder=0)
        ax.set_xlabel(r"$\lambda\ [\mu{\rm m}]$")
        ax.set_title(title)
        ax.text(0.04, 0.05, "UV", transform=ax.transAxes, fontsize=6, color="0.4")
        ax.text(0.74, 0.05, "FIR bump", transform=ax.transAxes, fontsize=6, color="0.5")
    axes[0].set_ylabel(r"$\lambda L_\lambda\ \mathrm{[erg\,s^{-1}\,(M_\odot\,yr^{-1})^{-1}]}$")

    sm = plt.cm.ScalarMappable(norm=cnorm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, ticks=dtm, format="%.2g", pad=0.01)
    cb.set_label(r"$f_{\rm dust}$  (dust-to-metal scaling)")
    cb.ax.minorticks_off()
    fig.suptitle("SFR-normalized SED vs $f_{\\rm dust}$   "
                 + f"({label}; BPASS chab100 bin, 10 Myr)", fontsize=8)
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
