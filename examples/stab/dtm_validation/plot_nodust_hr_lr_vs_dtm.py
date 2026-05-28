#!/usr/bin/env python3
r"""Dust-free (noDust) SED vs f_dust, for both lr and hr, at one parameter cell.

Shows the effect of the dust-to-metal scaling on the includeDust=false SEDs. The dust-free
continuum is the incident stellar spectrum plus the *unattenuated* nebular continuum; lr is
that continuum at R=300, hr adds the intrinsic emission lines at R=5e4. Because grains in the
Cloudy model still set the ionization/temperature, the nebular component depends on f_dust
even though no dust attenuation/emission appears in the output:

  continuum : stellar part is f_dust-invariant; the nebular part (free-free / free-bound /
              two-photon) weakens as f_dust rises (grains steal ionizing photons) - a ~10%
              effect, visible in the ratio row in the NIR/FIR.
  lines (hr): vary more strongly (recombination lines down, collisional/FS lines up).

The bottom row is L_lambda(f_dust) / L_lambda(f_dust=1), so deviations from unity are the
effect. These STABs are SFR-normalized over a 10 Myr constant-SFR history (no free time axis).

Usage::  python plot_nodust_hr_lr_vs_dtm.py --Z 0.02 --sfe 0.05 --ncl 160
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from _style import load_stab, cell_indices, cell_label

WL_LO, WL_HI = 0.1, 3000.0


def slices(filepath, Z, sfe, ncl):
    names, grids, values, wl = load_stab(filepath)
    iZ, iS, iN = cell_indices(names, grids, Z, sfe, ncl)
    dtm = grids[names.index("DTM")]
    seds = values[0, :, iZ, iS, iN, :]            # (lambda, DTM)
    label, _ = cell_label(grids, names, iZ, iS, iN)
    return wl, dtm, seds, label


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stab-dir", default=".")
    ap.add_argument("--prefix", default="BPASS_chab100_bin")
    ap.add_argument("--out", default="nodust_hr_lr_vs_dtm.png")
    ap.add_argument("--Z", type=float, default=0.02)
    ap.add_argument("--sfe", type=float, default=0.05)
    ap.add_argument("--ncl", type=float, default=160.0)
    args = ap.parse_args()

    base = f"{args.stab_dir}/ToddlersSFRNormalizedSEDFamily_{args.prefix}_noDust_DTM"
    data = {res: slices(f"{base}_{res}.stab", args.Z, args.sfe, args.ncl)
            for res in ("lr", "hr")}
    dtm = data["lr"][1]
    label = data["lr"][3]
    cnorm = mcolors.LogNorm(vmin=dtm.min(), vmax=dtm.max())
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(2, 2, figsize=(7.3, 5.0), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.4, 1]),
                           constrained_layout=True)
    for col, res in enumerate(("lr", "hr")):
        wl, _, seds, _ = data[res]
        win = (wl >= WL_LO) & (wl <= WL_HI)
        top, bot = ax[0, col], ax[1, col]
        lo, hi = np.inf, 0.0
        ref = seds[:, -1]                          # f_dust = 1 slice
        for k, fd in enumerate(dtm):
            col_k = cmap(cnorm(fd))
            top.loglog(wl, wl * seds[:, k], color=col_k, lw=0.9)
            pos = (wl * seds[:, k])[win]; pos = pos[pos > 0]
            if pos.size:
                lo, hi = min(lo, pos.min()), max(hi, pos.max())
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(ref > 0, seds[:, k] / ref, np.nan)
            bot.semilogx(wl, r, color=col_k, lw=0.8)
        top.set_xlim(WL_LO, WL_HI); top.set_ylim(lo * 0.4, hi * 3)
        top.set_title(f"noDust {res}" + ("  (continuum only)" if res == "lr"
                                         else "  (continuum + lines)"))
        bot.set_xlim(WL_LO, WL_HI)
        if res == "lr":                       # continuum-only: linear zoom shows the ~10% effect
            bot.set_yscale("linear"); bot.set_ylim(0.88, 1.22)
        else:                                 # lines span factors: log scale
            bot.set_yscale("log"); bot.set_ylim(0.3, 6)
        bot.axhline(1.0, color="0.5", lw=0.7, ls=":")
        bot.set_xlabel(r"$\lambda\ [\mu{\rm m}]$")
        if col == 0:
            top.set_ylabel(r"$\lambda L_\lambda$ [arb.]")
            bot.set_ylabel(r"$L_\lambda / L_\lambda(f_{\rm dust}{=}1)$")

    sm = plt.cm.ScalarMappable(norm=cnorm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, ticks=dtm, format="%.2g", pad=0.01)
    cb.set_label(r"$f_{\rm dust}$"); cb.ax.minorticks_off()
    fig.suptitle(r"Dust-free (noDust) SED vs $f_{\rm dust}$   "
                 + f"({label}; BPASS chab100 bin, 10 Myr)", fontsize=8)
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
