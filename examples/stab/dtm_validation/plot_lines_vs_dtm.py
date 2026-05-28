#!/usr/bin/env python3
r"""Sanity check 2 - emission-line response to f_dust, for lines across the spectrum.

Confirms that every line (not just H-alpha) carries the f_dust-varying nebular data, and
that the wavelength dependence is physical. Each line's peak flux is normalized to its
value at f_dust=1 and plotted against f_dust, coloured by rest wavelength:

  emergent (Dust)    : UV/optical lines suppressed by dust attenuation, graded by
                       wavelength (Ly-alpha most, recombination lines next, IR least);
                       FIR fine-structure lines ([OI]63, [CII]158) ENHANCED, via dust
                       photoelectric heating of the gas.
  intrinsic (noDust) : recombination lines mildly down (grains compete for ionizing
                       photons); collisional optical + FIR fine-structure lines up
                       (photoelectric heating raises the electron temperature).

Usage::

    python plot_lines_vs_dtm.py --stab-dir . --prefix BPASS_chab100_bin
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from _style import load_stab, cell_indices, cell_label

# name -> rest wavelength [micron], listed in wavelength order (UV -> FIR). Each line gets
# a distinct colour + marker (a continuous wavelength colourmap collapses the optical lines
# into indistinguishable hues); the legend stays wavelength-ordered to show the gradient.
LINES = {
    r"Ly$\alpha$ 0.122": 0.1216, "C IV 0.155": 0.1549, "[O II] 0.373": 0.3727,
    r"H$\beta$ 0.486": 0.4861, "[O III] 0.501": 0.5007, r"H$\alpha$ 0.656": 0.6563,
    "[N II] 0.658": 0.6584, "[S II] 0.673": 0.6731, "[S IV] 10.5": 10.51,
    "[Ne II] 12.8": 12.81, "[O I] 63": 63.18, "[O III] 88": 88.36, "[C II] 158": 157.7,
}
# 13 visually distinct colours + markers, one per line (wavelength order)
COLORS = ["#1f3b8c", "#3b7dd8", "#27c4c4", "#1a9850", "#7fbc41", "#bd9e00",
          "#f6a800", "#f46d20", "#e8261e", "#b2182b", "#d6379f", "#7b3294", "#000000"]
MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*", "h", "<", ">", "p", "d"]


def line_peak_vs_dtm(values, wl, idx, lam0, frac=0.01):
    iZ, iS, iN = idx
    sed = values[0, :, iZ, iS, iN, :]                    # (lambda, DTM)
    m = (wl >= lam0 * (1 - frac)) & (wl <= lam0 * (1 + frac))
    if not m.any():
        return None
    return sed[m].max(axis=0)                            # peak per DTM


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stab-dir", default=".")
    ap.add_argument("--prefix", default="BPASS_chab100_bin")
    ap.add_argument("--out", default="lines_vs_dtm.png")
    ap.add_argument("--Z", type=float, default=0.02)
    ap.add_argument("--sfe", type=float, default=0.05)
    ap.add_argument("--ncl", type=float, default=160.0)
    args = ap.parse_args()

    base = f"{args.stab_dir}/ToddlersSFRNormalizedSEDFamily_{args.prefix}"

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.5), sharex=True, sharey=True,
                             constrained_layout=True)
    label = None
    handles = None
    for ax, variant, title in ((axes[0], "_DTM_hr", "emergent (Dust)"),
                               (axes[1], "_noDust_DTM_hr", "intrinsic (noDust)")):
        names, grids, values, wl = load_stab(f"{base}{variant}.stab")
        idx = cell_indices(names, grids, args.Z, args.sfe, args.ncl)
        dtm = grids[names.index("DTM")]
        label, _ = cell_label(grids, names, *idx)
        for i, (name, lam0) in enumerate(LINES.items()):
            c = line_peak_vs_dtm(values, wl, idx, lam0)
            if c is None or not np.all(np.isfinite(c)) or c[-1] <= 0:
                continue
            ax.plot(dtm, c / c[-1], ls="-", marker=MARKERS[i], ms=3.5, lw=1.0,
                    color=COLORS[i], mec="0.2", mew=0.3, label=name)
        ax.axhline(1.0, color="0.5", lw=0.7, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$f_{\rm dust}$")
        ax.set_title(title)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    axes[0].set_ylabel(r"line peak / value at $f_{\rm dust}=1$")
    # single legend to the right (shared; both panels show the same lines)
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=6.5, handlelength=1.4, labelspacing=0.35, title="line (UV$\\to$FIR)",
               title_fontsize=7)
    fig.suptitle("Emission lines vs $f_{\\rm dust}$   "
                 + f"({label}; BPASS chab100 bin, 10 Myr)", fontsize=8)
    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
