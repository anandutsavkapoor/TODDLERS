#!/usr/bin/env python3
r"""Sanity check 3 - variable-DTM STAB at f_dust=1 versus the shipped SKIRT reference.

f_dust=1 is the published baseline (D/G/Z = 0.456 at solar), so the new STAB's last DTM
slice should reproduce the shipped single-DTM ``ToddlersSEDFamily_SFRNormalized_*`` library
at the overlapping (Z, SFE, n_cl) nodes. Two known, intended differences are expected:

  Dust   : a few-percent residual from the post-reference n_H He-conversion fix
           (the libraries are not byte-identical by design).
  noDust : the new build adds the unattenuated nebular continuum (free-free / free-bound),
           which the reference lacks, lifting the MIR/FIR off a near-zero baseline; this is
           an additive, energetically negligible (>~6 dex below the optical peak) tail.

Prints a per-band median/90th/max fractional-difference table and writes an SED overlay
with a fractional-residual subpanel. The reference library defaults to the SKIRT resources
tree; point ``--ref-dir`` at your local copy.

Usage::

    python compare_vs_reference.py --stab-dir . \
        --ref-dir /path/to/SKIRT9_Resources_TODDLERS/SED_TODDLERS --window 10Myr
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from _style import load_stab, nearest, cell_indices, cell_label

BANDS = {"UV 0.1-0.3": (0.1, 0.3), "opt 0.3-0.8": (0.3, 0.8), "NIR 0.8-5": (0.8, 5.0),
         "MIR 5-30": (5.0, 30.0), "FIR 30-1000": (30.0, 1000.0)}
WL_LO, WL_HI = 0.1, 3000.0


def ref_path(ref_dir, prefix, dust, res, window):
    return f"{ref_dir}/ToddlersSEDFamily_SFRNormalized_{prefix}_{dust}_{res}_{window}.stab"


def new_path(stab_dir, prefix, nd, res):
    return f"{stab_dir}/ToddlersSFRNormalizedSEDFamily_{prefix}{nd}_DTM_{res}.stab"


def overlapping_slices(new_f, ref_f, Z, sfe, ncl):
    nn, ng, nv, wl = load_stab(new_f)
    rn, rg, rv, _ = load_stab(ref_f)
    niZ, niS, niN = cell_indices(nn, ng, Z, sfe, ncl)
    iD = nearest(ng[nn.index("DTM")], 1.0)
    riZ = nearest(rg[rn.index("Z")], Z); riS = nearest(rg[rn.index("SFE")], sfe)
    riN = nearest(rg[rn.index("n_cl")], ncl)
    new = nv[0, :, niZ, niS, niN, iD]
    ref = rv[0, :, riZ, riS, riN]
    label, _ = cell_label(ng, nn, niZ, niS, niN)
    return wl, new, ref, label, (nn, ng)


def report_table(stab_dir, ref_dir, prefix, window):
    """Per-band fractional-difference table over all 8 overlapping nodes."""
    print(f"\nReference window: {window}")
    Zs, Ss, Ns = [0.008, 0.02], [0.025, 0.05], [80.0, 160.0]
    for dust, nd in (("Dust", ""), ("noDust", "_noDust")):
        for res in ("lr", "hr"):
            acc = {b: [] for b in BANDS}
            for Z in Zs:
                for sfe in Ss:
                    for ncl in Ns:
                        wl, new, ref, _, _ = overlapping_slices(
                            new_path(stab_dir, prefix, nd, res),
                            ref_path(ref_dir, prefix, dust, res, window), Z, sfe, ncl)
                        for b, (lo, hi) in BANDS.items():
                            m = (wl >= lo) & (wl <= hi) & (ref > 0) & (new > 0)
                            if m.any():
                                acc[b].append(np.abs(new[m] - ref[m]) / ref[m])
            print(f"  {dust} {res}:")
            for b, lst in acc.items():
                if lst:
                    a = np.concatenate(lst)
                    print(f"    {b:13s} median {np.median(a)*100:7.2f}%  "
                          f"90th {np.percentile(a,90)*100:7.2f}%  max {a.max()*100:7.1f}%")


def plot_overlay(stab_dir, ref_dir, prefix, window, Z, sfe, ncl, out):
    fig, ax = plt.subplots(2, 2, figsize=(7.3, 4.6), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 1]),
                           constrained_layout=True)
    label = None
    for col, (nd, dust) in enumerate(((("", "Dust")), ("_noDust", "noDust"))):
        wl, new, ref, label, _ = overlapping_slices(
            new_path(stab_dir, prefix, nd, "hr"),
            ref_path(ref_dir, prefix, dust, "hr", window), Z, sfe, ncl)
        top, bot = ax[0, col], ax[1, col]
        top.loglog(wl, wl * ref, color="0.5", lw=1.2, label=f"shipped SKIRT ({window})")
        top.loglog(wl, wl * new, color="#CB4035", lw=0.9, alpha=0.85,
                   label=r"new, $f_{\rm dust}=1$")
        win = (wl >= WL_LO) & (wl <= WL_HI)
        pos = (wl * new)[win][(wl * new)[win] > 0]
        top.set_xlim(WL_LO, WL_HI); top.set_ylim(pos.min() * 0.4, pos.max() * 3)
        top.set_title(dust); top.legend(loc="lower center", fontsize=6)
        if col == 0:
            top.set_ylabel(r"$\lambda L_\lambda$ [arb.]")
        m = (ref > 0) & (new > 0)
        bot.semilogx(wl[m], (new[m] - ref[m]) / ref[m] * 100, color="#4683DE", lw=0.6)
        bot.axhline(0, color="0.6", lw=0.7, ls=":")
        bot.set_ylim(-60, 60 if dust == "Dust" else 300)
        bot.set_xlabel(r"$\lambda\ [\mu{\rm m}]$")
        if col == 0:
            bot.set_ylabel("(new$-$ref)/ref [%]")
    fig.suptitle("New $f_{\\rm dust}=1$ vs shipped reference   "
                 + f"({label}; BPASS chab100 bin)", fontsize=8)
    fig.savefig(out)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stab-dir", default=".")
    ap.add_argument("--ref-dir",
                    default=os.path.expanduser(
                        "~/SKIRT/resources/SKIRT9_Resources_TODDLERS/SED_TODDLERS"))
    ap.add_argument("--prefix", default="BPASS_chab100_bin")
    ap.add_argument("--window", default="10Myr", help="reference SFH window token")
    ap.add_argument("--out", default="vs_reference.png")
    ap.add_argument("--Z", type=float, default=0.02)
    ap.add_argument("--sfe", type=float, default=0.05)
    ap.add_argument("--ncl", type=float, default=160.0)
    ap.add_argument("--no-table", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.ref_dir):
        raise SystemExit(f"reference dir not found: {args.ref_dir}\n"
                         "point --ref-dir at your SKIRT SED_TODDLERS resources tree.")
    if not args.no_table:
        report_table(args.stab_dir, args.ref_dir, args.prefix, args.window)
    plot_overlay(args.stab_dir, args.ref_dir, args.prefix, args.window,
                 args.Z, args.sfe, args.ncl, args.out)


if __name__ == "__main__":
    main()
