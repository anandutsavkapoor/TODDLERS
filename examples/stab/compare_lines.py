#!/usr/bin/env python3
"""Compare emission-line positions and intensities between our hr cloud-family STAB
and the shipped SKIRT reference hr STAB.

Confirms the lines sit at the same wavelengths and that their intensities track the
reference (any offset should be the known n_H He-correction, concentrated in the
nebular emission, not a wavelength/identification error).

Usage::
    python compare_lines.py OURS_DIR REF_DIR --prefix BPASS_chab100_bin --out plots
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

# optical lines (micron) for annotation / ratio readout
LINES = {0.486133: "Hb", 0.495891: "OIII4959", 0.500684: "OIII5007",
         0.656280: "Ha", 0.658345: "NII6584", 0.671644: "SII6716", 0.673082: "SII6731"}


def spectrum_at(d, vals, iT, iZ, iS, iN, iM):
    return np.asarray(vals)[0][:, iT, iZ, iS, iN, iM]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours_dir"); ap.add_argument("ref_dir")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", default="plots")
    ap.add_argument("--age", type=float, default=2.6, help="target age [Myr]")
    ap.add_argument("--line-map", help="path to line_map.pkl to compare ALL lines")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ours_dir, ref_dir = Path(args.ours_dir), Path(args.ref_dir)

    op = ours_dir / f"ToddlersCloudSEDFamily_{args.prefix}_hr.stab"
    rp = next((ref_dir / c for c in ref_candidates(args.prefix, "hr")
               if (ref_dir / c).exists()), None)
    our = read_stab_file(str(op)); ref = read_stab_file(str(rp))
    print("ours:", op.name, "  ref:", rp.name)

    wl = axis(our, "lambda") * 1e6          # meters -> micron (grids verified identical)
    iax = {ax: match_idx(axis(ref, ax), axis(our, ax)) for ax in PARAM_AXES}
    to, tr = axis(our, "time"), axis(ref, "time")
    kO = int(np.argmin(np.abs(to - args.age)))
    kR = int(np.argmin(np.abs(tr - to[kO])))
    print(f"age: ours t={to[kO]:.2f} Myr  ref t={tr[kR]:.2f} Myr")

    # representative cloud = last node on each axis
    cellO = (kO, 1, 1, 1, 1)
    cellR = (kR, iax["Z"][1], iax["SFE"][1], iax["n_cl"][1], iax["M_cl"][1])
    so = spectrum_at(our, our["values"], *cellO)
    sr = spectrum_at(ref, ref["values"], *cellR)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    # ---- optical overlay ----
    a = ax[0]
    o = (wl > 0.45) & (wl < 0.70)
    a.semilogy(wl[o], sr[o], "k", lw=1.4, alpha=0.6, label="SKIRT reference hr")
    a.semilogy(wl[o], so[o], "C3", lw=0.8, label="ours hr")
    for lam in LINES:
        a.axvline(lam, color="b", lw=0.3, ls=":", alpha=0.4)
    a.set_xlabel("wavelength [micron]"); a.set_ylabel("L (STAB units)")
    a.set_title(f"(a) optical, t~{to[kO]:.1f} Myr — line positions & intensities")
    a.legend(fontsize=9)

    # ---- Halpha complex zoom + ratio readout ----
    b = ax[1]
    z = (wl > 0.6535) & (wl < 0.6755)
    b.plot(wl[z], sr[z], "k.-", ms=3, lw=1.0, alpha=0.6, label="reference")
    b.plot(wl[z], so[z], "C3.-", ms=3, lw=0.8, label="ours")
    for lam, nm in LINES.items():
        if 0.6535 < lam < 0.6755:
            b.axvline(lam, color="b", lw=0.3, ls=":")
            b.text(lam, b.get_ylim()[1], nm, rotation=90, va="top", fontsize=7)
    b.set_xlabel("wavelength [micron]"); b.set_ylabel("L (STAB units)")
    b.set_title("(b) Halpha/[NII]/[SII] zoom"); b.legend(fontsize=9)

    # line-center peak ratios (max within +/-1 nm of nominal line center)
    print("\nline      lambda(um)  ours/ref(peak)   ours_peak     ref_peak")
    for lam, nm in LINES.items():
        m = (wl > lam - 1e-3) & (wl < lam + 1e-3)
        if not m.any():
            continue
        op_pk, rp_pk = np.nanmax(so[m]), np.nanmax(sr[m])
        # peak wavelength location in each (line position check)
        loc_o = wl[m][np.argmax(so[m])]; loc_r = wl[m][np.argmax(sr[m])]
        print(f"{nm:9s} {lam:8.5f}   {op_pk/rp_pk:8.4f}      {op_pk:.3e}  {rp_pk:.3e}"
              f"   loc ours={loc_o:.5f} ref={loc_r:.5f}")

    fig.tight_layout()
    fp = f"{args.out}/compare_lines_{args.prefix}.png"
    fig.savefig(fp, dpi=130); print("\nwrote", fp)

    # ---- ALL lines from line_map ----
    if args.line_map:
        import pickle
        lm = pickle.load(open(args.line_map, "rb"))["line_map"]   # {idx:(elem,ion,lam_um)}
        rows = []
        for idx, (elem, ion, lam) in lm.items():
            win = max(lam * 2e-3, 1.5e-4)                 # ~0.2% or 0.15nm min
            m = (wl > lam - win) & (wl < lam + win)
            if m.sum() < 3:
                continue
            io, ir = so[m], sr[m]
            # continuum baseline from window edges; line flux = peak - continuum
            cont_o = 0.5 * (io[0] + io[-1]); cont_r = 0.5 * (ir[0] + ir[-1])
            pk_o, pk_r = np.nanmax(io), np.nanmax(ir)
            loc_o = wl[m][np.argmax(io)]; loc_r = wl[m][np.argmax(ir)]
            exc_o = max(pk_o - cont_o, 0.0); exc_r = max(pk_r - cont_r, 0.0)
            sig_r = exc_r / cont_r if cont_r > 0 else 0.0          # line/continuum in ref
            flux_ratio = exc_o / exc_r if exc_r > 0 else np.nan
            rows.append((idx, f"{elem}{ion}", lam, ion, pk_o / pk_r, flux_ratio,
                         sig_r, loc_o - loc_r))
        # full table
        print(f"\n{'idx':>3} {'line':>8} {'lam_um':>10} {'ion':>3} {'peakR':>7} "
              f"{'fluxR':>7} {'line/cont':>9} {'dloc_um':>10}")
        for r in rows:
            print(f"{r[0]:3d} {r[1]:>8} {r[2]:10.5f} {r[3]:3d} {r[4]:7.3f} "
                  f"{r[5]:7.3f} {r[6]:9.2e} {r[7]:+10.2e}")
        # significant lines only (line/continuum > 0.1 in ref)
        strong = [r for r in rows if r[6] > 0.1 and np.isfinite(r[5])]
        maxloc = max(abs(r[7]) for r in rows)
        print(f"\nALL {len(rows)} lines: max |dloc| = {maxloc:.2e} um (0 => identical grid positions)")
        print(f"{len(strong)} lines with line/continuum>0.1 in reference; "
              f"flux-ratio median={np.median([r[5] for r in strong]):.3f} "
              f"min={min(r[5] for r in strong):.3f} max={max(r[5] for r in strong):.3f}")

        figL, axL = plt.subplots(figsize=(11, 5))
        lam_s = np.array([r[2] for r in strong]); fr = np.array([r[5] for r in strong])
        ion_s = np.array([r[3] for r in strong])
        scL = axL.scatter(lam_s, fr, c=ion_s, s=28, cmap="turbo", vmin=0, vmax=6,
                          edgecolor="k", lw=0.3)
        axL.axhline(1.0, color="r", lw=0.7, ls=":")
        axL.set_xscale("log"); axL.set_xlabel("wavelength [micron]")
        axL.set_ylabel("ours/ref line flux (continuum-subtracted)")
        axL.set_title(f"All significant lines ours/ref ({args.prefix}, t~{to[kO]:.1f} Myr)")
        figL.colorbar(scL, ax=axL, label="ionization stage")
        for r in strong:                                    # label the strongest
            if r[6] > 1.0:
                axL.annotate(r[1], (r[2], r[5]), fontsize=6, alpha=0.7)
        figL.tight_layout()
        fpL = f"{args.out}/compare_lines_all_{args.prefix}.png"
        figL.savefig(fpL, dpi=130); print("wrote", fpL)


if __name__ == "__main__":
    main()
