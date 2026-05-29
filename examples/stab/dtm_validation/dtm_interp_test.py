#!/usr/bin/env python3
r"""How few f_dust points can we COMPUTE and still recover the full library by interpolation?

Each computed f_dust point costs ~1.2M inodes / ~0.5 TB of Cloudy output for the full grid,
so dropping points is the most direct way to cut the tier-2 footprint. This leaves out
interior f_dust nodes, reconstructs them by interpolating from a candidate reduced grid in
the SAME space SKIRT uses (log axis if the DTM axis is log-scaled, log L_lambda if the
quantity is log), and reports the reconstruction error at the dropped nodes.
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from _style import load_stab, cell_indices

CELL = dict(Z=0.02, sfe=0.05, ncl=160.0)   # cell used for the detailed printout + plot
BANDS = {"UV 0.1-0.3": (0.1, 0.3), "opt 0.3-0.8": (0.3, 0.8), "NIR 0.8-5": (0.8, 5.0),
         "MIR 5-30": (5.0, 30.0), "FIR 30-1000": (30.0, 1000.0)}
# candidate reduced f_dust grids (must include the endpoints 0.02 and 1.0)
GRIDS = {"5pt": [0.02, 0.10, 0.40, 0.80, 1.00], "4pt": [0.02, 0.10, 0.40, 1.00]}


def _interp_funcs(dtm_scale, qscale):
    def xform_x(x):
        return np.log10(x) if str(dtm_scale).startswith("log") else np.asarray(x)
    def to_q(y):
        return np.log10(np.clip(y, 1e-300, None)) if str(qscale).startswith("log") else y
    def from_q(y):
        return 10.0**y if str(qscale).startswith("log") else y
    return xform_x, to_q, from_q


def reconstruct(cellvals, dtm, keep_vals, xform_x, to_q, from_q, wl):
    """Interpolate every dropped f_dust node from subgrid keep_vals. cellvals: (lambda, DTM)."""
    keep = np.array(sorted(keep_vals))
    ki = [int(np.argmin(np.abs(dtm - k))) for k in keep]
    xk = xform_x(dtm[ki])
    out = {}
    for j, fd in enumerate(dtm):
        if fd in keep:
            continue
        yj = np.array([np.interp(xform_x(fd), xk, to_q(cellvals[w, ki])) for w in range(wl.size)])
        out[fd] = from_q(yj)
    return out


def worst_band(cellvals, dtm, rec, wl):
    """Worst-band (name, median%, max%) over flux-significant wavelengths."""
    worst = ("", 0.0, 0.0)
    for b, (lo, hi) in BANDS.items():
        m = (wl >= lo) & (wl <= hi)
        errs = []
        for fd, y in rec.items():
            t = cellvals[m, list(dtm).index(fd)]
            ok = t > 0.1 * t.max()
            errs.append(np.abs(y[m][ok] - t[ok]) / t[ok])
        a = np.concatenate(errs)
        if a.max() * 100 > worst[2]:
            worst = (b, np.median(a) * 100, a.max() * 100)
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stab-dir", default=".")
    ap.add_argument("--prefix", default="BPASS_chab100_bin")
    args = ap.parse_args()
    import toddlers.stab.stab_io as sio
    base = f"{args.stab_dir}/ToddlersSFRNormalizedSEDFamily_{args.prefix}"

    # ---- cross-parameter sweep: every (Z, SFE, n_cl) cell, both variants ----
    print("Reconstruction error vs the full 7-point grid, worst band at flux-significant "
          "wavelengths,\nfor every parameter cell (lower Z = less dust):\n")
    print(f"{'variant':10s} {'Z':>6s} {'SFE':>6s} {'n_cl':>5s} "
          f"{'5pt worst (med|max%)':>26s} {'4pt worst (med|max%)':>26s}")
    summary = {}
    plotdata = {}
    worst_off = {"max": -1.0}      # worst-offender cell for the 4-point grid
    for variant, tag in (("Dust", "_DTM_hr"), ("noDust", "_noDust_DTM_hr")):
        d = sio.read_stab_file(f"{base}{tag}.stab")
        names = d["axisNames"]; grids = [np.asarray(g) for g in d["axisGrids"]]
        scales = d.get("axisScales"); qscale = d.get("quantityScales", ["log"])[0]
        wl = grids[names.index("lambda")] * 1e6
        dtm = grids[names.index("DTM")]
        dtm_scale = scales[names.index("DTM")] if scales else "log"
        xf, tq, fq = _interp_funcs(dtm_scale, qscale)
        V = np.asarray(d["values"])[0]    # (lambda, Z, SFE, n_cl, DTM)
        Zs, Ss, Ns = grids[names.index("Z")], grids[names.index("SFE")], grids[names.index("n_cl")]
        worst_all = {"5pt": 0.0, "4pt": 0.0}
        pd = {"labels": [], "5pt": [], "4pt": []}
        for iZ, Z in enumerate(Zs):
            for iS, S in enumerate(Ss):
                for iN, N in enumerate(Ns):
                    cv = V[:, iZ, iS, iN, :]
                    cells = {}
                    for gname, keep in GRIDS.items():
                        rec = reconstruct(cv, dtm, keep, xf, tq, fq, wl)
                        cells[gname] = worst_band(cv, dtm, rec, wl)
                        worst_all[gname] = max(worst_all[gname], cells[gname][2])
                    if cells["4pt"][2] > worst_off["max"]:
                        worst_off = {"max": cells["4pt"][2], "variant": variant, "tag": tag,
                                     "iZ": iZ, "iS": iS, "iN": iN, "Z": Z, "S": S, "N": N,
                                     "band": cells["4pt"][0]}
                    print(f"{variant:10s} {Z:6.3f} {S:6.3f} {N:5.0f} "
                          f"{cells['5pt'][0]:>9s} {cells['5pt'][1]:4.1f}|{cells['5pt'][2]:5.1f}   "
                          f"{cells['4pt'][0]:>9s} {cells['4pt'][1]:4.1f}|{cells['4pt'][2]:5.1f}")
                    pd["labels"].append(f"{Z:g}\n{S:g}\n{N:g}")
                    pd["5pt"].append(cells["5pt"][2]); pd["4pt"].append(cells["4pt"][2])
        summary[variant] = worst_all
        plotdata[variant] = pd
    print("\nWORST max over ALL cells:")
    for v, w in summary.items():
        print(f"  {v:8s} 5pt {w['5pt']:.1f}%   4pt {w['4pt']:.1f}%")

    # ---- cross-parameter robustness plot ----
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.4), sharey=True, constrained_layout=True)
    for ax, variant in zip(axes, ("Dust", "noDust")):
        pd = plotdata[variant]; x = np.arange(len(pd["labels"])); w = 0.38
        ax.bar(x - w / 2, pd["5pt"], w, label="5-point", color="#4683DE")
        ax.bar(x + w / 2, pd["4pt"], w, label="4-point", color="#FBA11B")
        ax.axhline(10, color="0.4", ls="--", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(pd["labels"], fontsize=5.5)
        ax.set_title(f"{variant} hr"); ax.set_xlabel(r"$Z$ / $\varepsilon_{\rm SF}$ / $n_{\rm cl}$ cell")
    axes[0].set_ylabel("worst-band max error vs 7-point [%]")
    axes[0].legend(fontsize=7, title="reduced grid", title_fontsize=7)
    fig.suptitle("DTM interpolation accuracy across all parameter cells "
                 "(reduced grid vs full 7-point)", fontsize=8)
    fig.savefig("dtm_interp_crossparam.png")
    print("wrote dtm_interp_crossparam.png")

    # ---- visual: 4-point grid at the WORST-OFFENDER cell (largest reconstruction error) ----
    keep4 = GRIDS["4pt"]
    d = sio.read_stab_file(f"{base}{worst_off['tag']}.stab")
    names = d["axisNames"]; grids = [np.asarray(g) for g in d["axisGrids"]]
    scales = d.get("axisScales"); qscale = d.get("quantityScales", ["log"])[0]
    wl = grids[names.index("lambda")] * 1e6
    dtm = grids[names.index("DTM")]
    xf, tq, fq = _interp_funcs(scales[names.index("DTM")] if scales else "log", qscale)
    vals = np.asarray(d["values"])[0, :, worst_off["iZ"], worst_off["iS"], worst_off["iN"], :]
    rec4 = reconstruct(vals, dtm, keep4, xf, tq, fq, wl)
    # among the dropped points, find the one whose error in the worst band is largest (for the zoom)
    lo, hi = BANDS[worst_off["band"]]
    bm = (wl >= lo) & (wl <= hi)
    worst_fd = max(rec4, key=lambda fd: (lambda t: np.abs(rec4[fd][bm][t > 0.1 * t.max()]
                   - t[t > 0.1 * t.max()]).max() / t[t > 0.1 * t.max()].max())(vals[bm, list(dtm).index(fd)]))

    cellstr = (f"{worst_off['variant']} hr; Z={worst_off['Z']:g}, "
               f"SFE={worst_off['S']:g}, n_cl={worst_off['N']:g}")
    fig, ax = plt.subplots(1, 2, figsize=(7.3, 3.3), constrained_layout=True)
    for j, fd in enumerate(dtm):
        c = plt.cm.viridis(j / (len(dtm) - 1))
        ax[0].loglog(wl, wl * vals[:, j], color=c, lw=1.0,
                     ls="-" if fd in keep4 else ":",
                     label=(f"keep {fd:g}" if fd in keep4 else f"drop {fd:g}"))
    for fd, y in rec4.items():
        ax[0].loglog(wl, wl * y, color="k", lw=0.8, ls="--")
    ax[0].set_xlim(0.1, 3000); ax[0].set_xlabel(r"$\lambda\ [\mu{\rm m}]$")
    ax[0].set_ylabel(r"$\lambda L_\lambda$ [arb.]")
    ax[0].set_title("4-pt grid: solid=keep, dotted=drop, black dash=interp")
    ax[0].legend(fontsize=6, ncol=2)
    # zoom on the worst band + worst dropped point: true vs interp
    bz = (wl >= lo * 0.7) & (wl <= hi * 1.4)
    ax[1].loglog(wl[bz], (wl * vals[:, list(dtm).index(worst_fd)])[bz], "C3-", lw=1.4,
                 label=f"computed f_dust={worst_fd:g}")
    ax[1].loglog(wl[bz], (wl * rec4[worst_fd])[bz], "k--", lw=1.2,
                 label="interpolated from 4-pt")
    ax[1].set_xlabel(r"$\lambda\ [\mu{\rm m}]$")
    ax[1].set_title(f"worst case: drop f_dust={worst_fd:g}, {worst_off['band']}")
    ax[1].legend(fontsize=7)
    fig.suptitle(f"4-point DTM interpolation, WORST-offender cell  ({cellstr}; "
                 f"max err {worst_off['max']:.1f}%)", fontsize=8)
    fig.savefig("dtm_interp_test.png")
    print(f"\nworst offender (4-pt): {cellstr}, band {worst_off['band']}, "
          f"max {worst_off['max']:.1f}%, worst dropped f_dust={worst_fd:g}")
    print("wrote dtm_interp_test.png")


if __name__ == "__main__":
    main()
