#!/usr/bin/env python3
r"""MIR (15-25 um) contribution: unattenuated nebular continuum vs dust thermal emission.

Uses the Tier-1 variable-DTM STAB (BPASS chab100 bin) where the noDust variant carries the
unattenuated diffuse nebular continuum (Paper Appendix B). Because the dust cross section in
the MIR is small, the nebular continuum in the noDust SED is a good proxy for the nebular
continuum that survives extinction inside the Dust SED. So in the 15-25 um band:

    L_MIR(Dust)   ~  L_MIR(stellar attenuated) + L_MIR(nebular continuum) + L_MIR(dust thermal)
    L_MIR(noDust) ~  L_MIR(stellar)            + L_MIR(nebular continuum)
    L_MIR(dust thermal)  ~~  L_MIR(Dust) - L_MIR(noDust)      (stellar MIR is the small term)
    nebular fraction of MIR ~~ L_MIR(noDust) / L_MIR(Dust)

The script reports the per-cell numbers at the fiducial f_dust=1.0 and across all 7 DTMs.
Output: mir_nebular_vs_dust.png + a printed table.
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import percentile_filter

from _style import load_stab, nearest


# Strong MIR emission lines (microns) present in noDust within 15-25 um. They show up as
# sharp spikes on top of the actual nebular continuum (free-free + bound-free) and dominate
# the naive band integral; mask them when isolating the continuum-only contribution.
MIR_LINES_UM = [15.55,   # [NeIII]
                17.04,   # H2 S(1)
                17.94,   # [FeII]
                18.71,   # [SIII]
                21.83,   # [ArIII]
                24.32,   # [NeV]
                25.89]   # [OIV]   (edge of band)


def continuum_estimate(wl_micron, L_lambda, pct=10, window_pts=25):
    """Robust line-free continuum estimate via a rolling low-percentile filter on log(L_lambda).

    Lines are sharp upward spikes in log-flux; the rolling percentile (e.g. 10th) traces the
    underlying continuum and ignores them. Works on the full array; you then mask to the band.
    """
    floor = np.where(L_lambda > 0, L_lambda, np.nan)
    log_L = np.log10(np.where(np.isfinite(floor), floor, 1e-60))
    cont_log = percentile_filter(log_L, percentile=pct, size=window_pts, mode="nearest")
    return 10 ** cont_log

WL_LO, WL_HI = 15.0, 25.0   # MIR window, micron


def integrate_band(wl_micron, L_lambda, lo, hi):
    """Integrate L_lambda over [lo, hi] microns. L_lambda is per-metre; wl converted to m.

    Returns the band-integrated luminosity in the native erg/s/(Msun/yr) of the SFR-normalized
    STAB (the metre cancels: integrand [erg/s/m/(Msun/yr)] x d-lambda [m] -> erg/s/(Msun/yr)).
    """
    wl_m = wl_micron * 1e-6
    mask = (wl_micron >= lo) & (wl_micron <= hi)
    return np.trapz(L_lambda[mask], wl_m[mask])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stab-dir", default=".")
    ap.add_argument("--prefix", default="BPASS_chab100_bin")
    ap.add_argument("--out", default=None,
                    help="output PNG path (default: mir_nebular_vs_dust_Z<Z>_SFE<sfe>_n<ncl>.png)")
    ap.add_argument("--Z", type=float, default=0.02,
                    help="metallicity of the cell for the left-panel SED comparison")
    ap.add_argument("--sfe", type=float, default=0.05,
                    help="star-formation efficiency of the cell for the left-panel")
    ap.add_argument("--ncl", type=float, default=160.0,
                    help="cloud density [cm^-3] of the cell for the left-panel")
    args = ap.parse_args()
    if args.out is None:
        args.out = f"mir_nebular_vs_dust_Z{args.Z:g}_SFE{args.sfe:g}_n{args.ncl:g}.png"

    base = f"{args.stab_dir}/ToddlersSFRNormalizedSEDFamily_{args.prefix}"
    nm_d, gr_d, vals_d, wl = load_stab(f"{base}_DTM_hr.stab")
    nm_n, gr_n, vals_n, _  = load_stab(f"{base}_noDust_DTM_hr.stab")
    assert nm_d == nm_n, "axis-name mismatch between Dust and noDust STABs"

    Zs   = gr_d[nm_d.index("Z")]
    sfes = gr_d[nm_d.index("SFE")]
    ns   = gr_d[nm_d.index("n_cl")]
    dtms = gr_d[nm_d.index("DTM")]
    i_fid = nearest(dtms, 1.0)
    print(f"# Tier-1 STAB cube: Z={list(Zs)}  SFE={list(sfes)}  n_cl={list(ns)}  DTM={list(dtms)}")
    print(f"# MIR band: {WL_LO}-{WL_HI} micron; fiducial DTM = {dtms[i_fid]:g}")
    print()

    # --- per-cell table at the fiducial DTM=1.0: raw integral vs line-masked continuum ---
    # Raw integral counts MIR emission lines ([NeIII], [SIII], [FeII], [NeV], [OIV], ...) as
    # "nebular" and overestimates the continuum contribution. The continuum estimate uses a
    # rolling 10th-percentile filter on log-flux to trace the line-free baseline.
    print("  cell                              raw%   cont%   L_neb_cont   L_dust_cont")
    rows = []
    for iZ, Z in enumerate(Zs):
        for iS, sfe in enumerate(sfes):
            for iN, n in enumerate(ns):
                Sd = vals_d[0, :, iZ, iS, iN, i_fid]
                Sn = vals_n[0, :, iZ, iS, iN, i_fid]
                Ld_raw = integrate_band(wl, Sd, WL_LO, WL_HI)
                Ln_raw = integrate_band(wl, Sn, WL_LO, WL_HI)
                Sd_cont = continuum_estimate(wl, Sd)
                Sn_cont = continuum_estimate(wl, Sn)
                Ld_cont = integrate_band(wl, Sd_cont, WL_LO, WL_HI)
                Ln_cont = integrate_band(wl, Sn_cont, WL_LO, WL_HI)
                Ldust_cont = max(Ld_cont - Ln_cont, 0.0)
                f_raw = Ln_raw / Ld_raw if Ld_raw > 0 else np.nan
                f_cont = Ln_cont / Ld_cont if Ld_cont > 0 else np.nan
                rows.append((Z, sfe, n, f_raw, f_cont, Ln_cont, Ldust_cont))
                print(f"  Z={Z:g} SFE={sfe:g} n_cl={n:g}    "
                      f"{100*f_raw:5.1f}  {100*f_cont:5.1f}  "
                      f"{Ln_cont:.2e}  {Ldust_cont:.2e}")
    f_raw_arr = np.array([r[3] for r in rows])
    f_cont_arr = np.array([r[4] for r in rows])
    print(f"\n  raw   nebular fraction (DTM=1.0): min {100*f_raw_arr.min():.1f}%, "
          f"median {100*np.median(f_raw_arr):.1f}%, max {100*f_raw_arr.max():.1f}%")
    print(f"  CONT  nebular fraction (DTM=1.0): min {100*f_cont_arr.min():.1f}%, "
          f"median {100*np.median(f_cont_arr):.1f}%, max {100*f_cont_arr.max():.1f}%")

    # --- continuum-only nebular fraction vs f_dust (median across 8 cells) ---
    f_cont_dtm = []
    for k in range(dtms.size):
        per_cell = []
        for iZ in range(Zs.size):
            for iS in range(sfes.size):
                for iN in range(ns.size):
                    Sd_c = continuum_estimate(wl, vals_d[0, :, iZ, iS, iN, k])
                    Sn_c = continuum_estimate(wl, vals_n[0, :, iZ, iS, iN, k])
                    Ldc = integrate_band(wl, Sd_c, WL_LO, WL_HI)
                    Lnc = integrate_band(wl, Sn_c, WL_LO, WL_HI)
                    per_cell.append(Lnc / Ldc if Ldc > 0 else np.nan)
        f_cont_dtm.append(np.nanmedian(per_cell))
    f_cont_dtm = np.array(f_cont_dtm)
    f_neb_dtm = f_cont_dtm     # kept for downstream plot code (was the naive version before)
    print(f"\n# continuum-only nebular fraction in MIR vs f_dust (median across 8 cells):")
    for fd, f in zip(dtms, f_cont_dtm):
        print(f"  f_dust={fd:.2g}: {100*f:.1f}%")

    # --- plot: SEDs in MIR band (Dust + noDust) for a representative cell, and DTM dependence ---
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.2), constrained_layout=True)

    # left: SEDs in 1-200 um for the requested cell at DTM=1.0
    iZ, iS, iN = nearest(Zs, args.Z), nearest(sfes, args.sfe), nearest(ns, args.ncl)
    Sd = vals_d[0, :, iZ, iS, iN, i_fid]
    Sn = vals_n[0, :, iZ, iS, iN, i_fid]
    Sd_cont = continuum_estimate(wl, Sd)
    Sn_cont = continuum_estimate(wl, Sn)
    win = (wl >= 1.0) & (wl <= 200.0)
    axes[0].loglog(wl[win], (wl*Sd)[win], color="#CB4035", lw=1.0, alpha=0.9, label="Dust (raw)")
    axes[0].loglog(wl[win], (wl*Sn)[win], color="#4683DE", lw=1.0, alpha=0.9, label="noDust (raw)")
    axes[0].loglog(wl[win], (wl*Sd_cont)[win], color="#7A1E10", lw=1.2, ls="--",
                   label="Dust continuum (line-masked)")
    axes[0].loglog(wl[win], (wl*Sn_cont)[win], color="#1F3F73", lw=1.2, ls="--",
                   label="noDust continuum (line-masked)")
    axes[0].axvspan(WL_LO, WL_HI, color="navajowhite", alpha=0.45, zorder=0)
    axes[0].set_xlabel(r"$\lambda\ [\mu{\rm m}]$")
    axes[0].set_ylabel(r"$\lambda L_\lambda\ [\mathrm{erg\,s^{-1}\,(M_\odot\,yr^{-1})^{-1}}]$")
    axes[0].set_title(rf"Z={Zs[iZ]:g}, $\varepsilon_{{\rm SF}}$={sfes[iS]:g}, "
                      rf"$n_{{\rm cl}}$={ns[iN]:g}, $f_{{\rm dust}}$=1.0")
    axes[0].legend(loc="lower right", fontsize=6)
    f_raw_cell  = integrate_band(wl, Sn,      WL_LO, WL_HI) / integrate_band(wl, Sd,      WL_LO, WL_HI)
    f_cont_cell = integrate_band(wl, Sn_cont, WL_LO, WL_HI) / integrate_band(wl, Sd_cont, WL_LO, WL_HI)
    axes[0].text(0.04, 0.95,
                 f"15-25 um nebular fraction:\n raw {100*f_raw_cell:.1f}% (lines + cont.)\n "
                 f"continuum-only {100*f_cont_cell:.1f}%",
                 transform=axes[0].transAxes, fontsize=7, va="top",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"))

    # right: continuum-only nebular fraction vs f_dust (median across 8 cells)
    axes[1].semilogx(dtms, 100*f_neb_dtm, "o-", color="#4BC55A", lw=1.2, ms=5)
    axes[1].set_xlabel(r"$f_{\rm dust}$")
    axes[1].set_ylabel("nebular continuum fraction of MIR 15-25 um (%)")
    axes[1].set_title("line-masked; median across 8 Tier-1 cells")
    axes[1].grid(True, which="both", alpha=0.3)
    for fd, fn in zip(dtms, f_neb_dtm):
        axes[1].annotate(f"{100*fn:.1f}", (fd, 100*fn), textcoords="offset points",
                         xytext=(0, 6), ha="center", fontsize=6, color="0.3")

    fig.suptitle("MIR 15-25 um: unattenuated nebular continuum vs dust thermal "
                 "(Tier-1 BPASS chab100 bin)", fontsize=8)
    fig.savefig(args.out, dpi=200)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
