#!/usr/bin/env python3
"""Physical-soundness checks on a freshly generated TODDLERS interpolant set.

Validates the nebular emission of the new runs against textbook expectations
(independent of any reference library):

  (a) Halpha declines with cluster age (ionizing budget drops as massive stars die);
  (b) intrinsic Halpha tracks the *absorbed* ionizing photon rate Q_abs = Q*(1-f_esc)
      at the case-B coefficient (L_Ha / Q_abs ~ 1.37e-12 erg, leak-corrected, dust-free);
  (c) the BPT diagram ([OIII]/Hb vs [NII]/Ha) lands in the star-forming locus;
  (d) the dust IR peak shifts with age (warm/compact when young -> cooler later).

Reads the interpolant tables written by ``toddlers.stab.interpolants`` for the
population selected via the TODDLERS_STAB_* environment (see toddlers.stab.config).

Usage::

    python physics_checks.py --interp-dir BPASS_chab100_bin_interp_tables --out plots
"""
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toddlers.stab import config as cfg
from toddlers.constants import M_SUN

# publication style, consistent with the paper figures (serif, four-sided ticks)
_STYLE = Path(__file__).resolve().parents[1] / "paper_figures" / "paper_style.mplstyle"
if _STYLE.exists():
    plt.style.use(str(_STYLE))

# line indices from line_map.pkl (verified for the chab100 line list)
L = dict(Ha=42, Hb=33, OIII5007=36, NII6584=43, SII6716=45, SII6731=46)
CASE_B_HA = 1.37e-12  # erg per ionizing photon (case B, 1e4 K; Hao+2011 / Kennicutt)
SEC_PER_MYR = 1e6 * 3.1557e7

# Interpolant tables store log10(luminosity); -99 is the "absent / zero" sentinel.


def node_lum(ll, line, dust, iZ, iS, iN, iM):
    """Stored line luminosity [erg/s] vs time at a grid node (no interpolation)."""
    v = np.asarray(ll.values)[line, dust, :, iZ, iS, iN, iM]
    return np.where(v <= -90, np.nan, 10.0**v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interp-dir", required=True)
    ap.add_argument("--out", default="plots")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    D = args.interp_dir

    ll = pickle.load(open(f"{D}/line_luminosities_interp.pkl", "rb"))
    t = np.asarray(ll.grid[2])              # Myr
    logZ = np.asarray(ll.grid[3]); Zvals = 10.0**logZ
    SFE = np.asarray(ll.grid[4])
    logn = np.asarray(ll.grid[5]); nvals = 10.0**logn
    logM = np.asarray(ll.grid[6])

    # decide which axis-1 slot is intrinsic (dust-free): the larger Halpha
    iZ = iS = iN = iM = 1
    ha0 = node_lum(ll, L["Ha"], 0, iZ, iS, iN, iM)
    ha1 = node_lum(ll, L["Ha"], 1, iZ, iS, iN, iM)
    DUST_INTR = 0 if np.nanmedian(ha0) >= np.nanmedian(ha1) else 1
    DUST_EMER = 1 - DUST_INTR
    print(f"intrinsic slot = {DUST_INTR} (median Ha {np.nanmedian(ha0):.2e} vs {np.nanmedian(ha1):.2e})")

    fig, ax = plt.subplots(2, 2, figsize=(8.4, 7.2), constrained_layout=True)

    # ---- (a) Halpha vs age, all 16 cells ----
    a = ax[0, 0]
    for iZ_ in range(2):
        for iS_ in range(2):
            for iN_ in range(2):
                for iM_ in range(2):
                    y = node_lum(ll, L["Ha"], DUST_EMER, iZ_, iS_, iN_, iM_)
                    a.semilogy(t, y, lw=0.8, alpha=0.6,
                               color="C0" if iZ_ == 0 else "C3")
    a.set_xlabel("age [Myr]"); a.set_ylabel(r"$L_{\rm H\alpha}$ (emergent) [erg/s]")
    a.set_title(r"(a) H$\alpha$ declines with age")
    a.plot([], [], "C0", label=f"Z={Zvals[0]:.3f}"); a.plot([], [], "C3", label=f"Z={Zvals[1]:.3f}")
    a.legend(fontsize=8)

    # ---- (b) intrinsic Halpha vs *absorbed* ionizing photon rate Q*(1-f_esc) ----
    # Cloudy's intrinsic Halpha is produced only by the ionizing photons actually
    # absorbed in the cloud; photons that leak out (escape fraction f_esc_i) do not
    # recombine. The case-B relation therefore holds against Q_abs = Q*(1-f_esc), not Q.
    b = ax[0, 1]
    try:
        from toddlers.stellar_feedback import StellarFeedback as SF
        fe = pickle.load(open(f"{D}/f_esc_i_interp.pkl", "rb"))  # axes: t,logZ,SFE,logn,logM
        fev = np.asarray(fe.values)
        ok = True
    except Exception as e:
        print("Q / f_esc setup failed:", e); ok = False
    if ok:
        sc_b = None
        for iZ_ in range(2):
            for iS_ in range(2):
                for iM_ in range(2):
                    M_cl = 10.0**logM[iM_] * M_SUN      # StellarFeedback expects grams
                    sf = SF(cfg.STELLAR_TEMPLATE, Zvals[iZ_], M_cl, SFE[iS_], [],
                            mode="burst", imf=cfg.IMF_TYPE, star_type=cfg.STAR_TYPE)
                    Q = np.array([sf.get_ionizing_photon_rate(ti * SEC_PER_MYR) for ti in t])
                    for iN_ in range(2):
                        f_esc = fev[:, iZ_, iS_, iN_, iM_]
                        Q_abs = Q * (1.0 - f_esc)
                        ha = node_lum(ll, L["Ha"], DUST_INTR, iZ_, iS_, iN_, iM_)
                        m = (Q_abs > 0) & np.isfinite(ha) & (ha > 0)
                        sc_b = b.scatter(Q_abs[m], ha[m], c=f_esc[m], s=7, alpha=0.7,
                                         cmap="plasma", vmin=0, vmax=1)
        qq = np.logspace(46, 54, 50)
        b.loglog(qq, CASE_B_HA * qq, "k--", lw=1.2,
                 label=r"case B: $1.37\times10^{-12}\,Q_{\rm abs}$")
        b.set_xlim(1e49, 3e53)
        b.set_xlabel(r"$Q_{\rm abs}=Q\,(1-f_{\rm esc})$ [photons/s]")
        b.set_ylabel(r"$L_{\rm H\alpha}$ intrinsic [erg/s]")
        b.set_title(r"(b) intrinsic H$\alpha$ vs absorbed $Q$ (leak-corrected)")
        b.legend(fontsize=8, loc="upper left")
        if sc_b is not None:
            fig.colorbar(sc_b, ax=b, label=r"$f_{\rm esc}$")

    # ---- (c) BPT ----
    c = ax[1, 0]
    for iZ_ in range(2):
        for iS_ in range(2):
            for iN_ in range(2):
                for iM_ in range(2):
                    o3 = node_lum(ll, L["OIII5007"], DUST_EMER, iZ_, iS_, iN_, iM_)
                    hb = node_lum(ll, L["Hb"], DUST_EMER, iZ_, iS_, iN_, iM_)
                    n2 = node_lum(ll, L["NII6584"], DUST_EMER, iZ_, iS_, iN_, iM_)
                    ha = node_lum(ll, L["Ha"], DUST_EMER, iZ_, iS_, iN_, iM_)
                    m = (o3 > 0) & (hb > 0) & (n2 > 0) & (ha > 0)
                    sc = c.scatter(np.log10(n2[m] / ha[m]), np.log10(o3[m] / hb[m]),
                                   c=t[m], s=8, cmap="viridis", vmin=0, vmax=15)
    # Kauffmann 2003 SF demarcation
    x = np.linspace(-2.0, 0.0, 100)
    c.plot(x, 0.61 / (x - 0.05) + 1.3, "k--", lw=1, label="Kauffmann+03")
    c.set_xlim(-2.0, 0.5); c.set_ylim(-1.5, 1.5)
    c.set_xlabel(r"log [NII]6584/H$\alpha$"); c.set_ylabel(r"log [OIII]5007/H$\beta$")
    c.set_title("(c) BPT (emergent), colored by age"); c.legend(fontsize=8)
    fig.colorbar(sc, ax=c, label="age [Myr]")

    # ---- (d) dust IR peak shift ----
    # Flux-weighted centroid of lambda*L_lambda over the IR (10-1000 micron) is a smooth,
    # grid-independent proxy for the dust-emission peak (argmax jumps between grid nodes).
    # Plotted only while the IR luminosity is within 1 dex of its peak, i.e. while there is
    # meaningful dust emission (after the cloud disperses the "peak" is ill-defined).
    d = ax[1, 1]
    try:
        sed = pickle.load(open(f"{D}/TODDLERS_totSED_lr_{cfg.MODEL_PREFIX}.pkl", "rb"))
        wl = 10.0**np.asarray(sed.grid[0])     # axis0 is log10(lambda/micron)
        ir = (wl >= 10.0) & (wl <= 1000.0)
        logwl = np.log10(wl[ir])
        for iN_ in range(2):
            cen, Lir = [], []
            for k in range(t.size):
                lL = (10.0**np.asarray(sed.values)[:, k, 1, 1, iN_, 1] * wl)[ir]
                cen.append(10.0**(np.sum(logwl * lL) / np.sum(lL)))
                Lir.append(np.trapz(lL / wl[ir], wl[ir]))
            cen, Lir = np.array(cen), np.array(Lir)
            keep = Lir > 0.1 * Lir.max()
            d.plot(t[keep], cen[keep], "-o", ms=3,
                   label=rf"$n_{{\rm cl}}={nvals[iN_]:.0f}\,{{\rm cm^{{-3}}}}$")
        d.set_xlabel("age [Myr]"); d.set_ylabel(r"dust IR centroid $\lambda$ [$\mu$m]")
        d.set_title(r"(d) dust IR peak shifts redward with age"); d.legend(fontsize=8)
    except Exception as e:
        d.text(0.1, 0.5, f"dust peak skipped:\n{e}", fontsize=8)

    fig.suptitle(f"Nebular physics checks - {cfg.MODEL_PREFIX} "
                 f"(emergent + intrinsic, baseline $f_{{\\rm dust}}=1$)", fontsize=11)
    fp = f"{args.out}/physics_checks_{cfg.MODEL_PREFIX}.png"
    fig.savefig(fp, dpi=150); print("wrote", fp)


if __name__ == "__main__":
    main()
