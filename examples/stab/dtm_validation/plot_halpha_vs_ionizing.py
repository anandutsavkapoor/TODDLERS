#!/usr/bin/env python3
r"""Visualize the recollapse-aware verification (verify_dtm's correlation check).

For one cell, overlay the intrinsic Halpha(t) from the Cloudy interpolant against the
evolution's absorbed ionizing rate Q_abs(t)=Q_ion*(1-f_esc) read from the .dat, and show the
log-log correlation. On a recollapse cell the two should jump together at each recollapse
time (a new generation lights up): a high correlation confirms the SED tracks the ionizing
output through the full history. Defaults to a 3-generation recollapse cell.
"""
import argparse
import json
import pickle

import numpy as np
import matplotlib.pyplot as plt

from _style import nearest                         # paper style is applied on import
from toddlers.hpc.verify_dtm import evolution_qabs, _index_dats, _nearest_dat, cell_logr, SEC_PER_MYR


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interp-dir", required=True)
    ap.add_argument("--evolution-dir", required=True)
    ap.add_argument("--prefix", default="BPASS_chab100_bin")
    ap.add_argument("--dtm", type=float, default=1.0)
    ap.add_argument("--Z", type=float, default=0.02)
    ap.add_argument("--eta", type=float, default=0.025)
    ap.add_argument("--ncl", type=float, default=160.0)
    ap.add_argument("--logM", type=float, default=6.75)
    ap.add_argument("--out", default="halpha_vs_ionizing.png")
    args = ap.parse_args()

    suffix = "" if abs(args.dtm - 1.0) < 1e-9 else f"_dtm{args.dtm:.2f}"
    hrf = f"{args.interp_dir}/TODDLERS_tot_hr_{args.prefix}_lines_emergent=False{suffix}.pkl"
    hr = pickle.load(open(hrf, "rb"))
    g = [np.asarray(x) for x in hr.grid]; wl = 10.0 ** g[0]; t_cl = g[1]; hv = np.asarray(hr.values)
    iZ = nearest(10.0 ** g[2], args.Z); iS = nearest(g[3], args.eta)
    iN = nearest(10.0 ** g[4], args.ncl); iM = nearest(g[5], args.logM)
    ha = (wl >= 0.6520) & (wl <= 0.6610)
    ha_t = np.array([np.max(10.0 ** hv[(ha, k, iZ, iS, iN, iM)]) for k in range(t_cl.size)])

    dp = _nearest_dat(_index_dats(args.evolution_dir), args.Z, args.eta, args.ncl, args.logM)
    if dp is None:
        raise SystemExit("no matching .dat for the requested cell")
    t_ev, qabs, ngen = evolution_qabs(dp)
    rec = [tr[1] / SEC_PER_MYR for gen in json.load(open(dp))["generations"]
           for tr in gen["phase_transitions"] if tr[0] == "phase2_to_recollapse"]

    m = (t_cl >= t_ev.min()) & (t_cl <= t_ev.max())
    q_cl = np.interp(t_cl[m], t_ev, qabs); h = ha_t[m]; tt = t_cl[m]
    good = np.isfinite(h) & (h > 0) & np.isfinite(q_cl) & (q_cl > 0)
    r = cell_logr(t_cl, ha_t, t_ev, qabs)

    fig, ax = plt.subplots(1, 2, figsize=(7.3, 3.2), constrained_layout=True)
    # (a) time series: intrinsic Halpha (left axis) + Q_abs (right axis), recollapse marked
    a = ax[0]
    a.semilogy(tt[good], h[good], "o-", ms=2.5, color="#CB4035", label=r"intrinsic H$\alpha$ (Cloudy)")
    a.set_xlabel("age [Myr]"); a.set_ylabel(r"$L_{\rm H\alpha}^{\rm intr}$ [erg/s]", color="#CB4035")
    a.tick_params(axis="y", colors="#CB4035")
    a2 = a.twinx()
    a2.semilogy(tt[good], q_cl[good], "s--", ms=2.5, color="#4683DE", label=r"$Q_{\rm abs}$ (evolution)")
    a2.set_ylabel(r"$Q_{\rm abs}=Q_{\rm ion}(1-f_{\rm esc})$ [s$^{-1}$]", color="#4683DE")
    a2.tick_params(axis="y", colors="#4683DE")
    for i, tr in enumerate(rec):
        a.axvline(tr, color="0.5", ls=":", lw=0.9, label="recollapse" if i == 0 else None)
    a.set_title(f"{ngen}-generation cell")
    a.legend(fontsize=6, loc="lower left")

    # (b) log-log correlation
    b = ax[1]
    sc = b.scatter(np.log10(q_cl[good]), np.log10(h[good]), c=tt[good], s=10, cmap="viridis")
    b.set_xlabel(r"$\log_{10} Q_{\rm abs}$"); b.set_ylabel(r"$\log_{10} L_{\rm H\alpha}^{\rm intr}$")
    b.set_title(f"case-B correlation:  r = {r:.3f}")
    fig.colorbar(sc, ax=b, label="age [Myr]")

    fig.suptitle(f"Intrinsic H$\\alpha$ vs evolution ionizing flux   "
                 f"(Z={10.0**g[2][iZ]:g}, eta={g[3][iS]:g}, n={10.0**g[4][iN]:g}, "
                 f"logM={g[5][iM]:g}; f_dust={args.dtm:g})", fontsize=8)
    fig.savefig(args.out)
    print(f"wrote {args.out}  (r={r:.3f}, {ngen} generations, recollapse at {[round(x,1) for x in rec]} Myr)")


if __name__ == "__main__":
    main()
