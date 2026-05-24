"""Phase 0 audit: sanity-check the clean track loader (HRD + surface-H evolution)."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from toddlers.pysb99.eep_interpolation import load_all_tracks

Z = "MW"
SHOW = [9, 15, 25, 40, 60, 85, 120, 300]   # masses to highlight

masses, tracks = load_all_tracks(Z)
sel = [(m, t) for m, t in zip(masses, tracks) if any(abs(m - s) < 0.6 for s in SHOW)]

fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)

# HRD
for m, t in sel:
    ax[0].plot(t["logTeff"], t["logL"], lw=1, label=f"{m:g}")
ax[0].invert_xaxis()
ax[0].set_xlabel(r"$\log T_\mathrm{eff}$"); ax[0].set_ylabel(r"$\log L/L_\odot$")
ax[0].set_title("HRD"); ax[0].legend(fontsize=7, ncol=2, title=r"$M\,[M_\odot]$")

# surface H vs age
for m, t in sel:
    ax[1].plot(t["age"] / 1e6, t["X_H"], lw=1)
ax[1].set_xlabel("age [Myr]"); ax[1].set_ylabel(r"surface $X(\mathrm{H})$")
ax[1].set_xlim(0, 12); ax[1].axhline(0.4, color="k", ls=":", lw=0.7)
ax[1].set_title("surface H (WR onset at 0.4)")

# lifetime vs mass
life = np.array([t["age"].max() / 1e6 for t in tracks])
ax[2].loglog(masses, life, "o-", ms=3)
ax[2].set_xlabel(r"$M\,[M_\odot]$"); ax[2].set_ylabel("lifetime [Myr]")
ax[2].set_title("lifetime vs mass")

out = os.path.join(HERE, "eep_phase0_audit.pdf")
fig.savefig(out)
print(f"Wrote {out}")

# print a small table for the highlighted massive tracks
print(f"\n{Z}: phase-marker sanity (massive tracks)")
print(f"{'M':>6} {'life/Myr':>9} {'X_H_end':>8} {'minX_H':>8} {'WR?':>5}")
for m, t in sel:
    if m >= 20:
        wr = (t["X_H"] < 0.4).any()
        print(f"{m:6g} {t['age'].max()/1e6:9.2f} {t['X_H'][-1]:8.3f} "
              f"{t['X_H'].min():8.3f} {str(wr):>5}")
