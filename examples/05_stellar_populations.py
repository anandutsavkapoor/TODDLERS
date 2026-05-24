"""Stellar population frameworks: SB99 vs BPASS binaries (needs feedback data).

TODDLERS supports Starburst99 (single-star) and BPASS (with binary evolution), among
others. Binaries keep the ionizing output harder and more sustained at late ages. This
runs the same cloud with each population and compares the shell expansion.

Requires: `python scripts/download_data.py` (incl. the BPASS tables).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toddlers.evolution import Evolution
from toddlers.constants import M_SUN, PC_TO_CM, MYR_TO_SEC

cases = [
    dict(label="SB99 (single)", template="SB99", imf="kroupa100", star_type="sin"),
    dict(label="BPASS (binary)", template="BPASS", imf="chab100", star_type="bin"),
]

fig, ax = plt.subplots(figsize=(5, 3.6))
for c in cases:
    label = c.pop("label")
    ev = Evolution(Z=0.02, eta_sf=0.05, n_cl=160.0, M_cl_init=1e6 * M_SUN,
                   profile_type="uniform", **c)
    g = ev.run_simulation()[0]
    ax.semilogy(g["time"] / MYR_TO_SEC, g["radius"] / PC_TO_CM, label=label)
    print(f"  {label}: final radius {g['radius'][-1] / PC_TO_CM:.0f} pc")

ax.set_xlabel("t [Myr]"); ax.set_ylabel("R [pc]")
ax.legend(frameon=False, fontsize=9)
fig.suptitle("Stellar population frameworks: SB99 vs BPASS")
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "05_stellar_populations.png")
fig.savefig(out, dpi=130)
print("wrote", out)
