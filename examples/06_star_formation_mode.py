"""Star formation mode: instantaneous burst vs constant SFR (needs feedback data).

The same total stellar mass can form in a single burst or be spread over a formation
timescale. Constant SFR delays and lowers the feedback peak, which shifts the
fragmentation time. This compares the two modes.

Requires: `python scripts/download_data.py`.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toddlers.evolution import Evolution
from toddlers.constants import M_SUN, PC_TO_CM, MYR_TO_SEC

base = dict(Z=0.02, eta_sf=0.05, n_cl=160.0, M_cl_init=1e6 * M_SUN,
            template="SB99", imf="kroupa100", star_type="sin", profile_type="uniform")
cases = [
    dict(label="burst", cluster_formation_mode="burst"),
    dict(label="constant SFR (2 Myr)", cluster_formation_mode="constant_sfr",
         formation_timescale=2.0 * MYR_TO_SEC),
]

fig, ax = plt.subplots(figsize=(5, 3.6))
for c in cases:
    label = c.pop("label")
    g = Evolution(**base, **c).run_simulation()[0]
    ax.semilogy(g["time"] / MYR_TO_SEC, g["radius"] / PC_TO_CM, label=label)
    tfrag = [tt for typ, tt, _ in g.get("phase_transitions", [])
             if typ == "phase1_to_fragmentation"]
    print(f"  {label}: fragmentation at "
          f"{tfrag[0] / MYR_TO_SEC:.2f} Myr" if tfrag else f"  {label}: no fragmentation")

ax.set_xlabel("t [Myr]"); ax.set_ylabel("R [pc]")
ax.legend(frameon=False, fontsize=9)
fig.suptitle("Star formation mode: burst vs constant SFR")
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "06_star_formation_mode.png")
fig.savefig(out, dpi=130)
print("wrote", out)
