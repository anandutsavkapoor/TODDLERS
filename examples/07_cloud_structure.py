"""Cloud density structure: uniform vs Bonnor-Ebert (needs feedback data).

The initial cloud structure controls how the shell sweeps up mass and when it
fragments. Centrally concentrated (Bonnor-Ebert) clouds fragment earlier than uniform
ones at fixed mass and mean density. This compares the shell evolution for each.

Requires: `python scripts/download_data.py`.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toddlers.evolution import Evolution
from toddlers.constants import M_SUN, PC_TO_CM, MYR_TO_SEC

base = dict(Z=0.02, eta_sf=0.05, n_cl=160.0, M_cl_init=1e6 * M_SUN,
            template="SB99", imf="kroupa100", star_type="sin")

fig, ax = plt.subplots(figsize=(5, 3.6))
for profile in ("uniform", "bonnor_ebert"):
    g = Evolution(**base, profile_type=profile).run_simulation()[0]
    ax.semilogy(g["time"] / MYR_TO_SEC, g["radius"] / PC_TO_CM, label=profile)
    tfrag = [tt for typ, tt, _ in g.get("phase_transitions", [])
             if typ == "phase1_to_fragmentation"]
    print(f"  {profile}: fragmentation at "
          f"{tfrag[0] / MYR_TO_SEC:.2f} Myr" if tfrag else f"  {profile}: none")

ax.set_xlabel("t [Myr]"); ax.set_ylabel("R [pc]")
ax.legend(frameon=False, fontsize=9)
fig.suptitle("Cloud structure: uniform vs Bonnor-Ebert")
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "07_cloud_structure.png")
fig.savefig(out, dpi=130)
print("wrote", out)
