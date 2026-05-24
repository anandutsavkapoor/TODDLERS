"""Deterministic shell evolution (needs feedback data).

Runs a single TODDLERS shell evolution for a uniform cloud with an SB99 population,
and plots the shell radius and ionizing-photon-rate histories. This is the cheap first
stage of the pipeline (no Cloudy).

Requires the feedback library: `python scripts/download_data.py`.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toddlers.evolution import Evolution
from toddlers.constants import M_SUN, PC_TO_CM, MYR_TO_SEC

ev = Evolution(Z=0.02, eta_sf=0.05, n_cl=160.0, M_cl_init=1e6 * M_SUN,
               template="SB99", imf="kroupa100", star_type="sin",
               profile_type="uniform")
results = ev.run_simulation()
g = results[0]

t = g["time"] / MYR_TO_SEC
R = g["radius"] / PC_TO_CM

fig, ax1 = plt.subplots(figsize=(5, 3.6))
ax1.semilogy(t, R, color="#CB4035", label="shell radius")
ax1.set_xlabel("t [Myr]"); ax1.set_ylabel("R [pc]", color="#CB4035")
for typ, tt, _ in g.get("phase_transitions", []):
    if typ == "phase1_to_fragmentation":
        ax1.axvline(tt / MYR_TO_SEC, ls="--", color="0.5", lw=1, label="fragmentation")
ax1.legend(frameon=False, fontsize=8)
fig.suptitle("Deterministic shell evolution (SB99, uniform cloud)")
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "03_run_evolution.png")
fig.savefig(out, dpi=130)
print("wrote", out)
print(f"  evolved to {t[-1]:.1f} Myr; final radius {R[-1]:.1f} pc; {len(t)} timepoints")
