"""Dynamic cloud density and variable covering fraction (needs feedback data).

Two v2 birth-cloud features, shown against the static/fully-covered baseline:
  - dynamic_cloud_density: escaping ionizing radiation expands the unswept cloud,
    lowering the back-pressure on the shell (most important at low metallicity).
  - add_cover_frac: after the cloud is swept, a covering fraction < 1 lets radiation
    and ram pressure leak through holes the 1D model cannot resolve directly.

Requires: `python scripts/download_data.py`.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toddlers.evolution import Evolution
from toddlers.constants import M_SUN, PC_TO_CM, MYR_TO_SEC

# low metallicity makes the dynamic-density channel most visible
base = dict(Z=0.001, eta_sf=0.05, n_cl=160.0, M_cl_init=1e6 * M_SUN,
            template="SB99", imf="kroupa100", star_type="sin", profile_type="uniform")
cases = [
    dict(label="baseline", dynamic_cloud_density=False),
    dict(label="dynamic density", dynamic_cloud_density=True),
    dict(label="covering frac 0.5", dynamic_cloud_density=False,
         add_cover_frac=True, post_sweep_covering_fraction=0.5),
]

fig, ax = plt.subplots(figsize=(5, 3.6))
for c in cases:
    label = c.pop("label")
    g = Evolution(**base, **c).run_simulation()[0]
    ax.semilogy(g["time"] / MYR_TO_SEC, g["radius"] / PC_TO_CM, label=label)
    print(f"  {label}: final radius {g['radius'][-1] / PC_TO_CM:.0f} pc")

ax.set_xlabel("t [Myr]"); ax.set_ylabel("R [pc]")
ax.legend(frameon=False, fontsize=9)
fig.suptitle(r"Dynamic density & covering fraction ($Z=0.001$)")
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "08_dynamic_density_and_covering.png")
fig.savefig(out, dpi=130)
print("wrote", out)
