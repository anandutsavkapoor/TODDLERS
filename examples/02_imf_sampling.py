"""Stochastic IMF sampling (no data required).

Draws a stellar population from the Kroupa IMF for a low-mass cluster using the
stop-after method, and shows the sampled mass distribution and how the most-massive
star varies between realizations (the origin of feedback scatter in low-mass clusters).
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toddlers.pysb99.stochastic.sampling import sample_imf

TOTAL_MASS = 1e3        # Msun; low-mass cluster -> IMF not fully sampled
N_REAL = 200

m_max = []
for seed in range(N_REAL):
    masses = sample_imf(total_mass=TOTAL_MASS, imf_name="kroupa",
                        m_min=0.08, m_max=120.0, seed=seed)
    m_max.append(masses.max())
m_max = np.array(m_max)

# one representative realization for the mass spectrum
masses = sample_imf(total_mass=TOTAL_MASS, imf_name="kroupa", m_min=0.08, m_max=120.0, seed=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.4))
ax1.hist(np.log10(masses), bins=30, color="#4683DE")
ax1.set_xlabel(r"$\log_{10}(m/M_\odot)$"); ax1.set_ylabel("N stars")
ax1.set_title(f"one realization ({len(masses)} stars, "
              fr"$\Sigma m$={masses.sum():.0f} $M_\odot$)", fontsize=8)
ax2.hist(m_max, bins=30, color="#CB4035")
ax2.set_xlabel(r"most massive star $[M_\odot]$"); ax2.set_ylabel("N realizations")
ax2.set_title(f"{N_REAL} realizations at {TOTAL_MASS:.0e} Msun", fontsize=8)
fig.suptitle("Stochastic IMF sampling")
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "02_imf_sampling.png")
fig.savefig(out, dpi=130)
print("wrote", out)
print(f"  most-massive star over {N_REAL} realizations: "
      f"min={m_max.min():.0f}, median={np.median(m_max):.0f}, max={m_max.max():.0f} Msun")
