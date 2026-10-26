"""Stochastic IMF sampling, as Evolution draws the stars of each generation (no data required).

For a low-mass cluster the IMF is not fully sampled, so feedback is built from an explicit
draw of individual stars. The IMF is named with its full mass range; each star-forming
generation draws its stars to its stellar mass (stop-after: the star that crosses the target
is kept) from its own random stream, derived from (base seed, generation). Stars below the
lowest evolutionary track (0.8 Msun) count toward the mass but give no feedback.

The figure shows one generation's mass spectrum and how the most massive star varies from
generation to generation, the origin of feedback scatter in low-mass clusters. In a run the
same draws come from
Evolution(template="pySB99", imf=None, stochastic=StochasticSpec(IMF, base_seed=BASE_SEED));
see 12_stochastic_population_pipeline.py.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toddlers.pysb99.stochastic.population import IMFSpec, draw_stop_after, generation_rng

IMF = IMFSpec("kroupa", m_low=0.1, m_up=100.0)
TOTAL_MASS = 1e3        # Msun per generation; low-mass cluster -> IMF not fully sampled
BASE_SEED = 0
N_GEN = 200             # generations drawn for the most-massive-star distribution
M_TRACK_MIN = 0.8       # Msun, lowest evolutionary track: lighter stars give no feedback

draws = [draw_stop_after(IMF, TOTAL_MASS, generation_rng(BASE_SEED, g)) for g in range(N_GEN)]
m_max = np.array([d.max() for d in draws])

for g in (0, 1):
    m = draws[g]
    print(f"generation {g}: {len(m)} stars, {m.sum():.1f} Msun (target {TOTAL_MASS:.0f}), "
          f"most massive {m.max():.1f} Msun; {(m >= M_TRACK_MIN).sum()} stars with feedback "
          f"({m[m >= M_TRACK_MIN].sum() / m.sum():.0%} of the mass), {(m > 8).sum()} above 8 Msun")
print(f"most massive star over {N_GEN} generations: min {m_max.min():.0f}, "
      f"median {np.median(m_max):.0f}, max {m_max.max():.0f} Msun")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.4))
m = draws[0]
ax1.hist(np.log10(m), bins=30, color="#4683DE")
ax1.axvline(np.log10(M_TRACK_MIN), color="0.4", ls="--", lw=1, label="lowest track (0.8 $M_\\odot$)")
ax1.set_xlabel(r"$\log_{10}(m/M_\odot)$"); ax1.set_ylabel("N stars"); ax1.legend(frameon=False, fontsize=7)
ax1.set_title(f"generation 0: {len(m)} stars, $\\Sigma m$ = {m.sum():.0f} $M_\\odot$", fontsize=8)
ax2.hist(m_max, bins=30, color="#CB4035")
ax2.set_xlabel(r"most massive star $[M_\odot]$"); ax2.set_ylabel("N generations")
ax2.set_title(f"{N_GEN} generations of $10^{{{np.log10(TOTAL_MASS):.0f}}}$ $M_\\odot$ (one stream each)", fontsize=8)
fig.suptitle(f"Stochastic IMF sampling (Kroupa, {IMF.m_low:g}-{IMF.m_up:g} $M_\\odot$)")
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_imf_sampling.png")
fig.savefig(out, dpi=130)
print("wrote", out)
