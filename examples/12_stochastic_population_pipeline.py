"""End-to-end pipeline for a STOCHASTICALLY sampled population (pySB99).

The stars are drawn at run time, one independent draw per star-forming generation:

    Evolution(template="pySB99", imf=None, stochastic=StochasticSpec(...)).run_simulation()
        each generation i draws its stars from the named IMF over its full range, to
        eta_sf (1 - eta_sf)^i M_cl_init, from a stream derived from (BASE_SEED, i), and
        gets the feedback of exactly those stars; star lists are saved with the output
    -> CloudySimulationManager(...).run_full_simulation()
        rebuilds each generation from its saved star list and writes one Cloudy star
        table per generation (snap mode only)

In constant_sfr mode the stars of a generation are born sequentially at its SFR
(t_k = (mass drawn before k + m_k/2) / SFR), and the table is not convolved again.

The cloud recollapses, so several generations form; the figure (12_stochastic_population_pipeline.png)
shows the ionizing photon rate of each generation's own stars against its age (per unit
stellar mass, so the draws compare directly) and the shell radius.

Cost: about ten minutes for the evolution (the per-generation feedback build reads the
single-star database once per metallicity); the Cloudy star tables take longer (one spectrum
query per star and age). Needs the single-star database (`python scripts/download_data.py
--stochastic-tracks`), plus `cloudy.exe` + CLOUDY_DATA_DIR for the Cloudy stage.

Usage:
    python examples/12_stochastic_population_pipeline.py            # evolution only
    python examples/12_stochastic_population_pipeline.py --run-cloudy --n-points 6
"""
import argparse
import os

import numpy as np

from toddlers.pysb99.stochastic.population import IMFSpec, StochasticSpec
from toddlers.evolution import Evolution
from toddlers.cloudy_simulation_manager import CloudySimulationManager
from toddlers.constants import M_SUN, MYR_TO_SEC, PC_TO_CM

# A cloud that recollapses: each star-forming generation draws its own stars.
M_CLOUD = 1e6          # initial cloud mass [Msun]
N_CL = 2560.0          # cloud density [cm-3]
BASE_SEED = 23
Z = 0.014              # MW (a metallicity of the single-star database)
ETA_SF = 0.01
SPEC = StochasticSpec(IMFSpec("kroupa", m_low=0.1, m_up=100.0), base_seed=BASE_SEED,
                      interpolation_mode="snap")


def plot_generations(gens, out):
    """Ionizing photon rate of each generation's own stars against its age, and the shell radius."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.4))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(gens)))
    lows = []
    for j in range(len(gens)):      # generation j's column in every later generation record
        t = np.concatenate([g["time"] for g in gens[j:]]) / MYR_TO_SEC
        Q = np.concatenate([np.asarray(g["stellar_feedback"]["Q_i"])[:, j] for g in gens[j:]])
        m_j = ETA_SF * (1 - ETA_SF) ** j * M_CLOUD
        ax1.semilogy(t - t[0], Q / m_j, color=colors[j], lw=1, label=f"generation {j}")
        age = t - t[0]
        lows.append((Q / m_j)[(age <= 10) & (Q > 0)].min())
    for g in gens:
        t = np.asarray(g["time"]) / MYR_TO_SEC
        ax2.plot(t, np.asarray(g["radius"]) / PC_TO_CM, color="#CB4035", lw=1)
        ax2.axvline(t[0], color="0.6", ls=":", lw=0.7)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(bottom=0.5 * min(lows))
    ax1.set_xlabel("age of the generation [Myr]")
    ax1.set_ylabel(r"$Q_{\rm H}$ per stellar mass [s$^{-1}$ $M_\odot^{-1}$]")
    ax1.legend(frameon=False, fontsize=6)
    ax2.set_xlabel("t [Myr]"); ax2.set_ylabel("shell radius [pc]")
    fig.suptitle("Stochastic population: one draw per generation (right, dotted: generation births)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description="Stochastic-population end-to-end pipeline.")
    ap.add_argument("--run-cloudy", action="store_true",
                    help="also run the Cloudy stage (needs cloudy.exe + CLOUDY_DATA_DIR)")
    ap.add_argument("--n-points", type=int, default=6,
                    help="Cloudy timepoints (uniform grid) when --run-cloudy is set")
    args = ap.parse_args()

    print("=== 1) shell evolution (template=pySB99, run-time stochastic draws) ===")
    ev = Evolution(Z=Z, eta_sf=ETA_SF, n_cl=N_CL, M_cl_init=M_CLOUD * M_SUN,
                   template="pySB99", imf=None, star_type="sin",
                   cluster_formation_mode="burst", formation_timescale=None,
                   profile_type="uniform", stochastic=SPEC)
    gens = ev.run_simulation()
    dat, _ = ev.get_output_paths()
    print("  evolution output:", dat, f"({len(gens)} generations)")
    plot_generations(gens, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "12_stochastic_population_pipeline.png"))

    # 2) Cloudy post-processing (optional; the expensive step).
    if args.run_cloudy:
        print("\n=== 2) Cloudy time series (bounded) ===")
        mgr = CloudySimulationManager(dat, method="uniform", n_points=args.n_points,
                                      add_DIG=False, continue_after_dissolution=True)
        mgr.run_full_simulation()
        print("  Cloudy output dir:", mgr.cloudy_run_dir)
        print("  -> post-process to a BPT as in examples/10_bpt_diagram.py "
              f"(`python examples/10_bpt_diagram.py {dat}`)")
    else:
        print("\n(Skipped Cloudy. Add --run-cloudy to run it.)")
    print("\nDONE")


if __name__ == "__main__":
    main()
