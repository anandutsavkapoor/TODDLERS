"""End-to-end pipeline for a STOCHASTICALLY sampled population (pySB99).

Example 09 stops at sampling a population; this carries it through the full chain:

    sample_imf_discrete + sample_ages              (draw a discrete population)
    -> build_stochastic_interpolants_2d / save     (its feedback vs age, Z)
    -> generate_cloudy_spectral_table_2d           (its Cloudy incident SED table)
    -> Evolution(template="pySB99", imf=<name>).run_simulation()
    -> CloudySimulationManager(...).run_full_simulation()

A stochastic population is just a specific realisation of stars (set by `SEED`); the
interpolant + SED table synthesised from it are written where TODDLERS looks for them
(`get_database_path()`'s dir and `CLOUDY_DATA_DIR`), keyed by `IMF_NAME`, so
`Evolution(template="pySB99", imf=IMF_NAME)` picks them up. Everything after evolution is
the standard pipeline (post-process to a BPT as in `11_bpt_diagram.py`).

Cost: synthesis takes a few minutes; the Cloudy stage is bounded here with a coarse
`--n-points` grid. Needs the single-star database (`python scripts/download_data.py
--stochastic-tracks`, or build it with `examples/build_stochastic_database.py`), plus
`cloudy.exe` + CLOUDY_DATA_DIR for the Cloudy stage.

Usage:
    python examples/13_stochastic_population_pipeline.py            # sample -> interpolants -> SED table -> evolution
    python examples/13_stochastic_population_pipeline.py --run-cloudy --n-points 6
"""
import argparse
import os

import numpy as np

from toddlers._paths import get_database_path
from toddlers.pysb99.stochastic.sampling import sample_imf_discrete, sample_ages
from toddlers.pysb99.stochastic.interpolants import (
    build_stochastic_interpolants_2d, save_stochastic_interpolants,
    generate_cloudy_spectral_table_2d)
from toddlers.evolution import Evolution
from toddlers.cloudy_simulation_manager import CloudySimulationManager
from toddlers.constants import M_SUN

IMF_NAME = "stochastic_kroupa_burst_5e4_seed42"
M_CLUSTER = 5e4        # total stellar mass to sample [Msun]
M_UPPER = 120.0        # truncate the track grid here before sampling
T_SF_MYR = 0.0         # 0 = instantaneous burst; >0 = constant SFR over this window
SEED = 42
Z = 0.014              # MW; interpolants/SED table built for this metallicity
Z_LABEL = "MW"
ETA_SF = 0.1


def main():
    ap = argparse.ArgumentParser(description="Stochastic-population end-to-end pipeline.")
    ap.add_argument("--run-cloudy", action="store_true",
                    help="also run the Cloudy stage (needs cloudy.exe + CLOUDY_DATA_DIR)")
    ap.add_argument("--n-points", type=int, default=6,
                    help="Cloudy timepoints (uniform grid) when --run-cloudy is set")
    args = ap.parse_args()

    db = get_database_path()
    db_dir = os.path.dirname(db)
    time_grid = np.logspace(np.log10(0.01), np.log10(50), 50)

    # 1) Draw the discrete population (exact track-grid masses) and per-star ages.
    print("=== 1) sample the stochastic population ===")
    masses = sample_imf_discrete(total_mass=M_CLUSTER, database_path=db, metallicity=Z_LABEL,
                                 m_upper=M_UPPER, imf_name="kroupa", seed=SEED, verbose=True)
    masses = np.around(masses, decimals=3)
    ages = sample_ages(n_stars=len(masses), t_sf_myr=T_SF_MYR, mode="uniform", seed=SEED * 2)
    print(f"  sampled {len(masses)} stars, total {masses.sum():.2e} Msun")

    # 2) Feedback interpolants for THIS realisation, saved where TODDLERS reads them.
    print("\n=== 2) feedback interpolants ===")
    if os.path.exists(os.path.join(db_dir, f"pySB99interpolation_{IMF_NAME}.obj")):
        print(f"  interpolant exists for '{IMF_NAME}', skipping")
    else:
        interp = build_stochastic_interpolants_2d(
            masses=masses, initial_ages=ages, database_path=db, metallicities=[Z_LABEL],
            time_grid_myr=time_grid, imf_name="kroupa", verbose=True)
        save_stochastic_interpolants(interpolants=interp, output_dir=db_dir,
                                     imf_name=IMF_NAME, overwrite=True)

    # 3) Cloudy incident-spectrum table for the same realisation (-> CLOUDY_DATA_DIR).
    print("\n=== 3) Cloudy spectral table ===")
    generate_cloudy_spectral_table_2d(
        masses=masses, initial_ages=ages, database_path=db, metallicities=[Z_LABEL],
        output_filename=f"pySB99_stochastic_2d_{IMF_NAME}_burst.ascii",
        time_grid_myr=time_grid, verbose=True)

    # 4) Evolution driven by the stochastic population's feedback.
    print("\n=== 4) shell evolution (template=pySB99, imf=stochastic) ===")
    ev = Evolution(Z=Z, eta_sf=ETA_SF, n_cl=100.0, M_cl_init=(M_CLUSTER / ETA_SF) * M_SUN,
                   template="pySB99", imf=IMF_NAME, star_type="sin",
                   cluster_formation_mode="burst", formation_timescale=None,
                   profile_type="uniform")
    ev.run_simulation()
    dat, _ = ev.get_output_paths()
    print("  evolution output:", dat)

    # 5) Cloudy post-processing (optional; the expensive step).
    if args.run_cloudy:
        print("\n=== 5) Cloudy time series (bounded) ===")
        mgr = CloudySimulationManager(dat, method="uniform", n_points=args.n_points,
                                      add_DIG=False, continue_after_dissolution=True)
        mgr.run_full_simulation()
        print("  Cloudy output dir:", mgr.cloudy_run_dir)
        print("  -> post-process to a BPT as in examples/11_bpt_diagram.py "
              f"(`python examples/11_bpt_diagram.py {dat}`)")
    else:
        print("\n(Skipped Cloudy. Add --run-cloudy to run it, or feed the .dat to "
              "examples/11_bpt_diagram.py / 04_cloudy_pipeline.py.)")
    print("\nDONE")


if __name__ == "__main__":
    main()
