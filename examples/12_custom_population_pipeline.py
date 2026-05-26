"""End-to-end pipeline for a CUSTOM stellar population (pySB99).

Unlike the built-in SB99/BPASS templates, you can define an arbitrary population and have
TODDLERS build its feedback + SED libraries on the fly, then run the normal pipeline:

    custom population  ->  generate_custom_population_interpolants   (feedback vs age, Z)
                       ->  SpectralTableGenerator.generate_spectral_table  (Cloudy SED table)
                       ->  Evolution(template="pySB99", imf=<name>).run_simulation()
                       ->  CloudySimulationManager(...).run_full_simulation()

The first two steps are the part that is *specific* to a custom population (they synthesise
the population's feedback and incident spectra from the single-star tracks); everything
after is the standard pipeline (post-process to a BPT exactly as in `11_bpt_diagram.py`).

The generated interpolant/SED libraries are written where TODDLERS looks for them
(`get_database_path()`'s dir and `CLOUDY_DATA_DIR`), keyed by `IMF_NAME`, so the
`Evolution(template="pySB99", imf=IMF_NAME)` call picks them up automatically.

Cost: interpolant + spectral-table synthesis take a few minutes; the Cloudy stage is bounded
here with a coarse `--n-points` grid (raise it, or loop all models, for production). Cached
steps are skipped on re-runs. Needs `python scripts/download_data.py`, plus `cloudy.exe` on
PATH and a valid CLOUDY_DATA_DIR for the Cloudy stage.

Usage:
    python examples/12_custom_population_pipeline.py            # interpolants + SED table + evolution
    python examples/12_custom_population_pipeline.py --run-cloudy --n-points 6   # + Cloudy SEDs
"""
import argparse
import os

import numpy as np

from toddlers._paths import get_database_path
from toddlers.pysb99.generate_pysb99_interpolants import generate_custom_population_interpolants
from toddlers.cloudy_stellar_spectra_generator import SpectralTableGenerator
from toddlers.evolution import Evolution
from toddlers.cloudy_simulation_manager import CloudySimulationManager
from toddlers.constants import M_SUN

# A custom population: explicit star counts (mass [Msun] -> number). Top-heavy here.
IMF_NAME = "custom_topheavy"
CUSTOM_STAR_NUMBERS = {20.0: 5, 40.0: 2, 80.0: 1}
Z = 0.014                  # MW; the interpolants/SED table are built for this metallicity
Z_LABEL = "MW"
ETA_SF = 0.05


def main():
    ap = argparse.ArgumentParser(description="Custom-population end-to-end pipeline.")
    ap.add_argument("--run-cloudy", action="store_true",
                    help="also run the Cloudy stage (needs cloudy.exe + CLOUDY_DATA_DIR)")
    ap.add_argument("--n-points", type=int, default=6,
                    help="Cloudy timepoints (uniform grid) when --run-cloudy is set")
    args = ap.parse_args()

    db_dir = os.path.dirname(get_database_path())   # where pySB99 interpolants are read from

    # 1) Synthesise the custom population's feedback interpolants (age x metallicity).
    print("=== 1) feedback interpolants for the custom population ===")
    if os.path.exists(os.path.join(db_dir, f"pySB99interpolation_{IMF_NAME}.obj")):
        print(f"  interpolant exists for '{IMF_NAME}', skipping")
    else:
        generate_custom_population_interpolants(
            custom_star_numbers=CUSTOM_STAR_NUMBERS, output_dir=db_dir, imf_name=IMF_NAME,
            metallicities=[Z_LABEL], time_start_myr=0.01, time_end_myr=31.0,
            time_step_myr=0.2, verbose=True)

    # 2) Build the Cloudy incident-spectrum table for the same population (-> CLOUDY_DATA_DIR).
    print("\n=== 2) Cloudy spectral table ===")
    time_points = np.logspace(np.log10(1e4), np.log10(50e6), 40)
    SpectralTableGenerator(template="pySB99", imf=IMF_NAME, star_type="sin",
                           max_age=time_points[-1], wavelength_resolution_factor=1
                           ).generate_spectral_table(time_points=time_points,
                                                      Z_values=[Z], formation_timescale=None)

    # 3) Evolution driven by the custom population's feedback.
    print("\n=== 3) shell evolution (template=pySB99, imf=custom) ===")
    m_stellar = sum(m * n for m, n in CUSTOM_STAR_NUMBERS.items())
    ev = Evolution(Z=Z, eta_sf=ETA_SF, n_cl=100.0, M_cl_init=(m_stellar / ETA_SF) * M_SUN,
                   template="pySB99", imf=IMF_NAME, star_type="sin",
                   cluster_formation_mode="burst", formation_timescale=None,
                   profile_type="uniform")
    ev.run_simulation()
    dat, _ = ev.get_output_paths()
    print("  evolution output:", dat)

    # 4) Cloudy post-processing (optional; the expensive step).
    if args.run_cloudy:
        print("\n=== 4) Cloudy time series (bounded) ===")
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
