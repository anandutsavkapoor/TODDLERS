"""End-to-end pipeline for a CUSTOM stellar population (pySB99).

Unlike the built-in SB99/BPASS templates, you can define an arbitrary population and have
TODDLERS build its feedback + SED libraries on the fly, then run the normal pipeline:

    custom population  ->  generate_custom_population_interpolants   (feedback vs age, Z)
                       ->  SpectralTableGenerator.generate_spectral_table  (Cloudy SED table)
                       ->  Evolution(template="pySB99", imf=<name>).run_simulation()
                       ->  CloudySimulationManager(...).run_full_simulation()

The first two steps are the part that is *specific* to a custom population (they synthesise
the population's feedback and incident spectra from the single-star tracks); everything
after is the standard pipeline (post-process to a BPT exactly as in `10_bpt_diagram.py`).

The generated interpolant/SED libraries are written where TODDLERS looks for them
(`get_database_path()`'s dir and `CLOUDY_DATA_DIR`), keyed by `IMF_NAME`, so the
`Evolution(template="pySB99", imf=IMF_NAME)` call picks them up automatically.

The figure (11_custom_population_pipeline.png) shows the population's feedback in the run and
the shell it drives.

Cost: interpolant + spectral-table synthesis take a few minutes; the Cloudy stage is bounded
here with a coarse `--n-points` grid (raise it, or loop all models, for production). Cached
steps are skipped on re-runs. Needs `python scripts/download_data.py`, plus `cloudy.exe` on
PATH and a valid CLOUDY_DATA_DIR for the Cloudy stage.

Usage:
    python examples/11_custom_population_pipeline.py            # interpolants + SED table + evolution
    python examples/11_custom_population_pipeline.py --run-cloudy --n-points 6   # + Cloudy SEDs
"""
import argparse
import os

import numpy as np

from toddlers._paths import get_database_path
from toddlers.pysb99.generate_pysb99_interpolants import generate_custom_population_interpolants
from toddlers.cloudy_stellar_spectra_generator import SpectralTableGenerator
from toddlers.evolution import Evolution
from toddlers.cloudy_simulation_manager import CloudySimulationManager
from toddlers.constants import M_SUN, MYR_TO_SEC, PC_TO_CM

# A custom population: explicit star counts (mass [Msun] -> number). Top-heavy here.
IMF_NAME = "custom_topheavy"
CUSTOM_STAR_NUMBERS = {20.0: 5, 40.0: 2, 80.0: 1}
Z = 0.014                  # MW; the interpolants/SED table are built for this metallicity
Z_LABEL = "MW"
ETA_SF = 0.05


def plot_run(g, out):
    """The custom population's feedback in the run, and the shell it drives."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.asarray(g["time"]) / MYR_TO_SEC
    sf = g["stellar_feedback"]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.0))
    axes[0].semilogy(t, np.asarray(sf["L_mech"]).sum(axis=1), color="#4683DE")
    axes[0].set_ylabel(r"$L_{\rm mech}$ [erg s$^{-1}$]"); axes[0].set_xlabel("t [Myr]")
    axes[1].plot(t, np.asarray(sf["Q_i"]).sum(axis=1) / 1e49, color="#4683DE")
    axes[1].set_ylabel(r"$Q_{\rm H}$ [$10^{49}$ s$^{-1}$]"); axes[1].set_xlabel("t [Myr]")
    axes[2].plot(t, np.asarray(g["radius"]) / PC_TO_CM, color="#CB4035")
    axes[2].set_ylabel("shell radius [pc]"); axes[2].set_xlabel("t [Myr]")
    stars = ", ".join(f"{n} x {m:g}" for m, n in CUSTOM_STAR_NUMBERS.items())
    fig.suptitle(f"Custom population ({stars} $M_\\odot$, Z = {Z})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


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
    if os.path.exists(os.path.join(db_dir, f"pySB99_{IMF_NAME}_feedback.h5")):
        print(f"  interpolant exists for '{IMF_NAME}', skipping")
    else:
        generate_custom_population_interpolants(
            custom_star_numbers=CUSTOM_STAR_NUMBERS, output_dir=db_dir, imf_name=IMF_NAME,
            metallicities=[Z_LABEL], time_start_myr=0.01, time_end_myr=31.0,
            time_step_myr=0.1, verbose=True)   # <= 0.1 Myr between 2 and 6 Myr (table rule)

    # 2) The Cloudy incident-spectrum table for the same population (-> CLOUDY_DATA_DIR); only
    #    the Cloudy stage reads it.
    if args.run_cloudy:
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
    gens = ev.run_simulation()
    dat, _ = ev.get_output_paths()
    print("  evolution output:", dat)
    plot_run(gens[0], os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "11_custom_population_pipeline.png"))

    # 4) Cloudy post-processing (optional; the expensive step).
    if args.run_cloudy:
        print("\n=== 4) Cloudy time series (bounded) ===")
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
