"""Build the single-star feedback database (`single_star_tracks.h5`).

This HDF5 database is the input to TODDLERS' **stochastic** IMF sampling (and the
EEP interpolation). It is NOT shipped with the package -- it is large (~0.5 GB for
the full grid) and is regenerated here from the committed pySB99 tracks/spectra, so
that what you run always matches the stellar library in the repo.

For each (metallicity, mass) it runs pySB99 for a single star and stores the full
feedback evolution. The result is written to the package data directory
(`$TODDLERS_DATA` or `<pkg>/database`), where the stochastic sampler looks for it.

Quick check (one metallicity, low mass cap -- ~2 min)::

    python examples/build_stochastic_database.py --quick

Full database (all metallicities, full mass grid -- slow, ~0.5 GB)::

    python examples/build_stochastic_database.py
"""
import argparse
import os
import time

from toddlers.pysb99.stochastic.database import generate_single_star_database
from toddlers._paths import get_database_path


def main():
    ap = argparse.ArgumentParser(description="Build single_star_tracks.h5")
    ap.add_argument("--quick", action="store_true",
                    help="small smoke build (MW only, mass <= 15 Msun, 5 Myr)")
    ap.add_argument("--output", default=None,
                    help="output path (default: the resolved package database path)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    output = args.output or get_database_path()  # <data>/single_star_tracks.h5
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    kw = dict(output_file=output, overwrite=args.overwrite)
    if args.quick:
        # smoke build: proves the pipeline end-to-end in ~2 min
        kw.update(metallicities=["MW"], max_mass=15.0,
                  time_max_myr=5.0, time_resolution_myr=0.5)
        print("Quick build (MW, mass <= 15 Msun, 5 Myr) -> ", output)
    else:
        # full database: all metallicities + full mass grid, 100 Myr at 0.1 Myr
        print("Full build (all metallicities, full mass grid) -> ", output)
        print("This is slow and writes ~0.5 GB.")

    t0 = time.time()
    generate_single_star_database(**kw)
    print(f"Done in {time.time() - t0:.0f}s -> {output}")


if __name__ == "__main__":
    main()
