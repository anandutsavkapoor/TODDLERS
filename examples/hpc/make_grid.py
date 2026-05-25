"""Emit an evolution grid JSON from the canonical parameter axes.

The paper parameter grid axes (metallicity, SFE, cloud density, cloud mass) are defined
once in ``toddlers.stab.config`` and selected by the stellar
template there. This script reads those arrays and writes the ``name -> [values]``
JSON that ``toddlers.hpc.generate_tasks evolution`` expands, so the grid stays in
sync with the STAB axes instead of being a hand-maintained static file.

DTM is intentionally NOT an axis here: for the paper it is swept at the Cloudy
stage (``generate_tasks cloudy --dust-to-metal ...``), so evolution runs once at
the fiducial DTM. See hpc/README.md.

Usage:
    python examples/hpc/make_grid.py -o examples/hpc/sb99_2023_grid.json
"""
import argparse
import json
import os

# the canonical axes live with the STAB code
from toddlers.stab import config as nc


def main():
    ap = argparse.ArgumentParser(description="Write the evolution grid JSON.")
    ap.add_argument("-o", "--out", default=os.path.join(os.path.dirname(__file__),
                    "sb99_2023_grid.json"))
    args = ap.parse_args()

    grid = {
        "Z": [float(z) for z in nc.METALLICITIES],
        "eta_sf": [float(e) for e in nc.STAR_FORMATION_EFFICIENCIES],
        "n_cl": [float(n) for n in nc.CLOUD_DENSITIES],
        "M_cl_init": [float(m) for m in nc.MASS_BIN_CENTERS],
        "template": [nc.STELLAR_TEMPLATE],
        "imf": [nc.IMF_TYPE],
        "star_type": [nc.STAR_TYPE],
        "profile_type": ["uniform"],
        "cluster_formation_mode": ["burst"],
    }
    n = len(grid["Z"]) * len(grid["eta_sf"]) * len(grid["n_cl"]) * len(grid["M_cl_init"])
    with open(args.out, "w") as f:
        json.dump(grid, f, indent=2)
    print(f"wrote {args.out}: {n} evolution runs "
          f"({nc.STELLAR_TEMPLATE}/{nc.IMF_TYPE}/{nc.STAR_TYPE}, fiducial DTM)")


if __name__ == "__main__":
    main()
