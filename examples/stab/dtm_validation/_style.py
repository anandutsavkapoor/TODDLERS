"""Shared plotting setup for the DTM-validation figures.

Applies the repository's A&A ``paper_style.mplstyle`` (serif, four-sided inward
ticks, project colour cycle) so the validation figures are publication-grade and
consistent with the paper figures, and exposes a small STAB reader helper.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toddlers.stab.stab_io import read_stab_file

_STYLE = Path(__file__).resolve().parents[2] / "paper_figures" / "paper_style.mplstyle"
if _STYLE.exists():
    plt.style.use(str(_STYLE))


def nearest(grid, value):
    """Index of the grid node nearest ``value``."""
    return int(np.argmin(np.abs(np.asarray(grid) - value)))


def load_stab(filepath):
    """Read a STAB and return (axisNames, [grids], values, lambda_micron)."""
    d = read_stab_file(filepath)
    names = d["axisNames"]
    grids = [np.asarray(g) for g in d["axisGrids"]]
    wl_micron = grids[names.index("lambda")] * 1e6   # SKIRT stores lambda in m
    values = np.asarray(d["values"])                 # (n_quantity=1, lambda, <params>)
    return names, grids, values, wl_micron


def cell_indices(names, grids, Z, sfe, ncl):
    """Param-axis node indices nearest the requested (Z, SFE, n_cl) cell."""
    return (nearest(grids[names.index("Z")], Z),
            nearest(grids[names.index("SFE")], sfe),
            nearest(grids[names.index("n_cl")], ncl))


def cell_label(grids, names, iZ, iS, iN):
    """Human-readable label of the actual grid node selected."""
    Z = grids[names.index("Z")][iZ]
    sfe = grids[names.index("SFE")][iS]
    ncl = grids[names.index("n_cl")][iN]
    return (rf"$Z={Z:g}$, $\varepsilon_{{\rm SF}}={sfe:g}$, "
            rf"$n_{{\rm cl}}={ncl:g}\,{{\rm cm^{{-3}}}}$"), (Z, sfe, ncl)
