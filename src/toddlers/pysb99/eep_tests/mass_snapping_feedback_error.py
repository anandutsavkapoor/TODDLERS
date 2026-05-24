#!/usr/bin/env python3
"""
Quantify the effect of mass-grid snapping on integrated stochastic feedback.

Referee point (p6c1): the stochastic sampler rounds each sampled star to the
nearest single-star track-grid mass so that feedback is read directly from the
database without interpolation in mass. The <1% deviation reported for the total
*mass* does not bound the error in the integrated *feedback*, which scales
non-linearly with stellar mass on a grid that is coarse at the high-mass end.

This script compares the population-integrated feedback (ionizing photon rate
Q(H), wind mechanical power, and wind momentum injection) computed two ways from
an identical set of sampled continuous masses:

    - "snapped" : each star rounded to the nearest track-grid mass (production behaviour)
    - "true"    : feedback interpolated (log-feedback vs. log-mass) at the continuous mass

It reports the distribution of the relative error (snapped - true)/true over many
realizations, both instantaneously at a few ages and time-integrated over 0-10 Myr.
It also repeats the test on a deliberately coarsened grid to expose the dependence
on grid sampling.

Outputs:
    analysis/mass_snapping_feedback_error.json   (numbers)
    analysis/mass_snapping_feedback_error.pdf    (figure)
"""
import os
import sys
import json

import numpy as np
import h5py

# project paths
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "pysb99"))
from stochastic.sampling import sample_imf  # continuous stop-after sampler

DB = os.path.join(ROOT, "src", "database", "single_star_tracks.h5")
ZGRP = "Z_0.0140"           # solar metallicity
QUANTS = ["Q_HI", "wind_power", "wind_momentum"]   # all stored as log10
QLABEL = {"Q_HI": r"$Q(\mathrm{H})$",
          "wind_power": r"$L_\mathrm{mech}$",
          "wind_momentum": r"$\dot{p}_\mathrm{wind}$"}

N_REAL = 500                # realizations per configuration
M_STAR_LIST = [1e3, 1e4]    # target stellar masses [Msun]
M_UPPER = 100.0             # upper mass limit (truncates grid before sampling)
AGES_MYR = [0.0, 1.0, 3.0, 5.0]   # instantaneous evaluation ages
T_INT_MYR = 10.0            # time-integration upper limit


def load_tracks(db, zgrp):
    with h5py.File(db, "r") as f:
        tg = np.array(f["metadata"]["time_grid"])      # Myr, (n_t,)
        g = f[zgrp]
        keys = [k for k in g.keys()
                if k.startswith("mass_") and isinstance(g[k], h5py.Group)]
        masses = np.array([float(g[k].attrs["mass_Msun"]) for k in keys])
        data = {q: np.array([np.array(g[k][q]) for k in keys]) for q in QUANTS}  # (n_m, n_t) log10
    order = np.argsort(masses)
    masses = masses[order]
    for q in QUANTS:
        data[q] = data[q][order]
    return tg, masses, data


def integrated_timeseries(query_masses, grid_masses, grid_logfb, snapped):
    """Population-integrated linear feedback time series, shape (n_t,)."""
    logm_g = np.log10(grid_masses)
    if snapped:
        idx = np.abs(grid_masses[None, :] - query_masses[:, None]).argmin(axis=1)
        lin = np.power(10.0, grid_logfb[idx, :])
    else:
        logm_q = np.log10(np.clip(query_masses, grid_masses.min(), grid_masses.max()))
        lin = np.empty((len(query_masses), grid_logfb.shape[1]))
        for j in range(grid_logfb.shape[1]):
            lin[:, j] = np.interp(logm_q, logm_g, grid_logfb[:, j])
        lin = np.power(10.0, lin)
    return lin.sum(axis=0)


def rel_errors_for_grid(snap_masses, snap_data, full_masses, full_data,
                        tg, m_star, n_real, seed0):
    """
    For n_real realizations, return relative errors (snapped - true)/true for each
    quantity, at the requested ages and time-integrated. 'true' always uses the FULL
    grid interpolation; 'snapped' snaps to snap_masses (which may be coarsened).
    Each *_data dict is aligned row-for-row with its corresponding *_masses array.
    """
    age_idx = [int(np.argmin(np.abs(tg - a))) for a in AGES_MYR]
    tmask = tg <= T_INT_MYR
    m_min, m_max = full_masses.min(), full_masses.max()

    out = {q: {"age": {a: [] for a in AGES_MYR}, "integrated": []} for q in QUANTS}
    for r in range(n_real):
        masses = sample_imf(total_mass=m_star, imf_name="kroupa",
                            m_min=m_min, m_max=m_max, seed=seed0 + r)
        masses = masses[masses <= M_UPPER]
        if masses.size == 0:
            continue
        for q in QUANTS:
            true_ts = integrated_timeseries(masses, full_masses, full_data[q], snapped=False)
            snap_ts = integrated_timeseries(masses, snap_masses, snap_data[q], snapped=True)
            for a, ai in zip(AGES_MYR, age_idx):
                t = true_ts[ai]
                if t > 0:
                    out[q]["age"][a].append((snap_ts[ai] - t) / t)
            ti_true = np.trapz(true_ts[tmask], tg[tmask])
            ti_snap = np.trapz(snap_ts[tmask], tg[tmask])
            if ti_true > 0:
                out[q]["integrated"].append((ti_snap - ti_true) / ti_true)
    return out


def summarize(errs):
    a = np.array(errs) * 100.0  # percent
    if a.size == 0:
        return None
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "p16": float(np.percentile(a, 16)),
        "p84": float(np.percentile(a, 84)),
        "p2.5": float(np.percentile(a, 2.5)),
        "p97.5": float(np.percentile(a, 97.5)),
        "max_abs": float(np.max(np.abs(a))),
    }


def main():
    tg, full_grid, data = load_tracks(DB, ZGRP)
    trunc = full_grid <= M_UPPER
    full_grid_trunc = full_grid[trunc]
    data_trunc = {q: data[q][trunc] for q in QUANTS}

    # coarsened grid: drop every other track above 20 Msun
    hi = full_grid_trunc >= 20.0
    keep = np.ones_like(full_grid_trunc, dtype=bool)
    hi_idx = np.where(hi)[0]
    keep[hi_idx[1::2]] = False        # drop alternate high-mass tracks
    coarse_grid = full_grid_trunc[keep]
    coarse_data = {q: data_trunc[q][keep] for q in QUANTS}

    # (grid masses, grid-aligned data) per configuration
    grids = {
        "production": (full_grid_trunc, data_trunc),
        "coarsened": (coarse_grid, coarse_data),
    }

    results = {
        "metadata": {
            "Z_group": ZGRP,
            "m_upper_Msun": M_UPPER,
            "n_realizations": N_REAL,
            "ages_myr": AGES_MYR,
            "t_int_myr": T_INT_MYR,
            "full_grid_above_20": [float(x) for x in np.sort(full_grid_trunc[full_grid_trunc >= 20])],
            "coarse_grid_above_20": [float(x) for x in np.sort(coarse_grid[coarse_grid >= 20])],
        },
        "results": {},
    }

    for m_star in M_STAR_LIST:
        results["results"][f"Mstar_{m_star:.0e}"] = {}
        for gname, (gmasses, gdata) in grids.items():
            errs = rel_errors_for_grid(gmasses, gdata, full_grid_trunc, data_trunc, tg,
                                       m_star, N_REAL, seed0=1000)
            block = {}
            for q in QUANTS:
                block[q] = {
                    "integrated": summarize(errs[q]["integrated"]),
                    "ages": {f"{a:g}Myr": summarize(errs[q]["age"][a]) for a in AGES_MYR},
                }
            results["results"][f"Mstar_{m_star:.0e}"][gname] = block

    out_json = os.path.join(HERE, "mass_snapping_feedback_error.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Wrote {out_json}")

    # console summary
    for m_star in M_STAR_LIST:
        print(f"\n=== M* = {m_star:.0e} Msun ===")
        for gname in grids:
            print(f"  [{gname} grid]")
            for q in QUANTS:
                ti = results["results"][f"Mstar_{m_star:.0e}"][gname][q]["integrated"]
                if ti:
                    print(f"    {q:14s} time-integrated err: "
                          f"median {ti['median']:+.1f}%, "
                          f"[16-84] {ti['p16']:+.1f}/{ti['p84']:+.1f}%, "
                          f"max|.| {ti['max_abs']:.1f}%")

    # figure: error distribution for Q(H), time-integrated, production grid
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(M_STAR_LIST), figsize=(8, 3.2), constrained_layout=True)
        if len(M_STAR_LIST) == 1:
            axes = [axes]
        for ax, m_star in zip(axes, M_STAR_LIST):
            for gname, (gmasses, gdata) in grids.items():
                errs = rel_errors_for_grid(gmasses, gdata, full_grid_trunc, data_trunc, tg,
                                           m_star, N_REAL, seed0=1000)
                a = np.array(errs["Q_HI"]["integrated"]) * 100.0
                ax.hist(a, bins=40, histtype="step", label=f"{gname}")
            ax.axvline(0, color="k", lw=0.6)
            ax.set_title(f"$M_*={m_star:.0e}\\,M_\\odot$")
            ax.set_xlabel(r"$Q(\mathrm{H})$ snapping error [%]")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("realizations")
        out_pdf = os.path.join(HERE, "mass_snapping_feedback_error.pdf")
        fig.savefig(out_pdf)
        print(f"Wrote {out_pdf}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
