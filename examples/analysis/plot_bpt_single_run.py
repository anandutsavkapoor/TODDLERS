"""Single-run BPT diagram from completed Cloudy output (output-handler based).

Verifies the analysis path end-to-end on ONE parameter set:

    Cloudy .out  ->  CloudyTableConsolidator (line-intensity tables)
                 ->  BPT line ratios across the run's completed timepoints
                 ->  log [O III]5007/Hbeta  vs  log [N II]6584/Halpha

It uses the ``no_dig`` consolidation (shell/unified models only), so it runs on
shell output alone -- no DIG models required. This is the same machinery the
multi-metallicity paper BPT figure uses, exercised on a single run.

Usage:
    python examples/analysis/plot_bpt_single_run.py <evolution_output.dat> [--with-dig]

Requires completed Cloudy output for that run (see the HPC examples / cloudy
pipeline). Needs the feedback data only to resolve the run's output directory.
"""
import argparse
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toddlers.cloudy_simulation_manager import CloudySimulationManager
from toddlers.cloudy_output_handler import CloudyTableConsolidator, ReadCloudyMainOutput

# The consolidator writes per-timepoint table files; the H-recombination lines
# (Halpha, Hbeta) live in the "H-like iso-sequence" table, the metal lines in
# "database lines". intensity_type is "intrinsic" or "emergent".
ELEMENT_H = "H-like_iso-sequence"
INTENSITY_TYPE = "intrinsic"

# BPT line keys as written by the consolidator ("<name>_<wavelength>").
OIII, HBETA, NII, HALPHA = "O3_5006.84A", "H1_4861.32A", "N2_6583.45A", "H1_6562.80A"


def parse_intensity_file(path):
    """Read a consolidated line table into {name_wavelength: intensity}."""
    intensities = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            try:
                intensities[f"{parts[0]}_{parts[1]}"] = 10 ** float(parts[2])
            except (IndexError, ValueError):
                continue
    return intensities


def consolidate_run(cloudy_run_dir, with_dig=False):
    """Build consolidated_tables_{no_dig,with_dig}/ for one run; return its path."""
    tag = "with_dig" if with_dig else "no_dig"
    out_dir = os.path.join(cloudy_run_dir, f"consolidated_tables_{tag}")
    consolidator = CloudyTableConsolidator(ReadCloudyMainOutput, output_dir=out_dir)

    model_types = ["shell", "unified"] + (["dig"] if with_dig else [])
    timesteps = set()
    for mt in model_types:
        for fn in os.listdir(cloudy_run_dir):
            if fn.startswith(mt + "_") and fn.endswith(".out"):
                try:
                    timesteps.add(float(fn.split("_")[1][:-4]))
                except (ValueError, IndexError):
                    pass

    n_ok = 0
    for t in sorted(timesteps):
        shell = os.path.join(cloudy_run_dir, f"shell_{t:.2f}.out")
        unified = os.path.join(cloudy_run_dir, f"unified_{t:.2f}.out")
        dig = os.path.join(cloudy_run_dir, f"dig_{t:.2f}.out") if with_dig else None
        if not (os.path.exists(shell) or os.path.exists(unified)):
            continue
        try:
            consolidator.consolidate_tables(
                timestep=t, shell_file=shell, unified_file=unified, dig_file=dig,
                is_within_cloud=os.path.exists(unified), add_dig=with_dig)
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 - skip a bad timestep, keep going
            print(f"  consolidate t={t:.2f} Myr failed: {exc}")
    print(f"consolidated {n_ok} timestep(s) -> {out_dir}")
    return out_dir


def draw_bpt_demarcations(ax):
    """Overlay the standard [N II]-BPT classification curves.

    Kauffmann+2003 (empirical pure-star-forming boundary) and Kewley+2001
    (theoretical maximum-starburst line). Points below Kauffmann are star-forming,
    between the two are composite, above Kewley are AGN-dominated. Each curve is
    only drawn up to its asymptote (x = 0.05 and x = 0.47 respectively).
    """
    xk = np.linspace(-2.0, 0.0, 300)      # Kauffmann valid for log[NII]/Ha < 0.05
    ax.plot(xk, 0.61 / (xk - 0.05) + 1.3, "k--", lw=1.0, label="Kauffmann+2003")
    xw = np.linspace(-2.0, 0.40, 300)     # Kewley valid for log[NII]/Ha < 0.47
    ax.plot(xw, 0.61 / (xw - 0.47) + 1.19, "k-", lw=1.0, label="Kewley+2001")


def gather_bpt(directory, intensity_type=INTENSITY_TYPE):
    """Return (times, log[NII]/Ha, log[OIII]/Hb) for all consolidated timepoints."""
    ts, x, y = [], [], []
    tag = f"{intensity_type} line intensities_"
    for fn in os.listdir(directory):
        if not (fn.startswith(f"database_lines_{tag}") and fn.endswith(".txt")):
            continue
        t = float(re.search(r"_([0-9.]+)\.txt$", fn).group(1))
        intensities = parse_intensity_file(os.path.join(directory, fn))
        h_file = os.path.join(directory, f"{ELEMENT_H}_{tag}{t:.2f}.txt")
        if os.path.exists(h_file):
            intensities.update(parse_intensity_file(h_file))
        o3, hb, n2, ha = (intensities.get(OIII), intensities.get(HBETA),
                          intensities.get(NII), intensities.get(HALPHA))
        if o3 and hb and n2 and ha:
            x.append(np.log10(n2 / ha))
            y.append(np.log10(o3 / hb))
            ts.append(t)
    order = np.argsort(ts)
    return np.array(ts)[order], np.array(x)[order], np.array(y)[order]


def main():
    ap = argparse.ArgumentParser(description="Single-run BPT from Cloudy output.")
    ap.add_argument("dat", help="evolution output .dat for the run")
    ap.add_argument("--with-dig", action="store_true",
                    help="include DIG models (needs completed dig_*.out)")
    ap.add_argument("--intensity", default=INTENSITY_TYPE,
                    choices=["intrinsic", "emergent"])
    args = ap.parse_args()

    run_dir = CloudySimulationManager(args.dat, complete_init=False).cloudy_run_dir
    print("cloudy run dir:", run_dir)

    cdir = consolidate_run(run_dir, with_dig=args.with_dig)
    ts, x, y = gather_bpt(cdir, args.intensity)
    if len(ts) == 0:
        raise SystemExit("No BPT points found -- need completed shell/unified Cloudy "
                         "output containing line intensities for this run.")

    print(f"BPT points ({len(ts)}):")
    for t, xi, yi in zip(ts, x, y):
        print(f"  t={t:5.2f} Myr   log[NII]/Ha={xi:+.3f}   log[OIII]/Hb={yi:+.3f}")

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    draw_bpt_demarcations(ax)
    if len(ts) > 1:
        ax.plot(x, y, "-", color="0.6", lw=0.8, zorder=2)
    sc = ax.scatter(x, y, c=ts, cmap="viridis", s=45, zorder=3)
    fig.colorbar(sc, label="t [Myr]")
    ax.set_xlim(-2.0, 0.6)
    ax.set_ylim(-1.0, 1.3)
    ax.set_xlabel(r"$\log_{10}$ [N II]$\lambda6584$ / H$\alpha$")
    ax.set_ylabel(r"$\log_{10}$ [O III]$\lambda5007$ / H$\beta$")
    ax.set_title(f"BPT, single run ({'with' if args.with_dig else 'no'} DIG)")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "plot_bpt_single_run.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
