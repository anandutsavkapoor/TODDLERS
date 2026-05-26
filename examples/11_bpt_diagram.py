"""Local end-to-end BPT diagram for one or more runs (Evolution -> Cloudy -> BPT).

For each requested parameter set this does the full TODDLERS chain on a single machine:

    Evolution().run_simulation()                  # 1D shell dynamics -> .dat
    CloudySimulationManager(...).run_full_simulation()   # Cloudy at every timepoint
    CloudyTableConsolidator / ReadCloudyMainOutput       # line-intensity tables
    -> log [O III]5007/Hbeta  vs  log [N II]6584/Halpha  # one BPT track per run

so you can see how a star-forming region moves through the BPT plane as it ages, and
how that track shifts with a physical parameter (the default compares two metallicities).

`run_full_simulation` skips any model whose Cloudy output is already valid, so the first
execution computes the Cloudy models (a handful of minutes each) and later runs are
effectively instant. To go faster, lower N_POINTS / trim RUNS, or point `--dat` at runs
whose Cloudy output already exists (e.g. produced by the HPC examples) to just draw the
BPT without recomputing.

Requires `python scripts/download_data.py`, `cloudy.exe` on PATH, and a valid
CLOUDY_DATA_DIR. DIG is off by default (BPT here uses the shell/unified models only).

Usage:
    python examples/11_bpt_diagram.py                 # built-in multi-run demo (full chain)
    python examples/11_bpt_diagram.py a.dat b.dat     # BPT existing runs (compute if missing)
    python examples/11_bpt_diagram.py --n-points 8 --with-dig
"""
import argparse
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toddlers.evolution import Evolution
from toddlers.cloudy_simulation_manager import CloudySimulationManager
from toddlers.cloudy_output_handler import CloudyTableConsolidator, ReadCloudyMainOutput
from toddlers.constants import M_SUN

# Built-in demo: two metallicities at otherwise identical conditions. Edit freely; each
# entry is forwarded to Evolution() (M_cl_init is given here in solar masses).
RUNS = [
    {"label": "Z=0.008", "Z": 0.008, "eta_sf": 0.05, "n_cl": 160.0, "M_cl_init": 1e6,
     "template": "SB99", "imf": "kroupa100", "star_type": "sin", "profile_type": "uniform"},
    {"label": "Z=0.02", "Z": 0.02, "eta_sf": 0.05, "n_cl": 160.0, "M_cl_init": 1e6,
     "template": "SB99", "imf": "kroupa100", "star_type": "sin", "profile_type": "uniform"},
]

INTENSITY_TYPE = "intrinsic"        # "intrinsic" or "emergent"
ELEMENT_H = "H-like_iso-sequence"   # H recombination lines table
OIII, HBETA, NII, HALPHA = "O3_5006.84A", "H1_4861.32A", "N2_6583.45A", "H1_6562.80A"


# --- BPT analysis (output-handler based) ------------------------------------------------

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
    print(f"  consolidated {n_ok} timestep(s)")
    return out_dir


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


def draw_bpt_demarcations(ax):
    """Overlay the standard [N II]-BPT classification curves (Kauffmann+2003, Kewley+2001)."""
    xk = np.linspace(-2.0, 0.0, 300)      # Kauffmann valid for log[NII]/Ha < 0.05
    ax.plot(xk, 0.61 / (xk - 0.05) + 1.3, "k--", lw=1.0, label="Kauffmann+2003")
    xw = np.linspace(-2.0, 0.40, 300)     # Kewley valid for log[NII]/Ha < 0.47
    ax.plot(xw, 0.61 / (xw - 0.47) + 1.19, "k-", lw=1.0, label="Kewley+2001")


# --- run the chain ----------------------------------------------------------------------

def ensure_output(dat_path, run=None, method="uniform", n_points=10, with_dig=False):
    """Make sure a run's Cloudy output exists; return its cloudy_run_dir.

    If ``run`` (an Evolution parameter dict) is given, evolution + the full Cloudy time
    series are computed when missing (completed models are skipped). With only ``dat_path``
    we just locate the run's output directory (it must already be populated)."""
    if run is not None:
        kwargs = {k: v for k, v in run.items() if k != "label"}
        kwargs["M_cl_init"] = run["M_cl_init"] * M_SUN   # RUNS gives M_cl_init in Msun
        ev = Evolution(**kwargs)
        ev.run_simulation()
        dat_path, _ = ev.get_output_paths()
    mgr = CloudySimulationManager(dat_path, method=method, n_points=n_points,
                                  add_DIG=with_dig)
    if run is not None:
        mgr.run_full_simulation()   # skips models whose output is already valid
    return mgr.cloudy_run_dir, dat_path


def main():
    ap = argparse.ArgumentParser(description="Local end-to-end BPT for one or more runs.")
    ap.add_argument("dat", nargs="*", help="evolution .dat for existing runs (optional; "
                    "default = the built-in multi-run demo computed from scratch)")
    ap.add_argument("--n-points", type=int, default=10,
                    help="Cloudy timepoints per run (uniform grid; fewer = faster)")
    ap.add_argument("--with-dig", action="store_true", help="include DIG models")
    ap.add_argument("--intensity", default=INTENSITY_TYPE,
                    choices=["intrinsic", "emergent"])
    args = ap.parse_args()

    # Either BPT the .dat given on the command line (compute output if absent) or run the
    # built-in demo set from scratch.
    if args.dat:
        jobs = [(os.path.splitext(os.path.basename(d))[0], d, None) for d in args.dat]
    else:
        jobs = [(r["label"], None, r) for r in RUNS]

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    draw_bpt_demarcations(ax)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(jobs), 1)))

    n_drawn = 0
    for (label, dat, run), color in zip(jobs, colors):
        print(f"=== {label} ===")
        run_dir, _ = ensure_output(dat, run=run, method="uniform",
                                   n_points=args.n_points, with_dig=args.with_dig)
        cdir = consolidate_run(run_dir, with_dig=args.with_dig)
        ts, x, y = gather_bpt(cdir, args.intensity)
        if len(ts) == 0:
            print(f"  no BPT points for {label} (no completed shell/unified output?)")
            continue
        ax.plot(x, y, "-", color=color, lw=0.9, alpha=0.7, zorder=2)
        ax.scatter(x, y, color=color, s=30, zorder=3, label=f"{label} ({len(ts)} pts)")
        ax.scatter(x[:1], y[:1], color=color, s=90, marker="*", zorder=4)  # youngest = star
        n_drawn += 1

    if n_drawn == 0:
        raise SystemExit("No BPT points found for any run.")

    ax.set_xlim(-2.0, 0.6)
    ax.set_ylim(-1.0, 1.3)
    ax.set_xlabel(r"$\log_{10}$ [N II]$\lambda6584$ / H$\alpha$")
    ax.set_ylabel(r"$\log_{10}$ [O III]$\lambda5007$ / H$\beta$")
    ax.set_title(f"BPT track(s), {'with' if args.with_dig else 'no'} DIG "
                 f"(★ = youngest)")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "11_bpt_diagram.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
