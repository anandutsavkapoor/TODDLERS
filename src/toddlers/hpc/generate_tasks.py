"""Generate flat task files for the TODDLERS HPC worker pool.

A task file is JSON-lines: one JSON object per line, each a self-contained task.
The worker pool consumes a modular slice of the lines (``line_index % N == id``), so
the line index is the stable task id used for status tracking and resume.

Two modes:

  evolution -- expand a parameter grid (JSON: name -> list of values) into one task
               per combination. ``M_cl_init`` values are in solar masses.

  cloudy    -- enumerate (simulation x timepoint x phase) work items from evolution
               ``.dat`` outputs, writing one task file per phase (phases must run as
               dependent jobs; see hpc/README.md).

Examples
--------
::

    python -m toddlers.hpc.generate_tasks evolution --grid grid.json -o tasks
    # STAB production (default time grid is 'toddlers_v1', as the interpolant requires):
    python -m toddlers.hpc.generate_tasks cloudy --input-dir evolution_output \
        --pattern '*.dat' -o tasks --dust-to-metal 0.02 0.10 0.20 0.40 0.60 0.80 1.00
"""
import argparse
import glob
import itertools
import json
import os
import sys
from pathlib import Path


def _write_jsonl(tasks, path):
    """Write an iterable of task dicts to a JSON-lines file; return the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for task in tasks:
            f.write(json.dumps(task, sort_keys=True) + "\n")
            n += 1
    return n


def cmd_evolution(args):
    """Expand a JSON parameter grid into an evolution task file."""
    with open(args.grid) as f:
        grid = json.load(f)
    if not isinstance(grid, dict) or not grid:
        sys.exit(f"Grid file {args.grid} must be a non-empty object of name -> [values].")

    # Allow scalars in the grid as a convenience (treated as a 1-element axis).
    axes = {k: (v if isinstance(v, list) else [v]) for k, v in grid.items()}
    names = list(axes)

    def combinations():
        for values in itertools.product(*(axes[n] for n in names)):
            task = {"stage": "evolution"}
            task.update(dict(zip(names, values)))
            yield task

    out = Path(args.outdir) / "evolution.tasks"
    n = _write_jsonl(combinations(), out)
    print(f"Wrote {n} evolution tasks -> {out}")


def cmd_cloudy(args):
    """Enumerate Cloudy work items and write one task file per phase."""
    from .enumerate_cloudy import enumerate_cloudy_tasks

    files = sorted(glob.glob(os.path.join(args.input_dir, "**", args.pattern),
                             recursive=True))
    if not files:
        sys.exit(f"No files matching '{args.pattern}' under {args.input_dir}")
    print(f"Found {len(files)} evolution output files")

    dtm = args.dust_to_metal  # None -> auto per file; list -> sweep
    tasks = enumerate_cloudy_tasks(
        files, method=args.method, n_points=args.n_points, add_dig=args.add_dig,
        logU_background=args.logU_background,
        continue_after_dissolution=args.continue_after_dissolution,
        dust_to_metal=dtm, verbose=True,
    )

    # Bucket by phase so each phase becomes a separately submittable (dependent) file.
    by_phase = {}
    for task in tasks:
        by_phase.setdefault(task["phase"], []).append(task)

    if not by_phase:
        print("No Cloudy work items found (all complete or none warranted).")
        return

    outdir = Path(args.outdir)
    total = 0
    for phase, plist in by_phase.items():
        out = outdir / f"cloudy_{phase}.tasks"
        n = _write_jsonl(plist, out)
        total += n
        print(f"  {phase}: {n} tasks -> {out}")
    print(f"Wrote {total} Cloudy tasks across {len(by_phase)} phase(s).")
    print("Submit phases as dependent jobs (shell/unified -> dig); see hpc/README.md.")


def build_parser():
    p = argparse.ArgumentParser(description="Generate TODDLERS HPC task files.")
    sub = p.add_subparsers(dest="mode", required=True)

    pe = sub.add_parser("evolution", help="expand a parameter grid")
    pe.add_argument("--grid", required=True, help="JSON file: param name -> [values]")
    pe.add_argument("-o", "--outdir", default="tasks", help="output directory")
    pe.set_defaults(func=cmd_evolution)

    pc = sub.add_parser("cloudy", help="enumerate Cloudy work items from .dat files")
    pc.add_argument("--input-dir", required=True, help="dir of evolution outputs")
    pc.add_argument("--pattern", default="*.dat", help="glob for evolution outputs")
    pc.add_argument("-o", "--outdir", default="tasks", help="output directory")
    pc.add_argument("--method", default="toddlers_v1",
                    choices=["adaptive", "uniform", "toddlers_v1"],
                    help="time-sampling method. Default 'toddlers_v1' is the canonical "
                         "TODDLERS (Kapoor+23) grid the SED interpolant / STAB pipeline "
                         "REQUIRES (the interpolant asserts the Cloudy timepoints match it). "
                         "Use 'adaptive'/'uniform' only for non-STAB experiments.")
    pc.add_argument("--n-points", type=int, default=None,
                    help="number of time points (uniform method)")
    pc.add_argument("--add-dig", action="store_true", help="include DIG models")
    pc.add_argument("--logU-background", type=float, default=None,
                    help="log ionization parameter for the DIG background")
    pc.add_argument("--continue-after-dissolution", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="run models past shell dissolution (default on; the STAB "
                         "interpolant grid spans the full time range, so keep this on for "
                         "STAB production. Use --no-continue-after-dissolution otherwise).")
    pc.add_argument("--dust-to-metal", type=float, nargs="+", default=None,
                    help="DTM value(s) to sweep; omit to read each file's own value")
    pc.set_defaults(func=cmd_cloudy)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
