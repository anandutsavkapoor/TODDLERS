"""Aggregate worker-pool results and emit unfinished tasks for a resume.

Given the original task file and the results file(s) produced by the workers, this
reports how many tasks succeeded, failed, or never ran, and (optionally) writes a
fresh task file containing every task not confirmed ``OK`` so the job can be
resubmitted against just the remaining work.

The resume file is a standalone task file: rerun the worker pool against it with a
new results file (its line indices are independent of the original).

Usage::

    python -m toddlers.hpc.check_status --task-file tasks/cloudy_shell.tasks \
        --results 'results/*.results' [-o tasks/cloudy_shell.resume.tasks]
"""
import argparse
import glob
import sys
from pathlib import Path


def parse_results(results_paths):
    """Return (ok_indices, fail_indices) read from one or more results files."""
    ok, fail = set(), set()
    for path in results_paths:
        with open(path) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                try:
                    idx = int(parts[0])
                except ValueError:
                    continue
                if parts[1] == "OK":
                    ok.add(idx)
                elif parts[1] == "FAIL":
                    fail.add(idx)
    return ok, fail


def main(argv=None):
    p = argparse.ArgumentParser(description="Aggregate worker results / emit resume.")
    p.add_argument("--task-file", required=True)
    p.add_argument("--results", required=True,
                   help="glob (quote it) or single path to results file(s)")
    p.add_argument("-o", "--out", default=None,
                   help="write unfinished tasks here for resubmission")
    args = p.parse_args(argv)

    lines = Path(args.task_file).read_text().splitlines()
    total = len([ln for ln in lines if ln.strip()])

    results_paths = sorted(glob.glob(args.results)) or (
        [args.results] if Path(args.results).exists() else [])
    if not results_paths:
        print(f"No results files matched {args.results!r}", file=sys.stderr)

    ok, fail = parse_results(results_paths)
    # A task confirmed OK is done; everything else (failed or never attempted) remains.
    remaining = [i for i in range(total) if i not in ok]

    print(f"task file : {args.task_file}  ({total} tasks)")
    print(f"results   : {len(results_paths)} file(s)")
    print(f"  OK      : {len(ok)}")
    print(f"  FAILED  : {len(fail)}")
    print(f"  pending : {total - len(ok) - len(fail)}")
    print(f"  remaining (failed + pending) : {len(remaining)}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for i in remaining:
                f.write(lines[i] + "\n")
        print(f"Wrote {len(remaining)} unfinished tasks -> {out}")

    # Non-zero exit if anything is unfinished, handy for scripted retry loops.
    sys.exit(1 if remaining else 0)


if __name__ == "__main__":
    main()
