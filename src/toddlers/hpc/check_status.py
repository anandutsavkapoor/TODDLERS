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
import hashlib
import sys
from pathlib import Path


def task_key(raw_line):
    """Stable content hash of a task line, independent of its position in the file.

    A resume task file is written as byte-identical copies of the original task lines
    (just a re-numbered subset), so a *content* key matches a result back to its task
    across rounds even though line indices restart at 0 in every resume file. Matching
    on the line index instead is the historic miscount: a resume round records "OK at
    index k" against the resume file, the gate then aggregates it against the original
    file, and index k there is a different task -- so already-done (skipped) tasks were
    never credited and got re-listed every round until the retry guard fired.
    """
    return hashlib.sha1(raw_line.strip().encode()).hexdigest()[:16]


def parse_results(results_paths):
    """Aggregate worker results files.

    Returns ``(ok_keys, fail_keys, ok_idx, fail_idx)``. Current workers write
    ``idx<TAB>STATUS<TAB>KEY<TAB>info`` (4+ fields) and are matched by content KEY;
    legacy 3-field ``idx<TAB>STATUS[<TAB>info]`` lines fall back to index matching so
    old results files still parse.
    """
    ok_keys, fail_keys, ok_idx, fail_idx = set(), set(), set(), set()
    for path in results_paths:
        with open(path) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                status = parts[1]
                if status not in ("OK", "FAIL"):
                    continue
                if len(parts) >= 4:                       # current: idx, status, key, info
                    (ok_keys if status == "OK" else fail_keys).add(parts[2])
                else:                                     # legacy: idx, status[, info]
                    try:
                        idx = int(parts[0])
                    except ValueError:
                        continue
                    (ok_idx if status == "OK" else fail_idx).add(idx)
    return ok_keys, fail_keys, ok_idx, fail_idx


def main(argv=None):
    p = argparse.ArgumentParser(description="Aggregate worker results / emit resume.")
    p.add_argument("--task-file", required=True)
    p.add_argument("--results", required=True,
                   help="glob (quote it) or single path to results file(s)")
    p.add_argument("-o", "--out", default=None,
                   help="write unfinished tasks here for resubmission")
    args = p.parse_args(argv)

    lines = Path(args.task_file).read_text().splitlines()
    indices = [i for i, ln in enumerate(lines) if ln.strip()]
    total = len(indices)

    results_paths = sorted(glob.glob(args.results)) or (
        [args.results] if Path(args.results).exists() else [])
    if not results_paths:
        print(f"No results files matched {args.results!r}", file=sys.stderr)

    ok_keys, fail_keys, ok_idx, fail_idx = parse_results(results_paths)
    keys = {i: task_key(lines[i]) for i in indices}

    def is_ok(i):
        return keys[i] in ok_keys or i in ok_idx

    def is_fail(i):
        return not is_ok(i) and (keys[i] in fail_keys or i in fail_idx)

    # A task confirmed OK is done; everything else (failed or never attempted) remains.
    remaining = [i for i in indices if not is_ok(i)]
    n_ok = total - len(remaining)
    n_fail = sum(1 for i in indices if is_fail(i))

    print(f"task file : {args.task_file}  ({total} tasks)")
    print(f"results   : {len(results_paths)} file(s)")
    print(f"  OK      : {n_ok}")
    print(f"  FAILED  : {n_fail}")
    print(f"  pending : {total - n_ok - n_fail}")
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
