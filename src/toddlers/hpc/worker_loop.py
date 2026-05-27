"""Persistent worker for the TODDLERS HPC worker pool.

One ``worker_loop`` process is launched per allocated core. Each reads the whole
task file but only executes the lines in its modular slice
(``line_index % n_workers == worker_id``), calling the inline task function
directly. Python startup is therefore paid once per worker, not once per task.

Each worker writes its own results file ``<results_file>.<worker_id>`` (one line per
task) -- no shared file, no locking, so the pool is safe across multiple nodes
(cross-node ``flock`` on a shared filesystem is unreliable). Each result line is::

    <task_index>\t<OK|FAIL>\t<content_key>\t<info-or-error>

``content_key`` is a hash of the task line (:func:`toddlers.hpc.check_status.task_key`);
it ties a result back to its task by *content*, so :mod:`toddlers.hpc.check_status`
credits it correctly even when a resume round re-numbers the task file. The leading
``task_index`` is kept for human debugging only.

Usage (normally invoked by the submit template via ``srun``; the worker reads its
slot from $SLURM_PROCID / $SLURM_NTASKS, which are global across all nodes)::

    srun --cpu-bind=cores python -m toddlers.hpc.worker_loop \
        --task-file tasks/evolution.tasks \
        --results-file results/${SLURM_JOB_ID}.results [--cloudy-exe /path/cloudy.exe]
"""
import argparse
import json
import os
import sys
import time
import traceback


def worker_results_path(results_file, worker_id):
    """Per-worker results file: ``<results_file>.<worker_id>``.

    Each worker owns its own file, so there is NO shared-file locking -- this is
    what makes the pool safe across multiple nodes (cross-node ``flock`` on a shared
    filesystem is not reliable). ``check_status`` aggregates them via a glob.
    """
    return f"{results_file}.{worker_id:05d}"


def run(task_file, n_workers, worker_id, results_file, cloudy_exe=None):
    from .runner import dispatch
    from .check_status import task_key

    out_path = worker_results_path(results_file, worker_id)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with open(task_file) as f:
        lines = f.readlines()

    tag = f"[w{worker_id:03d}]"
    t0 = time.time()
    n_done = n_ok = n_fail = 0

    # Sole writer of this file -> plain appends, no flock. Flush+fsync each line so
    # partial progress survives a walltime kill and is visible to check_status.
    with open(out_path, "a") as out:
        def record(line):
            out.write(line + "\n")
            out.flush()
            os.fsync(out.fileno())

        for idx, raw in enumerate(lines):
            if idx % n_workers != worker_id:
                continue
            raw = raw.strip()
            if not raw:
                continue
            key = task_key(raw)
            try:
                row = json.loads(raw)
                dispatch(row, cloudy_exe=cloudy_exe)
                record(f"{idx}\tOK\t{key}\t")
                n_ok += 1
            except Exception as exc:  # noqa: BLE001 - one task must not kill the worker
                msg = repr(exc).replace("\t", " ").replace("\n", " ")
                record(f"{idx}\tFAIL\t{key}\t{msg}")
                n_fail += 1
                print(f"{tag} task {idx} FAILED: {msg}", file=sys.stderr)
                traceback.print_exc()
            n_done += 1

    dt = time.time() - t0
    print(f"{tag} DONE: {n_done} tasks ({n_ok} ok, {n_fail} fail) in {dt:.1f}s -> {out_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description="TODDLERS HPC worker-pool process.")
    p.add_argument("--task-file", required=True)
    p.add_argument("--n-workers", type=int, default=None,
                   help="number of workers; default $SLURM_NTASKS (srun launch)")
    p.add_argument("--worker-id", type=int, default=None,
                   help="this worker's id in [0, n_workers); default $SLURM_PROCID")
    p.add_argument("--results-file", required=True)
    p.add_argument("--cloudy-exe", default=os.environ.get("CLOUDY_EXE"),
                   help="path to cloudy.exe (Cloudy stage); default $CLOUDY_EXE")
    args = p.parse_args(argv)

    # When launched via `srun`, each task is one worker: take its slot from the
    # SLURM environment (SLURM_PROCID in [0, SLURM_NTASKS)). Explicit flags win,
    # so the same module also works under a local bash launch loop.
    n_workers = (args.n_workers if args.n_workers is not None
                 else int(os.environ.get("SLURM_NTASKS", 1)))
    worker_id = (args.worker_id if args.worker_id is not None
                 else int(os.environ.get("SLURM_PROCID", 0)))

    if not (0 <= worker_id < n_workers):
        p.error(f"worker_id ({worker_id}) must satisfy 0 <= id < n_workers ({n_workers})")

    run(args.task_file, n_workers, worker_id, args.results_file,
        cloudy_exe=args.cloudy_exe)


if __name__ == "__main__":
    main()
