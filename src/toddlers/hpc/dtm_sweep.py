#!/usr/bin/env python3
r"""Per-DTM sweep coordinator: keep the live Cloudy footprint bounded by running ONE DTM at a
time and deleting its (bulky, truth) cloudy_output only after it is verified.

The variable-DTM library is a full-grid Cloudy sweep over several f_dust; all DTMs' Cloudy
output at once would be several TiB. This coordinator instead, for each f_dust in turn:

    submit a single-DTM campaign (--stab none -> builds only that DTM's interpolant .pkl)
      -> wait for it to finish (the .stab_build_complete sentinel)
      -> verify_dtm (integrity / completeness / physical / recollapse-correlation / ...)
      -> on PASS: delete that DTM's cloudy_output (~0.5 TiB);  on FAIL: HALT and preserve
      -> next f_dust
    then assemble the 5D SFR-normalized STAB from the kept .pkls (no Cloudy re-run).

It is fully RESUMABLE from on-disk state (so it survives its own death -- relaunching just
continues) and prints a complete plan under --dry-run without submitting or deleting anything.

State per DTM, read from disk:
    interpolant .pkl present AND its cloudy_output gone   -> DONE (skip)
    interpolant .pkl present AND cloudy_output still there -> NEEDS_GATE (re-verify, then delete)
    neither                                                -> NEEDS_RUN (submit campaign)
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SENTINEL_NAME = ".stab_build_complete"     # written by the campaign gate when a build finishes
DONE_NAME = ".dtm_sweep_done"              # written by this coordinator when the whole sweep is done


def _suffix(dtm):
    return "" if abs(float(dtm) - 1.0) < 1e-9 else f"_dtm{float(dtm):g}"   # mirrors utils.dtm_label


def _log(args, msg):
    line = f"[dtm_sweep {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if not args.dry_run:
        with open(os.path.join(args.work_root, "dtm_sweep.log"), "a") as fh:
            fh.write(line + "\n")


def _interp_pkl(args, dtm):
    """The per-DTM totSED interpolant (its presence == that DTM's interpolant was built)."""
    return os.path.join(args.stab_dir, f"{args.prefix}_interp_tables",
                        f"TODDLERS_totSED_lr_{args.prefix}{_suffix(dtm)}.pkl")


def _cloudy_dirs(args, dtm):
    """Leaf Cloudy model dirs belonging to this DTM.

    Model dirs are named like ``Z0.02_eta0.05_n160.0_logM6.00[_dtm0.80]`` -- the DTM appears as
    a ``_dtm<val>`` token, and DTM=1.0 carries no token. Match on that (and on ``logM`` so only
    leaf model dirs, not the template/imf/... parents, are returned)."""
    s = _suffix(dtm)
    out = []
    for d in glob.glob(os.path.join(args.cloudy_output, "**", "*logM*"), recursive=True):
        if not os.path.isdir(d):
            continue
        base = os.path.basename(d)
        if s:
            if s.strip("_") in base:                 # e.g. "dtm0.80" in the model-dir name
                out.append(d)
        elif "_dtm" not in base:                     # DTM=1.0: model dirs with no _dtm token
            out.append(d)
    return out


def cell_state(args, dtm):
    has_pkl = os.path.exists(_interp_pkl(args, dtm))
    has_cloudy = len(_cloudy_dirs(args, dtm)) > 0
    if has_pkl and not has_cloudy:
        return "DONE"
    if has_pkl and has_cloudy:
        return "NEEDS_GATE"
    return "NEEDS_RUN"


def _run(args, cmd, *, capture=False):
    """Run a command (or just print it under --dry-run)."""
    if args.dry_run:
        print("  $ " + " ".join(str(c) for c in cmd), flush=True)
        return ""
    r = subprocess.run([str(c) for c in cmd], capture_output=capture, text=True)
    if r.returncode != 0 and not capture:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(map(str, cmd))}")
    return (r.stdout or "") if capture else ""


def _sentinel(args):
    return os.path.join(os.path.abspath(args.stab_dir), SENTINEL_NAME)


def _campaign_cmd(args, dtm):
    work = os.path.join(args.work_root, f"dtm{_suffix(dtm) or '1.00'}")
    cmd = [sys.executable, "-m", "toddlers.hpc.campaign",
           "--evolution-dir", args.evolution_dir,
           "--dust-to-metal", str(dtm),
           "--small-to-large-ratio", str(args.small_to_large_ratio),
           "--stab", "none",
           "--work-dir", work,
           "--cache-dir", args.cache_dir,
           "--stab-dir", args.stab_dir,
           "--account", args.account, "--partition", args.partition,
           "--ntasks", str(args.ntasks), "--max-nodes", str(args.max_nodes),
           "--walltime", args.walltime, "--stab-walltime", args.stab_walltime,
           "--toddlers-src", args.toddlers_src,
           "--cloudy-exe", args.cloudy_exe, "--cloudy-data", args.cloudy_data,
           "--python-module", args.python_module,
           "--max-resume-rounds", str(args.max_resume_rounds)]
    if args.activate_env:
        cmd += ["--activate-env", args.activate_env]   # extra module-load lines for every campaign job
    if args.output_root:
        cmd += ["--output-root", args.output_root]     # where cloudy_output/ is written+read (scratch)
    return cmd


def _jobs_running(args):
    """True if any campaign jobs are still queued/running (by job-name prefix)."""
    try:
        out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""),
                              "-o", "%j"], capture_output=True, text=True).stdout
    except Exception:                                            # noqa: BLE001
        return False
    return any(n.strip().startswith("campaign") for n in out.splitlines())


def wait_for_completion(args, dtm):
    """Poll until the build sentinel appears (success) or the campaign vanishes (failure)."""
    sen = _sentinel(args)
    if args.dry_run:
        print(f"  (poll {sen} until present; fail if campaign jobs vanish without it)", flush=True)
        return True
    t0 = time.time()
    empties = 0
    while True:
        if os.path.exists(sen):
            return True
        if _jobs_running(args):
            empties = 0
        else:
            empties += 1
            if empties >= 3:                                    # no jobs for 3 polls, no sentinel
                return False
        if time.time() - t0 > args.max_wait_hours * 3600:
            _log(args, f"TIMEOUT waiting for DTM {dtm} after {args.max_wait_hours} h")
            return False
        time.sleep(args.poll_seconds)


def run_gate(args, dtm, prev_dtm):
    cmd = [sys.executable, "-m", "toddlers.hpc.verify_dtm",
           "--interp-dir", os.path.join(args.stab_dir, f"{args.prefix}_interp_tables"),
           "--prefix", args.prefix, "--dtm", str(dtm),
           "--evolution-dir", args.evolution_dir,
           "--cloudy-output", args.cloudy_output]
    if prev_dtm is not None:
        cmd += ["--prev-dtm", str(prev_dtm)]
    if args.dry_run:
        print("  $ " + " ".join(str(c) for c in cmd) + "   (exit 0 == safe to delete)", flush=True)
        return True
    return subprocess.run([str(c) for c in cmd]).returncode == 0


def delete_cloudy(args, dtm):
    dirs = _cloudy_dirs(args, dtm)
    _log(args, f"deleting {len(dirs)} cloudy_output dirs for DTM {dtm}")
    for d in dirs:
        if args.dry_run:
            print(f"  $ rm -rf {d}", flush=True)
        else:
            shutil.rmtree(d, ignore_errors=True)


def final_assembly(args):
    interp = os.path.join(args.stab_dir, f"{args.prefix}_interp_tables")
    dtm_flag = ["--dust-to-metal"] + [str(d) for d in sorted(args.dtms)]
    _log(args, "final assembly: recollapse data + resample + 5D SFR-normalized STAB")
    # recollapse data (run-specific; regenerated here, just before STAB generation)
    _run(args, [sys.executable, "-m", "toddlers.stab.recollapse"])
    for st in ("Dust", "noDust"):
        for res in ("lr", "hr"):
            _run(args, [sys.executable, "-m", "toddlers.stab.sfr_scaled_seds",
                        "--sed-type", st, "--resolution", res] + dtm_flag)
    _run(args, [sys.executable, "-m", "toddlers.stab.sfr_normalized_stab"])
    if args.cloud_family:
        _run(args, [sys.executable, "-m", "toddlers.stab.cloud_family_stab"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtms", type=float, nargs="+", required=True,
                    help="f_dust grid, e.g. 1e-3 0.02 0.1 0.4 1.0 (processed ascending)")
    ap.add_argument("--evolution-dir", required=True)
    ap.add_argument("--work-root", required=True, help="coordinator state + per-DTM work dirs")
    ap.add_argument("--cloudy-output", required=True,
                    help="Cloudy output root this coordinator scans + deletes per DTM. Must equal "
                         "<output-root>/cloudy_output (where the campaign jobs actually write).")
    ap.add_argument("--output-root", default="",
                    help="base dir for cloudy_output/, forwarded to campaign as --output-root "
                         "(exported as TODDLERS_OUTPUT_ROOT). Set it to the run root on scratch.")
    ap.add_argument("--stab-dir", required=True, help="holds <prefix>_interp_tables (.pkls accumulate)")
    ap.add_argument("--cache-dir", required=True, help="interpolant cache (on scratch)")
    ap.add_argument("--prefix", default="BPASS_chab100_bin")
    ap.add_argument("--small-to-large-ratio", type=float, default=0.40)
    ap.add_argument("--cloud-family", action="store_true", help="also build the cloud-family STAB")
    # campaign passthrough
    ap.add_argument("--account", required=True); ap.add_argument("--partition", required=True)
    ap.add_argument("--ntasks", type=int, default=96); ap.add_argument("--max-nodes", type=int, default=8)
    ap.add_argument("--walltime", default="3-00:00:00"); ap.add_argument("--stab-walltime", default="12:00:00")
    ap.add_argument("--toddlers-src", required=True)
    ap.add_argument("--cloudy-exe", required=True); ap.add_argument("--cloudy-data", required=True)
    ap.add_argument("--python-module", default="SciPy-bundle/2024.05-gfbf-2024a")
    ap.add_argument("--activate-env", default="",
                    help="extra shell lines injected into every campaign job after the primary "
                         "--python-module load (e.g. additional 'module load' lines for clusters "
                         "where matplotlib/h5py are separate modules). Use embedded newlines for "
                         "multiple lines.")
    ap.add_argument("--max-resume-rounds", type=int, default=10)
    # coordinator polling
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--max-wait-hours", type=float, default=72.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.dry_run:
        os.makedirs(args.work_root, exist_ok=True)
    done_path = os.path.join(args.work_root, DONE_NAME)
    if os.path.exists(done_path):
        _log(args, "sweep already complete (DONE sentinel present) -> nothing to do")
        return 0

    dtms = sorted(args.dtms)
    _log(args, f"DTM sweep over {dtms} (ascending); prefix {args.prefix}")
    prev = None
    for dtm in dtms:
        st = cell_state(args, dtm)
        _log(args, f"DTM {dtm}: state={st}")
        if st == "DONE":
            prev = dtm
            continue
        if st == "NEEDS_RUN":
            if not args.dry_run and os.path.exists(_sentinel(args)):
                os.remove(_sentinel(args))                       # avoid reading the previous DTM's
            _log(args, f"submitting single-DTM campaign for f_dust={dtm}")
            _run(args, _campaign_cmd(args, dtm))
            if not wait_for_completion(args, dtm):
                _log(args, f"HALT: campaign for DTM {dtm} did not complete -- preserving everything")
                return 2
        if not run_gate(args, dtm, prev):
            _log(args, f"HALT: verify_dtm FAILED for DTM {dtm} -- preserving its cloudy_output")
            return 3
        delete_cloudy(args, dtm)
        prev = dtm
        _log(args, f"DTM {dtm}: verified and cloudy_output deleted")

    final_assembly(args)
    if not args.dry_run:
        Path(done_path).touch()
    _log(args, "DTM sweep COMPLETE -> 5D STAB assembled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
