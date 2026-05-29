#!/bin/bash
# ---------------------------------------------------------------------------
# TODDLERS Cloudy stage -- universal SLURM worker-pool submission (per phase).
#
# Cloudy phases run in a LINEAR chain: `unified` reads the `shell` model's output
# (density structure), and `dig` consumes the transmitted continuum of its inner
# model. So submit one phase per job, each depending on the previous via afterok:
#
#   sh=$(sbatch --parsable --export=ALL,PHASE=shell                           submit_cloudy.sh)
#   un=$(sbatch --parsable --dependency=afterok:$sh --export=ALL,PHASE=unified submit_cloudy.sh)
#   sbatch          --dependency=afterok:$un --export=ALL,PHASE=dig            submit_cloudy.sh
#
# Generate the per-phase task files first:
#   python -m toddlers.hpc.generate_tasks cloudy --input-dir evolution_output \
#       --pattern '*.dat' -o tasks --add-dig
# ---------------------------------------------------------------------------
#SBATCH --job-name=tdl_cloudy
#SBATCH --account=@ACCOUNT@
#SBATCH --partition=@PARTITION@
#SBATCH --nodes=1
#SBATCH --ntasks=@NTASKS@
#SBATCH --time=@WALLTIME@
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true

module load @PYTHON_MODULE@
@ACTIVATE_ENV@
export PYTHONPATH=@TODDLERS_SRC@:${PYTHONPATH:-}
export TODDLERS_DATA=@TODDLERS_DATA@

# Cloudy: keep intermediates in node RAM, and pass the cluster's cloudy.exe.
export TMPDIR=/dev/shm
export CLOUDY_EXE=@CLOUDY_EXE@        # e.g. /data/gent/436/vsc43602/cloudy/source/cloudy.exe
# Small-to-large grain mass ratio for the Cloudy grain distribution. Empty -> package default
# (0.10, Orion-like, the v2 value); the campaign sets 0.40 (ISM-like) for the v2-DTM grid.
export TODDLERS_SMALL_TO_LARGE_RATIO=@SMALL_TO_LARGE_RATIO@

PHASE=${PHASE:?set PHASE=shell|unified|dig|dissolved via --export}
# TASKFILE defaults to the full phase task list, but can be overridden (e.g. with a
# `*.resume.tasks` file) via --export=ALL,TASKFILE=... to re-run only the unfinished tasks.
TASKFILE=${TASKFILE:-@TASKDIR@/cloudy_${PHASE}.tasks}
RESULTS=@RESULTSDIR@/cloudy_${PHASE}_${SLURM_JOB_ID}.results
mkdir -p "$(dirname "$RESULTS")"

if [[ ! -s "$TASKFILE" ]]; then
    echo "No task file $TASKFILE (phase '$PHASE' has no work). Nothing to do."
    exit 0
fi

echo "Cloudy phase=$PHASE: launching ${SLURM_NTASKS} workers on $(hostname) for $TASKFILE"

# One worker per task via srun, each bound to its own core (slot from
# $SLURM_PROCID/$SLURM_NTASKS). A bash `... &` loop is NOT used: its children
# inherit the batch step's single-core affinity on some clusters and time-share
# one core. Cloudy intermediates go to $TMPDIR=/dev/shm (set above).
srun --cpu-bind=cores python -m toddlers.hpc.worker_loop \
    --task-file "$TASKFILE" \
    --results-file "$RESULTS" \
    --cloudy-exe "$CLOUDY_EXE"

echo "Phase $PHASE finished. Check status with:"
echo "  python -m toddlers.hpc.check_status --task-file $TASKFILE --results '$RESULTS.*'"
