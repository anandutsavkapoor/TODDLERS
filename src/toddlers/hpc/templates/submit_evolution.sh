#!/bin/bash
# ---------------------------------------------------------------------------
# TODDLERS evolution stage -- universal SLURM worker-pool submission.
#
# Fill in the @PLACEHOLDERS@ for your cluster, then:
#     sbatch submit_evolution.sh
#
# This launches one persistent worker per allocated core; each worker runs its
# modular slice of tasks/evolution.tasks. Generate that file first with:
#     python -m toddlers.hpc.generate_tasks evolution --grid grid.json -o tasks
# ---------------------------------------------------------------------------
#SBATCH --job-name=tdl_evo
#SBATCH --account=@ACCOUNT@            # e.g. starting_2026_035 (Tier-1) / gvo00053 (Tier-2)
#SBATCH --partition=@PARTITION@       # e.g. cpu_milan_rhel9 (Tier-1)
#SBATCH --nodes=1
#SBATCH --ntasks=@NTASKS@             # cores per node; portable form (NOT --ntasks-per-node)
#SBATCH --time=@WALLTIME@             # HH:MM:SS
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# NOTE: on Tier-2 (Gent) add  #SBATCH --cluster=<name>  and submit with
#       `module swap cluster/<name> && sbatch ...`. Tier-1 is single-cluster: no --cluster.

set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true

# --- cluster environment ---------------------------------------------------
module load @PYTHON_MODULE@           # e.g. SciPy-bundle/2024.05-gfbf-2024a
# Activate the toddlers env however your site provides it (conda OR venv OR module):
#   source /path/to/conda/etc/profile.d/conda.sh && conda activate toddlers
@ACTIVATE_ENV@

# Prepend, never overwrite: keeps the module's site-packages on the path.
export PYTHONPATH=@TODDLERS_SRC@:${PYTHONPATH:-}

# Large data lives outside the package; point the resolver at it.
export TODDLERS_DATA=@TODDLERS_DATA@

# --- run -------------------------------------------------------------------
TASKFILE=@TASKFILE@                   # e.g. $PWD/tasks/evolution.tasks
RESULTS=@RESULTSDIR@/${SLURM_JOB_ID}.results
mkdir -p "$(dirname "$RESULTS")"

echo "Launching ${SLURM_NTASKS} workers on $(hostname) for $TASKFILE"

# Launch one worker per task via srun: each worker is bound to its own core and
# takes its slot from $SLURM_PROCID / $SLURM_NTASKS. Do NOT use a bash `... &`
# loop -- background children inherit the batch step's single-core affinity on
# some clusters and end up time-sharing one core (catastrophically slow).
srun --cpu-bind=cores python -m toddlers.hpc.worker_loop \
    --task-file "$TASKFILE" \
    --results-file "$RESULTS"

echo "All workers finished. Check status with:"
echo "  python -m toddlers.hpc.check_status --task-file $TASKFILE --results '$RESULTS.*'"
