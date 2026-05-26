#!/usr/bin/env bash
#
# The two TODDLERS -> SKIRT STAB campaigns behind the paper, as runnable commands.
# Both use the same driver (`python -m toddlers.hpc.campaign`); they differ only in where
# evolution comes from and whether the Cloudy stage sweeps the dust-to-metal ratio.
#
#   1) Grid-from-scratch (production): generate a BPASS chab100 grid, then evolution -> Cloudy
#      -> STAB in one dependent chain (fiducial DTM, no DTM axis).
#   2) Active variable-DTM sweep: start from the evolution produced by (1) and sweep DTM at
#      the Cloudy stage, scaled across nodes -> the paper's 5D DTM-axis SFR-normalised STAB.
#
# SAFE BY DEFAULT: runs with --dry-run, so it only PRINTS the generate_tasks/sbatch commands
# and submits nothing. Edit the CONFIG block for your cluster, then set DRYRUN="" to submit.
# Run from the repo root on the cluster login node (the driver calls sbatch).
#
# See examples/hpc/README.md for the full explanation.
set -euo pipefail

# ---------------------------------------------------------------------------- CONFIG
ACCOUNT="my_account"                         # SLURM account
PARTITION="cpu_partition"                     # SLURM partition
PYTHON_MODULE="SciPy-bundle/2024.05-gfbf-2024a"   # cluster module providing numpy/scipy/h5py
TODDLERS_SRC="$PWD/src"                        # path to the package src/
CLOUDY_EXE="/path/to/cloudy/source/cloudy.exe"
CLOUDY_DATA="/path/to/cloudy/data"
NTASKS=128                                     # cores per node (worker-pool size per node)
MAX_NODES=8                                    # cap on nodes a phase scales across
DRYRUN="--dry-run"                             # set DRYRUN="" to actually submit
# ---------------------------------------------------------------------------------

DTM_AXIS="0.02 0.10 0.20 0.40 0.60 0.80 1.00"

common_args=(
  --work-dir runs_chab100 --stab-dir examples/stab
  --account "$ACCOUNT" --partition "$PARTITION"
  --ntasks "$NTASKS" --max-nodes "$MAX_NODES"
  --python-module "$PYTHON_MODULE" --toddlers-src "$TODDLERS_SRC"
  --cloudy-exe "$CLOUDY_EXE" --cloudy-data "$CLOUDY_DATA"
)

# Evolution output paths are deterministic, so the two campaigns hand off via this leaf
# with no run registry:
#   (1) --grid RUNS evolution: it derives this leaf from the grid's single-valued path axes
#       (template/imf/star_type/cluster_mode/profile_type) and writes one sim_*.dat per
#       Z x eta x n x M combination into it.
#   (2) --evolution-dir DISCOVERS runs: it globs '*.dat' in this leaf and reads the Z/SFE/n
#       from the filenames (and the population from the path) -- it does not run evolution.
# So (1) computes the dynamics once and (2) sweeps DTM at the Cloudy stage on that output;
# keep this leaf consistent with the grid above.
EVO_LEAF="$TODDLERS_SRC/evolution_output/template_BPASS/imf_chab100/star_type_bin/cluster_mode_burst/profile_type_uniform"

echo "############################################################"
echo "# 1) Grid-from-scratch: BPASS chab100 grid, evolution -> Cloudy -> STAB (fiducial DTM)"
echo "############################################################"
# Small demo grid (examples/hpc/bpass_chab100_grid.json, 16 runs). For the full production
# grid, regenerate the JSON from toddlers.stab.config with make_grid.py (BPASS template).
python -m toddlers.hpc.campaign \
  --grid examples/hpc/bpass_chab100_grid.json \
  "${common_args[@]}" $DRYRUN

echo
echo "############################################################"
echo "# 2) Active campaign: variable-DTM sweep on the SAME evolution -> 5D SFR-norm STAB"
echo "############################################################"
# Starts from the evolution produced by (1); sweeps DTM only at the Cloudy stage (the shell
# dynamics are computed once and reused). --max-nodes spreads the large Cloudy grid for
# wall-clock speed at the same CPU-hours.
python -m toddlers.hpc.campaign \
  --evolution-dir "$EVO_LEAF" \
  --dust-to-metal $DTM_AXIS \
  "${common_args[@]}" $DRYRUN

echo
echo "Done. (Dry-run prints commands only; set DRYRUN=\"\" in CONFIG to submit.)"
