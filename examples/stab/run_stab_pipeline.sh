#!/bin/bash
# Full TODDLERS -> SKIRT STAB pipeline, end to end on one machine.
#
# Usage:
#   ./run_stab_pipeline.sh <evolution-output leaf dir>
#
# where the leaf is .../template_<T>/imf_<I>/star_type_<S>/cluster_mode_*/profile_type_*/
# containing the evolution .dat files (with their Cloudy output alongside).
#
# Population/grid are taken from the environment (see README.md); defaults below.
# For a cluster, use `toddlers.hpc.campaign` instead (see ../hpc/README.md).
set -euo pipefail

EVO="${1:?usage: run_stab_pipeline.sh <evolution-output leaf dir>}"

: "${TODDLERS_STAB_TEMPLATE:=SB99}"
: "${TODDLERS_STAB_IMF:=kroupa100}"
: "${TODDLERS_STAB_STARTYPE:=sin}"
export TODDLERS_STAB_TEMPLATE TODDLERS_STAB_IMF TODDLERS_STAB_STARTYPE
# optional: export TODDLERS_STAB_Z / TODDLERS_STAB_SFE / TODDLERS_STAB_NCL to override the grid axes

PREFIX="$(python -c 'from toddlers.stab import config; print(config.MODEL_PREFIX)')"
echo "MODEL_PREFIX = $PREFIX   (evolution: $EVO)"

echo "=== Stage 1: SED interpolant + recollapse data ==="
python -m toddlers.stab.interpolants \
    --evolution-dir "$EVO" \
    --output-dir "${PREFIX}_interp_tables" \
    --dust-to-metal 1.0
mkdir -p hdf5
cp "${PREFIX}_interp_tables/recollapse_data.h5" "hdf5/recollapse_data_${PREFIX}.hdf5"

echo "=== Stage 2a: cloud-family STAB ==="
python -m toddlers.stab.cloud_family_stab        # -> cloud_family_stab_output/*.stab

echo "=== Stage 2b: SFR-scaled SED grids + SFR-normalized STAB ==="
for st in Dust noDust; do
  for res in lr hr; do
    python -m toddlers.stab.sfr_scaled_seds --sed-type "$st" --resolution "$res"
  done
done
python -m toddlers.stab.sfr_normalized_stab      # -> stab_output/*.stab

echo "=== done: cloud_family_stab_output/*.stab and stab_output/*.stab ==="
