# Building SKIRT `.stab` SED libraries from TODDLERS

This is the worked, runnable example for turning TODDLERS evolution + Cloudy output into
the `.stab` SED libraries that [SKIRT](https://skirt.ugent.be)'s `ToddlersSEDFamily` reads.

The pipeline itself now lives in the installed package, `toddlers.stab` — you run its
stages as `python -m toddlers.stab.<stage>`. This directory holds only the **example
driver, the validation walkthrough, and a couple of inspection scripts**; there is no
library code to edit here.

- `run_stab_pipeline.sh` — one script that runs the full pipeline end-to-end on an
  evolution-output leaf (the quickest way to see the whole thing work).
- `VALIDATION.md` — the byte-identical check against the shipped `TODDLERS_Cloud_*` SKIRT
  reference (the acceptance test).
- `inspect_lines.py`, `inspect_sed_interp.py`, `plot_RGI.py` — diagnostics for eyeballing
  line lists, interpolant SEDs, and the grid interpolator.

## Configuration (no files to edit)

The population and grid are read from the environment by `toddlers.stab.config`
(defaults: `SB99` / `kroupa100` / `sin`). Set them once per run:

```bash
export TODDLERS_STAB_TEMPLATE=BPASS   # or SB99
export TODDLERS_STAB_IMF=chab100      # or kroupa100
export TODDLERS_STAB_STARTYPE=bin     # or sin
# optional grid-axis overrides (else the template defaults are used):
export TODDLERS_STAB_Z=0.008,0.02
export TODDLERS_STAB_SFE=0.025,0.05
export TODDLERS_STAB_NCL=80.0,160.0
```

`MODEL_PREFIX = <template>_<imf>_<startype>` (e.g. `BPASS_chab100_bin`) and the SED/
wavelength units follow from the template (BPASS and SB99 both: `micron`, `erg/s/micron`,
converted to SKIRT's `m` / `W/m` on write).

## The pipeline (run in this order)

**Stage 1 — SED interpolant + recollapse data**
```bash
python -m toddlers.stab.interpolants \
    --evolution-dir <evolution_output>/template_<T>/imf_<I>/star_type_<S>/cluster_mode_burst/profile_type_uniform \
    --output-dir    ${MODEL_PREFIX}_interp_tables \
    --dust-to-metal 1.0
mkdir -p hdf5 && cp ${MODEL_PREFIX}_interp_tables/recollapse_data.h5 hdf5/recollapse_data_${MODEL_PREFIX}.hdf5
```
Reads the evolution `.dat` and their Cloudy `.cont`/`.phy` output, builds the 6D
`RegularGridInterpolator` (`…_interp_tables/*.pkl`) and the recollapse HDF5. (The copy step
renames the recollapse file to the path the SFR stage expects.)

**Stage 2a — cloud-family STAB** (one 6D library per cloud)
```bash
python -m toddlers.stab.cloud_family_stab     # -> cloud_family_stab_output/*.stab
```

**Stage 2b — SFR-normalized STAB** (SFR-weighted recollapse family, 4D / 5D-with-DTM)
First resample the interpolant into SFR-scaled SED grids (one folder per SED type x
resolution); for a DTM sweep pass all DTM values so the SEDs carry `_dtm` suffixes:
```bash
for st in Dust noDust; do for res in lr hr; do
    python -m toddlers.stab.sfr_scaled_seds --sed-type $st --resolution $res
done; done
python -m toddlers.stab.sfr_normalized_stab   # -> stab_output/*.stab
```

All of the above is wrapped in `run_stab_pipeline.sh` — pass it the evolution-output leaf.

## On a cluster

`toddlers.hpc.campaign` runs this entire chain (Cloudy + all STAB stages) as a SLURM
worker-pool job, deriving the population/grid env vars from the evolution path
automatically. See `../hpc/README.md`.

## Inspecting / consuming STABs

```python
from toddlers.stab.stab_io import read_stab_file     # parse a .stab StoredTable
from toddlers.stab import compare                     # cell-by-cell STAB comparison
from toddlers.stab import rename, resample            # SKIRT-family rename; apply to particles
```
`python -m toddlers.stab.compare OURS_DIR REF_DIR --prefix <prefix>` runs the cloud-family
comparison used in `VALIDATION.md`.

## Dependencies

The STAB code is `toddlers.stab`; it writes STABs via the vendored `toddlers.pts`
(SKIRT toolkit, AGPL-3.0; see `src/toddlers/pts/LICENSE.txt`).
