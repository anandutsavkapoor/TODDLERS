# Validating the STAB pipeline against the shipped reference

This documents an end-to-end check that the public pipeline reproduces the official SKIRT
`TODDLERS_Cloud_*` cloud-family STABs. The strongest form uses the **verified production
evolution runs** as input: same input + correct pipeline ⇒ the overlapping STAB cells are
byte-for-byte identical to the reference.

The worked example below uses **BPASS chab100 (binary)**, a 2×2×2×2 = 16-run subset, and
compares against `TODDLERS_Cloud_BPASS_chab100`. (Any template works; BPASS is the cleanest
because the shipped library and the public pipeline share the same code era and line list.)

## Why a subset is enough

A cloud-family STAB stores `interpolator(node)` at each grid node, and a
`RegularGridInterpolator` returns the stored node value exactly. So the STAB cell at a node
is the original Cloudy SED for that cloud, not an interpolation. A small subset grid
therefore yields the *same* cell values the full library has at those nodes — we just
compare where the grids overlap. (Two points per axis is the minimum the interpolant
accepts.)

## Prerequisites

1. **Verified evolution runs.** The `.dat` files that produced the shipped library (not a
   fresh re-run — fresh runs differ at the ~percent level). For the worked example these are
   the BPASS chab100 production runs; stage the 16-run subset into an evolution leaf:
   ```
   evolution_output/template_BPASS/imf_chab100/star_type_bin/cluster_mode_burst/profile_type_uniform/
       sim_Z{0.008,0.02}_eta{0.025,0.05}_n{80,160}_logM{5.00,6.00}.dat   (16 files)
   ```
2. **Population set via environment** (read by `toddlers.stab.config`):
   ```bash
   export TODDLERS_STAB_TEMPLATE=BPASS TODDLERS_STAB_IMF=chab100 TODDLERS_STAB_STARTYPE=bin
   ```
   This selects the BPASS branch (wavelength `micron`, SED `erg/s/micron`) and
   `MODEL_PREFIX = BPASS_chab100_bin`. (The cluster campaign sets these automatically from
   the evolution path.)
3. **BPASS Cloudy spectral table** on the run host's Cloudy data dir
   (e.g. `BPASS_chab100_bin_burst_resFac10.ascii`). For byte-identity it must be the same
   table (resolution) the reference used.

## Step 1 — produce the cloud-family STAB (run on the cluster)

From-existing campaign, cloud-family only (no DTM sweep, no SFR-norm needed for this check):

```bash
python -m toddlers.hpc.campaign \
    --evolution-dir evolution_output/template_BPASS/imf_chab100/star_type_bin/cluster_mode_burst/profile_type_uniform \
    --work-dir runs --stab-dir examples/stab \
    --stab cloud \
    --account <your_account> --partition <partition> --ntasks 128 \
    --python-module <SciPy-bundle/...> --toddlers-src $PWD/src \
    --cloudy-exe /path/to/cloudy.exe --cloudy-data /path/to/cloudy/data \
    --dry-run        # drop --dry-run to submit
```

This runs Cloudy (shell → unified → dissolved, with lines) → interpolant → cloud-family
STAB, writing `examples/stab/cloud_family_stab_output/ToddlersCloudSEDFamily_BPASS_chab100_bin_{,noDust_}{lr,hr}.stab`.

## Step 2 — get the reference (35 GB, e.g. on a separate analysis host)

The cloud-family reference is a SKIRT resource pack. Either run the SKIRT downloader and
answer `y` for `TODDLERS_Cloud_BPASS_chab100`:

```bash
cd /path/to/SKIRT/git && ./downloadResources.sh
```

or fetch the one pack directly:

```bash
URL=https://sciences.ugent.be/skirtextdat/SKIRT9/Resources/
wget --no-check-certificate ${URL}SKIRT9_Resources_TODDLERS_Cloud_BPASS_chab100_v2.zip
unzip SKIRT9_Resources_TODDLERS_Cloud_BPASS_chab100_v2.zip
```

## Step 3 — compare

With the `toddlers` package importable, run:

```bash
python -m toddlers.stab.compare \
    cloud_family_stab_output \
    /path/to/SKIRT9_Resources_TODDLERS_Cloud_BPASS_chab100/SED_TODDLERS \
    --prefix BPASS_chab100_bin
```

It iterates the four variants (Dust/noDust × lr/hr), matches the overlapping
(Z, SFE, n_cl, M_cl) nodes and the time grid, and reports either **BYTE-IDENTICAL** or the
median/p95/max relative difference. With the verified evolution runs and a matching Cloudy
version, expect byte-identity (or machine precision). A few-percent offset in the Dust
variants only, with noDust matching, points at a Cloudy-version difference in the dust /
nebular reprocessing rather than a pipeline error.
