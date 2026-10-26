# Cloudy artifacts

What TODDLERS ships, downloads and generates for Cloudy, and how each generated file is reused. All md5 sums below are of the files in this repository unless stated otherwise.

## Shipped in the repository

| Artifact | Where | Notes |
|---|---|---|
| Patch `save diffuse continuum unattenuated` | `cloudy_patches/cloudy_unattenuated_diffuse_continuum.patch` (md5 b1ef8776b8ce92cb2fb5d59c340969cd) and the seven patched sources | Optional. The released grids were run with Cloudy master `013dc0bf` plus this patch; the patch was made against master `01c4cfc6` and applies to both. Details and known limitations: `cloudy_patches/README.md`. |
| Line list | `src/toddlers/database/lines_list/cloudy_lines_TODDLERS_v2.dat` (148 lines, md5 57bb536912c30d399f0173de14ac3140) | Built by `get_lines_list.py` (same folder) from Cloudy's `LineList_HII.dat`, `LineList_PDR_H2.dat` and `LineList_HeH.dat`, shipped next to it; a test regenerates the list and its `_with_comments`, `_sorted` and `_removed` variants byte for byte. All 148 labels appear verbatim in the line lists of Cloudy `013dc0bf`. The only accepted `line_list_version` is `'v2'`; the v1 list of the first TODDLERS build is in `docs/legacy/` for comparison only. |
| Ionized-gas temperature table | `src/toddlers/database/Hii_temperature_interpolator.obj` (md5 5d21c759e1601ec8dd734b61228ec43b) | Used only with `Evolution(use_Tion_model=True)`. A pickled linear `RegularGridInterpolator` over (log10 Z, log10 n [cm^-3], log10 phi [photons cm^-2 s^-1]) giving log10 T [K], from single-zone Cloudy models. Z is a mass fraction: 0.001 to 0.04 (5 nodes); n 1e-3 to 1e6 (10); phi 10^4.22 to 10^18.22 (11). Built from 550 single-zone models run with Cloudy master `013dc0bf` (BPASS chab100 binary burst at 1 Myr, GASS abundances with `metals Z/0.014`, cosmic-ray background, `stop zone 1`); T is the temperature of that zone. Rebuilding the grid from those 550 outputs reproduces the shipped table bitwise. The model outputs are not distributed. 74 of the 550 grid values are below 1000 K (low ionizing flux); `utils.get_shell_inner_face_density` floors the temperature it returns at 1000 K, while its density root search uses the table value. |

Abundances and depletions are written into each Cloudy input by `cloudy_abundances_generator.py` (GASS10 solar scale by default); grain commands by `cloudy_grains_generator.py` (below).

## Downloaded by `scripts/download_data.py`

- **TODDLERS Cloudy resources** (Google Drive bundle, unpacked into the Cloudy data directory): the released star tables. After unpacking, each is checked against its released md5 (`RELEASED_TABLE_MD5` in `cloudy_stellar_spectra_generator.py`).
- **Full-resolution BPASS v2.2.1 burst tables** (`--full-bpass`, or `--bpass-imf` with `--bpass-stars`), from data.nublado.org, into `src/others/Cloudy Spectra/` (git-ignored), which is where `SpectralTableGenerator` reads them. The md5 of each `.xz` and of the decompressed table is pinned (`BPASS_SOURCE_MD5`); a mismatch is refused.
- **Single-star track database** (`--stochastic-tracks`), md5 checked.

## Generated, and how reuse is keyed

| Artifact | Written by | Reused when |
|---|---|---|
| Star tables `<name>.ascii` (Cloudy data directory) | `SpectralTableGenerator`, the stochastic pySB99 writer | A sidecar `<name>.ascii.meta.json` records the md5 of the table and a content key (template, IMF, star type, formation timescale, resolution and resampling, maximum age, code version, md5 of the source spectra). The table is reused only if the sidecar md5 matches the file, the key matches the request and the table's own age/Z lines match; a table without a sidecar only if its md5 is a released one. Anything else is moved to `stale_tables/<UTC time>/` (never deleted) and written again. Tables are written to a temporary file and renamed into place. |
| Cloudy star index `<name>.idx` | Cloudy, only when `compile stars` is run on the table; the index goes to the directory Cloudy runs in (stars.cpp `GridCompile` and `lgCompileAtmosphere`, cpu.cpp `p_getPathList`, Cloudy master `01c4cfc6`) | TODDLERS never runs `compile stars`. `table star` reads an index if it finds one on Cloudy's search path with the right version number, and does not check that it was built from the table it sits next to; without one it reads the `.ascii` directly and writes no index (stars.cpp `InitGrid`). TODDLERS therefore moves `<name>.idx` (and any `<name>.mod`) next to a table to `stale_tables/` whenever it moves that table aside or writes a table of that name. An index elsewhere on the search path, for example in a run directory, is not moved. |
| Grain size distribution `ratio<NNN>_exp<n>.szd` | `CloudyGrainsGenerator` | Written only when its content changes (temporary file, then rename). NNN is the small-to-large mass ratio x 1000; ratios must have at most 3 decimals. |
| Grain opacity `<material>_ratio<NNN>_exp<n>_<bins>.opc` | Cloudy `compile grain`, run by `CloudyGrainsGenerator` | A sidecar `<opc>.meta.json` records the key (md5 of the .szd, of the `.rfi` and of the Cloudy executable, n_bins, ratio) and the Cloudy version line of the .opc header. Reused only on a matching sidecar; otherwise compiled again, once, under a lock directory `<opc>.lock` (parallel workers wait), with the old file moved to `stale_tables/`. The lock records its host and pid; a waiter on the same host breaks the lock of a dead process at once, and a lock without an owner record after 60 s. The compile input is `compile_<material>_<szd stem>_<bins>.in` in the Cloudy data directory. A compile that exits non-zero or writes no .opc raises. A ratio below 0.05 uses Cloudy's own `<material>_orion_10.opc`, with a warning and a comment line in the Cloudy input. |
| Unified-model density laws (`.ini`), Cloudy inputs and run directories | the model generators | Run directories are reused when they exist. They do not record the Cloudy build, patch version, grain setup or line list. |

## Patch detection

A stock Cloudy does not reject `save diffuse continuum unattenuated`: it writes the standard diffuse continuum with a `ConEmitLocal` header. The reader and the HPC runner therefore read the header of `.diffContUnatt`: `DiffContUnatt` in its first line means the patched build. Without it, a one-time `[patch-check]` note is printed and the noDust / variable-DTM SEDs use the attenuated diffuse continuum.

## BPASS star tables are reduced-resolution

The BPASS Cloudy star tables TODDLERS ships and builds by default (`_resFac10`) keep every 10th wavelength point of the full-resolution BPASS spectra (`resample='decimate'`, the method of the released tables). `resample='average'` (files `_resAvg10`) instead averages each block of 10 points with cell weights, keeping the integral of the spectrum, and never mixes points across the He II, He I and H I edges. The feedback quantities that drive the dynamics (Q_HI, luminosities, in the HDF5 feedback tables) come from the full-resolution BPASS data. So the ionizing photon rate implied by a decimated star table differs from the feedback Q_HI: by up to 0.013 dex for the released binary chab100 tables and up to 0.092 dex for single-star chab100 (at 9.47 Myr). Cloudy takes the spectral shape from the star table and the normalization from the feedback, so this is a difference in spectral shape, documented rather than corrected.
