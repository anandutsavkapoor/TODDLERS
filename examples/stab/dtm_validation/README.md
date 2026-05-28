# Variable-DTM STAB: sanity checks

This directory holds the validation of the **variable dust-to-metal (DTM) STAB library**,
i.e. the SKIRT `.stab` SED families that carry a dust-to-metal scaling axis `f_dust`
(`f_dust = 1` is the published baseline, D/G / Z = 0.456 at solar; lower `f_dust` scales the
grain abundance per H atom down). The checks confirm that the `f_dust` axis modifies the
SEDs and the emission lines correctly, and that the baseline slice reproduces the published
library.

Three checks are specific to the DTM axis (scripts here); one is the general nebular-physics
check (`../physics_checks.py`). All figures use the repository A&A style
(`../../paper_figures/paper_style.mplstyle`) and are produced for one representative cloud
cell, `Z = 0.02`, `eps_SF = 0.05`, `n_cl = 160 cm^-3` (BPASS chab100 bin, 10 Myr SFH), which
is changeable via `--Z/--sfe/--ncl`.

> The `.stab` binaries used to make these figures are **not** committed (see `.gitignore`).
> Regenerate them with the STAB pipeline (`examples/stab/run_stab_pipeline.sh`) or copy the
> built `ToddlersSFRNormalizedSEDFamily_*_DTM_*.stab` into this directory before plotting.

## The checks

### 1. SED vs f_dust  - `plot_sed_vs_dtm.py` -> `sed_vs_dtm.png`
Overplots `lambda*L_lambda` for every `f_dust` node, Dust and noDust.
- **Dust**: UV/optical continuum suppressed and the FIR dust bump enhanced, **monotonically**
  in `f_dust` (energy absorbed in the UV/opt reappears in the FIR). Measured for this cell:
  UV down x2.4, optical down x2, FIR bump up x8.4 over `f_dust` 0.02 -> 1.
- **noDust** (negative control): the continuum is `f_dust`-invariant (no grains, no dust
  emission/attenuation); only the emission lines move (nebular gas-state response). A flat
  Dust panel, or a varying noDust *continuum*, would signal a degenerate axis.

#### 1b. Dust-free SED, lr vs hr  - `plot_nodust_hr_lr_vs_dtm.py` -> `nodust_hr_lr_vs_dtm.png`
Zooms in on the `includeDust=false` variants at one cell: the dust-free SED for all 7
`f_dust`, lr (R=300 continuum only) and hr (continuum + lines), with a ratio-to-`f_dust=1`
subpanel. The stellar UV/optical continuum is `f_dust`-invariant; the unattenuated nebular
continuum weakens by ~10-15 % from low to high `f_dust` in the NIR/FIR (where it is a larger
fraction of the faint continuum), monotonically; in hr the lines add per-line variation.
Confirms the dust-free SEDs carry the `f_dust`-dependent nebular emission (see Table
`tab:sed_components`: lr = unattenuated nebular continuum, hr = continuum + intrinsic lines).

### 2. Emission lines vs f_dust  - `plot_lines_vs_dtm.py` -> `lines_vs_dtm.png`
Each line's peak flux, normalized to its `f_dust = 1` value, vs `f_dust`, for lines from the
UV to the FIR. Confirms that **all** lines (not only H-alpha) carry the `f_dust`-varying data,
with the right wavelength dependence:
- **emergent (Dust)**: UV/optical lines suppressed, graded by wavelength (Ly-alpha x15.6,
  recombination lines ~x4, tapering into the IR); FIR fine-structure lines [O I]63, [C II]158
  *enhanced* (dust photoelectric heating of the gas).
- **intrinsic (noDust)**: recombination lines mildly down (grains compete for ionizing
  photons); collisional optical and FIR fine-structure lines up (higher electron temperature).

### 3. vs shipped SKIRT reference  - `compare_vs_reference.py` -> `vs_reference.png`
New STAB at `f_dust = 1` against the shipped single-DTM
`ToddlersSEDFamily_SFRNormalized_*_10Myr.stab`, at the 8 overlapping
(Z, SFE, n_cl) nodes. Prints a per-band fractional-difference table and an SED overlay with a
residual subpanel.
- **Dust**: reproduces the published library to within a few percent
  (median ~0.5-7 % per band) - the expected residual from the post-reference n_H
  He-conversion fix (the libraries are **not** byte-identical by design).
- **noDust**: matches in the UV/optical/NIR; in the MIR/FIR the new build *adds* the
  unattenuated nebular continuum (free-free / free-bound) that the reference lacks, lifting it
  off a near-zero baseline. This is additive and energetically negligible (~7 dex below the
  optical peak) with no spurious dust bump, so the large *relative* MIR/FIR differences in the
  table are off a ~zero reference, not a defect.

Requires a local copy of the SKIRT TODDLERS resources; pass
`--ref-dir /path/to/SKIRT9_Resources_TODDLERS/SED_TODDLERS`.

### 4. Nebular physics  - `../physics_checks.py` -> `physics_checks_<prefix>.png`
Textbook checks on the interpolants at the baseline `f_dust = 1`, independent of the DTM axis:
H-alpha declines with cluster age; intrinsic H-alpha sits on the case-B line
`L_Halpha = 1.37e-12 * Q_abs` (absorbed ionizing photon rate `Q*(1-f_esc)`); the BPT diagram
lands in the star-forming locus below Kauffmann+03; the dust IR peak shifts with age. Reads
the population selected via the `TODDLERS_STAB_*` environment, so set
`TODDLERS_STAB_TEMPLATE=BPASS TODDLERS_STAB_IMF=chab100 TODDLERS_STAB_STARTYPE=bin` for the
BPASS chab100 bin library (the default is SB99).

## Regenerate

```bash
cd examples/stab/dtm_validation
python plot_sed_vs_dtm.py           # sed_vs_dtm.png
python plot_lines_vs_dtm.py         # lines_vs_dtm.png
python compare_vs_reference.py --ref-dir <SKIRT_SED_TODDLERS>   # vs_reference.png

# nebular physics (run from examples/stab, needs the interpolant tables):
cd ..
TODDLERS_STAB_TEMPLATE=BPASS TODDLERS_STAB_IMF=chab100 TODDLERS_STAB_STARTYPE=bin \
  python physics_checks.py --interp-dir BPASS_chab100_bin_interp_tables --out physics_plots
```
