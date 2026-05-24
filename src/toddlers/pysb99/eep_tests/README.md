# EEP interpolation — validation tests

Validation/diagnostic scripts for the phase-aligned (Equivalent Evolutionary
Point) mass interpolation implemented in `pysb99/eep_interpolation.py`.

Run any script from this directory, e.g. `python3 eep_shape_validation.py MW`.
Most accept an optional metallicity list (default `MW LMC SMC MWC`). Figures
(PDF) are written next to the scripts.

## Method (following MIST, Dotter 2016)
1. Detect comprehensive primary markers on each track (`detect_eeps_comprehensive`):
   proxies for the MIST primary EEPs (ZAMS, MS turn-off, RSG tip, WN/WC onset, blue
   return, END), taken from the Teff/L/surface-abundance histories.
2. For a pair of bracketing grid masses, the shared markers (longest common
   subsequence) are the phase-aligned segment boundaries.
3. Within each segment, secondary EEPs are placed at equal increments of the MIST
   HRD metric (cumulative path length in logTeff, logL, each normalised to span
   [0, 1] over the track).
4. Feedback is interpolated in log at fixed EEP, with log-mass as the independent
   variable.

## Scripts
| script | what it checks |
|---|---|
| `eep_phase0_audit.py` | clean track loader: HRD, surface-H evolution, lifetimes |
| `eep_anchor_comparison.py` | which structural variable best aligns each feedback channel (justifies the logTeff+logL metric; rejects surface-H) |
| `eep_config_validation.py [Z...]` | leave-one-out, time-integrated budget error per channel for each within-segment sampler |
| `eep_shape_validation.py [Z...]` | leave-one-out, feedback *history* (shape) error vs the integral — shows the sampler choice is fidelity-neutral, hence MIST on principle |
| `eep_index_alignment.py [Z...]` | feedback plotted vs age (smeared) vs vs EEP index (aligned), masses on the radiation skeleton — visual confirmation that phases align on the index |
| `eep_curvature_anchor_test.py [Z...]` | geometry-only (RDP high-change) anchors vs the semantic markers — confirms the Teff-threshold markers earn their keep, geometry alone is no better |
| `eep_reconstruction_plots.py [Z...]` | figures: MIST EEP placement on the HRD; curated 3-mass/2-channel reconstruction; full-coverage `eep_recon_grid_<chan>_<Z>.pdf` (every interior mass) vs truth vs naive fixed-age |
| `mass_snapping_feedback_error.py` | population-level impact of nearest-track snapping vs interpolation |

## Key results
- The within-segment sampling coordinate (time / Teff / logL / MIST metric) does
  not change the reconstructed feedback *history*: shape error is flat across
  samplers at every metallicity (Q(H) ~0.05 dex, winds ~0.02-0.05 dex). The
  integral differences are a few percent and not robust across Z/channel.
- Because fidelity does not select a winner, the within-segment coordinate is
  chosen on principle: the structural MIST HRD metric (an equivalent-evolutionary-
  point coordinate), not clock age.
- Naive fixed-age mass interpolation collapses prematurely at the late-time
  transition (it blends a still-luminous star with a dead one); EEP interpolation
  tracks the truth — see `eep_reconstruction_*.pdf`.
- Default production behaviour remains nearest-track snapping (consistent with
  Starburst99's internal scheme); EEP is the validated optional interpolation mode.
