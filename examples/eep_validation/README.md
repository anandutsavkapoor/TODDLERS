# EEP interpolation — validation

Validation/diagnostic scripts for the phase-aligned (Equivalent Evolutionary
Point) mass interpolation implemented in `toddlers/pysb99/eep_interpolation.py`,
the optional interpolation mode used by the stochastic sampler.

Run any script from this directory, e.g. `python3 eep_config_validation.py MW`.
Each accepts an optional metallicity list (default `MW LMC SMC MWC`). Figures
(PDF) are written next to the scripts. Everything here runs from the committed
pySB99 tracks; no extra data download is needed.

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

## The production configuration

`eep_interpolation.py` uses the **MIST HRD-metric sampler** (`sampler_mist`) with the
weak `MS_MID` marker **dropped** (`use_ms_mid=False`). The scripts here justify and
illustrate that choice:

| script | what it shows |
|---|---|
| `eep_config_validation.py [Z...]` | **the decision.** Leave-one-out reconstruction of each held-out grid mass, sweeping the two knobs (within-segment sampler `time`/`teff`/`phys`/`mist` × `MS_MID` on/off) and reporting the median integrated-budget error per channel. All configurations land within ~1%, so the coordinate is chosen on principle (the structural MIST metric) and `MS_MID` is dropped as immaterial. |
| `eep_index_alignment.py [Z...]` | feedback plotted vs age (smeared across masses) vs vs EEP index (aligned): the same phase transitions snap onto the same index across masses — the mechanism the interpolation relies on. (For a single shared index axis the figure groups on the reduced common skeleton `ZAMS-MS_TO-T_DOWN-TEFF_MIN-END` that every track has; the real pairwise interpolation uses the per-pair shared markers, which can additionally include the WR markers `WN_ON`/`WC_ON`. Same detector and MIST sampler in both.) |
| `eep_reconstruction_plots.py [Z...]` | the payoff. MIST EEP placement on the HRD (`eep_hrd_placement_<Z>.pdf`); a curated 3-mass/2-channel reconstruction and a full-coverage grid over every interior mass (`eep_recon_grid_<chan>_<Z>.pdf`), each vs the true track and vs naive fixed-age interpolation. |

## Key results
- The within-segment sampling coordinate (time / Teff / logL / MIST metric) does not
  change the reconstructed feedback budget: the leave-one-out error is flat across
  samplers at every metallicity, and the `MS_MID` marker has no measurable effect.
- Because fidelity does not pick a winner, the coordinate is chosen on principle: the
  structural MIST HRD metric (an equivalent-evolutionary-point coordinate), not clock age.
- Naive fixed-age mass interpolation collapses prematurely at the late-time transition
  (it blends a still-luminous star with a dead one); EEP interpolation tracks the truth
  (see `eep_reconstruction_*.pdf`).
- Default production behaviour remains nearest-track snapping (consistent with
  Starburst99's internal scheme); EEP is the validated optional interpolation mode.
