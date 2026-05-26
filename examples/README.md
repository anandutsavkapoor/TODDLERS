# TODDLERS examples

Each script demonstrates one capability and runs standalone. Run from anywhere after
`pip install -e .`.

| Script | Shows | Needs |
|---|---|---|
| `01_density_profiles.py` | cloud density profiles (uniform / Bonnor-Ebert / modified-BE) | nothing |
| `02_imf_sampling.py` | drawing a stochastic stellar population from an IMF | nothing |
| `03_run_evolution.py` | a deterministic shell evolution and its history | feedback data |
| `04_cloudy_pipeline.py` | full pipeline: evolution -> Cloudy -> emergent model | feedback data + Cloudy |
| `05_stellar_populations.py` | SB99 vs BPASS (binary) populations | feedback data |
| `06_star_formation_mode.py` | instantaneous burst vs constant SFR | feedback data |
| `07_cloud_structure.py` | uniform vs Bonnor-Ebert cloud structure | feedback data |
| `08_dynamic_density_and_covering.py` | dynamic cloud density + variable covering fraction | feedback data |
| `09_stochastic_pipeline.py` | discrete population -> feedback interpolants | stellar database |
| `10_fast_feedback_queries.py` | fast precomputed queries (MCMC-style repeated evaluation) | feedback data |
| `11_bpt_diagram.py` | local end-to-end BPT track(s): evolution -> full Cloudy time series -> [O III]/Hβ vs [N II]/Hα for one or more runs | feedback data + Cloudy |
| `12_custom_population_pipeline.py` | full chain for a **custom** population: synthesise its feedback interpolants + Cloudy SED table -> evolution -> Cloudy | feedback data (+ Cloudy for `--run-cloudy`) |
| `13_stochastic_population_pipeline.py` | full chain for a **stochastically sampled** population: sample -> interpolants + Cloudy SED table -> evolution -> Cloudy | stellar database (+ Cloudy for `--run-cloudy`) |

**Data:** scripts marked "feedback data" / "stellar database" need the downloaded/built
libraries (`python scripts/download_data.py`). The Cloudy example additionally needs
`cloudy.exe` on `PATH` and a valid `CLOUDY_DATA_DIR`.

Figures are written next to each script as `<name>.png`.

## More

Beyond the numbered scripts, this directory has:

| Path | What |
|---|---|
| `build_stochastic_database.py` | build the single-star feedback database (`single_star_tracks.h5`, ~556 MB) the stochastic sampler needs; `--quick` for a smoke build. (Or download it: `python scripts/download_data.py --stochastic-tracks`.) |
| `paper_figures/` | regenerate every figure in the TODDLERS 2.0 paper (`generate_all_paper_figures.py --fig ...`) — see its README |
| `stab/` | build SKIRT `.stab` SED libraries (cloud-family + SFR-normalized) from evolution+Cloudy output |
| `eep_validation/` | validation/diagnostic scripts for the EEP (phase-aligned) mass interpolation used by the stochastic sampler — see its README |

For running the two expensive stages (evolution grids, Cloudy post-processing) on a SLURM
cluster, see `hpc/` here and `src/toddlers/hpc/README.md`.
