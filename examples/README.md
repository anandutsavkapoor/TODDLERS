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

**Data:** scripts marked "feedback data" / "stellar database" need the downloaded/built
libraries (`python scripts/download_data.py`). The Cloudy example additionally needs
`cloudy.exe` on `PATH` and a valid `CLOUDY_DATA_DIR`.

Figures are written next to each script as `<name>.png`.
