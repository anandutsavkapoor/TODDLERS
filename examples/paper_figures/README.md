# Reproducing the TODDLERS 2.0 paper figures

`generate_all_paper_figures.py` regenerates every figure in the TODDLERS 2.0 paper
from the public package. The physics is imported from `toddlers` (the same code you
installed); only the plotting and figure layout live here.

## Usage

```bash
# all figures
python generate_all_paper_figures.py

# selected figures (space-separated keys)
python generate_all_paper_figures.py --fig 3 5 meth

# replot from cached simulation output (skip the simulations)
python generate_all_paper_figures.py --fig 7 --plot-only
```

Figures are written to `figures/`; intermediate simulation/Cloudy output is cached in
sibling directories so `--plot-only` can re-render without recomputing.

## Figure keys

| key | figure | needs |
|---|---|---|
| `meth` | methodology: density profiles + ram-force history | feedback data |
| `3` | profile / star-formation-mode comparison | feedback data |
| `4` | dynamic cloud density | feedback data |
| `5` | IMF comparison | feedback data |
| `6` | custom (stochastic) feedback | stellar database |
| `7` | stochastic population survey | stellar database |
| `8` | BPT / diffuse-ionized-gas comparison | Cloudy + a with-DIG multi-Z grid |
| `grain` | appendix: grain SED comparison | Cloudy (+ `grain_comparison/*.in`) |
| `graindist` | appendix: grain size distribution | feedback data |

## Data and external tools

- "feedback data" / "stellar database": the downloaded/built libraries
  (`python ../../scripts/download_data.py`; the stochastic database is built by
  `../build_stochastic_database.py`).
- Figures `8`, `grain` need **Cloudy** (`cloudy.exe` on `PATH`, a valid
  `CLOUDY_DATA_DIR`). Figure `8` additionally needs a precomputed with-DIG,
  multi-metallicity Cloudy grid (produced by the HPC pipeline, see `../hpc/` and
  `src/toddlers/hpc/README.md`); it is the one figure not reproducible from the
  committed inputs alone.

## Style

All panels use `paper_style.mplstyle` (A&A style: serif, inward ticks on all four
sides, minor ticks), shipped alongside the script.
