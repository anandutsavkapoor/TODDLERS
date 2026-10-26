# TODDLERS

**T**ime evolution of **O**bservables including **D**ust **D**iagnostics and **L**ine **E**mission from **R**egions containing young **S**tars.

TODDLERS couples 1D feedback-driven shell dynamics with [Cloudy](https://www.nublado.org/) photoionization to predict the time-evolving, UV-to-mm observables of star-forming regions (full SEDs, nebular and fine-structure lines, dust continuum). It can be used standalone to model individual regions or as a sub-grid emission model in galaxy-scale radiative-transfer pipelines.

This is **TODDLERS 2.1.0**. It models flexible stellar populations (Starburst99, BPASS single and binary stars, and pySB99 populations from any IMF, custom star lists or stochastic draws made at run time, with per-star supernova feedback), non-uniform and dynamic cloud density, a variable covering fraction, and Cloudy dust and diffuse-ionized-gas treatment. Feedback tables are HDF5 files with units and provenance, every evolution output records the inputs and code version it was made with, and outputs can go to any directory. Changes since 2.0 are listed in the [release notes](https://github.com/anandutsavkapoor/TODDLERS/releases). The scripts behind the figures of the TODDLERS 2.0 paper are in the `v2.0` tag (`examples/paper_figures/`).

## Installation

```bash
git clone https://github.com/anandutsavkapoor/TODDLERS.git
cd TODDLERS
python3 -m venv .venv && source .venv/bin/activate   # recommended; required on PEP 668
                                                     # systems (recent Ubuntu/Debian)
pip install -e .                 # core install
pip install -e ".[dev]"          # optional: also installs the test/lint tools
                                 # (pytest, pytest-cov, flake8, black)
```

`import toddlers` then works from anywhere. The `[dev]` extra is only needed if you
plan to run the test suite or linters; plain `pip install -e .` is enough to use the
package.

Requires Python 3.10+. It runs on current scientific stacks (verified on a clean host
with Python 3.12, NumPy 2.4, SciPy 1.17) as well as older ones; there are no upper
version pins on NumPy/SciPy.

### External requirements
- **Cloudy** for the photoionization post-processing (the pure-dynamics part runs without it). Install separately, put `cloudy.exe` on your `PATH`, and run `python scripts/download_data.py` (below) so the TODDLERS stellar-continua tables land in Cloudy's data directory (the photoionization models need them). Verified to build and run with the current release (C25.00); C23+ is expected to work.
- **Cloudy patch (optional).** `cloudy_patches/` adds one `save` command that exports the gas-phase nebular continuum with dust extinction removed. It is only needed if you want that unattenuated nebular continuum folded into the `noDust` SEDs (the variable-DTM library); the standard pipeline and the default cloud-family SED libraries work with stock Cloudy. Apply it with `cloudy_patches/apply_patch.sh /path/to/cloudy` (verified on C22.00, the 2025 master, and C25.00). The released grids were run with Cloudy master `013dc0bf` plus this patch; see [`cloudy_patches/README.md`](cloudy_patches/README.md) and [`docs/cloudy_artifacts.md`](docs/cloudy_artifacts.md).

### Data
The stellar feedback tables (Starburst99, BPASS, pySB99) ship with the package, so the dynamics run without any download. The data the other stages need are versioned with the code (`src/toddlers/data_resources.json`):

```bash
python scripts/download_data.py --list                # what this version needs, and what is installed
python scripts/download_data.py                       # Cloudy stellar-continua tables (needs Cloudy)
python scripts/download_data.py --full-bpass          # optional: full-resolution BPASS tables
python scripts/download_data.py --stochastic-tracks   # single-star database for stochastic IMFs (~583 MB; no Cloudy needed)
```

The single-star database can also be built locally with `python examples/build_stochastic_database.py`. A database that does not match this version is moved aside to `stale_data/`, never deleted.

## Quickstart

```python
from toddlers.evolution import Evolution
from toddlers.constants import M_SUN, MYR_TO_SEC

ev = Evolution(Z=0.02, eta_sf=0.05, n_cl=160.0, M_cl_init=1e6 * M_SUN,
               template="SB99", imf="kroupa100", star_type="sin", profile_type="uniform")
results = ev.run_simulation()
g = results[0]
print(g["time"] / MYR_TO_SEC, g["radius"])   # shell radius vs time
```

A stochastically sampled population draws the stars of each star-forming generation at run time (needs the single-star database):

```python
from toddlers.pysb99.stochastic.population import IMFSpec, StochasticSpec

ev = Evolution(Z=0.014, eta_sf=0.01, n_cl=2560.0, M_cl_init=1e6 * M_SUN, template="pySB99",
               imf=None, stochastic=StochasticSpec(IMFSpec("kroupa", 0.1, 100.0), base_seed=17),
               profile_type="uniform")
```

### Outputs
Evolution and Cloudy outputs go under one root: `Evolution(..., output_root=...)`, else `$TODDLERS_OUTPUT_ROOT`, else the directory that holds the package (`src/` in a source checkout). The Cloudy stage writes next to the evolution output it reads. Each `.dat` records its inputs, the feedback table and the code version, and is reused only when they match.

See [`examples/`](examples/) for one runnable script per capability (deterministic and constant-SFR runs, custom and stochastic IMFs, BPASS binaries, non-uniform/dynamic density, Cloudy post-processing, and querying the precomputed SED tables).

## Documentation
Built with Sphinx under [`docs/`](docs/) (API reference from the docstrings). What is shipped, downloaded and generated, and how Cloudy files are reused: [`docs/cloudy_artifacts.md`](docs/cloudy_artifacts.md). Running grids on a SLURM cluster: [`src/toddlers/hpc/README.md`](src/toddlers/hpc/README.md).

## Tests

```bash
pip install -e ".[dev]"
pytest                       # fast unit tests by default
pytest -m slow               # long-running tests
pytest -m "cloudy or data"   # tests needing Cloudy or downloaded data
```

## Citation
If you use TODDLERS, please cite Kapoor et al. (2023, MNRAS 526, 3871), Kapoor et al. (2024, A&A 692, A79), and the TODDLERS 2.0 paper (Kapoor et al. 2026, A&A 713, A227, [doi:10.1051/0004-6361/202660433](https://doi.org/10.1051/0004-6361/202660433)). See [`CITATION.cff`](CITATION.cff).

## License
See [`LICENSE`](LICENSE).
