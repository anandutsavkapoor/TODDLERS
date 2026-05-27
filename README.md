# TODDLERS

**T**ime evolution of **O**bservables including **D**ust **D**iagnostics and **L**ine **E**mission from **R**egions containing young **S**tars.

TODDLERS couples 1D feedback-driven shell dynamics with [Cloudy](https://www.nublado.org/) photoionization to predict the time-evolving, UV-to-mm observables of star-forming regions (full SEDs, nebular and fine-structure lines, dust continuum). It can be used standalone to model individual regions or as a sub-grid emission model in galaxy-scale radiative-transfer pipelines.

This is **TODDLERS 2.0**, which adds flexible stellar populations (Starburst99 and BPASS, single- and binary-star evolution, arbitrary and stochastically sampled IMFs), non-uniform and dynamic cloud density, a variable covering fraction, and an improved Cloudy dust and diffuse-ionized-gas treatment.

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
- **Cloudy** for the photoionization post-processing (the pure-dynamics part runs without it). Install separately, put `cloudy.exe` on your `PATH`, and run `python scripts/download_data.py` (below) so the TODDLERS stellar-continua tables land in Cloudy's data directory — the photoionization models need them. Verified to build and run with the current release (C25.00); C23+ is expected to work.
- **Cloudy patch (optional).** `cloudy_patches/` adds one `save` command that exports the gas-phase nebular continuum with dust extinction removed. It is only needed if you want that unattenuated nebular continuum folded into the `noDust` SEDs (the variable-DTM library); the standard pipeline and the default cloud-family SED libraries work with stock Cloudy. Apply it with `cloudy_patches/apply_patch.sh /path/to/cloudy` (verified on C22.00, the 2025 master, and C25.00).

### Data
The code ships only small parameter/grid inputs. The large stellar-atmosphere and track libraries, and the BPASS tables, are downloaded on demand; heavy synthesized products (SEDs, interpolants) are built on first use.

```bash
python scripts/download_data.py        # fetch base libraries + BPASS tables
```

The single-star feedback database (`single_star_tracks.h5`, ~556 MB) the **stochastic** IMF sampler needs is distributed separately (it does not require Cloudy). Either download it or build it locally:

```bash
python scripts/download_data.py --stochastic-tracks   # download (Google Drive)
python examples/build_stochastic_database.py          # or build it locally
```

## Quickstart

```python
from toddlers.evolution import Evolution
from toddlers.constants import M_SUN, MYR_TO_SEC

ev = Evolution(Z=0.02, eta_sf=0.05, n_cl=160.0, M_cl_init=1e6 * M_SUN,
               template="SB99_100", profile_type="uniform")
results = ev.run_simulation()
g = results[0]
print(g["time"] / MYR_TO_SEC, g["radius"])   # shell radius vs time
```

See [`examples/`](examples/) for one runnable script per capability (deterministic and constant-SFR runs, custom and stochastic IMFs, BPASS binaries, non-uniform/dynamic density, Cloudy post-processing, and querying the precomputed SED tables).

## Documentation
Built with Sphinx under [`docs/`](docs/).

## Tests

```bash
pip install -e ".[dev]"
pytest                       # fast unit tests by default
pytest -m slow               # long-running tests
pytest -m "cloudy or data"   # tests needing Cloudy or downloaded data
```

## Citation
If you use TODDLERS, please cite Kapoor et al. (2023, MNRAS 526, 3871) and the TODDLERS 2.0 paper (Kapoor et al., submitted). See [`CITATION.cff`](CITATION.cff).

## License
See [`LICENSE`](LICENSE).
