# TODDLERS

**T**ime evolution of **O**bservables including **D**ust **D**iagnostics and **L**ine **E**mission from **R**egions containing young **S**tars.

TODDLERS couples 1D feedback-driven shell dynamics with [Cloudy](https://www.nublado.org/) photoionization to predict the time-evolving, UV-to-mm observables of star-forming regions (full SEDs, nebular and fine-structure lines, dust continuum). It can be used standalone to model individual regions or as a sub-grid emission model in galaxy-scale radiative-transfer pipelines.

This is **TODDLERS 2.0**, which adds flexible stellar populations (Starburst99 and BPASS, single- and binary-star evolution, arbitrary and stochastically sampled IMFs), non-uniform and dynamic cloud density, a variable covering fraction, and an improved Cloudy dust and diffuse-ionized-gas treatment.

## Installation

```bash
git clone https://github.com/anandutsavkapoor/toddlers-public.git
cd toddlers-public
pip install -e .                 # core install
pip install -e ".[dev]"          # optional: also installs the test/lint tools
                                 # (pytest, pytest-cov, flake8, black)
```

`import toddlers` then works from anywhere. The `[dev]` extra is only needed if you
plan to run the test suite or linters; plain `pip install -e .` is enough to use the
package.

### External requirements
- **Cloudy** (C23+) for the photoionization post-processing. Install separately and put `cloudy.exe` on your `PATH`. The pure-dynamics part runs without Cloudy.

### Data
The code ships only small parameter/grid inputs. The large stellar-atmosphere and track libraries, and the BPASS tables, are downloaded on demand; heavy synthesized products (the single-star feedback database, SEDs, interpolants) are built on first use.

```bash
python scripts/download_data.py        # fetch base libraries + BPASS tables
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
