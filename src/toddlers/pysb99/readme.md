# pySB99 in TODDLERS

## Overview
pySB99 synthesises stellar populations from single-star tracks (Geneva) and stellar atmospheres, for any IMF or an explicit list of stars. TODDLERS uses it to build feedback tables and Cloudy stellar-continua tables for populations that Starburst99 and BPASS do not provide. Core-collapse supernovae are included in the mechanical feedback as point events at each star's death time, in a channel separate from the winds.

Stochastically sampled populations (a new draw of stars for every star-forming generation) do not go through these generators: `Evolution(..., template="pySB99", imf=None, stochastic=StochasticSpec(...))` builds them at run time (see `examples/12_stochastic_population_pipeline.py` and `stochastic/`).

## Main files
- `generate_pysb99_interpolants.py`: builds the feedback table of an IMF population (`generate_kroupa_like_interpolants`) or of a custom star list (`generate_custom_population_interpolants`).
- `pysb99_core.py`: the population synthesis engine.
- `../cloudy_stellar_spectra_generator.py`: writes the Cloudy stellar-continua table of a population (`SpectralTableGenerator`).
- `stochastic/`: run-time stochastic populations (IMF specification, draws, the single-star database).

## Quick start

### IMF population
```python
import os
from toddlers._paths import get_data_dir
from toddlers.pysb99.generate_pysb99_interpolants import generate_kroupa_like_interpolants

generate_kroupa_like_interpolants(
    imf_exponents=[1.3, 2.3],
    imf_mass_limits=(0.1, 0.5, 100.0),
    output_dir=get_data_dir(),          # where TODDLERS reads feedback tables
    imf_name='my_kroupa',
    metallicities=['MW'],
)
```

### Custom population
```python
from toddlers._paths import get_data_dir
from toddlers.pysb99.generate_pysb99_interpolants import generate_custom_population_interpolants

generate_custom_population_interpolants(
    custom_star_numbers={50.0: 10, 100.0: 5, 120.0: 1},   # stars per initial mass [Msun]
    output_dir=get_data_dir(),
    imf_name='my_cluster',
    metallicities=['MW'],
)
```
A custom population is normalized to its own stellar mass and stored per 1e6 Msun, like the IMF tables.

### Cloudy stellar-continua table
```python
import numpy as np
from toddlers.cloudy_stellar_spectra_generator import SpectralTableGenerator

time_points = np.logspace(np.log10(1e4), np.log10(50e6), 50)   # yr
SpectralTableGenerator(template='pySB99', imf='my_cluster', star_type='sin'
                       ).generate_spectral_table(time_points=time_points, Z_values=[0.014])
```

### Evolution with the population
```python
from toddlers.evolution import Evolution
from toddlers.constants import M_SUN

ev = Evolution(Z=0.014, eta_sf=0.05, n_cl=160.0, M_cl_init=1e6 * M_SUN,
               template='pySB99', imf='my_cluster', star_type='sin', profile_type='uniform')
ev.run_simulation()
```
`examples/11_custom_population_pipeline.py` runs this chain end to end, including Cloudy.

## Outputs
- **Feedback table:** `pySB99_<name>_feedback.h5` in the data directory (`$TODDLERS_DATA`, default `src/toddlers/database/`). One HDF5 table (`toddlers.feedback_tables`) with L_bol, L_ion, Q_HI, Q_HeI, Q_HeII, L_LyW and the wind and supernova mechanical channels per 1e6 Msun, with units, the IMF record and the generator configuration. Hardness and the mean ionizing photon energy are derived on load:

```python
from toddlers.feedback_tables import load_table
tab = load_table('pySB99', 'my_cluster')
tab.get('mean_ionizing_photon_energy_eV', 3.0, Z=0.014)
```
- **Cloudy table:** `pySB99_<name>_sin_burst.ascii` in `CLOUDY_DATA_DIR`, with a sidecar `<table>.meta.json` recording its content key (a table is reused only when the key matches).

## Metallicities
IZw18 (Z = 0.0004), SMC (0.002), LMC (0.006), MW (0.014), MWC (0.020).

## Tests
`tests/test_pysb99_*.py` (IMF segments, supernovae, metadata keys, ionizing luminosity, cool supergiants).
