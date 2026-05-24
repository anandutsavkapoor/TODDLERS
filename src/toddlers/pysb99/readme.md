# pySB99 Integration with TODDLERS

## Overview
This directory contains pySB99 stellar population synthesis tools integrated with TODDLERS for flexible IMF and custom stellar population modeling.

## Main Files

### generate_pysb99_interpolants.py
Generates stellar feedback interpolants for TODDLERS. Provides helper functions:
- `generate_kroupa_like_interpolants()` - IMF-based populations
- `generate_custom_population_interpolants()` - Custom stellar mixes
- Handles all 5 standard metallicities (IZw18, SMC, LMC, MW, MWC)
- Saves interpolants + metadata JSON to `src/database/`

### pysb99_core.py
Core stellar population synthesis engine.

### cloudy_stellar_spectra_generator.py
Generates Cloudy ASCII spectral tables from pySB99 interpolants.

## Quick Start

### IMF-based Population
```python
from generate_pysb99_interpolants import generate_kroupa_like_interpolants

generate_kroupa_like_interpolants(
    imf_exponents=[1.3, 2.3],
    imf_mass_limits=(0.1, 0.5, 100.0),
    output_dir='src/database',
    imf_name='kroupa100',
    metallicities=['LMC', 'MW', 'MWC']
)
```

### Custom Population
```python
from generate_pysb99_interpolants import generate_custom_population_interpolants

generate_custom_population_interpolants(
    custom_star_numbers={120.0: 1},  # Single 120 Msun star
    output_dir='src/database',
    imf_name='single_120msun',
    metallicities=['LMC', 'MW', 'MWC']
)
```

### Generate Cloudy Tables
```python
from src.cloudy_stellar_spectra_generator import SpectralTableGenerator
import numpy as np

time_points = np.logspace(np.log10(1e4), np.log10(50e6), 50)
generator = SpectralTableGenerator(
    template='pySB99',
    imf='kroupa100',
    star_type='sin'
)
generator.generate_spectral_table(
    time_points=time_points,
    Z_values=[0.006, 0.014, 0.020]
)
```

## Custom Populations

Specify stellar mix instead of IMF:
```python
custom_star_numbers = {
    50.0: 10,   # 10 stars at 50 Msun
    100.0: 5,   # 5 stars at 100 Msun
    120.0: 1    # 1 star at 120 Msun
}
```

**Normalization:** Custom populations are correctly normalized to actual stellar mass, then scaled to "per 1e6 Msun" for interpolants.

## Output Locations

**Interpolants:** `src/database/`
- `pySB99interpolation_<name>.obj`
- `pySB99interpolation_<name>_LumLymanWerner.obj`
- `pySB99interpolation_<name>_mean_ionizing_photon_energy.obj`
- `pySB99interpolation_<name>_hardness.obj`
- `pySB99interpolation_<name>_metadata.json`

**Cloudy Tables:** `CLOUDY_DATA_DIR` (from constants.py)
- `pySB99_<name>_sin_burst.ascii`

## Metallicities

Standard: IZw18 (0.0004), SMC (0.002), LMC (0.006), MW (0.014), MWC (0.020)

## Usage in TODDLERS

Evolution simulations use these via `stellar_feedback.py`:
```python
from src.evolution import Evolution

evolution = Evolution(
    template='pySB99',
    imf='kroupa100',  # or 'single_120msun'
    star_type='sin',
    Z=0.014,
    M_cl_init=1e6 * M_SUN,
    ...
)
```

See `main/execute_pysb99_examples.py` for complete workflow examples.

## Testing

Run `tests/test_pysb99_stellarFeedback.py` to test interpolant generation and stellar feedback integration.