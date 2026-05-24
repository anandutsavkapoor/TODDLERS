# TODDLERS Evolution/Observables Overview

## Evolution Core Modules

### Evolution Class (`evolution.py`)
The main class coordinating the evolution of stellar feedback-driven shells.

#### Key Components:
- **Phase Management**: Handles transitions between phase1, fragmentation, phase2, and dissolution
- **Stellar Feedback**: Integrates mechanical and radiative feedback from young stars
- **Shell Properties**: Tracks radius, velocity, mass, and energetics
- **State Management**: Maintains and updates physical state variables
- **Cloud Profile**: Manages dynamic cloud density evolution

#### Main Methods:
```python
def run_simulation(self)
def integrated_evolution(self, t, x)
def get_acceleration(self)
def get_mass_change_rate(self)
def get_forces(self)
```

### Phase Classes
Specialized classes handling different evolutionary phases:

#### Phase1 (`phase1.py`)
Initial pressure-driven expansion phase.
```python
def solve(self)
def check_fragmentation_condition(self)
def check_gravitational_instability(self)
```

#### Fragmentation (`fragmentation.py`)
Handles shell breakup and energy-loss.
```python
def solve(self)
def get_cooling_rate_fragmentation(self)
```

#### Phase2 (`phase2.py`)
Post-fragmentation evolution momentum-driven expansion.
```python
def solve(self)
def check_dissolution(self)
```

#### Dissolution (`dissolution.py`)
Final phase handling shell dissipation.
```python
def solve(self)
```

## Cloud Physics Modules

### CloudDensityProfile (`cloud_density_profiles.py`)
Base class and implementations for different cloud density profiles.

#### Available Profiles:
- UniformDensity
- UniformDensityWithCavity
- BonnorEbertSphere
- ModifiedBonnorEbertProfile
- GaussianProfile

#### Key Methods:
```python
def density(self, r)
def mass_enclosed(self, r)
def binding_energy(self)
def pressure(self, r)
```

### CloudEnergyInjection (`cloud_energy_injection.py`)
Handles energy injection into the uniform cloud and resulting density profile changes.

```python
def inject_energy(self, E_inj)
def get_current_cloud(self)
def verify_binding_energy(self)
```

## Observables: Cloudy-TODDLERS interface Modules

### CloudySimulationManager (`cloudy_simulation_manager.py`)
Coordinates Cloudy calculations.

#### Key Features:
- Manages multiple model types (shell, unified, DIG)
- Handles time sampling (adaptive/uniform)
- Processes Cloudy outputs
- Error handling and logging

```python
def run_full_simulation(self)
def determine_model_to_run(self, t)
def run_simulation(self, t, model)
```

### Model Generators
Classes generating Cloudy input files for different models:

#### BaseCloudyInputGenerator (`cloudy_base_input_generator.py`)
Base class providing common functionality:
```python
def get_primary_source_options(self, t)
def get_grain_options(self)
def get_abundance_options(self)
def get_cr_options(self, t)
def get_output_options(self, grains=True, dissolved=False, save_elements=False, 
                        save_lines=True, save_overview=True, save_heating=True, 
                        save_cooling=True, save_opacities=False, line_list_version='v2')
```
- The output options are self-explanatory
- The lines' lists sit in src/database/lines_list/
- v2 option is compatible with Cloudy-23

#### ShellModelGenerator (`cloudy_shell_model_generator.py`)
Generates shell-only models:
```python
def get_density_options(self, t)
def get_geometry_options(self, t)
def get_stopping_criteria(self)
```

#### UnifiedModelGenerator (`cloudy_unified_model_generator.py`)
Creates combined shell+cloud models:
```python
def get_density_and_geometry_options(self, t)
def create_density_law(self, t)
```
- The density law creates an overall profile using inner shell and the unswept cloud.
- Works with all cloud profiles.

#### DIGModelGenerator (`cloudy_dig_model_generator.py`)
Handles diffuse ionized gas models:
```python
def get_source_options(self, t)
def get_old_stellar_source(self)
def calculate_old_stellar_luminosity(self, t)
```

### CloudyOutputHandler (`cloudy_output_handler.py`)
Processes and analyzes Cloudy output files:
```python
def get_density_structure(self)
def get_radial_structure(self)
def get_temperature_structure(self)
def get_ionization_structure(self)
def calculate_escape_fraction(self)
```

## Utility Modules

### Constants (`constants.py`)
Physical constants and model parameters.

### Logging (`evolution_logging.py`, `cloudy_logging.py`)
Logging for evolution and Cloudy simulations.

### Tracking (`track_simulation.py`)
Handles data collection and storage during evolution simulations.


# Detailed : TODDLERS-Cloudy interface / input file generation

1. **Base Generator** (`BaseCloudyInputGenerator`)
   - Provides common functionality for all model types
   - Handles shared parameters like abundances, grains, and cosmic rays

2. **Model-Specific Generators**
   - `ShellModelGenerator`: Shell-only models
   - `UnifiedModelGenerator`: Combined shell and cloud models
   - `DIGModelGenerator`: Diffuse ionized gas models
   - `DissolvedModelGenerator`: Post-dissolution direct stellar radiation

3. **Simulation Manager** (`CloudySimulationManager`)
   - Handles evolution output through interpolant generation
   - Coordinates model generation and execution
   - Handles time sampling and model selection

## Base Generator Class

### BaseCloudyInputGenerator

The foundation class providing common functionality for all Cloudy models.

#### Initialization
```python
def __init__(self, simulation_params, t_list_collapse, dissolution_time, 
             interpolants, cloudy_run_dir, speedup=False, logger=None):
    """
    Args:
        simulation_params (dict): Basic simulation parameters
        t_list_collapse (list): Times of shell recollapse events
        dissolution_time (float): Time of shell dissolution
        interpolants (tuple): Functions for interpolating physical quantities
        cloudy_run_dir (str): Directory for Cloudy runs
        speedup (bool): Whether to use speedup options
        logger: Logger instance for output
    """
```

#### Key Methods

```python
def get_primary_source_options(self, t, scaling=1):
    """Generate stellar source commands.
    
    Args:
        t (float): Current time in seconds
        scaling (float): Luminosity scaling factor
    
    Returns:
        list: Cloudy source commands
    """

def get_grain_options(self, skip_pah=False):
    """Generate grain-related commands.
    
    Args:
        skip_pah (bool): Whether to skip PAH grains
    
    Returns:
        list: Cloudy grain commands
    """

def get_abundance_options(self):
    """Generate abundance commands.
    
    Returns:
        list: Cloudy abundance commands
    """

def get_cr_options(self, t, scale_with_Z=True):
    """Generate cosmic ray background commands.
    
    Args:
        t (float): Current time
        scale_with_Z (bool): Whether to scale with metallicity
    
    Returns:
        list: Cloudy cosmic ray commands
    """
```

## Model-Specific Generators

### ShellModelGenerator

Generates input for shell models.

```python
def get_density_options(self, t):
    """Generate density commands for the shell.
    
    Uses:
        - Inner shell density from evolution simulation
        - Constant pressure assumption
    """

def get_geometry_options(self, t):
    """Generate geometry commands.
    
    Includes:
        - Spherical geometry
        - Inner radius from evolution
        - Covering factor
    """
```

### UnifiedModelGenerator

Handles combined shell and cloud models with density law tables.

```python
def get_density_and_geometry_options(self, t):
    """Generate density law and geometry for unified model.
    
    Creates:
        - Density law combining shell and cloud
        - Smooth transition between components
        - Position-dependent density table
    """

def _create_density_law(self, t):
    """Create the density law table.
    
    Handles:
        - Shell density profile
        - Cloud density profile
        - Transition region
    """
```

### DIGModelGenerator

Manages DIG models with source combinations.

```python
def get_source_options(self, t, inner_model_prefix):
    """Generate source specifications for DIG.
    
    Components:
        1. Direct stellar radiation
        2. Transmitted radiation from inner model
        3. Old stellar population (optional)
    """

def calculate_old_stellar_luminosity(self, t):
    """Calculate old stellar population contribution.
    
    Uses:
        - Specified ionization parameter
        - DIG density
        - Geometry
    """
```

### DissolvedModelGenerator
Essentially outputs the naked stellar spectra by running a cheap Cloudy simulation.

## Input file creation flow

1. **Model Selection**
```python
models_to_run, status = simulation_manager.determine_model_to_run(t)
```

2. **Input Generation**
```python
for model in models_to_run:
    if model == 'shell':
        model_generator = ShellModelGenerator(base_generator)
    elif model == 'unified':
        model_generator = UnifiedModelGenerator(base_generator)
    elif model == 'dig':
        model_generator = DIGModelGenerator(base_generator)
        
    model_generator.write_input_file(t)
```

3. **File Generation**
```python
def write_input_file(self, t):
    """
    1. Write header comments
    2. Add source specifications
    3. Add density/geometry options
    4. Add abundances
    5. Add grains
    6. Add cosmic rays
    7. Add other physics
    8. Add stopping criteria
    9. Add output options
    """
```
# Dust Grains

### CloudyGrainsGenerator

This class manages the generation of custom grain size distributions and opacity file generation for Cloudy simulations. 
See Hazy for more details of the grains code in Cloudy. CloudyGrainsGenerator also generates input commands for Cloudy.

#### Size Distribution Creation

```python
def create_modified_orion_distribution(self, small_to_large_ratio, n=3):
    """
    Create modified Orion-like grain distribution.
    
    Args:
        small_to_large_ratio (float): Ratio of mass in small (<0.03μm) 
                                     to large grains
        n (int): Power of exponential cutoff (1-3)
    
    Key Parameters:
        - A_0: Lower grain size limit (0.001μm)
        - A_L: Cutoff for exponential function (0.03μm)
        - α: Power law index (-3.5)
    """
```
 **Modified Orion Distribution**
   - Modified from Cloudy's default (0.030 - 0.25μm) MRN power-law with no cutoff
   - Adjusted small-to-large (SL) grain ratio by adding small grains through exponential cutoff
   - Valid SL ratios: 0.01 (Orion-like) to 0.40 (ISM-like)
   - We use SL ratio of 0.1 in SF regions.
   - Some plots available in tests/test_cloudy_grains
   - For exponential cutoff, we use n=3
   - No modification to PAH size distribution
