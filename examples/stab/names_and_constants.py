import numpy as np
import os

# Stellar population parameters
STELLAR_TEMPLATE = "SB99"     # Alternative: "BPASS"
IMF_TYPE = "kroupa100"        # Alternative: "chab100" 
STAR_TYPE = "sin"            # Alternative: "bin"

# Create consistent model identifier
MODEL_PREFIX = f"{STELLAR_TEMPLATE}_{IMF_TYPE}_{STAR_TYPE}"
# print(f'recollapse_data_{MODEL_PREFIX}.hdf5') # recollapse_data_BPASS_chab100_bin.hdf5

# Directory paths
HDF5_FILENAME = f'hdf5/recollapse_data_{MODEL_PREFIX}.hdf5'

# Constants
AGE_LIMIT = 10.1             # Myr
AGE_START = 0.1              # Myr
N_TEMPORAL_BINS = 5
M_STAR_MEAN = 10000        # Solar masses
SAMPLE_SIZE = 10**6        # For cloud mass distribution
EXPONENT = -1.8            # Cloud mass function slope
SAMPLE_SIZE_TIME = 1000    # For synthetic sfr data

# Simulation constants
LENGTH_UNIT = 'kpc'
AGE_UNIT = 'Myr'
SPREAD_RADIUS = 0.0       # kpc
SMOOTH_AGE_SCALE = 0.0    # Myr

# Parameter space
if STELLAR_TEMPLATE == 'SB99': # TODDLERS v1
    METALLICITIES = np.array([0.001, 0.004, 0.008, 0.020, 0.040])
    STAR_FORMATION_EFFICIENCIES = np.array([.01, .025, .05, .075, .1, .125, .15])
    CLOUD_DENSITIES = np.around(10**np.arange(1, 3.5, .30102), 0)  # ~[10-2560] cm^-3
    MASS_BIN_CENTERS = np.array([10**np.around(i, 2) for i in np.arange(5.0, 6.8, .25)])
    SED_UNIT = 'W/m'
    WAVELENGTH_UNIT = 'm'
elif STELLAR_TEMPLATE == 'BPASS': # v2
    METALLICITIES = np.array([0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.010, 0.014, 0.020, 0.030, 0.040])
    STAR_FORMATION_EFFICIENCIES = np.array([.01, .025, .05, .075, .1])
    CLOUD_DENSITIES = np.array([40., 80., 160., 320., 640.])
    MASS_BIN_CENTERS = np.array([10**np.around(i, 2) for i in np.arange(5.0, 6.8, .25)])
    SED_UNIT = 'erg/s/micron'
    WAVELENGTH_UNIT = 'micron'
else:
    # No grid axes defined for this template. Failing here (rather than leaving
    # METALLICITIES/SED_UNIT/... undefined for a downstream NameError) makes the
    # cause explicit. To use another template (e.g. 'pySB99'), add a branch above
    # defining its grid axes, units and recollapse-HDF5 version.
    raise ValueError(
        f"names_and_constants: no grid axes defined for STELLAR_TEMPLATE="
        f"{STELLAR_TEMPLATE!r}. Supported: 'SB99', 'BPASS'. Add a branch for it.")

# Recollapse-HDF5 schema version. This is GENERATOR-determined, not template-
# determined: RecollapseDataCollector (the interpolant generator) always writes the
# nested "v2" schema (recollapse_data/simulations/<key>/recollapse_times_myr), which
# TODDLERS_recollapse_handling reads and converts to its fast flat form on the fly.
# (The legacy flat "v1" files are no longer produced, so do not set this to 'v1'.)
RECOLLAPSE_HDF5_VER = 'v2'

################################## SFR normalized templates: Currently using SB99/Kroupa100/sin only
# Input/Output directories
SED_OUTPUT_DIR =  f"{MODEL_PREFIX}" + "_sed_output_Dust_lr"  # Example: "sed_output_Dust_lr", sed_output_noDust_lr, "sed_output_noDust_hr", "sed_output_Dust_hr" 
RECOLLAPSE_SIM_DIR = f"{MODEL_PREFIX}" +  "_recollapse_sims"

# Extract configuration from SED_OUTPUT_DIR
_dir_parts = SED_OUTPUT_DIR.split('_')
IS_NODUST = 'noDust' in _dir_parts
IS_HR = 'hr' in _dir_parts

# Base paths
_INTERPOLATOR_BASE = f"{MODEL_PREFIX}" + "_interp_tables"

# Interpolator file selection based on resolution and dust settings
def _get_interpolator_filename():
    if IS_HR:
        # High resolution cases
        return f"TODDLERS_tot_hr_{MODEL_PREFIX}_lines_emergent={'False' if IS_NODUST else 'True'}.pkl"
    else:
        # Low resolution cases
        return f"TODDLERS_inciSED_lr_{MODEL_PREFIX}.pkl" if IS_NODUST else f"TODDLERS_totSED_lr_{MODEL_PREFIX}.pkl"


SED_INTERPOLATOR_FILE = os.path.join(_INTERPOLATOR_BASE, _get_interpolator_filename())

# Parameter validation ranges
VALID_RANGES = {
    'Z': (0.0001, 0.05),
    'eta': (0.001, 0.2),
    'n_cl': (1.0, 5000.0)
}