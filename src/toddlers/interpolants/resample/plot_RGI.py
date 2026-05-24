import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import sys
import scipy.interpolate
sys.modules['scipy.interpolate._interpolate'] = scipy.interpolate

# Configuration
MODEL_PARAMS = {
    'stellar_template': 'BPASS',  # Stellar template
    'imf': 'chab300',             # IMF
    'star_type': 'bin',           # Binary stars
    'sed_mode': 'Cloud',          # Cloud mode
    'include_dust': True,         # Include dust
    'resolution': 'High',         # High resolution
    'Z': 0.020,                   # Metallicity
    'SFE': 0.050,                 # Star formation efficiency
    'n_cl': 320.0,                # Cloud number density (cm^-3)
    'M_cl': 5.0e+05,              # Cloud mass (M☉)
    'time': 5.0,                  # Time (Myr)
}

# Construct interpolator path
model_prefix = f"{MODEL_PARAMS['stellar_template']}_{MODEL_PARAMS['imf']}_{MODEL_PARAMS['star_type']}"
_INTERPOLATOR_BASE = f"{model_prefix}_interp_tables"

# Choose the appropriate interpolator file based on dust and resolution options
if MODEL_PARAMS['resolution'] == 'High':
    # High resolution cases - emergent=True means include dust
    filename = f"TODDLERS_tot_hr_{model_prefix}_lines_emergent={'True' if MODEL_PARAMS['include_dust'] else 'False'}.pkl"
else:
    # Low resolution cases
    if MODEL_PARAMS['include_dust']:
        filename = f"TODDLERS_totSED_lr_{model_prefix}.pkl"
    else:
        filename = f"TODDLERS_inciSED_lr_{model_prefix}.pkl"

interpolator_path = Path(_INTERPOLATOR_BASE) / filename
print(f"Looking for interpolator at: {interpolator_path}")

# Load the interpolator
try:
    with open(interpolator_path, 'rb') as f:
        interpolator = pickle.load(f)
    print("Successfully loaded interpolator")
    
    # Get grid information from the interpolator
    log_wl_grid = interpolator.grid[0]
    time_grid = interpolator.grid[1]
    log_Z_grid = interpolator.grid[2]
    eta_grid = interpolator.grid[3]
    log_n_grid = interpolator.grid[4]
    log_M_grid = interpolator.grid[5]
    
    # Print grid ranges
    print("Interpolator grid ranges:")
    print(f"  Wavelength: {10**log_wl_grid.min():.3e} to {10**log_wl_grid.max():.3e}")
    print(f"  Time: {time_grid.min():.1f} to {time_grid.max():.1f} Myr")
    print(f"  Z: {10**log_Z_grid.min():.3e} to {10**log_Z_grid.max():.3e}")
    print(f"  SFE: {eta_grid.min():.3f} to {eta_grid.max():.3f}")
    print(f"  n_cl: {10**log_n_grid.min():.1f} to {10**log_n_grid.max():.1f} cm^-3")
    print(f"  M_cl: {10**log_M_grid.min():.1e} to {10**log_M_grid.max():.1e} M☉")
    
    # Check if requested parameters are within grid bounds
    if (np.log10(MODEL_PARAMS['Z']) < log_Z_grid.min() or np.log10(MODEL_PARAMS['Z']) > log_Z_grid.max()):
        print(f"Warning: Z value {MODEL_PARAMS['Z']} is outside interpolator grid bounds")
    if (MODEL_PARAMS['SFE'] < eta_grid.min() or MODEL_PARAMS['SFE'] > eta_grid.max()):
        print(f"Warning: SFE value {MODEL_PARAMS['SFE']} is outside interpolator grid bounds")
    if (np.log10(MODEL_PARAMS['n_cl']) < log_n_grid.min() or np.log10(MODEL_PARAMS['n_cl']) > log_n_grid.max()):
        print(f"Warning: n_cl value {MODEL_PARAMS['n_cl']} is outside interpolator grid bounds")
    if (np.log10(MODEL_PARAMS['M_cl']) < log_M_grid.min() or np.log10(MODEL_PARAMS['M_cl']) > log_M_grid.max()):
        print(f"Warning: M_cl value {MODEL_PARAMS['M_cl']} is outside interpolator grid bounds")
    if (MODEL_PARAMS['time'] < time_grid.min() or MODEL_PARAMS['time'] > time_grid.max()):
        print(f"Warning: time value {MODEL_PARAMS['time']} is outside interpolator grid bounds")
    
    # Ensure values are within interpolator bounds
    log_Z = np.clip(np.log10(MODEL_PARAMS['Z']), log_Z_grid.min(), log_Z_grid.max())
    SFE = np.clip(MODEL_PARAMS['SFE'], eta_grid.min(), eta_grid.max())
    log_n_cl = np.clip(np.log10(MODEL_PARAMS['n_cl']), log_n_grid.min(), log_n_grid.max())
    log_M_cl = np.clip(np.log10(MODEL_PARAMS['M_cl']), log_M_grid.min(), log_M_grid.max())
    time = np.clip(MODEL_PARAMS['time'], time_grid.min(), time_grid.max())
    
    # Create input array for interpolation
    inputs = np.column_stack([
        log_wl_grid,  # log wavelength
        np.full_like(log_wl_grid, time),  # time
        np.full_like(log_wl_grid, log_Z),  # log Z
        np.full_like(log_wl_grid, SFE),  # SFE
        np.full_like(log_wl_grid, log_n_cl),  # log n_cl
        np.full_like(log_wl_grid, log_M_cl)  # log M_cl
    ])
    
    # Perform interpolation to get the SED
    log_sed = interpolator(inputs)
    
    # Handle NaN values if any
    nan_mask = np.isnan(log_sed)
    if np.any(nan_mask):
        print(f"Warning: {np.sum(nan_mask)} NaN values found in interpolation result")
        log_sed[nan_mask] = -300.0  # Replace with very small value in log space
    
    # Convert from log to linear space
    sed = 10**log_sed
    
    # Convert wavelength from log to linear
    wavelength = 10**log_wl_grid
    
    # If the wavelength is in microns, convert to meters for consistency with SKIRT
    WAVELENGTH_UNIT = 'micron'  # Change to 'm' if interpolator uses meters
    if WAVELENGTH_UNIT.lower() in ('micron', 'μm', 'um'):
        wavelength_m = wavelength * 1e-6
        wavelength_micron = wavelength
    else:
        wavelength_m = wavelength
        wavelength_micron = wavelength * 1e6
    
    # Apply unit conversion if needed
    SED_UNIT = 'erg/s/micron'  # Change to match interpolator's units ('W/m', 'erg/s/micron', or 'erg/s/Angstrom')
    if SED_UNIT == 'erg/s/micron':
        # Convert erg/s/micron to W/m for consistency with SKIRT
        # 1 erg/s/micron = 0.1 W/m (1e-7 W·s/erg * 1e6 micron/m)
        sed_Wm = sed * 0.1
    elif SED_UNIT == 'erg/s/Angstrom':
        # Convert erg/s/Angstrom to W/m
        # 1 erg/s/Angstrom = 1000 W/m (1e-7 W·s/erg * 1e10 Angstrom/m)
        sed_Wm = sed * 1000.0
    else:
        # Already in W/m
        sed_Wm = sed
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot λLλ vs λ (in microns) for smoother visualization
    plt.loglog(wavelength_micron, sed_Wm * wavelength_m, 'b-', linewidth=2)
    
    # Add labels and title
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('λLλ (W)')
    title = f"TODDLERS SED - {model_prefix}\n"
    title += f"Cloud, {'Dust' if MODEL_PARAMS['include_dust'] else 'noDust'}, {MODEL_PARAMS['resolution']}\n"
    title += f"Z={MODEL_PARAMS['Z']:.3f}, SFE={MODEL_PARAMS['SFE']:.3f}, n={MODEL_PARAMS['n_cl']:.1f} cm⁻³, M={MODEL_PARAMS['M_cl']:.1e} M☉, t={MODEL_PARAMS['time']:.1f} Myr"
    plt.title(title)
    
    # Add grid
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Set axis limits for better visualization
    plt.xlim(0.1, 3000)
    
    # Save the plot
    output_file = f"TODDLERS_SED_{model_prefix}_Z{MODEL_PARAMS['Z']:.3f}_SFE{MODEL_PARAMS['SFE']:.3f}_n{MODEL_PARAMS['n_cl']:.1f}_M{MODEL_PARAMS['M_cl']:.1e}_t{MODEL_PARAMS['time']:.1f}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"SED plot saved to {output_file}")
    
    # Save the SED data to a text file for further analysis
    data_file = f"TODDLERS_SED_{model_prefix}_Z{MODEL_PARAMS['Z']:.3f}_SFE{MODEL_PARAMS['SFE']:.3f}_n{MODEL_PARAMS['n_cl']:.1f}_M{MODEL_PARAMS['M_cl']:.1e}_t{MODEL_PARAMS['time']:.1f}.txt"
    np.savetxt(data_file, np.column_stack((wavelength_micron, sed_Wm)), fmt='%.18e', 
               header='Wavelength(micron) SED(W/m)')
    
    print(f"SED data saved to {data_file}")
    
except FileNotFoundError:
    print(f"Error: Interpolator file not found at {interpolator_path}")
    print("Please make sure the interpolator file exists in the correct location.")
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()