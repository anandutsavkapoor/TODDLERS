import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

# Set paths to interpolant files
base_path = Path("resample/sed_pickles/")
low_res_file = base_path / "TODDLERS_totSED_lr_BPASS_chab300_bin.pkl"
high_res_file = base_path / "sed_high_res_dust.pkl"

# Load interpolants
with open(low_res_file, 'rb') as f:
    low_res_interp = pickle.load(f)
# with open(high_res_file, 'rb') as f:
#     high_res_interp = pickle.load(f)

# Print interpolator information to debug
print("\nLow resolution interpolator info:")
print(f"Type: {type(low_res_interp)}")
# print("\nHigh resolution interpolator info:")
# print(f"Type: {type(high_res_interp)}")

if isinstance(low_res_interp, RegularGridInterpolator):
    # Access grid points directly from interpolator
    low_wave = low_res_interp.grid[0]
    time_points = low_res_interp.grid[1]
    Z_points = low_res_interp.grid[2]
    eta_points = low_res_interp.grid[3]
    n_points = low_res_interp.grid[4]
    logM_points = low_res_interp.grid[5]
    
    # high_wave = high_res_interp.grid[0]
    
    print("\nDetailed grid information:")
    print("\nWavelength grid:")
    print(f"Low resolution wavelength points: {low_wave}")
    print(f"Number of points: {len(low_wave)}")
    
    print("\nTime grid:")
    print(f"Time points (Myr): {time_points}")
    print(f"Number of time points: {len(time_points)}")
    
    print("\nMetallicity grid:")
    print(f"log10(Z/Z⊙) points: {Z_points}")
    print(f"Number of Z points: {len(Z_points)}")
    
    print("\nEta grid:")
    print(f"η points: {eta_points}")
    print(f"Number of η points: {len(eta_points)}")
    
    print("\nDensity grid:")
    print(f"log10(n/cm⁻³) points: {n_points}")
    print(f"Number of density points: {len(n_points)}")
    
    print("\nMass grid:")
    print(f"log10(M/M⊙) points: {logM_points}")
    print(f"Number of mass points: {len(logM_points)}")
else:
    # If the structure is different, print its attributes
    print("\nAvailable attributes for low_res_interp:")
    print(dir(low_res_interp))
    raise ValueError("Please check the structure of the interpolator objects")


# Define some test parameters
test_params = [
    {
        'time': 1.0,      # Myr
        'Z': 0.02,        # Z
        'eta': 0.1,       # Dimensionless
        'n': 640.0,       # cm^-3
        'logM': 6.0,      # Solar masses (log)
        'label': 'Early time'
    },
    {
        'time': 4.0,     # Myr
        'Z': 0.02,       # Z
        'eta': 0.1,       # Dimensionless
        'n': 640.0,      # cm^-3
        'logM': 6.0,      # Solar masses (log)
        'label': 'intermdt time'
    },
    {
        'time': 10.0,     # Myr
        'Z': 0.02,       # Z
        'eta': 0.1,       # Dimensionless
        'n': 640.0,      # cm^-3
        'logM': 6.0,      # Solar masses (log)
        'label': 'Later time'
    }
]

# Create figure
plt.figure(figsize=(15, 10))
colors = ['red', 'green', 'blue']
# Plot SEDs for each parameter set
for i, params in enumerate(test_params):
    # Create input points for interpolation
    input_points = np.array([
        [wave,
         params['time'],
         np.log10(params['Z']),
         params['eta'],
         np.log10(params['n']),
         params['logM']]
        for wave in low_wave
    ])
    # input_points_hr = np.array([
    #     [wave,
    #      params['time'],
    #      np.log10(params['Z']),
    #      params['eta'],
    #      np.log10(params['n']),
    #      params['logM']]
    #     for wave in high_wave
    # ])
    
    # Get SEDs
    low_res_sed = low_res_interp(input_points)
    #high_res_sed = high_res_interp(input_points_hr)
    
    # Plot
    plt.plot(10**low_wave, 10**low_res_sed, colors[i], alpha=0.7, label='Low Resolution')
    #plt.plot(10**high_wave, 10**high_res_sed, 'r-', alpha=0.7, label='High Resolution')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel('Wavelength (microns)')
    plt.ylabel('Specific Luminosity (erg/s/micron)')
    
    # Add parameter info as text
    param_text = (f'Time = {params["time"]:.2f} Myr\n'
                 f'Z = {params["Z"]:.2f} \n'
                 f'eta_Sf = {params["eta"]:.2f}\n'
                 f'n = {params["n"]:.0f} cm⁻³\n'
                 f'log(M) = {params["logM"]:.2f}')
    plt.text(0.05 + i * 0.125, 0.25, param_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=10, color=colors[i],
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.xlim(0.1, 3005)
    plt.ylim(1e33, 1e43)

plt.savefig('sed_resolution_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Print some statistics about the wavelength grids
print("\nWavelength grid statistics:")
print(f"Low resolution points: {len(low_wave)}")
# print(f"High resolution points: {len(high_wave)}")
print(f"\nWavelength range (microns):")
print(f"Low res: {10**low_wave.min():.2e} to {10**low_wave.max():.2e}")
# print(f"High res: {10**high_wave.min():.2e} to {10**high_wave.max():.2e}")