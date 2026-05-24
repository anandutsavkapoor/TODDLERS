import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import re
from pathlib import Path
import argparse
import warnings

# Configure warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Set matplotlib style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12 
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

def parse_sed_filename(filename):
    """Extract parameters from SED filename."""
    pattern = r'sed_sfr_scaled_(\w+)_(\w+)_(\w+)_Z_(\d+\.\d+)_eta_(\d+\.\d+)_n_(\d+\.\d+)\.txt'
    match = re.match(pattern, filename)
    
    if match:
        template, imf, star_type, z, eta, n = match.groups()
        return {
            'template': template,
            'imf': imf,
            'star_type': star_type,
            'Z': float(z),
            'eta': float(eta),
            'n': float(n)
        }
    else:
        return None

def load_sed_file(filepath):
    """
    Load wavelength and flux data from SED file, properly handling wavelength units from headers.
    """
    try:
        # First check the header to determine wavelength units
        wavelength_unit = "unknown"
        with open(filepath, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    break
                
                # Look for wavelength unit in header
                wl_unit_match = re.search(r'Wavelength\[([^\]]+)\]', line) or re.search(r'wavelength.*\(([^)]+)\)', line, re.IGNORECASE)
                if wl_unit_match:
                    wavelength_unit = wl_unit_match.group(1).lower().strip()
                    break
        
        # Load the data
        data = np.loadtxt(filepath, comments='#')
        
        # Extract wavelength and flux
        if data.shape[1] == 2:  # Simple format with wavelength and flux
            wavelength, flux = data[:, 0], data[:, 1]
        else:
            # Assuming first column is wavelength and second column is flux
            wavelength, flux = data[:, 0], data[:, 1]

        # Convert wavelength to microns based on detected unit
        if wavelength_unit != "unknown":
            if "angstrom" in wavelength_unit or "å" in wavelength_unit:
                wavelength = wavelength / 10000  # Angstroms to microns
            elif "nm" in wavelength_unit or "nanometer" in wavelength_unit:
                wavelength = wavelength / 1000  # Nanometers to microns
            elif "m" == wavelength_unit.strip():
                wavelength = wavelength * 1e6  # Meters to microns
            elif "mm" in wavelength_unit:
                wavelength = wavelength * 1000  # Millimeters to microns
            elif "cm" in wavelength_unit:
                wavelength = wavelength * 10000  # Centimeters to microns
            # If already in microns (or "micron" is in the unit string), no conversion needed
        else:
            # Fallback to guessing based on scale if no unit is found in header
            if np.median(wavelength) > 1000:
                wavelength = wavelength / 10000  # Likely Angstroms to microns
            elif np.median(wavelength) < 1e-3:
                wavelength = wavelength * 1e6  # Likely meters to microns
            # Otherwise assume it's already in microns
            
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
    
    return wavelength, flux

def organize_sed_files(directory):
    """Organize SED files by parameters."""
    organized_files = {
        'by_Z': {},
        'by_eta': {},
        'by_n': {}
    }
    
    all_params = []
    
    for filename in os.listdir(directory):
        if filename.startswith('sed_sfr_scaled_') and filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            params = parse_sed_filename(filename)
            
            if params:
                # Store by Z value
                if params['Z'] not in organized_files['by_Z']:
                    organized_files['by_Z'][params['Z']] = []
                organized_files['by_Z'][params['Z']].append((filepath, params))
                
                # Store by eta value
                if params['eta'] not in organized_files['by_eta']:
                    organized_files['by_eta'][params['eta']] = []
                organized_files['by_eta'][params['eta']].append((filepath, params))
                
                # Store by n value
                if params['n'] not in organized_files['by_n']:
                    organized_files['by_n'][params['n']] = []
                organized_files['by_n'][params['n']].append((filepath, params))
                
                all_params.append(params)
    
    # Extract unique parameter values
    unique_Z = sorted(organized_files['by_Z'].keys())
    unique_eta = sorted(organized_files['by_eta'].keys())
    unique_n = sorted(organized_files['by_n'].keys())
    
    return organized_files, unique_Z, unique_eta, unique_n, all_params

def plot_seds_by_parameter(organized_files, param_name, output_dir):
    """Create plots grouped by a specific parameter."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set fixed wavelength range (0.1 to 1000 microns)
    wavelength_min = 0.1  # microns
    wavelength_max = 1000  # microns
    
    for param_value, file_list in organized_files[f'by_{param_name}'].items():
        plt.figure(figsize=(12, 8))
        
        # Track max flux within wavelength range for y-axis scaling
        max_flux = float('-inf')
        all_curves = []
        
        # Define colormap based on other parameters
        if param_name == 'Z':
            # Color by density (n)
            other_params = sorted(set(params['n'] for _, params in file_list))
            cmap = plt.cm.viridis
            norm = plt.Normalize(min(other_params), max(other_params))
            
            for filepath, params in file_list:
                wavelength, flux = load_sed_file(filepath)
                
                # Store curve data for later plotting
                all_curves.append({
                    'wavelength': wavelength,
                    'flux': flux,
                    'params': params,
                    'color': cmap(norm(params['n'])),
                    'label': f"eta={params['eta']:.3f}, n={params['n']:.1f}"
                })
                
                # Update max flux (only consider points within our x-range)
                mask = (wavelength >= wavelength_min) & (wavelength <= wavelength_max)
                if np.any(mask) and np.max(flux[mask]) > max_flux:
                    max_flux = np.max(flux[mask])
                
            title = f"SEDs for Z={param_value:.4f}"
            
        elif param_name == 'eta':
            # Color by density (n)
            other_params = sorted(set(params['n'] for _, params in file_list))
            cmap = plt.cm.viridis
            norm = plt.Normalize(min(other_params), max(other_params))
            
            for filepath, params in file_list:
                wavelength, flux = load_sed_file(filepath)
                
                # Store curve data for later plotting
                all_curves.append({
                    'wavelength': wavelength,
                    'flux': flux,
                    'params': params,
                    'color': cmap(norm(params['n'])),
                    'label': f"Z={params['Z']:.4f}, n={params['n']:.1f}"
                })
                
                # Update max flux (only consider points within our x-range)
                mask = (wavelength >= wavelength_min) & (wavelength <= wavelength_max)
                if np.any(mask) and np.max(flux[mask]) > max_flux:
                    max_flux = np.max(flux[mask])
                
            title = f"SEDs for eta={param_value:.4f}"
            
        elif param_name == 'n':
            # Color by metallicity (Z)
            other_params = sorted(set(params['Z'] for _, params in file_list))
            cmap = plt.cm.plasma
            norm = plt.Normalize(min(other_params), max(other_params))
            
            for filepath, params in file_list:
                wavelength, flux = load_sed_file(filepath)
                
                # Store curve data for later plotting
                all_curves.append({
                    'wavelength': wavelength,
                    'flux': flux,
                    'params': params,
                    'color': cmap(norm(params['Z'])),
                    'label': f"Z={params['Z']:.4f}, eta={params['eta']:.3f}"
                })
                
                # Update max flux (only consider points within our x-range)
                mask = (wavelength >= wavelength_min) & (wavelength <= wavelength_max)
                if np.any(mask) and np.max(flux[mask]) > max_flux:
                    max_flux = np.max(flux[mask])
                
            title = f"SEDs for n={param_value:.1f}"
        
        # Now plot all curves with consistent axes
        for curve in all_curves:
            plt.loglog(curve['wavelength'], curve['flux'], 
                     label=curve['label'],
                     color=curve['color'])
        
        # Set axis limits
        plt.xlim(wavelength_min, wavelength_max)
        if max_flux > float('-inf'):
            max_flux = max_flux * 0.5
            min_flux = max_flux * 1e-5  # 10 orders of magnitude below max
            plt.ylim(1e39, 1e44) 
        
        plt.xlabel("Wavelength [µm]")
        plt.ylabel("SED per SFR [W/M⊙ yr⁻¹]")
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncols=2, fontsize=8)
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, f"seds_{param_name}_{param_value}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"Plots for parameter {param_name} saved to {output_dir}")

def create_parameter_grid(all_params, unique_Z, unique_eta, unique_n, output_dir):
    """Create a grid plot showing coverage of parameter space."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot Z vs eta
    ax1 = axes[0]
    for params in all_params:
        ax1.scatter(params['Z'], params['eta'], c=np.log10(params['n']), cmap='viridis', s=50, alpha=0.7)
    
    ax1.set_xlabel('Metallicity (Z)')
    ax1.set_ylabel('Star Formation Efficiency (eta)')
    ax1.set_title('Z vs eta (colored by log n)')
    ax1.grid(True, alpha=0.3)
    
    # Plot Z vs n
    ax2 = axes[1]
    for params in all_params:
        ax2.scatter(params['Z'], params['n'], c=params['eta'], cmap='plasma', s=50, alpha=0.7)
    
    ax2.set_xlabel('Metallicity (Z)')
    ax2.set_ylabel('Density (n) [cm⁻³]')
    ax2.set_title('Z vs n (colored by eta)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Plot eta vs n
    ax3 = axes[2]
    for params in all_params:
        ax3.scatter(params['eta'], params['n'], c=params['Z'], cmap='cividis', s=50, alpha=0.7)
    
    ax3.set_xlabel('Star Formation Efficiency (eta)')
    ax3.set_ylabel('Density (n) [cm⁻³]')
    ax3.set_title('eta vs n (colored by Z)')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "parameter_grid.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Parameter grid plot saved to {output_file}")

def create_wavelength_comparison(organized_files, output_dir):
    """Create plots comparing SEDs at specific wavelengths as a function of parameters."""
    # Sample wavelengths to extract across the spectrum (UV, optical, near-IR, mid-IR, far-IR)
    sample_wavelengths = [0.15, 0.5, 2.0, 10.0, 100.0]  # in microns
    
    # Collect data for plotting
    plot_data = {wl: {'Z': [], 'eta': [], 'n': [], 'flux': []} for wl in sample_wavelengths}
    
    # Process all files
    for param_group in organized_files['by_Z'].values():
        for filepath, params in param_group:
            wavelength, flux = load_sed_file(filepath)
            
            # Find closest wavelength points and extract flux
            for target_wl in sample_wavelengths:
                idx = np.abs(wavelength - target_wl).argmin()
                actual_wl = wavelength[idx]
                
                # Only use if we're reasonably close to the target wavelength
                if abs(actual_wl - target_wl) / target_wl < 0.1:  # within 10%
                    plot_data[target_wl]['Z'].append(params['Z'])
                    plot_data[target_wl]['eta'].append(params['eta'])
                    plot_data[target_wl]['n'].append(params['n'])
                    plot_data[target_wl]['flux'].append(flux[idx])
    
    # Create plots for each wavelength
    for wl in sample_wavelengths:
        if not plot_data[wl]['Z']:  # Skip if no data for this wavelength
            continue
            
        # Convert to numpy arrays for easier manipulation
        Z = np.array(plot_data[wl]['Z'])
        eta = np.array(plot_data[wl]['eta'])
        n = np.array(plot_data[wl]['n'])
        flux = np.array(plot_data[wl]['flux'])
        
        # Find global min/max flux for consistent y-axis
        max_flux = np.max(flux)
        min_flux = max_flux * 1e-2  # 3 orders of magnitude below max for better visibility
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Flux vs Z for different eta values
        unique_eta_values = np.unique(eta)
        for eta_val in unique_eta_values:
            mask = eta == eta_val
            if np.any(mask):
                # Sort by n to get consistent line coloring
                sort_idx = np.argsort(n[mask])
                axes[0].loglog(Z[mask][sort_idx], flux[mask][sort_idx], 'o', 
                             label=f'eta={eta_val:.3f}')
        
        axes[0].set_xlabel('Metallicity (Z)')
        axes[0].set_ylabel(f'Flux at {wl} µm [W/M⊙ yr⁻¹]')
        axes[0].set_title(f'Flux vs Z at {wl} µm')
        axes[0].set_ylim(min_flux, max_flux * 1.5)
        axes[0].grid(True, which="both", ls="-", alpha=0.2)
        axes[0].legend()
        
        # Flux vs eta for different Z values
        unique_Z_values = np.unique(Z)
        for Z_val in unique_Z_values:
            mask = Z == Z_val
            if np.any(mask):
                sort_idx = np.argsort(n[mask])
                axes[1].loglog(eta[mask][sort_idx], flux[mask][sort_idx], 'o', 
                             label=f'Z={Z_val:.4f}')
        
        axes[1].set_xlabel('Star Formation Efficiency (eta)')
        axes[1].set_ylabel(f'Flux at {wl} µm [W/M⊙ yr⁻¹]')
        axes[1].set_title(f'Flux vs eta at {wl} µm')
        axes[1].set_ylim(min_flux, max_flux * 1.5)
        axes[1].grid(True, which="both", ls="-", alpha=0.2)
        axes[1].legend()
        
        # Flux vs n for different Z values
        for Z_val in unique_Z_values:
            for eta_val in unique_eta_values:
                mask = (Z == Z_val) & (eta == eta_val)
                if np.sum(mask) > 1:  # Only plot if multiple points exist
                    sort_idx = np.argsort(n[mask])
                    axes[2].loglog(n[mask][sort_idx], flux[mask][sort_idx], 'o', 
                                 label=f'Z={Z_val:.4f}, eta={eta_val:.3f}')
        
        axes[2].set_xlabel('Density (n) [cm⁻³]')
        axes[2].set_ylabel(f'Flux at {wl} µm [W/M⊙ yr⁻¹]')
        axes[2].set_title(f'Flux vs n at {wl} µm')
        axes[2].set_ylim(min_flux, max_flux * 1.5)
        axes[2].grid(True, which="both", ls="-", alpha=0.2)
        #axes[2].legend()
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"wavelength_comparison_{wl:.2f}micron.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create composite plot with all wavelengths
    plt.figure(figsize=(12, 8))
    
    # Choose a specific parameter combination to compare wavelengths
    # Use the first Z value and eta value, but compare across different n values
    if len(unique_Z_values) > 0 and len(unique_eta_values) > 0:
        ref_Z = unique_Z_values[0]
        ref_eta = unique_eta_values[0]
        
        for wl in sample_wavelengths:
            if not plot_data[wl]['Z']:
                continue
                
            Z = np.array(plot_data[wl]['Z'])
            eta = np.array(plot_data[wl]['eta'])
            n = np.array(plot_data[wl]['n'])
            flux = np.array(plot_data[wl]['flux'])
            
            # Filter for reference Z and eta
            mask = (Z == ref_Z) & (eta == ref_eta)
            if np.any(mask):
                # Sort by n
                sort_idx = np.argsort(n[mask])
                plt.loglog(n[mask][sort_idx], flux[mask][sort_idx], 'o', 
                         label=f'λ = {wl} µm')
    
    plt.xlabel('Density (n) [cm⁻³]')
    plt.ylabel('Flux [W/M⊙ yr⁻¹]')
    plt.title(f'Flux vs Density across different wavelengths (Z={ref_Z:.4f}, eta={ref_eta:.3f})')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(ncols=2, fontsize=8)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "wavelength_comparison_combined.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Wavelength comparison plots saved to {output_dir}")

def create_summary_plot(organized_files, unique_Z, unique_eta, unique_n, output_dir):
    """Create a summary plot showing sample SEDs across parameter space."""
    plt.figure(figsize=(12, 10))
    
    # Set wavelength range
    wavelength_min, wavelength_max = 0.1, 1000  # microns
    
    # Track max flux for y-axis scaling
    max_flux = float('-inf')
    min_flux = float('inf')
    
    # Select a subset of parameters to show (to avoid overcrowding)
    sample_Z = unique_Z[:min(2, len(unique_Z))]
    sample_eta = unique_eta[:min(2, len(unique_eta))]
    sample_n = unique_n[::max(1, len(unique_n)//3)]  # Take ~3 values spread out
    
    # Define markers and linestyles
    markers = ['', '-', '--', '-.', ':']
    
    # Define line colors by Z
    Z_colors = plt.cm.viridis(np.linspace(0, 1, len(sample_Z)))
    
    # Define line styles by eta
    eta_linestyles = markers[1:len(sample_eta)+1]
    
    # Define brightness by density
    n_alphas = np.linspace(0.4, 1.0, len(sample_n))
    
    # Create legend handles and labels
    Z_handles, Z_labels = [], []
    eta_handles, eta_labels = [], []
    n_handles, n_labels = [], []
    
    # Plot selected SEDs
    for i, Z in enumerate(sample_Z):
        for j, eta in enumerate(sample_eta):
            for k, n in enumerate(sample_n):
                # Find matching files
                found = False
                for param_group in organized_files['by_Z'].values():
                    for filepath, params in param_group:
                        if (abs(params['Z'] - Z) < 1e-6 and 
                            abs(params['eta'] - eta) < 1e-6 and 
                            abs(params['n'] - n) < 1e-6):
                            
                            wavelength, flux = load_sed_file(filepath)
                            
                            # Skip if no data
                            if len(wavelength) == 0 or len(flux) == 0:
                                continue
                                
                            # Plot with Z determining color, eta determining line style, n determining alpha
                            lw = 1.5
                            alpha = n_alphas[k]
                            color = Z_colors[i]
                            linestyle = eta_linestyles[j]
                            
                            plt.loglog(wavelength, flux, 
                                      color=color, 
                                      linestyle=linestyle,
                                      alpha=alpha,
                                      linewidth=lw)
                            
                            # Update max and min flux (only consider points within our x-range)
                            mask = (wavelength >= wavelength_min) & (wavelength <= wavelength_max)
                            if np.any(mask):
                                if np.max(flux[mask]) > max_flux:
                                    max_flux = np.max(flux[mask])
                                if np.min(flux[mask]) < min_flux:
                                    min_flux = np.min(flux[mask])
                            
                            found = True
                            break
                    if found:
                        break
    
    # Create custom legend elements
    for i, Z in enumerate(sample_Z):
        line = plt.Line2D([0], [0], color=Z_colors[i], linewidth=2)
        Z_handles.append(line)
        Z_labels.append(f'Z = {Z:.4f}')
        
    for j, eta in enumerate(sample_eta):
        line = plt.Line2D([0], [0], color='black', linestyle=eta_linestyles[j], linewidth=2)
        eta_handles.append(line)
        eta_labels.append(f'η = {eta:.3f}')
        
    for k, n in enumerate(sample_n):
        line = plt.Line2D([0], [0], color='gray', alpha=n_alphas[k], linewidth=2)
        n_handles.append(line)
        n_labels.append(f'n = {n:.1f}')
    
    # Add legends
    l1 = plt.legend(Z_handles, Z_labels, title='Metallicity (Z)', loc='upper left')
    plt.gca().add_artist(l1)
    
    # Place eta legend to the right
    l2 = plt.legend(eta_handles, eta_labels, title='Star Formation Efficiency (η)', 
                   loc='upper right')
    plt.gca().add_artist(l2)
    
    # Place n legend below
    l3 = plt.legend(n_handles, n_labels, title='Cloud Density (n)', 
                   loc='lower left')
    
    # Set axis limits
    plt.xlim(wavelength_min, wavelength_max)
    if max_flux > float('-inf'):
        min_y = max_flux * 1e-10  # 10 orders of magnitude below max
        plt.ylim(1e39, 1e44)  # Add some space above max
    
    plt.xlabel("Wavelength [µm]")
    plt.ylabel("SED per SFR [W/M⊙ yr⁻¹]")
    plt.title("Representative SEDs Across Parameter Space")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Mark important spectral regions
    regions = [
        (0.1, 0.4, 'UV', 'purple'),
        (0.4, 0.7, 'Optical', 'green'),
        (0.7, 5, 'Near-IR', 'red'),
        (5, 30, 'Mid-IR', 'orange'),
        (30, 1000, 'Far-IR', 'brown')
    ]
    
    y_pos = min_y * 1.5  # Position text just above the bottom
    
    for start, end, label, color in regions:
        # Only include regions within our plot range
        if end >= wavelength_min and start <= wavelength_max:
            actual_start = max(start, wavelength_min)
            actual_end = min(end, wavelength_max)
            
            # Add shaded region
            plt.axvspan(actual_start, actual_end, alpha=0.1, color=color)
            
            # Add text label in the middle (in log space)
            mid_point = np.sqrt(actual_start * actual_end)
            plt.text(mid_point, y_pos, label, ha='center', va='bottom', 
                    color=color, fontweight='bold', alpha=0.7)
    
    # Save figure
    output_file = os.path.join(output_dir, "sed_summary.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved to {output_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze and visualize SED files.')
    parser.add_argument('--dir', type=str, default='sed_output_lr',
                        help='Directory containing SED files')
    parser.add_argument('--output', type=str, default='sed_plots',
                        help='Directory for output plots')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Organize SED files
    print(f"Analyzing SED files in {args.dir}...")
    organized_files, unique_Z, unique_eta, unique_n, all_params = organize_sed_files(args.dir)
    
    # Print summary
    print(f"Found {len(all_params)} SED files")
    print(f"Unique Z values: {unique_Z}")
    print(f"Unique eta values: {unique_eta}")
    print(f"Unique n values: {unique_n}")
    
    # Create plots by parameter
    print("Creating parameter-based plots...")
    plot_seds_by_parameter(organized_files, 'Z', args.output)
    plot_seds_by_parameter(organized_files, 'eta', args.output)
    plot_seds_by_parameter(organized_files, 'n', args.output)
    
    # Create parameter grid plot
    print("Creating parameter grid plot...")
    create_parameter_grid(all_params, unique_Z, unique_eta, unique_n, args.output)
    
    # Create wavelength comparison plots
    print("Creating wavelength comparison plots...")
    create_wavelength_comparison(organized_files, args.output)
    
    print(f"All plots have been saved to {args.output}")


if __name__ == "__main__":
    main()