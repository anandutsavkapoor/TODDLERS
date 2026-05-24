#!/usr/bin/env python3
"""
Cache Analysis Script for TODDLERS Interpolant Cache Files

This script analyzes cached simulation data from TODDLERS interpolant cache files.
It provides visualizations and statistics for key physical quantities over time.

Usage:
    python analyze_cache.py path/to/cache/file.pkl
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from toddlers.constants import MYR_TO_SEC, M_SUN, PC_TO_CM
from toddlers.cloudy_output_handler import CloudyOutputHandler

def load_cache_file(file_path):
    """Load and validate cache file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        if not isinstance(data, dict):
            raise ValueError("Cache file does not contain a dictionary")
            
        required_keys = {'evolution', 'cloudy', 'timepoints', 'dissolution_time', 'paths'}
        if not all(key in data for key in required_keys):
            raise ValueError("Cache file missing required keys")
            
        return data
    except Exception as e:
        print(f"Error loading cache file: {str(e)}")
        sys.exit(1)

def print_basic_info(data):
    """Print basic information about the simulation."""
    print("\nBasic Simulation Information:")
    print("-" * 50)
    
    # Get parameters from path
    sim_path = Path(data['paths']['evolution'])
    param_str = sim_path.stem.replace('sim_', '')
    print(f"Parameter set: {param_str}")
    
    # Evolution information
    n_generations = len(data['evolution']['results'])
    print(f"\nNumber of generations: {n_generations}")
    
    # Dissolution information
    if data['dissolution_time'] is not None:
        print(f"Dissolution time: {data['dissolution_time'] / MYR_TO_SEC:.2f} Myr")  # Convert seconds to Myr
    else:
        print("No dissolution occurred")
        
    # Generation information
    for i, gen in enumerate(data['evolution']['results'], 1):
        print(f"\nGeneration {i}:")
        print(f"  Status: {gen['status']}")
        print(f"  Cloud mass: {gen['cloud_mass']/M_SUN:.2e} M☉")  # Convert to solar masses
        print(f"  Cloud radius: {gen['cloud_radius']/PC_TO_CM:.2f} pc")  # Convert to parsecs
        
        # Print phase transitions
        if 'phase_transitions' in gen and gen['phase_transitions']:
            print("  Phase transitions:")
            for transition in gen['phase_transitions']:
                print(f"    {transition[0]} at {transition[1]/MYR_TO_SEC:.2f} Myr")

def plot_evolution_data(data):
    """Create comprehensive plots of evolution data."""
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)
    fig.suptitle("Evolution Analysis", fontsize=14, y=0.95)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    for gen_idx, gen_data in enumerate(data['evolution']['results']):
        print(gen_idx)
        time = np.array(gen_data['time']) / MYR_TO_SEC  # Convert to Myr
        
        # Plot 1: Radius evolution

        radius = np.array(gen_data['radius']) / PC_TO_CM  # Convert to pc
        ax1.plot(time, radius, label=f'Gen {gen_idx+1}')
        ax1.set_xlabel('Time (Myr)')
        ax1.set_ylabel('Radius (pc)')
        ax1.set_title('Shell Radius Evolution')
        ax1.grid(True)
        ax1.legend()
        
        # Plot 2: Velocity evolution
        velocity = np.array(gen_data['velocity']) / 1e5  # Convert to km/s
        print(velocity.shape)
        ax2.plot(time, velocity, label=f'Gen {gen_idx+1}')
        ax2.set_xlabel('Time (Myr)')
        ax2.set_ylabel('Velocity (km/s)')
        ax2.set_title('Shell Velocity Evolution')
        ax2.grid(True)
        ax2.legend()
        
        # Plot 3: Mass evolution
        mass = np.array(gen_data['mass']) / M_SUN  # Convert to solar masses
        ax3.plot(time, mass, label=f'Gen {gen_idx+1}')
        ax3.set_xlabel('Time (Myr)')
        ax3.set_ylabel('Mass (M☉)')
        ax3.set_title('Shell Mass Evolution')
        ax3.grid(True)
        ax3.legend()
        
        # Plot 4: Cloud density evolution
        density = np.array(gen_data['n_cloud_avg'])  # Already in cm^-3
        ax4.plot(time, density, label=f'Gen {gen_idx+1}')
        ax4.set_xlabel('Time (Myr)')
        ax4.set_ylabel('Average Cloud Density (cm⁻³)')
        ax4.set_title('Cloud Density Evolution')
        ax4.set_yscale('log')
        ax4.grid(True)
        ax4.legend()
        
        # Plot 5: Shell properties
        n_shell_in = np.array(gen_data['shell_properties']['n_shell_in'])
        n_shell_max = np.array(gen_data['shell_properties']['n_shell_max'])
        ax5.plot(time, n_shell_in, label=f'Inner n (Gen {gen_idx+1})', linestyle='-')
        ax5.plot(time, n_shell_max, label=f'Max n (Gen {gen_idx+1})', linestyle='--')
        ax5.set_xlabel('Time (Myr)')
        ax5.set_ylabel('Shell Density (cm⁻³)')
        ax5.set_title('Shell Density Evolution')
        ax5.set_yscale('log')
        ax5.grid(True)
        ax5.legend()
        
        # Plot 6: Escape fractions
        f_esc_i = np.array(gen_data['shell_properties']['f_esc_i'])
        f_esc_uv = np.array(gen_data['shell_properties']['f_esc_uv'])
        ax6.plot(time, f_esc_i, label=f'f_esc_i (Gen {gen_idx+1})', linestyle='-')
        ax6.plot(time, f_esc_uv, label=f'f_esc_uv (Gen {gen_idx+1})', linestyle='--')
        ax6.set_xlabel('Time (Myr)')
        ax6.set_ylabel('Escape Fraction')
        ax6.set_title('Escape Fraction Evolution')
        ax6.grid(True)
        ax6.legend()
    
    return fig


def plot_seds(data):
    """Create comparative plots of SEDs showing time evolution."""
    if not data['cloudy']:
        print("\nNo Cloudy data available for SED plots")
        return None
        
    # Get all timepoints and sort them
    timepoints = sorted(data['cloudy'].keys())
    print(f"\nPlotting SEDs for {len(timepoints)} timepoints")
    
    # Create figure with three subplots
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1])
    fig.suptitle("SED Time Evolution Analysis", fontsize=14, y=0.95)
    
    # Plot full range SEDs with time evolution
    ax1 = fig.add_subplot(gs[0, :])
    
    # Color map for time evolution
    colors = plt.cm.viridis(np.linspace(0, 1, len(timepoints)))
    
    # Store wavelength range for zoom plots
    wave_min = float('inf')
    wave_max = -float('inf')
    
    # Plot SEDs for each timepoint
    for i, time in enumerate(timepoints):
        time_data = data['cloudy'][time]
        wave = time_data['continuum']['nu']  # wavelength in microns
        
        # Update wavelength range
        wave_min = min(wave_min, wave.min())
        wave_max = max(wave_max, wave.max())
        
        # Plot incident and observed SEDs
        incident = time_data['continuum']['incident']
        observed = time_data['continuum']['observed']
        
        # Plot with transparency based on time
        alpha = 0.3 if i < len(timepoints)-1 else 1.0  # Make last timepoint solid
        
        ax1.loglog(wave, incident, '-', color=colors[i], alpha=alpha,
                  label=f't = {time:.1f} Myr' if i == len(timepoints)-1 or i == 0 else "")
        ax1.loglog(wave, observed, '--', color=colors[i], alpha=alpha)
    
    ax1.set_xlabel('Wavelength (μm)')
    ax1.set_ylabel('νLν (erg/s)')
    ax1.set_xlim(0.1, 3005)
    ax1.set_ylim(1e32, 1e42)
    ax1.set_title('SED Evolution (Solid: Incident, Dashed: Observed)')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend()
    
    # Add colorbar to show time evolution
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                              norm=plt.Normalize(min(timepoints), max(timepoints)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1)
    cbar.set_label('Time (Myr)', rotation=270, labelpad=15)
    
    # Plot zoom regions with first and last timepoint comparison
    first_time = timepoints[0]
    last_time = timepoints[-1]
    
    # UV region
    ax2 = fig.add_subplot(gs[1, 0])
    for time, ls, label in [(first_time, '-', 'First'), (last_time, '--', 'Last')]:
        time_data = data['cloudy'][time]
        wave = time_data['continuum']['nu']
        mask_uv = (wave >= 0.1) & (wave <= 0.3)
        
        ax2.loglog(wave[mask_uv], time_data['continuum']['incident'][mask_uv], 
                  ls, color='b', label=f'{label} Incident', alpha=0.7)
        ax2.loglog(wave[mask_uv], time_data['continuum']['observed'][mask_uv], 
                  ls, color='r', label=f'{label} Observed', alpha=0.7)
    
    ax2.set_xlabel('Wavelength (μm)')
    ax2.set_ylabel('νLν (erg/s)')
    ax2.set_title(f'UV Region Comparison (t = {first_time:.1f} vs {last_time:.1f} Myr)')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend()
    
    # Optical region
    ax3 = fig.add_subplot(gs[1, 1])
    for time, ls, label in [(first_time, '-', 'First'), (last_time, '--', 'Last')]:
        time_data = data['cloudy'][time]
        wave = time_data['continuum']['nu']
        mask_opt = (wave >= 0.3) & (wave <= 1.0)
        
        ax3.loglog(wave[mask_opt], time_data['continuum']['incident'][mask_opt], 
                  ls, color='b', label=f'{label} Incident', alpha=0.7)
        ax3.loglog(wave[mask_opt], time_data['continuum']['observed'][mask_opt], 
                  ls, color='r', label=f'{label} Observed', alpha=0.7)
    
    ax3.set_xlabel('Wavelength (μm)')
    ax3.set_ylabel('νLν (erg/s)')
    ax3.set_title(f'Optical Region Comparison (t = {first_time:.1f} vs {last_time:.1f} Myr)')
    ax3.grid(True, which="both", ls="-", alpha=0.2)
    ax3.legend()
    
    # Plot evolution of specific wavelengths
    ax4 = fig.add_subplot(gs[2, :])
    wavelengths_to_track = [0.15, 0.5, 2.0]  # UV, optical, and IR
    
    for wavelength in wavelengths_to_track:
        incident_evolution = []
        observed_evolution = []
        
        for time in timepoints:
            time_data = data['cloudy'][time]
            wave = time_data['continuum']['nu']
            
            # Find closest wavelength index
            idx = np.abs(wave - wavelength).argmin()
            
            incident_evolution.append(time_data['continuum']['incident'][idx])
            observed_evolution.append(time_data['continuum']['observed'][idx])
        
        ax4.semilogy(timepoints, incident_evolution, '-', 
                    label=f'Incident {wavelength:.2f}μm', alpha=0.7)
        ax4.semilogy(timepoints, observed_evolution, '--', 
                    label=f'Observed {wavelength:.2f}μm', alpha=0.7)
    
    ax4.set_xlabel('Time (Myr)')
    ax4.set_ylabel('νLν (erg/s)')
    ax4.set_title('Evolution at Specific Wavelengths')
    ax4.grid(True, which="both", ls="-", alpha=0.2)
    ax4.legend()
    
    
    return fig

def analyze_cloudy_data(data):
    """Analyze and print Cloudy data statistics."""
    if not data['cloudy']:
        print("\nNo Cloudy data available")
        return
        
    print("\nCloudy Data Analysis:")
    print("-" * 50)
    
    # Count model types and track radii
    model_counts = {'shell': 0, 'unified': 0, 'dissolved': 0}
    model_radii = {'shell': [], 'unified': [], 'dissolved': []}
    
    for time, time_data in data['cloudy'].items():
        model_type = time_data.get('model_type')
        if model_type in model_counts:
            model_counts[model_type] += 1
            
            # Get radius if available (not available for dissolved model)
            if model_type != 'dissolved' and 'radius' in time_data:
                try:
                    radius = time_data['radius'][0][0]  # First point of radius array
                    model_radii[model_type].append(radius)
                except Exception as e:
                    print(f"Error getting radius at time {time}: {str(e)}")
            
    print("\nModel type distribution:")
    for model, count in model_counts.items():
        print(f"  {model}: {count} timepoints")
        if model_radii[model]:  # Only print radius stats if we have data
            radii = np.array(model_radii[model])
            print(f"    Initial radius: {radii[0]/PC_TO_CM:.2f} pc")
            print(f"    Average radius: {np.mean(radii)/PC_TO_CM:.2f} pc")
            print(f"    Min radius: {np.min(radii)/PC_TO_CM:.2f} pc")
            print(f"    Max radius: {np.max(radii)/PC_TO_CM:.2f} pc")
        
    # Analyze ionization fractions
    print("\nAverage ionization fractions:")
    total_h_ion = 0
    total_he_ion = 0
    count = 0
    
    for time_data in data['cloudy'].values():
        if 'ionization' in time_data:
            ion_data = time_data['ionization']
            if 'H' in ion_data and 'He' in ion_data:
                total_h_ion += ion_data['H']['H+'][0]  # First point
                total_he_ion += ion_data['He']['He+'][0] + 2*ion_data['He']['He++'][0]  # Include double ionization
                count += 1
                
    if count > 0:
        print(f"  Average H ionization fraction: {total_h_ion/count:.3f}")
        print(f"  Average He ionization fraction: {total_he_ion/count:.3f}")

def normalize_wavelength(wavelength_value):
    """
    Normalize wavelength value to consistent format for comparison.
    
    Args:
        wavelength_value (Union[str, float]): Wavelength value, either as:
            - String with optional 'm' suffix (e.g., '0.097702m')
            - Float (e.g., 0.097702)
            
    Returns:
        float: Normalized wavelength value with consistent precision
                (e.g., 0.097702)
                
    Example:
        >>> self._normalize_wavelength('0.09770200000001m')
        0.097702
        >>> self._normalize_wavelength(0.09770200000001)
        0.097702
    """
    if isinstance(wavelength_value, str):
        # Remove 'm' if present and convert to float
        wavelength_value = float(wavelength_value.rstrip('m'))
    return float(f"{wavelength_value:.6f}")

def create_line_map():
    """
    Create mapping between numerical IDs and line identifiers.
    
    Uses CloudyOutputHandler's line list reading functionality to create
    a consistent mapping between line IDs and their physical properties.
    
    Returns:
        dict: Mapping of line IDs to normalized tuples where each tuple contains:
            - element (str): Element symbol (e.g., 'C', 'O', 'Fe')
            - ion (int): Ionization state (0 for molecules)
            - wavelength (float): Normalized wavelength in microns
            
    Note:
        Wavelengths are normalized using _normalize_wavelength to ensure
        consistent comparison with Cloudy output.
    """
    handler = CloudyOutputHandler('shell', 0.0)  
    lines = handler._read_line_list()
    
    line_map = {}
    for line_id, line_info in enumerate(lines):
        try:
            wavelength = normalize_wavelength(line_info['wavelength'])
            
            line_id_tuple = (
                line_info['element'],
                line_info['ion'],
                wavelength
            )
            line_map[line_id] = line_id_tuple
                
        except (KeyError, ValueError) as e:
            print(f"Error processing line info: {line_info}")
            print(f"Error details: {str(e)}")
            continue
            
    return line_map

def process_luminosities(line_map, cloudy_data, time_point):
    """
    Process line luminosities for a single time point.
    
    Extracts both intrinsic and emergent line luminosities from Cloudy output,
    normalizing wavelengths to ensure consistent comparison with line map.
    
    Args:
        cloudy_data (dict): Dictionary containing Cloudy output data
        time_point (float): Time point in Myr
        
    Returns:
        tuple: Arrays of (intrinsic_luminosities, emergent_luminosities), where:
            - Each array has shape (n_lines,)
            - Array indices correspond to line_map keys
            - Luminosities are in the units provided by Cloudy
            
    Raises:
        ValueError: If any lines from the line map are missing in the Cloudy output
        
    Note:
        Uses _normalize_wavelength to ensure consistent wavelength comparison
        between line map and Cloudy output.
    """
    if time_point not in cloudy_data:
        return None, None
            
    # Get the original data
    intrinsic_orig = cloudy_data[time_point]['lines_intrinsic']
    emergent_orig = cloudy_data[time_point]['lines_emergent']
    
    # Create normalized versions of the dictionaries
    intrinsic = {
        (t[0], t[1], normalize_wavelength(t[2])): v 
        for t, v in intrinsic_orig.items()
    }
    emergent = {
        (t[0], t[1], normalize_wavelength(t[2])): v 
        for t, v in emergent_orig.items()
    }
    
    intr_lums = np.zeros(len(line_map))
    emer_lums = np.zeros(len(line_map))
    
    missing_lines = []
    for line_id, line_tuple in line_map.items():
        if line_tuple not in intrinsic or line_tuple not in emergent:
            missing_lines.append((line_id, line_tuple))
            
    if missing_lines:
        print("\nDetailed comparison for first missing line:")
        missing_id, missing_tuple = missing_lines[0]
        print(f"Missing line tuple: {missing_tuple}")
        print(f"Missing wavelength normalized: {normalize_wavelength(missing_tuple[2])}")
        print("Available tuples that look similar:")
        for available_tuple in list(intrinsic.keys())[:5]:
            print(f"  {available_tuple} (normalized wavelength: "
                    f"{normalize_wavelength(available_tuple[2])})")
            
    if missing_lines:
        raise ValueError(f"Missing line data for the following lines: {missing_lines}")
        
    # Fill arrays using normalized comparisons
    for line_id, line_tuple in line_map.items():
        intr_lums[line_id] = intrinsic[line_tuple]
        emer_lums[line_id] = emergent[line_tuple]
        print(line_id, line_tuple, intr_lums[line_id], emer_lums[line_id])
    
    return intr_lums, emer_lums

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_cache.py path/to/cache/file.pkl")
        sys.exit(1)
        
    cache_file = Path(sys.argv[1])
    if not cache_file.exists():
        print(f"Error: Cache file {cache_file} does not exist")
        sys.exit(1)
        
    # Load and analyze data
    data = load_cache_file(cache_file)
    print_basic_info(data)
    analyze_cloudy_data(data)
    
    line_map = create_line_map()
    print(process_luminosities(line_map, data['cloudy'], 0.1))

    # Create evolution plots
    fig_evo = plot_evolution_data(data)
    output_file_evo = cache_file.with_stem(cache_file.stem + '_evolution')
    output_file_evo = output_file_evo.with_suffix('.png')
    fig_evo.savefig(output_file_evo, dpi=300, bbox_inches='tight')
    print(f"\nEvolution plot saved to: {output_file_evo}")
    plt.close(fig_evo)
    
    # Create SED plots
    fig_sed = plot_seds(data)
    if fig_sed is not None:
        output_file_sed = cache_file.with_stem(cache_file.stem + '_sed')
        output_file_sed = output_file_sed.with_suffix('.png')
        fig_sed.savefig(output_file_sed, dpi=300, bbox_inches='tight')
        print(f"SED plot saved to: {output_file_sed}")
        plt.close(fig_sed)

if __name__ == "__main__":
    main()
