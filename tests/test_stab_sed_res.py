import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path
from matplotlib.gridspec import GridSpec
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # repo src/
from toddlers.cloudy_output_handler import CloudyOutputHandler
from toddlers.stab.line_profiles import LineProfileGenerator
from toddlers.constants import MYR_TO_SEC

def test_sed_resolution(cloudy_dir, time=0.1, model='shell', min_spacing=1e-16):
    """
    Generate and compare low and high resolution SEDs using real Cloudy output.
    
    Args:
        cloudy_dir (str|Path): Path to cloudy output directory
        time (float): Time in Myr to analyze
        model (str): Model type ('shell', 'unified', or 'dissolved')
        min_spacing (float): Minimum relative spacing between wavelength points
    """
    cloudy_dir = Path(cloudy_dir)
    
    # Initialize handlers
    handler = CloudyOutputHandler(model, time*MYR_TO_SEC, absolute_path=str(cloudy_dir))
    line_gen = LineProfileGenerator()
    
    # Check if Cloudy run exists and was successful
    if not handler.check_cloudy_success():
        raise RuntimeError(f"Cloudy run not successful for {model} at {time} Myr")
        
    print(f"\nProcessing {model} model at t = {time} Myr")
    
    # Get continuum data
    cont_data = handler.get_continuum()
    wave_data = cont_data['nu']  # Wavelengths in microns
    
    # Filter wavelength range
    wave_min, wave_max = 0.01, 3005  # microns
    valid_mask = (wave_data >= wave_min) & (wave_data <= wave_max)
    base_waves = wave_data[valid_mask]
    base_sed = cont_data['observed_no_lines'][valid_mask]  # Use observed continuum
    base_sed /= base_waves
    # Check low resolution grid spacing
    print("\nAnalyzing wavelength grids:")
    rel_spacing_low = abs(base_waves[1:] - base_waves[:-1]) / base_waves[:-1]
    low_res_overlap = np.all(rel_spacing_low >= min_spacing)
    if low_res_overlap:
        print("✓ Low resolution grid spacing OK")
        print(f"  Minimum spacing: {np.min(rel_spacing_low):.2e}")
        print(f"  Maximum spacing: {np.max(rel_spacing_low):.2e}")
    else:
        overlap_idx = np.where(rel_spacing_low < min_spacing)[0]
        print("✗ Found overlapping points in low resolution grid:")
        for idx in overlap_idx:
            print(f"  Points {base_waves[idx]:.6f}, {base_waves[idx+1]:.6f}, "
                  f"relative spacing: {rel_spacing_low[idx]:.2e}")
    
    # Get line data for high resolution grid
    print("\nProcessing line data...")
    line_data = handler.get_line_luminosities(use_emergent=True)
    print(f"Found {len(line_data)} lines")
    # Filter line data to wavelength range
    filtered_line_data = {
        k: v for k, v in line_data.items()
        if wave_min <= k[2] <= wave_max
    }    
    # Create line profiles
    line_profiles = line_gen.create_line_profiles(filtered_line_data, resolution=5e4)
    
    # Generate high resolution SED using updated merge_profiles
    print("\nGenerating high resolution SED...")
    high_res_waves, high_res_sed = line_gen.merge_profiles(base_waves, base_sed, line_profiles)
    
    # Check high resolution grid spacing
    rel_spacing_high = abs(high_res_waves[1:] - high_res_waves[:-1]) / high_res_waves[:-1]
    high_res_overlap = np.all(rel_spacing_high >= min_spacing)
    if high_res_overlap:
        print("\n✓ High resolution grid spacing OK")
        print(f"  Minimum spacing: {np.min(rel_spacing_high):.2e}")
        print(f"  Maximum spacing: {np.max(rel_spacing_high):.2e}")
    else:
        overlap_idx = np.where(rel_spacing_high < min_spacing)[0]
        print("\n✗ Found overlapping points in high resolution grid:")
        for idx in overlap_idx:
            print(f"  Points {high_res_waves[idx]:.6f}, {high_res_waves[idx+1]:.6f}, "
                  f"relative spacing: {rel_spacing_high[idx]:.2e}")
    
    print(f"Number of points in low resolution grid: {len(base_waves)}")
    print(f"Number of points in high resolution grid: {len(high_res_waves)}")
    
    results = {
        'wavelengths_lr': base_waves,
        'wavelengths_hr': high_res_waves,
        'low_res_sed': base_sed,
        'high_res_sed': high_res_sed,
        'low_res_overlap': low_res_overlap,
        'high_res_overlap': high_res_overlap
    }
    
    # Create plots
    plot_sed_comparison(results, save_path=cloudy_dir / f'sed_comparison_{model}_{time}Myr.png')
    plot_wavelength_spacing(results, save_path=cloudy_dir / f'wavelength_spacing_{model}_{time}Myr.png')
    
def plot_sed_comparison(results, save_path=None, figsize=(15, 10)):
    """
    Create a detailed comparison plot of low and high resolution SEDs.
    
    Args:
        results (dict): Dictionary containing wavelength and SED data
        save_path (str|Path, optional): Path to save figure. If None, displays instead.
        figsize (tuple): Figure size in inches
    """
    # Create figure with grid layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.15)

    # Main SED plot
    ax1 = fig.add_subplot(gs[0])
    
    # Plot both SEDs
    ax1.loglog(results['wavelengths_lr'], results['low_res_sed'], 
               'b-', label='Low Resolution', alpha=0.7, linewidth=1.5)
    ax1.loglog(results['wavelengths_hr'], results['high_res_sed'], 
               'r-', label='High Resolution', alpha=0.7, linewidth=1.5)
    
    ax1.set_xlabel('Wavelength (μm)')
    ax1.set_ylabel('Specific luminosity (erg/s/micron)')
    ax1.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.set_title('SED Resolution Comparison', pad=10)
    ax1.set_xlim(0.1, 3000)
    ax1.set_ylim(1e35, 1e44)

    # Residuals plot
    ax2 = fig.add_subplot(gs[1])

    with np.errstate(divide='ignore'):  # Suppress warnings about log(0)
        log_sed = np.log10(np.maximum(results['low_res_sed'], 1e-99))
        
    # Interpolate in log space
    low_res_interp = interp1d(results['wavelengths_lr'], log_sed,
                            kind='linear', bounds_error=False, 
                            fill_value=np.log10(1e-99))

    # Convert back to linear space after interpolation
    low_res_on_high_grid = 10**low_res_interp(results['wavelengths_hr'])
    
    # Calculate relative difference between interpolated low res and high res
    with np.errstate(divide='ignore', invalid='ignore'):  # Handle zero/nan values
        relative_diff = (results['high_res_sed'] / low_res_on_high_grid)
    
    # Plot residuals
    ax2.loglog(results['wavelengths_hr'], relative_diff, 
                 'k-', alpha=0.7, linewidth=1)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Wavelength (μm)')
    ax2.set_ylabel('Relative values \nHigh Res / Low Res')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.set_xlim(0.1, 3000)
    # Add information about wavelength grid checks
    info_text = (
        f"Low resolution grid overlap check: {'✓' if results['low_res_overlap'] else '✗'}\n"
        f"High resolution grid overlap check: {'✓' if results['high_res_overlap'] else '✗'}"
    )
    plt.figtext(0.02, 0.02, info_text, fontsize=8, ha='left')
    
    # Save or display
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved SED comparison plot to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_wavelength_spacing(results, save_path=None, figsize=(12, 6)):
    """
    Create a plot showing the wavelength spacing distribution.
    
    Args:
        results (dict): Dictionary containing wavelength data
        save_path (str|Path, optional): Path to save figure. If None, displays instead.
        figsize (tuple): Figure size in inches
    """
    waves = results['wavelengths_hr']
    rel_spacing = (waves[1:] - waves[:-1]) / waves[:-1]
    
    plt.figure(figsize=figsize)
    
    # Plot relative spacing
    plt.loglog(waves[:-1], rel_spacing, 'k.', alpha=0.01, 
                 label='Point-to-point spacing')    
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Relative spacing (Δλ/λ)')
    plt.title('Wavelength Grid Spacing Distribution')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlim(0.1, 3000)
    plt.legend()
    
    # Save or display
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved wavelength spacing plot to {save_path}")
        plt.close()
    else:
        plt.show()

# Example usage:
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python test_sed_res.py <cloudy_output_dir> [time_Myr] [model]\n"
                 "  <cloudy_output_dir>: a Cloudy model output directory "
                 "(e.g. .../Z..._eta..._n..._logM.../)")
    cloudy_dir = sys.argv[1]
    time = float(sys.argv[2]) if len(sys.argv) > 2 else 0.85
    model = sys.argv[3] if len(sys.argv) > 3 else 'unified'
    test_sed_resolution(cloudy_dir, time=time, model=model)