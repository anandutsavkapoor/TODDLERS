import numpy as np
from pathlib import Path
import struct
from typing import Dict, List, Tuple, Union, Optional
import logging
import matplotlib.pyplot as plt

def plot_sed_slice(stab_data: dict, 
                  axis_indices: dict,
                  show_wavelength_in_microns: bool = True,
                  show_lambda_times_L: bool = True,
                  ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot SED for a specific slice of parameter values.
    
    Args:
        stab_data: Dictionary containing STAB file data
        axis_indices: Dictionary of axis name to index for slicing
                     e.g. {"Z": 2, "SFE": 1, "n_cl": 3}
        show_wavelength_in_microns: If True, convert wavelength to microns
        show_lambda_times_L: If True, multiply luminosity by wavelength
        ax: Optional matplotlib axes for plotting. If None, creates new figure
        
    Returns:
        matplotlib.pyplot.Axes object
    """
    axis_names = stab_data["axisNames"]
    
    # Build the slice tuple
    slice_list = [0]  # First quantity
    for name in axis_names:
        if name == 'lambda':
            slice_list.append(slice(None))  # Take all wavelengths
        elif name in axis_indices:
            slice_list.append(axis_indices[name])
        else:
            raise ValueError(f"Missing index for axis {name}")
    
    # Get wavelength grid and SED values
    wavelength_idx = axis_names.index('lambda')
    wavelengths = stab_data['axisGrids'][wavelength_idx]
    sed_values = stab_data['values'][tuple(slice_list)]
    
    # Convert wavelength to microns if requested
    if show_wavelength_in_microns:
        wavelengths = wavelengths * 1e6
        wavelength_unit = 'μm'
    else:
        wavelength_unit = 'm'
    
    # Multiply by wavelength if requested
    if show_lambda_times_L:
        if show_wavelength_in_microns:
            sed_values = sed_values * (wavelengths * 1e-6)  # Convert back to meters for calculation
        else:
            sed_values = sed_values * wavelengths
    
    # Create or get axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the SED
    ax.loglog(wavelengths, sed_values, 'b-', linewidth=1.5)
    
    # Add labels
    ax.set_xlabel(f'Wavelength ({wavelength_unit})')
    y_label = 'λLλ' if show_lambda_times_L else 'Lλ'
    y_label += f' ({stab_data["quantityUnits"][0]})'
    ax.set_ylabel(y_label)
    
    # Create title showing parameter values
    title_parts = []
    for name, idx in axis_indices.items():
        if name != 'lambda':
            grid_value = stab_data["axisGrids"][axis_names.index(name)][idx]
            unit = stab_data["axisUnits"][axis_names.index(name)]
            title_parts.append(f"{name}={grid_value:.3e} {unit}")
    
    ax.set_title('SED for ' + ', '.join(title_parts))
    
    # Add grid
    ax.grid(True, which='both', ls='-', alpha=0.2)
    
    return ax

def plot_sed_parameter_comparison(stab_data: dict,
                                vary_parameter: str,
                                fixed_parameters: dict,
                                show_wavelength_in_microns: bool = True,
                                show_lambda_times_L: bool = True) -> None:
    """
    Plot multiple SEDs varying one parameter while keeping others fixed.
    
    Args:
        stab_data: Dictionary containing STAB file data
        vary_parameter: Name of parameter to vary (e.g., "Z", "SFE", "n_cl")
        fixed_parameters: Dictionary of fixed parameter indices
        show_wavelength_in_microns: If True, convert wavelength to microns
        show_lambda_times_L: If True, multiply luminosity by wavelength
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get the grid for the varying parameter
    param_idx = stab_data["axisNames"].index(vary_parameter)
    param_grid = stab_data["axisGrids"][param_idx]
    param_unit = stab_data["axisUnits"][param_idx]
    
    # Plot SED for each value of the varying parameter
    colors = plt.cm.viridis(np.linspace(0, 1, len(param_grid)))
    
    for i, param_value in enumerate(param_grid):
        # Create parameter dictionary for this plot
        params = fixed_parameters.copy()
        params[vary_parameter] = i
        
        # Plot the SED
        ax = plot_sed_slice(stab_data, params, 
                          show_wavelength_in_microns=show_wavelength_in_microns,
                          show_lambda_times_L=show_lambda_times_L,
                          ax=ax)
        
        # Update line color and add to legend
        ax.get_lines()[-1].set_color(colors[i])
        ax.get_lines()[-1].set_label(f'{vary_parameter}={param_value:.3e} {param_unit}')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent legend overlap
    plt.tight_layout()


def analyze_stab_slice(stab_data: dict, axis_indices: dict = None) -> None:
    """
    Analyze a specific slice of SKIRT .stab data.
    
    Args:
        stab_data: Dictionary returned by read_stab_file()
        axis_indices: Dictionary of axis name to index for slicing
                     e.g. {"Z": 2, "SFE": 1}
    """
    values = stab_data["values"]
    axis_names = stab_data["axisNames"]
    
    print(f"\nData Structure Analysis:")
    print("=====================")
    
    # Show overall shape
    print(f"\nOverall shape: {values.shape}")
    print(f"Dimensions:")
    print(f"- Dimension 0: {values.shape[0]} quantities")
    for i, (name, size) in enumerate(zip(axis_names, values.shape[1:]), 1):
        print(f"- Dimension {i}: {name} ({size} points)")
    
    # If specific slice requested
    if axis_indices:
        # Build the slice tuple
        slice_list = [0]  # First quantity
        for name in axis_names:
            if name in axis_indices:
                slice_list.append(axis_indices[name])
            else:
                slice_list.append(slice(None))
        
        # Get the slice
        data_slice = values[tuple(slice_list)]
        
        print(f"\nRequested Slice:")
        print("- Fixed dimensions:")
        for name, idx in axis_indices.items():
            grid_value = stab_data["axisGrids"][axis_names.index(name)][idx]
            print(f"  {name} = {grid_value:.3e} {stab_data['axisUnits'][axis_names.index(name)]}")
        
        print(f"- Slice shape: {data_slice.shape}")
        non_zero_mask = data_slice != 0
        if np.any(non_zero_mask):
            non_zero_values = data_slice[non_zero_mask]
            print(f"- Value range: {np.min(non_zero_values):.3e} to {np.max(non_zero_values):.3e}")
            print(f"- Mean (non-zero): {np.mean(non_zero_values):.3e}")
            print(f"- Non-zero elements: {np.count_nonzero(data_slice):,} "
                  f"({np.count_nonzero(data_slice)/data_slice.size*100:.1f}% of slice)")

def read_stab_file(filepath: Union[str, Path]) -> Dict:
    """
    Read a SKIRT .stab binary file and return its contents.
    
    Args:
        filepath: Path to the .stab file
        
    Returns:
        Dictionary containing:
        - axisNames: List of axis names
        - axisUnits: List of corresponding units
        - axisScales: List of corresponding scales (lin/log)
        - axisGrids: List of numpy arrays with grid points for each axis
        - quantityNames: List of quantity names
        - quantityUnits: List of corresponding units
        - quantityScales: List of corresponding scales
        - values: Numpy array containing the quantity values
        
    Raises:
        ValueError: If file format is invalid
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)
    
    def read_string(file) -> str:
        """Read 8-byte string and strip whitespace."""
        return str(file.read(8).strip(), 'utf8')
    
    def read_int(file) -> int:
        """Read 8-byte little-endian unsigned integer."""
        return struct.unpack('<Q', file.read(8))[0]
    
    def read_array(file, shape: Tuple) -> np.ndarray:
        """Read array of doubles with given shape."""
        count = np.prod(shape)
        arr = np.fromfile(file, dtype='<f8', count=count)
        return np.reshape(arr, tuple(reversed(shape))).T
    
    try:
        with open(filepath, 'rb') as f:
            # Verify header
            if read_string(f) != "SKIRT X" or read_int(f) != 0x010203040A0BFEFF:
                raise ValueError("Invalid SKIRT stored table format")
            
            # Read axes information
            num_axes = read_int(f)
            axis_names = [read_string(f) for _ in range(num_axes)]
            axis_units = [read_string(f) for _ in range(num_axes)]
            axis_scales = [read_string(f) for _ in range(num_axes)]
            
            # Read grid points for each axis
            axis_grids = []
            grid_points = []
            for _ in range(num_axes):
                num_points = read_int(f)
                grid_points.append(num_points)
                axis_grids.append(read_array(f, (num_points,)))
            
            # Read quantities information
            num_quantities = read_int(f)
            quantity_names = [read_string(f) for _ in range(num_quantities)]
            quantity_units = [read_string(f) for _ in range(num_quantities)]
            quantity_scales = [read_string(f) for _ in range(num_quantities)]
            
            # Read values
            shape = tuple([num_quantities] + grid_points)
            values = read_array(f, shape)
            
            # Verify EOF tag
            if read_string(f) != "STABEND":
                raise ValueError("Missing or invalid end-of-file tag")
            
            return {
                "axisNames": axis_names,
                "axisUnits": axis_units,
                "axisScales": axis_scales,
                "axisGrids": axis_grids,
                "quantityNames": quantity_names,
                "quantityUnits": quantity_units,
                "quantityScales": quantity_scales,
                "values": values
            }
            
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {filepath}")
    except Exception as e:
        raise ValueError(f"Error reading .stab file: {str(e)}")

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        sys.exit("usage: python read_stab.py <file.stab>")

    try:
        stab_data = read_stab_file(sys.argv[1])
        
        print("\nFile Contents Summary:")
        print("=====================")
        
        # Print axis information with ranges
        print(f"\nAxes ({len(stab_data['axisNames'])}):")
        for i, (name, unit, scale, grid) in enumerate(zip(
            stab_data['axisNames'], 
            stab_data['axisUnits'], 
            stab_data['axisScales'],
            stab_data['axisGrids'])):
            print(f"  {i+1}. {name} ({unit}) [{scale}]")
            print(f"     - Points: {len(grid)}")
            print(f"     - Range: {grid[0]:.3e} to {grid[-1]:.3e}")
        
        # Print quantity information
        print(f"\nQuantities ({len(stab_data['quantityNames'])}):")
        values = stab_data['values']
        
        for i, (name, unit, scale) in enumerate(zip(
            stab_data['quantityNames'],
            stab_data['quantityUnits'],
            stab_data['quantityScales'])):
            
            quantity_values = values[i]
            
            # Calculate statistics for non-zero values to handle sparse data
            non_zero_mask = quantity_values != 0
            if np.any(non_zero_mask):
                non_zero_values = quantity_values[non_zero_mask]
                value_min = np.min(non_zero_values)
                value_max = np.max(non_zero_values)
                value_mean = np.mean(non_zero_values)
                value_median = np.median(non_zero_values)
            else:
                value_min = value_max = value_mean = value_median = 0
            
            print(f"  {i+1}. {name} ({unit}) [{scale}]")
            print(f"     - Overall range: {value_min:.3e} to {value_max:.3e}")
            print(f"     - Mean (non-zero): {value_mean:.3e}")
            print(f"     - Median (non-zero): {value_median:.3e}")
            print(f"     - Zero values: {np.sum(quantity_values == 0)} "
                  f"({np.mean(quantity_values == 0)*100:.1f}% of total)")
            
            # If the quantity has a log scale, also show log range
            if scale == 'log' and value_max > 0:
                log_min = np.log10(max(value_min, np.finfo(float).tiny))
                log_max = np.log10(value_max)
                print(f"     - Log10 range: {log_min:.2f} to {log_max:.2f}")
        
        # Print array information
        print(f"\nArray Information:")
        print(f"  Shape: {values.shape}")
        print(f"  Memory usage: {values.nbytes / 1e6:.1f} MB")
        print(f"  Total elements: {values.size:,}")
        print(f"  Non-zero elements: {np.count_nonzero(values):,} "
              f"({np.count_nonzero(values)/values.size*100:.1f}% of total)")
        
        # check a slice
        analyze_stab_slice(stab_data, {"Z": 0, "SFE": 1})
        
        # Plot single SED
        plot_sed_slice(stab_data, {"Z": 0, "SFE": 1, "n_cl": 3})
        plt.savefig('single_sed.pdf')
        plt.close()
        
        # Plot parameter comparison
        plot_sed_parameter_comparison(
            stab_data,
            vary_parameter="Z",
            fixed_parameters={"SFE": 4, "n_cl": 3}
        )
        plt.savefig('sed_metallicity_comparison.pdf')
        plt.close()

    except Exception as e:
        logging.error(f"Failed to read .stab file: {str(e)}")