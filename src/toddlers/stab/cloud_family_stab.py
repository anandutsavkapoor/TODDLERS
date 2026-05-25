import numpy as np
from pathlib import Path
import logging
import pickle
from enum import Enum
from typing import Optional, Set
import matplotlib.pyplot as plt
from .config import *
from ..pts import storedtable as stab
import sys
import scipy.interpolate
sys.modules['scipy.interpolate._interpolate'] = scipy.interpolate
import multiprocessing as mp
from functools import partial
import argparse

class SEDType(Enum):
    DUST = "Dust"
    NODUST = "noDust"

class Resolution(Enum):
    LOW = "lr"
    HIGH = "hr"

class TODDLERSTimeSeriesStabGenerator:
    """Generates STAB files for SKIRT from TODDLERS SED data, without SFR normalization."""
    
    def __init__(self, out_base_dir: Path = None, num_time_steps: int = None, dust_to_metal: float = 1.0):
        """
        Initialize the generator.
        
        Args:
            out_base_dir: Base output directory (defaults to current directory)
            num_time_steps: Number of time steps to include in the output (default: None = use all time steps)
        """
        self.dust_to_metal = dust_to_metal
        self.out_base_dir = Path(out_base_dir) if out_base_dir else Path("")
        self.outdir_stab = self.out_base_dir / "cloud_family_stab_output"
        self.outdir_stab.mkdir(parents=True, exist_ok=True)
        
        # Number of time steps to sample (None means use all time steps from interpolator)
        self.num_time_steps = num_time_steps
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Parameter space definition from names_and_constants.py
        # These are not used for grid definition, but kept for reference and potential filtering
        self.Z_ref = METALLICITIES
        self.etaSF_ref = STAR_FORMATION_EFFICIENCIES
        self.n_cl_ref = CLOUD_DENSITIES
        
        # Fixed model parameters (from names_and_constants.py)
        self.stellar_template = STELLAR_TEMPLATE
        self.imf = IMF_TYPE
        self.star_type = STAR_TYPE
        
        # For sed_unit conversion
        self.sed_unit = SED_UNIT
        self.wavelength_unit = WAVELENGTH_UNIT
        
        self.logger.info(f"Initialized generator with:")
        self.logger.info(f"  - Model: {self.stellar_template}_{self.imf}_{self.star_type}")
        self.logger.info(f"  - SED unit: {self.sed_unit}")
        self.logger.info(f"  - Wavelength unit: {self.wavelength_unit}")
        self.logger.info(f"  - Number of time steps: {self.num_time_steps}")
        
    def get_interpolator_filename(self, sed_type: SEDType, resolution: Resolution) -> Path:
        """
        Get the appropriate interpolator file path.
        
        Args:
            sed_type: Type of SED (DUST or NODUST)
            resolution: Resolution (LOW or HIGH)
            
        Returns:
            Path to the interpolator file
        """
        model_prefix = f"{self.stellar_template}_{self.imf}_{self.star_type}"
        _INTERPOLATOR_BASE = f"{model_prefix}_interp_tables"
        
        dtm_suffix = f"_dtm{self.dust_to_metal:.2f}" if self.dust_to_metal != 1.0 else ""

        if resolution == Resolution.HIGH:
            return Path(_INTERPOLATOR_BASE) / f"TODDLERS_tot_hr_{model_prefix}_lines_emergent={'False' if sed_type == SEDType.NODUST else 'True'}{dtm_suffix}.pkl"
        else:
            if sed_type == SEDType.NODUST:
                return Path(_INTERPOLATOR_BASE) / f"TODDLERS_inciSED_lr_{model_prefix}{dtm_suffix}.pkl"
            else:
                return Path(_INTERPOLATOR_BASE) / f"TODDLERS_totSED_lr_{model_prefix}{dtm_suffix}.pkl"
                
    def load_interpolator(self, sed_type: SEDType, resolution: Resolution):
        """
        Load the SED interpolator from file.
        
        Args:
            sed_type: Type of SED (DUST or NODUST)
            resolution: Resolution (LOW or HIGH)
            
        Returns:
            Loaded interpolator object
        """
        interpolator_file = self.get_interpolator_filename(sed_type, resolution)
        
        self.logger.info(f"Loading interpolator from {interpolator_file}")
        
        try:
            with open(interpolator_file, 'rb') as f:
                interpolator = pickle.load(f)
            return interpolator
        except FileNotFoundError:
            self.logger.error(f"Interpolator file not found: {interpolator_file}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading interpolator: {str(e)}")
            raise
            
    def get_stab_filename(self, sed_type: SEDType, resolution: Resolution) -> Path:
        """Construct STAB filename based on parameters."""
        model_str = f"ToddlersCloudSEDFamily_{self.stellar_template}_{self.imf}_{self.star_type}"
        if sed_type == SEDType.NODUST:
            model_str += "_noDust"
        if self.dust_to_metal != 1.0:
            model_str += f"_dtm{self.dust_to_metal:.2f}"
        return self.outdir_stab / f"{model_str}_{resolution.value}.stab"
        
    def extract_wavelength_grid(self, interpolator) -> np.ndarray:
        """
        Extract wavelength grid directly from interpolator to ensure correct bounds.
        
        Args:
            interpolator: The loaded interpolator
            
        Returns:
            Wavelength grid in meters
        """
        # Get the log wavelength grid directly from the interpolator
        log_wl_grid = interpolator.grid[0]
        
        # Instead of generating our own grid, use exactly the grid points from the interpolator
        # This ensures we stay within the valid bounds
        wl_grid_linear = 10**log_wl_grid  # Convert from log to linear
        
        # Apply the correct unit conversion if needed
        if self.wavelength_unit.lower() in ('micron', 'μm', 'um'):
            # If the interpolator uses microns, convert to meters for SKIRT
            wl_grid_meters = wl_grid_linear * 1e-6  # Convert microns to meters
            self.logger.info(f"Converting wavelength grid from microns to meters")
        else:
            # If the interpolator already uses meters, no conversion needed
            wl_grid_meters = wl_grid_linear
            
        self.logger.info(f"Extracted wavelength grid with {len(wl_grid_meters)} points")
        self.logger.info(f"Wavelength range: {wl_grid_meters.min():.3e} to {wl_grid_meters.max():.3e} meters")
        
        return wl_grid_meters
        
    def extract_time_grid(self, interpolator, num_points=None) -> np.ndarray:
        """
        Extract time grid from interpolator.
        
        Args:
            interpolator: The loaded interpolator
            num_points: Number of time points to include (if None, use all points)
            
        Returns:
            Time grid in Myr
        """
        # Get the time grid directly from the interpolator
        original_time_grid = interpolator.grid[1]
        
        # If requested to use all points, return the entire grid
        if num_points is None or num_points >= len(original_time_grid):
            self.logger.info(f"Using full time grid from interpolator: {len(original_time_grid)} points from {original_time_grid[0]:.2f} to {original_time_grid[-1]:.2f} Myr")
            return original_time_grid
            
        # Otherwise, create linearly spaced points, but ensure extremes match interpolator grid
        if num_points > 2:
            # Create linearly spaced interior points
            t_min = original_time_grid[0]
            t_max = original_time_grid[-1]
            
            # Create num_points-2 interior points linearly spaced between min and max
            interior_points = np.linspace(t_min, t_max, num_points)[1:-1]
            
            # Ensure first and last points are exactly from the interpolator grid
            sampled_time_grid = np.concatenate(([t_min], interior_points, [t_max]))
            
        elif num_points == 2:
            # Just first and last points
            sampled_time_grid = np.array([original_time_grid[0], original_time_grid[-1]])
        else:
            # Just the first point if only one point requested
            sampled_time_grid = np.array([original_time_grid[0]])
        
        self.logger.info(f"Created time grid with {len(sampled_time_grid)} points")
        self.logger.info(f"Time range: {sampled_time_grid[0]:.2f} to {sampled_time_grid[-1]:.2f} Myr")
        
        return sampled_time_grid

    def handle_invalid_values(self, log_sed, parameter_info=None):
        """
        Handle invalid values (NaN, Inf) in the interpolation results.
        
        Args:
            log_sed: Array of log SED values
            parameter_info: Optional string with parameter information for logging
            
        Returns:
            Array with invalid values replaced by a small number
        """
        # Check for NaNs or Inf values
        nan_mask = np.isnan(log_sed)
        inf_mask = np.isinf(log_sed)
        
        if np.any(nan_mask) or np.any(inf_mask):
            num_invalid = np.sum(nan_mask) + np.sum(inf_mask)
            
            # Log the issue
            msg = f"Found {num_invalid} invalid values in interpolation result"
            if parameter_info:
                msg += f" for {parameter_info}"
            self.logger.warning(msg)
            
            # Replace NaNs and Infs with a very small value in log space
            replacement_value = -300.0  # Very small in log space, essentially zero in linear space
            log_sed_fixed = log_sed.copy()
            log_sed_fixed[nan_mask | inf_mask] = replacement_value
            return log_sed_fixed
        
        return log_sed

    def create_sed_grid(self, interpolator, time_grid):
        """
        Create SED grid for all parameter combinations without SFR normalization.
        
        Args:
            interpolator: The loaded interpolator
            time_grid: Array of time values to compute SEDs for
            
        Returns:
            Dictionary with grid information and SED values
        """
        # Extract grids directly from interpolator, but in meters for stab
        wl_grid_meters = self.extract_wavelength_grid(interpolator)
        
        # Extract Z, eta, n_cl, and M_cl grids directly from interpolator
        log_wl_grid = interpolator.grid[0]
        Z_grid      = 10**interpolator.grid[2]  # Convert from log10 to linear
        etaSF_grid  = interpolator.grid[3]  # Already linear
        n_cl_grid   = 10**interpolator.grid[4]  # Convert from log10 to linear
        M_cl_grid   = 10**interpolator.grid[5]  # Convert from log10 to linear
        
        self.logger.info(f"Using grid values directly from interpolator:")
        self.logger.info(f"  Z: {len(Z_grid)} values from {Z_grid.min():.3e} to {Z_grid.max():.3e}")
        self.logger.info(f"  etaSF: {len(etaSF_grid)} values from {etaSF_grid.min():.3f} to {etaSF_grid.max():.3f}")
        self.logger.info(f"  n_cl: {len(n_cl_grid)} values from {n_cl_grid.min():.1f} to {n_cl_grid.max():.1f}")
        self.logger.info(f"  M_cl: {len(M_cl_grid)} values from {M_cl_grid.min():.1e} to {M_cl_grid.max():.1e}")
        
        # Create array to store SED values
        # Shape: (wavelengths, times, Z, etaSF, n_cl, M_cl)
        sed_shape = (len(wl_grid_meters), len(time_grid), len(Z_grid), 
                    len(etaSF_grid), len(n_cl_grid), len(M_cl_grid))
        
        self.logger.info(f"Creating SED grid with shape {sed_shape}")
        sed_values = np.zeros(sed_shape)
        
        # Get grid bounds from interpolator for parameter validation
        log_wl_bounds = (interpolator.grid[0].min(), interpolator.grid[0].max())
        time_bounds   = (interpolator.grid[1].min(), interpolator.grid[1].max())
        log_Z_bounds  = (interpolator.grid[2].min(), interpolator.grid[2].max())
        etaSF_bounds  = (interpolator.grid[3].min(), interpolator.grid[3].max())
        log_n_bounds  = (interpolator.grid[4].min(), interpolator.grid[4].max())
        log_M_bounds  = (interpolator.grid[5].min(), interpolator.grid[5].max())
        
        # Log grid bounds
        self.logger.info("Interpolator grid bounds:")
        self.logger.info(f"  log_wavelength: {log_wl_bounds}")
        self.logger.info(f"  time: {time_bounds}")
        self.logger.info(f"  log_Z: {log_Z_bounds} ({10**log_Z_bounds[0]:.3e} to {10**log_Z_bounds[1]:.3e})")
        self.logger.info(f"  etaSF: {etaSF_bounds}")
        self.logger.info(f"  log_n: {log_n_bounds} ({10**log_n_bounds[0]:.1f} to {10**log_n_bounds[1]:.1f})")
        self.logger.info(f"  log_M: {log_M_bounds} ({10**log_M_bounds[0]:.1e} to {10**log_M_bounds[1]:.1e})")
                
        # Set up a tracking counter
        total_combinations = len(Z_grid) * len(etaSF_grid) * len(n_cl_grid) * len(M_cl_grid)
        processed = 0
        
        # Now compute SEDs for each parameter combination
        # Since we're using the exact grid points from the interpolator,
        # we don't need to clamp values - they're already valid
        for i, Z in enumerate(Z_grid):
            log_Z = np.log10(Z)
            
            for j, eta in enumerate(etaSF_grid):
                for k, n in enumerate(n_cl_grid):
                    log_n = np.log10(n)
                    
                    for m, M in enumerate(M_cl_grid):
                        log_M = np.log10(M)
                        
                        # Create input arrays for interpolator for all wavelengths and times
                        inputs = []
                        for t in time_grid:
                            # Create 2D grid for wavelength × time point
                            wl_grid_2d = np.column_stack([
                                log_wl_grid,  # log wavelength
                                np.full_like(log_wl_grid, t),  # time
                                np.full_like(log_wl_grid, log_Z),  # log Z
                                np.full_like(log_wl_grid, eta),  # etaSF
                                np.full_like(log_wl_grid, log_n),  # log n
                                np.full_like(log_wl_grid, log_M)  # log M
                            ])
                            inputs.append(wl_grid_2d)
                        
                        # Stack all time points
                        all_inputs = np.vstack(inputs)
                        
                        # Interpolate for all wavelengths and times in one call
                        log_sed = interpolator(all_inputs)

                        # Check for NaNs or Inf values
                        parameter_info = f"Z={Z:.3e}, eta={eta:.3f}, n={n:.1f}, M={M:.1e}"
                        log_sed = self.handle_invalid_values(log_sed, parameter_info)

                        # Convert to linear space
                        sed = 10**log_sed
                        
                        # Reshape to (n_times, n_wavelengths)
                        sed = sed.reshape(len(time_grid), len(wl_grid_meters))
                        
                        # Transpose to get (n_wavelengths, n_times)
                        sed = sed.T
                        
                        # Store in sed_values array
                        sed_values[:, :, i, j, k, m] = sed
                        
                        # Update progress
                        processed += 1
                        if processed % 10 == 0 or processed == total_combinations:
                            progress = 100.0 * processed / total_combinations
                            self.logger.info(f"Progress: {processed}/{total_combinations} ({progress:.1f}%)")
        
        # Apply unit conversion if needed
        if self.sed_unit == 'erg/s/micron':
            # Convert erg/s/micron to W/m
            # 1 erg/s/micron = 0.1 W/m (1e-7 W·s/erg * 1e6 micron/m)
            sed_values *= 0.1
            self.logger.info("Applied unit conversion: erg/s/micron to W/m")
        elif self.sed_unit == 'erg/s/Angstrom':
            # Convert erg/s/Angstrom to W/m
            # 1 erg/s/Angstrom = 1000 W/m (1e-7 W·s/erg * 1e10 Angstrom/m)
            sed_values *= 1000.0
            self.logger.info("Applied unit conversion: erg/s/Angstrom to W/m")
        # W/m needs no conversion
        
        # Return grid and values
        return {
            'wavelength_grid': wl_grid_meters,
            'time_grid': time_grid,
            'Z_grid': Z_grid,
            'etaSF_grid': etaSF_grid,
            'n_cl_grid': n_cl_grid,
            'M_cl_grid': M_cl_grid,
            'sed_values': sed_values
        }
        
    def plot_sample_seds(self, grid_data, sed_type, resolution, idx_Z=0, idx_eta=2, idx_n=2, idx_M=3):
        """
        Create a plot of sample SEDs for different time steps.
        
        Args:
            grid_data: SED grid data from create_sed_grid()
            sed_type: Type of SED (DUST or NODUST)
            resolution: Resolution (LOW or HIGH)
            idx_Z, idx_eta, idx_n, idx_M: Indices to plot
        """
        # Extract data
        wl_grid = grid_data['wavelength_grid']
        time_grid = grid_data['time_grid']
        # Clamp the sample indices to the available grid: the defaults assume a large
        # production grid, but the grid may be smaller (e.g. a minimal test grid).
        idx_Z = min(idx_Z, len(grid_data['Z_grid']) - 1)
        idx_eta = min(idx_eta, len(grid_data['etaSF_grid']) - 1)
        idx_n = min(idx_n, len(grid_data['n_cl_grid']) - 1)
        idx_M = min(idx_M, len(grid_data['M_cl_grid']) - 1)
        Z = grid_data['Z_grid'][idx_Z]
        eta = grid_data['etaSF_grid'][idx_eta]
        n = grid_data['n_cl_grid'][idx_n]
        M = grid_data['M_cl_grid'][idx_M]
        sed_values = grid_data['sed_values']
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot SEDs for different time steps
        # Use log spacing for time indices to show evolution clearly
        num_time_points = min(10, len(time_grid))
        time_indices = np.unique(np.round(np.logspace(
            0, np.log10(len(time_grid)-1), num_time_points
        )).astype(int))
        
        cmap = plt.cm.viridis
        colors = [cmap(i/len(time_indices)) for i in range(len(time_indices))]
        
        for i, time_idx in enumerate(time_indices):
            t = time_grid[time_idx]
            sed = sed_values[:, time_idx, idx_Z, idx_eta, idx_n, idx_M]
            
            # Plot λ*Lλ vs λ
            plt.loglog(wl_grid*1e6, sed*wl_grid, label=f't = {t:.1f} Myr', color=colors[i])
        
        plt.xlabel('Wavelength (μm)')
        plt.ylabel('λ * Lλ (W)')
        plt.title(f'Time Evolution: Z={Z:.3f}, η={eta:.3f}, n={n:.1f} cm⁻³, M={M:.1e} M☉')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend(loc='best')
        plt.xlim(0.1, 3000)
        plt.ylim(1e-2)
        
        # Save figure
        dust_str = 'noDust' if sed_type == SEDType.NODUST else 'Dust'
        res_str = resolution.value
        fig_name = f'sample_seds_{self.stellar_template}_{self.imf}_{self.star_type}_{dust_str}_{res_str}.png'
        fig_path = self.outdir_stab / fig_name
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Sample SED plot saved to {fig_path}")
    
    def write_stab_file(self, grid_data, sed_type, resolution):
        """
        Write the SED grid to a STAB file.
        
        Args:
            grid_data: SED grid data from create_sed_grid()
            sed_type: Type of SED (DUST or NODUST)
            resolution: Resolution (LOW or HIGH)
        """
        # Extract data
        wl_grid = grid_data['wavelength_grid']
        time_grid = grid_data['time_grid']
        Z_grid = grid_data['Z_grid']
        etaSF_grid = grid_data['etaSF_grid']
        n_cl_grid = grid_data['n_cl_grid']
        M_cl_grid = grid_data['M_cl_grid']
        sed_values = grid_data['sed_values']
        
        # Get STAB filename
        stab_file = self.get_stab_filename(sed_type, resolution)
        
        self.logger.info(f"Writing STAB file: {stab_file}")
        self.logger.info(f"Dimensions: wavelength({len(wl_grid)}) × time({len(time_grid)}) × "
                        f"Z({len(Z_grid)}) × etaSF({len(etaSF_grid)}) × "
                        f"n({len(n_cl_grid)}) × M({len(M_cl_grid)})")
        
        try:
            # Write STAB file
            stab.writeStoredTable(
                str(stab_file),
                ['lambda', 'time', 'Z', 'SFE', 'n_cl', 'M_cl'],
                ['m', 'Myr', '1', '1', '1/cm3', 'Msun'],
                ['log', 'lin', 'log', 'lin', 'log', 'log'],
                [wl_grid, time_grid, Z_grid, etaSF_grid, n_cl_grid, M_cl_grid],
                ['Llambda'],
                ['W/m'],
                ['log'],
                [sed_values]
            )
            self.logger.info(f"Successfully wrote STAB file: {stab_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error writing STAB file: {str(e)}")
            return False
        

def process_combination(generator, sed_type, resolution):
    """Process a single combination of SED type and resolution."""
    try:
        generator.logger.info(f"Processing {sed_type.value} {resolution.value} SEDs...")
        
        # Load interpolator
        interpolator = generator.load_interpolator(sed_type, resolution)
        
        # Extract time grid
        time_grid = generator.extract_time_grid(interpolator, generator.num_time_steps)
        
        # Create SED grid
        grid_data = generator.create_sed_grid(interpolator, time_grid)
        
        # Create sample plot (diagnostic only; must never block the STAB write)
        try:
            generator.plot_sample_seds(grid_data, sed_type, resolution)
        except Exception as e:
            generator.logger.warning(
                f"Sample-SED diagnostic plot skipped for {sed_type.value} {resolution.value}: {e}"
            )

        # Write STAB file
        success = generator.write_stab_file(grid_data, sed_type, resolution)
        
        if success:
            generator.logger.info(f"Successfully processed {sed_type.value} {resolution.value} SEDs")
            return True
        else:
            generator.logger.error(f"Failed to process {sed_type.value} {resolution.value} SEDs")
            return False
            
    except Exception as e:
        generator.logger.error(f"Error processing {sed_type.value} {resolution.value} SEDs: {str(e)}")
        import traceback
        generator.logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate STAB files for TODDLERS time-series SEDs")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--time-steps", type=int, default=None, 
                       help="Number of time steps to include (default: use all time steps)")
    parser.add_argument("--sed-types", choices=["dust", "nodust", "both"], default="both", 
                       help="Which SED types to process")
    parser.add_argument("--resolutions", choices=["high", "low", "both"], default="both",
                       help="Which resolutions to process")
    parser.add_argument("--processes", type=int, default=None,
                       help="Number of parallel processes to use (default: number of CPU cores)")
    
    args = parser.parse_args()
    
    # Set up SED types and resolutions based on command-line arguments
    sed_types = set()
    if args.sed_types in ["dust", "both"]:
        sed_types.add(SEDType.DUST)
    if args.sed_types in ["nodust", "both"]:
        sed_types.add(SEDType.NODUST)
        
    resolutions = set()
    if args.resolutions in ["high", "both"]:
        resolutions.add(Resolution.HIGH)
    if args.resolutions in ["low", "both"]:
        resolutions.add(Resolution.LOW)
    
    # Create generator
    generator = TODDLERSTimeSeriesStabGenerator(
        out_base_dir=Path(args.output_dir),
        num_time_steps=args.time_steps
    )
    
    # Create combinations of SED types and resolutions
    combinations = [(sed_type, resolution) for sed_type in sed_types for resolution in resolutions]
    
    # Use default number of processes if not specified
    num_processes = args.processes or mp.cpu_count()
    
    # Limit number of processes to number of combinations
    num_processes = min(num_processes, len(combinations))
    
    # Log parallelization info
    generator.logger.info(f"Processing {len(combinations)} combinations using {num_processes} parallel processes")
    
    # Create partial function with generator fixed
    process_func = partial(process_combination, generator)
    
    # Run processing in parallel
    if num_processes > 1:
        with mp.Pool(processes=num_processes) as pool:
            results = pool.starmap(process_func, combinations)
            
        # Log results
        successes = sum(1 for result in results if result)
        generator.logger.info(f"Completed {successes} of {len(combinations)} combinations successfully")
    else:
        # Run sequentially if only using 1 process
        generator.logger.info("Running in single-process mode")
        results = []
        for combination in combinations:
            results.append(process_func(*combination))
        
        # Log results
        successes = sum(1 for result in results if result)
        generator.logger.info(f"Completed {successes} of {len(combinations)} combinations successfully")