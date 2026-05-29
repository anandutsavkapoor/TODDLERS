"""
TODDLERS Interpolant Generator

This module generates time-dependent interpolants for TODDLERS evolution and Cloudy 
post-processing by combining data from evolution outputs and Cloudy outputs.

Capabilities:
- Automatic parameter space extraction from file structure
- Handles both low and high resolution SEDs
- Support for dust and no-dust spectral processing
- Property interpolant generation
"""

import os
import sys
import pickle
from pathlib import Path
import h5py
import datetime
import traceback
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from toddlers.track_simulation import load_output_file
from toddlers.cloudy_timegrid_generator import TimeGridGenerator
from toddlers.cloudy_output_handler import CloudyOutputHandler
from toddlers.constants import *
from toddlers.utils import dtm_label
import argparse
from .line_profiles import LineProfileGenerator
from .config import INCLUDE_NEBULAR_CONTINUUM

def handle_error(func):
    """
    Decorator for error handling with detailed tracebacks.
    Adds error context and preserves the full traceback.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"\nError in {func.__name__}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nFull traceback:")
            traceback.print_exc()
            raise
    return wrapper


class TODDLERSInterpolantGenerator:
    """
    Generates interpolants for TODDLERS by combining data from evolution outputs
    and Cloudy outputs. Automatically determines all paths from evolution directory.
    """
    @handle_error
    def __init__(self, evolution_dir, dust_to_metal=1.0):
        """
        Initialize generator with evolution directory path.

        Args:
            evolution_dir (str|Path): Full path to evolution output directory containing
                                    template_X/imf_Y/star_type_Z/... structure
            dust_to_metal (float): Dust-to-metal scaling factor relative to solar.
                This multiplicative factor scales the grain abundance per hydrogen
                atom in the Cloudy input (1.0 = full dust, 0.5 = half). It is
                Z-independent and should not be confused with D/G divided by Z.
                When != 1.0, Cloudy output directories are expected to have a
                _dtmX.XX suffix. Defaults to 1.0.
        """
        self.dust_to_metal = dust_to_metal
        self.evolution_dir = Path(evolution_dir)
        if not self.evolution_dir.exists():
            raise RuntimeError(f"Evolution directory not found: {self.evolution_dir}")

        # Get root directory by finding evolution_output in path
        parts = self.evolution_dir.parts
        try:
            evo_idx = parts.index("evolution_output")
            self.root_dir = Path(*parts[:evo_idx])
            self.rel_path = Path(*parts[evo_idx+1:])
        except ValueError:
            raise RuntimeError("Path must contain 'evolution_output' directory")

        # Derive Cloudy directory
        self.cloudy_dir = self.root_dir / "cloudy_output" / self.rel_path
        
        print("\nDirectory Setup:")
        print(f"Root directory: {self.root_dir}")
        print(f"Evolution directory: {self.evolution_dir}")
        print(f"Cloudy directory: {self.cloudy_dir}")

        # Verify directories exist
        self._verify_directories()
        
        # Extract template configuration from path
        self.template_config = self._extract_template_config()
        print("\nTemplate Configuration:")
        for key, value in self.template_config.items():
            print(f"  {key}: {value}")
        
        # Extract parameter space from evolution files
        self.parameter_space = self._extract_parameter_space()
        print("\nParameter Space:")
        for param, values in self.parameter_space.items():
            print(f"  {param}: {values}")

        # Initialize data manager
        self.data_manager = DataManager(self)

    @handle_error
    def _verify_directories(self):
        """Verify required directory structure exists."""
        missing = []
        for name, path in [
            ("Evolution", self.evolution_dir),
            ("Cloudy output", self.cloudy_dir)
        ]:
            if not path.exists():
                missing.append(f"{name} directory at {path}")
        
        if missing:
            raise RuntimeError("Missing directories:\n" + "\n".join(missing))

    @handle_error
    def _extract_template_config(self):
        """Extract template configuration from directory path."""
        parts = self.evolution_dir.parts
        config = {}
        
        # Map of expected prefixes to config keys
        prefix_map = {
            'template_': 'template',
            'imf_': 'imf',
            'star_type_': 'star_type',
            'cluster_mode_': 'cluster_mode',
            'profile_type_': 'profile_type'
        }
        
        # Extract configuration
        for part in parts:
            for prefix, key in prefix_map.items():
                if part.startswith(prefix):
                    config[key] = part[len(prefix):]
                    break
                    
        # Verify all required config items were found
        missing_configs = set(prefix_map.values()) - set(config.keys())
        if missing_configs:
            raise ValueError(f"Missing required configuration items: {missing_configs}")
            
        return config

    @handle_error
    def _extract_parameter_space(self):
        """Extract parameter space from evolution simulation files."""
        param_space = {
            'Z': set(),
            'eta_sf': set(),
            'n_cl': set(),
            'logM': set()
        }

        simulation_files = list(self.evolution_dir.glob("sim_*.dat"))
        if not simulation_files:
            raise ValueError(f"No simulation files found in {self.evolution_dir}")

        # Parameter extraction patterns
        param_patterns = {
            'Z': lambda p: p.startswith('Z'),
            'eta_sf': lambda p: p.startswith('eta'),
            'n_cl': lambda p: p.startswith('n') and not p.startswith('n_'),
            'logM': lambda p: p.startswith('logM')
        }

        for sim_file in simulation_files:
            try:
                # Extract parameters from filename
                name = sim_file.stem[4:]  # Remove 'sim_' prefix
                parts = name.split('_')
                
                for part in parts:
                    for param, pattern in param_patterns.items():
                        if pattern(part):
                            try:
                                if param == 'Z':
                                    value = float(part[1:])
                                elif param == 'eta_sf':
                                    value = float(part[3:])
                                elif param == 'n_cl':
                                    value = float(part[1:])
                                elif param == 'logM':
                                    value = float(part[4:])
                                param_space[param].add(value)
                            except ValueError as e:
                                print(f"Error parsing {param} from {part} in {sim_file}: {str(e)}")
                                traceback.print_exc()
            except Exception as e:
                print(f"Error processing file {sim_file}: {str(e)}")
                traceback.print_exc()

        # Verify we found parameters
        empty_params = [param for param, values in param_space.items() if not values]
        if empty_params:
            raise ValueError(f"No values found for parameters: {empty_params}")

        # Convert sets to sorted arrays
        return {key: np.sort(np.array(list(values))) 
                for key, values in param_space.items()}

    @handle_error
    def get_paths_for_params(self, Z, eta_sf, n_cl, logM, dust_to_metal=None):
        """
        Get paths to relevant files for a specific parameter set.

        Args:
            Z (float): Metallicity value
            eta_sf (float): Star formation efficiency
            n_cl (float): Cloud number density
            logM (float): Log of cloud mass
            dust_to_metal (float): Dust-to-metal ratio relative to solar.
                Appends _dtmX.XX to Cloudy directory name when != 1.0.

        Returns:
            dict: Dictionary with paths to evolution and cloudy files

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if Z <= 0 or eta_sf <= 0 or n_cl <= 0:
            raise ValueError(f"Invalid parameter values: Z={Z}, eta_sf={eta_sf}, n_cl={n_cl}")

        if dust_to_metal is None:
            dust_to_metal = self.dust_to_metal

        param_str = f"Z{Z:.3g}_eta{eta_sf:.3g}_n{n_cl:.1f}_logM{logM:.2f}"

        cloudy_dir_name = param_str
        cloudy_dir_name += dtm_label(dust_to_metal)

        paths = {
            'evolution': self.evolution_dir / f"sim_{param_str}.dat",
            'cloudy': self.cloudy_dir / cloudy_dir_name
        }

        # Verify evolution file exists
        if not paths['evolution'].exists():
            raise FileNotFoundError(f"Evolution file not found: {paths['evolution']}")

        print(f"\nPaths for parameters:")
        print(f"  Evolution file: {paths['evolution']}")
        print(f"  Cloudy directory: {paths['cloudy']}")

        return paths

    @handle_error
    def generate_sed_interpolant(self, resolution='low', dust=True):
        """Generate SED interpolant."""
        builder = SEDInterpolantBuilder(self.data_manager)
        return builder.build(resolution=resolution, dust=dust)

    @handle_error
    def generate_property_interpolant(self, property_name):
        """Generate physical property interpolant."""
        builder = PropertyInterpolantBuilder(self.data_manager)
        return builder.build(property_name)

    @handle_error
    def generate_dissolution_interpolant(self):
        """Generate dissolution time prediction interpolant."""
        builder = DissolutionInterpolantBuilder(self.data_manager)
        return builder.build()

    @handle_error
    def save_recollapse_data(self, filename=None):
        """Save recollapse data to HDF5."""
        collector = RecollapseDataCollector(self.data_manager)
        if filename is None:
            filename = self.evolution_dir / "recollapse_data.h5"
        collector.collect_and_save(filename)

    @handle_error
    def generate_line_interpolant(self):
        """Generate line luminosity interpolant."""
        builder = LineInterpolantBuilder(self.data_manager)
        return builder.build()


class DataManager:
    """
    Manages data loading and caching for interpolant generation.
    Uses individual cache files for each parameter combination with selective data loading.
    """
    
    def __init__(self, generator):
        self.generator = generator
        self.cached_data = {}
        
        # Interpolant build cache. This is a large transient (the parsed Cloudy output; its
        # size scales with the grid -- one entry per cloud), so the TODDLERS_INTERP_CACHE env
        # var lets it live on scratch instead of the (often quota-limited) code/home filesystem
        # -- important for large DTM sweeps on a cluster. Default is the package dir, fine for
        # casual single runs.
        cache_env = os.environ.get("TODDLERS_INTERP_CACHE")
        self.cache_dir = Path(cache_env) if cache_env else (Path(__file__).resolve().parent / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCache directory initialized at: {self.cache_dir}")
        
        # Create TimeGridGenerator instance
        self.grid_generator = TimeGridGenerator(
            t_start=T_INIT_OBSERVABLES,
            dissolution_time=None,
            logger=None
        )

    def _get_cache_path(self, Z, eta_sf, n_cl, logM):
        """Generate cache file path for specific parameters.

        The dust-to-metal scaling is part of the key: a DTM sweep runs one
        ``interpolants`` invocation per DTM, all sharing this cache dir, and each
        DTM reads a different set of Cloudy output (the ``_dtmX.XX`` dirs). Without
        DTM in the key the first DTM's cache entry is reused for every later DTM, so
        all per-DTM interpolants come out identical (the DTM axis is degenerate).
        DTM=1.0 carries no suffix, matching the no-suffix baseline Cloudy dirs.
        """
        dtm = getattr(self.generator, "dust_to_metal", 1.0)
        param_str = f"Z{Z:.3g}_eta{eta_sf:.3g}_n{n_cl:.1f}_logM{logM:.2f}"
        param_str += dtm_label(dtm)
        return self.cache_dir / f"{param_str}.pkl"

    def _load_cache_entry(self, cache_path, params, data_keys=None):
        """
        Load a single cache entry with selective data loading.
        
        Args:
            cache_path (Path): Path to cache file
            params (tuple): Parameter tuple (Z, eta_sf, n_cl, logM)
            data_keys (dict, optional): Specifies which data to load. Example::

                {
                    'continuum': ['incident', 'observed'],
                    'lines': 'intrinsic' or 'emergent' or 'both',
                    'evolution': True or False
                }

        Returns:
            dict or None: Cached data if valid, None otherwise
        """
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)

            # Validate cache structure
            if not isinstance(cached_data, dict):
                print(f"Invalid cache format in {cache_path}")
                return None

            # If no data_keys specified, return full structure
            if data_keys is None:
                if not self._validate_cache_entry(params, cached_data):
                    print(f"Invalid cache content in {cache_path}")
                    cache_path.unlink()
                    return None
                print(f"Successfully loaded full cache from {cache_path}")
                return cached_data

            # Initialize result with minimal required data
            result = {
                'paths': cached_data.get('paths', {}),
                'dissolution_time': cached_data.get('dissolution_time'),
                'timepoints': cached_data.get('timepoints'),
            }

            # Load evolution data if requested
            if data_keys.get('evolution', True):
                result['evolution'] = cached_data.get('evolution')

            # Initialize cloudy data structure
            result['cloudy'] = {}

            # Process each timepoint selectively
            cloudy_data = cached_data.get('cloudy', {})
            for timepoint, full_data in cloudy_data.items():
                result['cloudy'][timepoint] = {}
                
                # Add model type
                result['cloudy'][timepoint]['model_type'] = full_data.get('model_type')

                # Process continuum data selectively
                if 'continuum' in full_data:
                    cont_data = full_data['continuum']
                    filtered_cont = {'nu': cont_data['nu']}  # Always need wavelength grid
                    
                    # Add requested continuum types
                    cont_types = data_keys.get('continuum', [])
                    for cont_type in cont_types:
                        if cont_type in cont_data:
                            filtered_cont[cont_type] = cont_data[cont_type]
                    
                    result['cloudy'][timepoint]['continuum'] = filtered_cont

                # Process line data selectively
                line_type = data_keys.get('lines')
                if line_type:
                    if line_type == 'intrinsic' and 'lines_intrinsic' in full_data:
                        result['cloudy'][timepoint]['lines_intrinsic'] = full_data['lines_intrinsic']
                    elif line_type == 'emergent' and 'lines_emergent' in full_data:
                        result['cloudy'][timepoint]['lines_emergent'] = full_data['lines_emergent']
                    elif line_type == 'both':
                        if 'lines_intrinsic' in full_data:
                            result['cloudy'][timepoint]['lines_intrinsic'] = full_data['lines_intrinsic']
                        if 'lines_emergent' in full_data:
                            result['cloudy'][timepoint]['lines_emergent'] = full_data['lines_emergent']

            print(f"Successfully loaded selective cache from {cache_path}")
            return result

        except Exception as e:
            print(f"Error loading cache file {cache_path}: {str(e)}")
            traceback.print_exc()
            return None

    def _validate_cache_entry(self, params, data):
        """Validate a cache entry's structure and content."""
        # Check basic structure
        required_keys = {'evolution', 'cloudy', 'timepoints', 'dissolution_time', 'paths'}
        if not isinstance(data, dict) or not all(k in data for k in required_keys):
            return False

        Z, eta_sf, n_cl, logM = params
        
        # Check parameter ranges
        if Z not in self.generator.parameter_space['Z']:
            return False
        if eta_sf not in self.generator.parameter_space['eta_sf']:
            return False
        if n_cl not in self.generator.parameter_space['n_cl']:
            return False
        if logM not in self.generator.parameter_space['logM']:
            return False

        # Basic data validation
        if data['evolution'] is None:
            return False

        return True

    def _save_cache_entry(self, cache_path, data):
        """Save a single cache entry atomically."""
        temp_path = cache_path.with_suffix('.tmp')
        
        try:
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f)
            
            temp_path.replace(cache_path)
            print(f"Successfully saved cache to {cache_path}")
            
        except Exception as e:
            print(f"Error saving cache to {cache_path}: {str(e)}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def load_data_for_params(self, Z, eta_sf, n_cl, logM, data_keys=None):
        """
        Load data for a specific parameter set, using cache when available.
        
        Args:
            Z (float): Metallicity value
            eta_sf (float): Star formation efficiency
            n_cl (float): Cloud number density
            logM (float): Log of cloud mass
            data_keys (dict, optional): Specifies which data to load
            
        Returns:
            dict: Dictionary containing loaded data
        """
        params = (Z, eta_sf, n_cl, logM)
        cache_path = self._get_cache_path(Z, eta_sf, n_cl, logM)
        
        # Try to load from cache first
        cached_data = self._load_cache_entry(cache_path, params, data_keys)
        if cached_data is not None:
            return cached_data

        # Get paths for this parameter set
        paths = self.generator.get_paths_for_params(Z, eta_sf, n_cl, logM)
        
        # Load evolution data
        evolution_data = self._load_evolution_data(paths['evolution'])
        if evolution_data is None:
            raise ValueError(f"Failed to load evolution data for parameters: Z={Z}, eta_sf={eta_sf}, n={n_cl}, logM={logM}")

        # Get dissolution time
        dissolution_time = None
        for gen_data in evolution_data['results']:
            for transition in gen_data.get('phase_transitions', []):
                if transition[0] == 'phase2_to_dissolution':
                    dissolution_time = transition[1]
                    break
            if dissolution_time:
                break

        # Load Cloudy data and generate time grid
        cloudy_data = self._load_cloudy_data(paths['cloudy'])
        
        # Update grid generator with dissolution time
        self.grid_generator.dissolution_time = dissolution_time
        
        # Generate time grid
        time_points = self.grid_generator.generate_grid(
            method='toddlers_v1',
            continue_after_dissolution=True
        )

        # Combine all data
        combined_data = {
            'evolution': evolution_data,
            'cloudy': cloudy_data,
            'timepoints': time_points,
            'dissolution_time': dissolution_time,
            'paths': paths
        }
        
        # Save full data to cache
        self._save_cache_entry(cache_path, combined_data)
        
        # If data_keys specified, filter the data before returning
        if data_keys is not None:
            return self._load_cache_entry(cache_path, params, data_keys)
        
        return combined_data

    def clear_cache_files(self):
        """Clear all cache files from cache directory."""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            for temp_file in self.cache_dir.glob("*.tmp"):
                temp_file.unlink()
            print(f"Cleared all cache files from {self.cache_dir}")
        except Exception as e:
            print(f"Error clearing cache: {str(e)}")

    def clear_cache(self):
        """Clear loaded cached data."""
        self.cached_data.clear()
        
    def get_cache_size(self):
        """
        Get total size of cache directory in MB.
        
        Returns:
            float: Cache size in megabytes
        """
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
        return total_size / (1024 * 1024)  # Convert to MB

    def list_cached_params(self):
        """
        List all parameter combinations that are currently cached.
        
        Returns:
            list: List of tuples (Z, eta_sf, n_cl, logM, evolution_only)
                 representing cached parameter combinations
        """
        cached_params = []
        for cache_file in self.cache_dir.glob("*.pkl"):
            # Parse parameters from filename
            param_str = cache_file.stem
            parts = param_str.split("_")
            
            try:
                Z = float(parts[0][1:])        # Remove 'Z'
                eta_sf = float(parts[1][3:])   # Remove 'eta'
                n_cl = float(parts[2][1:])     # Remove 'n'
                logM = float(parts[3][4:])     # Remove 'logM'
                evolution_only = len(parts) > 4 and parts[4] == "evo"
                
                cached_params.append((Z, eta_sf, n_cl, logM, evolution_only))
            except Exception as e:
                print(f"Error parsing parameters from {cache_file}: {str(e)}")
                continue
                
        return cached_params

    @handle_error
    def _load_evolution_data(self, evolution_path):
        """Load evolution simulation data with error handling."""
        if not evolution_path.exists():
            raise FileNotFoundError(f"Evolution file not found: {evolution_path}")
            
        try:
            simulation_params, all_results = load_output_file(evolution_path)
            
            # Extract dissolution time and other key data
            dissolution_time = None
            t_list_collapse = []
            
            for gen_data in all_results:
                for transition in gen_data.get('phase_transitions', []):
                    if transition[0] == 'phase2_to_dissolution':
                        dissolution_time = transition[1]
                    elif transition[0] == 'phase2_to_recollapse':
                        t_list_collapse.append(transition[1])
            
            return {
                'params': simulation_params,
                'results': all_results,
                'dissolution_time': dissolution_time,
                't_list_collapse': t_list_collapse
            }
            
        except Exception as e:
            print(f"Error loading evolution file {evolution_path}: {str(e)}")
            traceback.print_exc()
            return None

    @handle_error
    def _load_cloudy_data(self, cloudy_dir):
        """Load Cloudy output data with proper path handling."""
        if not cloudy_dir.exists():
            print(f"Warning: Cloudy directory does not exist: {cloudy_dir}")
            return None
            
        print(f"Loading Cloudy data from: {cloudy_dir}")
        cloudy_data = {}
        
        # Get all timepoints from files
        timepoints = set()
        model_files = {
            'shell': [],
            'unified': [], 
            'dissolved': []
        }
        
        # First collect all output files
        for model in model_files.keys():
            model_files[model] = list(cloudy_dir.glob(f"{model}_*.out"))
            print(f"Found {len(model_files[model])} {model} files")
            for outfile in model_files[model]:
                try:
                    time = float(outfile.stem.split('_')[1])
                    timepoints.add(time)
                except (ValueError, IndexError) as e:
                    print(f"Error parsing time from {outfile}: {str(e)}")
                    continue
        
        if not timepoints:
            print(f"Warning: No valid timepoints found in {cloudy_dir}")
            return None
            
        print(f"Found {len(timepoints)} timepoints")
        for model, files in model_files.items():
            print(f"  {model}: {len(files)} files")

        # Load data for each timepoint
        for time in sorted(timepoints):
            try:
                data_at_time = {}
                
                # Determine which model to use
                unified_path = cloudy_dir / f"unified_{time:.2f}.out"
                shell_path = cloudy_dir / f"shell_{time:.2f}.out"
                dissolved_path = cloudy_dir / f"dissolved_{time:.2f}.out"
                
                if unified_path.exists():
                    print(f"Loading unified model at t={time:.2f}")
                    handler = CloudyOutputHandler('unified', time*MYR_TO_SEC, absolute_path=str(cloudy_dir))
                    data_at_time['model_type'] = 'unified'
                elif shell_path.exists():
                    print(f"Loading shell model at t={time:.2f}")
                    handler = CloudyOutputHandler('shell', time*MYR_TO_SEC, absolute_path=str(cloudy_dir))
                    data_at_time['model_type'] = 'shell'
                elif dissolved_path.exists():
                    print(f"Loading dissolved model at t={time:.2f}")
                    handler = CloudyOutputHandler('dissolved', time*MYR_TO_SEC, absolute_path=str(cloudy_dir))
                    data_at_time['model_type'] = 'dissolved'
                else:
                    print(f"Warning: No output file found for t={time:.2f}")
                    continue
                    
                # Only load if run was successful
                if handler.check_cloudy_success(print_traceback=True):
                    try:
                        data_at_time.update({
                            'continuum': handler.get_continuum(),
                            'lines_intrinsic': handler.get_line_luminosities(use_emergent=False),
                            'lines_emergent': handler.get_line_luminosities(use_emergent=True),
                            'f_esc_i': handler.calculate_escape_fraction()
                        })
                        
                        if data_at_time['model_type'] != 'dissolved':
                            data_at_time.update({
                                'radius': handler.get_radial_structure(),
                                'density': handler.get_density_structure(),
                                'temperature': handler.get_temperature_structure(),
                                'ionization': handler.get_ionization_structure()
                            })
                        
                        cloudy_data[time] = data_at_time
                        print(f"Successfully loaded data for t={time:.2f} Myr ({data_at_time['model_type']} model)")
                        
                    except Exception as e:
                        print(f"Error loading data for t={time:.2f}: {str(e)}")
                        continue
                else:
                    print(f"Warning: Cloudy run not successful for t={time:.2f}")
                    
            except Exception as e:
                print(f"Error processing time {time}: {str(e)}")
                traceback.print_exc()
                continue
        
        print(f"Successfully loaded data for {len(cloudy_data)} timepoints")
        return cloudy_data


class SEDInterpolantBuilder:
    """
    Builds SED interpolants by combining continuum and line data.
    
    This class handles the creation of wavelength grids and interpolants for both
    low and high resolution SEDs, properly handling line profiles and continuum merging.
    """
    
    @handle_error
    def __init__(self, data_manager):
        """
        Initialize the SED interpolant builder.
        
        Args:
            data_manager: DataManager instance for accessing simulation data
        """
        self.data_manager = data_manager
        self.generator = data_manager.generator
        self.line_profile_gen = LineProfileGenerator()
        
        # Get parameter arrays from generator
        self.Z_arr = self.generator.parameter_space['Z']
        self.eta_arr = self.generator.parameter_space['eta_sf']
        self.n_arr = self.generator.parameter_space['n_cl']
        self.logM_arr = self.generator.parameter_space['logM']

        # Define wavelength limits in microns
        self.wave_min = 0.01  # 0.01 micron
        self.wave_max = 3005  # 3005 micron

        # high resolution SED resolution
        self.hr_sed_resolution = 5e4
        
        # Cache for wavelength grids
        self._low_res_wave_grid = None
        self._high_res_wave_grid = None

    @handle_error
    def _verify_timepoint_consistency(self, data, Z, eta, n, logM):
        """
        Verify that Cloudy timepoints match grid generator points.
        
        Args:
            data (dict): Data dictionary containing timepoints and Cloudy data
            Z (float): Metallicity
            eta (float): Star formation efficiency 
            n (float): Cloud number density
            logM (float): Log of cloud mass
            
        Returns:
            np.ndarray: Array of verified timepoints in Myr
            
        Raises:
            AssertionError: If timepoints don't match within tolerance
            ValueError: If required data is missing
        """
        if 'timepoints' not in data or 'cloudy' not in data:
            raise ValueError("Missing required timepoint data")
            
        # Get time points from the grid generator (in seconds)
        grid_points = data['timepoints']
        
        # Convert grid points to Myr for comparison with Cloudy data
        grid_points_myr = grid_points / MYR_TO_SEC
        
        # Get Cloudy timepoints (in Myr)
        cloudy_points = np.array(sorted(data['cloudy'].keys()))
        
        # Log timepoint information for debugging
        print(f"\nTimepoint comparison for Z={Z}, eta={eta}, n={n}, logM={logM}:")
        print(f"Grid points (Myr): {sorted(grid_points_myr)}")
        print(f"Cloudy points (Myr): {sorted(cloudy_points)}")
        
        np.testing.assert_allclose(
            sorted(cloudy_points), 
            sorted(grid_points_myr),
            rtol=1e-10, 
            err_msg=f"Mismatch between Cloudy timepoints and grid points for Z={Z}, eta={eta}, n={n}, logM={logM}"
        )
        
        return grid_points_myr

    @handle_error
    def _get_wavelength_grid(self, resolution='low', param_data=None, data_keys=None):
        """
        Get or generate wavelength grid for given resolution.
        
        Args:
            resolution (str): 'low' or 'high'
            param_data (dict, optional): Parameter data needed for high resolution grid
            data_keys (dict, optional): Dictionary specifying which data was loaded
            
        Returns:
            np.ndarray: Sorted wavelength grid
        """
        if resolution == 'low':
            if self._low_res_wave_grid is None:
                # Get representative Cloudy data from first available timepoint
                first_data = next(iter(param_data.values()))
                first_tp = min(first_data['cloudy'].keys())
                wave_data = first_data['cloudy'][first_tp]['continuum']['nu']
                
                # Filter wavelengths within limits
                valid_waves = wave_data[(wave_data >= self.wave_min) & 
                                      (wave_data <= self.wave_max)]
                self._low_res_wave_grid = valid_waves
                
            return self._low_res_wave_grid
            
        elif resolution == 'high':
            if self._high_res_wave_grid is None:
                                
                # Only process line data if it was actually loaded
                if data_keys and 'lines' in data_keys:
                    # Get representative line data to generate profiles
                    first_data = next(iter(param_data.values()))
                    first_tp = min(first_data['cloudy'].keys())
                    
                    # Use the appropriate line data based on what was loaded
                    line_type = data_keys['lines']
                    if line_type == 'intrinsic':
                        line_data = first_data['cloudy'][first_tp]['lines_intrinsic']
                    elif line_type == 'emergent':
                        line_data = first_data['cloudy'][first_tp]['lines_emergent']
                    else:  # 'both' - use intrinsic by default
                        line_data = first_data['cloudy'][first_tp]['lines_intrinsic']
                    
                    # Filter line data to wavelength range
                    filtered_line_data = {
                        k: v for k, v in line_data.items()
                        if self.wave_min <= k[2] <= self.wave_max
                    }
                    
                    line_profiles = self.line_profile_gen.create_line_profiles(
                        filtered_line_data, resolution=self.hr_sed_resolution
                    )
                    
                    # Get base continuum SED for this timepoint
                    cont_data = first_data['cloudy'][first_tp]['continuum']
                    valid_indices = ((cont_data['nu'] >= self.wave_min) & 
                                   (cont_data['nu'] <= self.wave_max))
                    if 'observed_no_lines' in data_keys['continuum']:
                        base_sed = cont_data['observed_no_lines']
                    elif 'observed' in data_keys['continuum']:
                        base_sed = cont_data['observed']
                    else:
                        base_sed = cont_data['incident']
                    filtered_waves = cont_data['nu'][valid_indices]
                    filtered_sed = base_sed[valid_indices]

                    # Merge profiles with continuum wavelength grid 
                    waves, _ = self.line_profile_gen.merge_profiles(
                        filtered_waves,
                        filtered_sed,
                        line_profiles
                    )
                    
                    self._high_res_wave_grid = waves
                
            return self._high_res_wave_grid
        else:
            raise ValueError(f"Invalid resolution: {resolution}")

    @handle_error
    def _process_single_sed(self, cont_data, line_data=None, resolution='low', dust=True):
        """
        Process a single SED, optionally including line profiles.
        
        Args:
            cont_data (dict): Continuum data from Cloudy
            line_data (dict, optional): Line data from Cloudy if resolution='high'
            resolution (str): 'low' or 'high'
            dust (bool): Whether to use dust-attenuated spectrum
            
        Returns:
            np.ndarray: low / high resolution SED with / without dust
        """
        
        # Get appropriate continuum
        if dust:
            if resolution == 'low':
                base_sed = cont_data['observed']
            else: # lines will be added later
                base_sed = cont_data['observed_no_lines']
        else:
            base_sed = cont_data['incident']
            # Optionally add the unattenuated diffuse nebular continuum (free-bound,
            # free-free, two-photon) so the noDust SED carries intrinsic gas emission,
            # not just the incident stellar field (paper Appendix B, v2-DTM). Requires
            # the patched Cloudy ".diffContUnatt" output; index-aligned with 'incident'.
            if INCLUDE_NEBULAR_CONTINUUM and 'nebular_unatt' in cont_data:
                neb = cont_data['nebular_unatt']
                base_sed = base_sed + neb
                self._neb_added = getattr(self, '_neb_added', 0) + 1
                if self._neb_added == 1:
                    # report once per build, with the optical fractional contribution as a sanity value
                    opt = (cont_data['nu'] > 0.4) & (cont_data['nu'] < 0.7)
                    frac = float(np.nanmedian(neb[opt] / np.maximum(base_sed[opt], 1e-99))) if opt.any() else float('nan')
                    print(f"  [nebular] noDust SED: adding unattenuated nebular continuum from "
                          f".diffContUnatt (DiffContUnatt col); optical median fraction ~{frac:.2f}")
            elif INCLUDE_NEBULAR_CONTINUUM and not getattr(self, '_neb_warned', False):
                self._neb_warned = True
                print("  [nebular] WARNING: INCLUDE_NEBULAR_CONTINUUM set but no 'nebular_unatt' in "
                      "continuum data (unpatched Cloudy / missing .diffContUnatt?); noDust = incident only.")

        # Filter to wavelength range
        valid_indices = ((cont_data['nu'] >= self.wave_min) & 
                        (cont_data['nu'] <= self.wave_max))
        filtered_waves = cont_data['nu'][valid_indices]
        filtered_sed = base_sed[valid_indices]

        # convert to specific luminosity here
        filtered_sed /= filtered_waves
        
        if resolution == 'low':
            # Just interpolate continuum onto common grid
            sed = np.maximum(filtered_sed, 1e-99)
        else:
            # Get appropriate line data 
            if dust:
                line_data = line_data['lines_emergent']
            else:
                line_data = line_data['lines_intrinsic']
                
            # Create and merge line profiles
            try:
                # Filter line data to wavelength range
                filtered_line_data = {
                    k: v for k, v in line_data.items()
                    if self.wave_min <= k[2] <= self.wave_max
                }
                
                line_profiles = self.line_profile_gen.create_line_profiles(
                    filtered_line_data, resolution=self.hr_sed_resolution
                )
                _, sed = self.line_profile_gen.merge_profiles(
                    filtered_waves,
                    filtered_sed,
                    line_profiles
                )
            except Exception as e:
                print(f"Error processing lines: {str(e)}")
                traceback.print_exc()
        
        return sed

    @handle_error
    def build(self, resolution='low', dust=True):
        """
        Build SED interpolant with selective data loading.
        
        Args:
            resolution (str): 'low' or 'high'
            dust (bool): Whether to use dust-attenuated spectrum
            
        Returns:
            RegularGridInterpolator: Interpolant for SEDs
            
        The interpolator takes inputs in the order:
        (log10_wavelength, time, log10_Z, eta, log10_n, logM)
        
        All inputs must be in the appropriate units:
        - wavelength: microns
        - time: Myr
        - Z: solar units
        - eta: dimensionless
        - n: cm^-3
        - M: solar masses (in log)
        
        Returns log10 of luminosity.
        """
        if resolution not in ['low', 'high']:
            raise ValueError("Resolution must be 'low' or 'high'")

        # Determine which data to load based on resolution and dust settings
        data_keys = {
            'evolution': False,  # Don't need evolution data for SEDs
            'continuum': []
        }

        # Set continuum based on dust and resolution
        if dust:
            if resolution == 'low':
                data_keys['continuum'].append('observed')
            else:  # high resolution
                data_keys['continuum'].append('observed_no_lines')
        else:
            data_keys['continuum'].append('incident')
            # Also retain the unattenuated nebular continuum so it survives the
            # selective cache filter and reaches the noDust SED assembly.
            if INCLUDE_NEBULAR_CONTINUUM:
                data_keys['continuum'].append('nebular_unatt')

        # Only load line data for high resolution
        if resolution == 'high':
            data_keys['lines'] = 'emergent' if dust else 'intrinsic'

        # First pass: collect data and verify timepoints
        timepoints = set()
        param_data = {}
        print(f"\nLoading data with settings: resolution={resolution}, dust={dust}")
        print(f"Data keys: {data_keys}")

        for Z in self.Z_arr:
            for eta in self.eta_arr:
                for n in self.n_arr:
                    for logM in self.logM_arr:
                        data = self.data_manager.load_data_for_params(
                            Z, eta, n, logM,
                            data_keys=data_keys
                        )
                        if not data['cloudy']:
                            continue
                            
                        param_data[(Z, eta, n, logM)] = data
                        timepoints.update(
                            self._verify_timepoint_consistency(data, Z, eta, n, logM)
                        )

        if not timepoints:
            raise ValueError("No valid data found for interpolant construction")

        # Get wavelength grid
        wave_arr = self._get_wavelength_grid(resolution, param_data, data_keys)
        
        # Convert timepoints to sorted array
        self.time_arr = np.sort(np.array(list(timepoints)))  # In Myr

        # Initialize SED array
        sed_array = np.zeros((
            len(wave_arr),
            len(self.time_arr),
            len(self.Z_arr),
            len(self.eta_arr),
            len(self.n_arr),
            len(self.logM_arr)
        ))

        print(f"\nSED array shape: {sed_array.shape}")
        print(f"Memory usage: {sed_array.nbytes / 1e9:.2f} GB")

        # Second pass: fill SED array
        for i_Z, Z in enumerate(self.Z_arr):
            for i_eta, eta in enumerate(self.eta_arr):
                for i_n, n in enumerate(self.n_arr):
                    for i_m, logM in enumerate(self.logM_arr):
                        data = param_data.get((Z, eta, n, logM))
                        if not data:
                            continue
                            
                        for i_t, t in enumerate(self.time_arr):
                            if t not in data['cloudy']:
                                continue
                            
                            # Get SED data
                            cloudy_data = data['cloudy'][t] # line data for _process_single_sed
                            cont_data = cloudy_data['continuum']
                            
                            # Process SED
                            sed = self._process_single_sed(
                                cont_data,
                                cloudy_data if resolution == 'high' else None,
                                resolution=resolution,
                                dust=dust
                            )

                            assert len(wave_arr) == len(sed), (
                                f"Size mismatch: len(wave_arr)={len(wave_arr)}, len(sed)={len(sed)}\n"
                                f"Parameters: Z={Z}, eta={eta}, n={n}, logM={logM}, t={t}, "
                                f"indices: i_Z={i_Z}, i_eta={i_eta}, i_n={i_n}, i_m={i_m}, i_t={i_t}"
                            )

                            # Replace zeros and store
                            sed = np.clip(sed, 1e-99, np.inf)  # Replace zeros with small value
                            sed_array[:, i_t, i_Z, i_eta, i_n, i_m] = sed

        # Create interpolator points
        points = (
            np.log10(wave_arr),  # log wavelength
            self.time_arr,        # linear time
            np.log10(self.Z_arr), # log Z
            self.eta_arr,         # linear eta
            np.log10(self.n_arr), # log n
            self.logM_arr         # already log M
        )
        
        # Replace zeros in final array before taking log
        sed_array = np.clip(sed_array, 1e-99, np.inf)
        
        return RegularGridInterpolator(
            points=points,
            values=np.log10(sed_array),  # log luminosity
            bounds_error=True
        )


class PropertyInterpolantBuilder:
    """Builds interpolants for physical properties."""

    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.generator = data_manager.generator
        
        # Get parameter arrays
        self.Z_arr = self.generator.parameter_space['Z']
        self.eta_arr = self.generator.parameter_space['eta_sf']
        self.n_arr = self.generator.parameter_space['n_cl']
        self.logM_arr = self.generator.parameter_space['logM']

    @handle_error
    def build(self, property_name):
        """Build interpolant for a physical property."""
        evolution_properties = {'velocity'}
        cloudy_properties = {'radius', 'density', 'temperature', 'ionization_H', 
                           'ionization_He', 'f_esc_i'}
        
        if property_name not in evolution_properties | cloudy_properties:
            raise ValueError(f"Invalid property name. Must be one of: "
                           f"{evolution_properties | cloudy_properties}")
            
        if property_name in evolution_properties:
            return self._build_evolution_interpolant(property_name)
        else:
            return self._build_cloudy_interpolant(property_name)

    @handle_error
    def _build_evolution_interpolant(self, property_name):
        """Build interpolant for properties from evolution data."""
        # First pass to get data
        param_data = {}
        
        for Z in self.Z_arr:
            for eta in self.eta_arr:
                for n in self.n_arr:
                    for logM in self.logM_arr:
                        try:
                            data = self.data_manager.load_data_for_params(Z, eta, n, logM)
                            if not data['evolution']:
                                continue
                                
                            param_data[(Z, eta, n, logM)] = data

                        except Exception as e:
                            print(f"Error loading data for Z={Z}, eta={eta}, n={n}, logM={logM}: {str(e)}")
                            continue
 
        time_arr = np.linspace(T_INIT_OBSERVABLES, MAX_SIMULATION_TIME, num=500, endpoint=True)

        # Initialize array
        prop_array = np.zeros((
            len(time_arr),
            len(self.Z_arr),
            len(self.eta_arr), 
            len(self.n_arr),
            len(self.logM_arr)
        ))

        # Fill array with property data
        for i_Z, Z in enumerate(self.Z_arr):
            for i_eta, eta in enumerate(self.eta_arr):
                for i_n, n in enumerate(self.n_arr):
                    for i_m, logM in enumerate(self.logM_arr):
                        data = param_data.get((Z, eta, n, logM))
                        if not data:
                            continue

                        try:

                            # Get evolution data
                            time_list, property_list = [], []
                            for _, gen_data in enumerate(data['evolution']['results']):
                                time = gen_data["time"]
                                time_list.append(time)
                                property_list.append(gen_data[property_name])
                            evo_times = np.concatenate(time_list)
                            values = np.concatenate(property_list)

                            # Interpolate onto common time grid
                            prop_array[:, i_Z, i_eta, i_n, i_m] = np.interp(
                                time_arr, evo_times, values,
                                left=values[0], right=values[-1]
                            )
                        except Exception as e:
                            print(f"Error processing property {property_name} for parameters "
                                  f"Z={Z}, eta={eta}, n={n}, logM={logM}: {str(e)}")
                            continue

        # Create interpolator
        points = (
            time_arr,            # linear time
            np.log10(self.Z_arr), # log Z 
            self.eta_arr,        # linear eta
            np.log10(self.n_arr), # log n
            self.logM_arr        # already log M
        )

        return RegularGridInterpolator(
            points=points,
            values=prop_array,
            bounds_error=True
        )

    @handle_error
    def _build_cloudy_interpolant(self, property_name):
        """Build interpolant for properties from Cloudy outputs."""
        # First pass to get timepoints
        timepoints = set()
        param_data = {}
        
        for Z in self.Z_arr:
            for eta in self.eta_arr:
                for n in self.n_arr:
                    for logM in self.logM_arr:
                        try:
                            data = self.data_manager.load_data_for_params(Z, eta, n, logM)
                            if not data['cloudy']:
                                continue
                                
                            param_data[(Z, eta, n, logM)] = data
                            timepoints.update(data['cloudy'].keys())
                        except Exception as e:
                            print(f"Error loading Cloudy data for Z={Z}, eta={eta}, n={n}, logM={logM}: {str(e)}")
                            continue

        if not timepoints:
            raise ValueError("No valid timepoints found for Cloudy interpolant")

        # Convert to sorted array
        time_arr = np.sort(np.array(list(timepoints)))

        # Initialize property array
        prop_array = np.zeros((
            len(time_arr),
            len(self.Z_arr),
            len(self.eta_arr),
            len(self.n_arr),
            len(self.logM_arr)
        ))

        # Fill property array
        for i_Z, Z in enumerate(self.Z_arr):
            for i_eta, eta in enumerate(self.eta_arr):
                for i_n, n in enumerate(self.n_arr):
                    for i_m, logM in enumerate(self.logM_arr):                            
                        for i_t, t in enumerate(time_arr):
                                
                            try:
                                data = param_data.get((Z, eta, n, logM))
                                cloudy_data = data['cloudy'][t]
                                if cloudy_data['model_type'] != 'dissolved':
                                    # Get property based on name
                                    value = self._get_cloudy_property(cloudy_data, property_name)
                                else:
                                    value = self._get_cloudy_property(cloudy_data, property_name, dissolved=True)
                                prop_array[i_t, i_Z, i_eta, i_n, i_m] = value

                            except Exception as e:
                                print(f"Error processing property {property_name} at time {t} for parameters "
                                      f"Z={Z}, eta={eta}, n={n}, logM={logM}: {str(e)}")
                                continue

        # Create interpolator
        points = (
            time_arr,            # linear time
            np.log10(self.Z_arr), # log Z
            self.eta_arr,        # linear eta
            np.log10(self.n_arr), # log n
            self.logM_arr        # already log M
        )
        
        # Determine if property should be logged
        log_properties = {'density', 'temperature'}
        if property_name in log_properties:
            values = np.log10(prop_array)
        else:
            values = prop_array

        return RegularGridInterpolator(
            points=points,
            values=values,
            bounds_error=True
        )

    @handle_error
    def _get_cloudy_property(self, data, property_name, dissolved=False):
        """Extract specific property from Cloudy data."""
        if property_name == 'radius':
            if dissolved:
                return 1e-99
            else:    
                return data['radius'][0][0]  # First point of radius array
        elif property_name == 'density':
            if dissolved:
                return 1e-99
            else:
                return data['density'][0]  # First point of density array
        elif property_name == 'temperature':
            if dissolved:
                return 1e-99
            else:
                return data['temperature'][0]  # First point of temperature array
        elif property_name == 'f_esc_i':
            if dissolved:
                return 1
            else:
                return data['f_esc_i']
        else:
            raise ValueError(f"Unknown Cloudy property: {property_name}")

class DissolutionInterpolantBuilder:
    """Builds interpolant for predicting dissolution time."""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.generator = data_manager.generator

        # Get parameter arrays
        self.Z_arr = self.generator.parameter_space['Z']
        self.eta_arr = self.generator.parameter_space['eta_sf']
        self.n_arr = self.generator.parameter_space['n_cl']
        self.logM_arr = self.generator.parameter_space['logM']

    @handle_error
    def build(self):
        """
        Build dissolution time prediction interpolant.
        
        For simulations that don't dissolve, uses MAX_SIMULATION_TIME + 1 * MYR_TO_SEC
        as the dissolution time.
        """
        # Initialize dissolution time array
        time_array = np.zeros((
            len(self.Z_arr),
            len(self.eta_arr),
            len(self.n_arr),
            len(self.logM_arr)
        ))
        time_array.fill(np.nan)  # Initialize with NaN

        valid_points = 0
        # Fill array with dissolution times
        for i_Z, Z in enumerate(self.Z_arr):
            for i_eta, eta in enumerate(self.eta_arr):
                for i_n, n in enumerate(self.n_arr):
                    for i_m, logM in enumerate(self.logM_arr):
                        try:
                            data = self.data_manager.load_data_for_params(Z, eta, n, logM)
                            if data:
                                if data.get('dissolution_time') is not None:
                                    time_array[i_Z, i_eta, i_n, i_m] = data['dissolution_time']
                                else:
                                    # Use MAX_SIMULATION_TIME + MYR_TO_SEC (31 Myr) for runs that don't dissolve
                                    time_array[i_Z, i_eta, i_n, i_m] = MAX_SIMULATION_TIME + MYR_TO_SEC
                                valid_points += 1
                        except Exception as e:
                            print(f"Error processing dissolution time for Z={Z}, eta={eta}, n={n}, logM={logM}: {str(e)}")
                            continue

        if valid_points == 0:
            raise ValueError("No valid dissolution times found for interpolant construction")

        # Create interpolator
        points = (
            np.log10(self.Z_arr), # log Z
            self.eta_arr,         # linear eta
            np.log10(self.n_arr), # log n
            self.logM_arr         # already log M
        )

        return RegularGridInterpolator(
            points=points,
            values=time_array,    # linear time
            bounds_error=True
        )


class RecollapseDataCollector:
    """Collects and stores recollapse data in HDF5 format."""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.generator = data_manager.generator

    @handle_error
    def collect_and_save(self, filename):
        """Collect recollapse data and save to HDF5."""
        # Create parameter combinations
        params_list = [
            (Z, eta, n, logM) 
            for Z in self.generator.parameter_space['Z']
            for eta in self.generator.parameter_space['eta_sf']
            for n in self.generator.parameter_space['n_cl']
            for logM in self.generator.parameter_space['logM']
        ]

        with h5py.File(filename, 'w') as f:
            # Create main group
            group = f.create_group("recollapse_data")
            
            # Add metadata
            group.attrs.update({
                'template': self.generator.template_config['template'],
                'imf': self.generator.template_config['imf'],
                'star_type': self.generator.template_config['star_type'],
                'cluster_mode': self.generator.template_config['cluster_mode'],
                'profile_type': self.generator.template_config['profile_type'],
                'date_generated': str(datetime.datetime.now())
            })
            
            # Add dataset to store parameter grid
            param_group = group.create_group("parameter_grid")
            param_group.create_dataset("Z", data=self.generator.parameter_space['Z'])
            param_group.create_dataset("eta_sf", data=self.generator.parameter_space['eta_sf'])
            param_group.create_dataset("n_cl", data=self.generator.parameter_space['n_cl'])
            param_group.create_dataset("logM", data=self.generator.parameter_space['logM'])
            
            # Create group for simulation data
            sims_group = group.create_group("simulations")
            
            # Process each parameter combination
            successful_saves = 0
            failed_saves = 0
            
            for Z, eta, n, logM in params_list:
                try:
                    # Create parameter key
                    key = f"Z{Z:.3g}_eta{eta:.3g}_n{n:.1f}_logM{logM:.2f}"
                    data = self.data_manager.load_data_for_params(Z, eta, n, logM)
                    
                    if not data or 't_list_collapse' not in data['evolution']:
                        continue
                        
                    sim_group = sims_group.create_group(key)
                    
                    # Store recollapse times in seconds and Myr
                    times_sec = np.array(data['evolution']['t_list_collapse'])
                    times_myr = times_sec / MYR_TO_SEC
                    sim_group.create_dataset('recollapse_times_sec', data=times_sec)
                    sim_group.create_dataset('recollapse_times_myr', data=times_myr)
                    
                    # Store parameters as attributes
                    sim_group.attrs.update({
                        'Z': Z,
                        'eta_sf': eta,
                        'n_cl': n,
                        'logM': logM
                    })
                    
                    # Extract data about each recollapse event
                    events_group = sim_group.create_group("events")
                    for gen_idx, result in enumerate(data['evolution']['results']):
                        for trans in result.get('phase_transitions', []):
                            if trans[0] == 'phase2_to_recollapse':
                                event_group = events_group.create_group(f"generation_{gen_idx}")
                                event_group.attrs.update({
                                    'time_sec': trans[1],
                                    'time_myr': trans[1] / MYR_TO_SEC
                                })
                                
                                # Store additional data if available
                                for field in ['stellar_mass', 'cloud_mass', 'cloud_radius']:
                                    if field in result:
                                        event_group.attrs[field] = result[field]
                    
                    successful_saves += 1
                    
                except Exception as e:
                    print(f"Error processing parameters Z={Z}, eta={eta}, n={n}, logM={logM}: {str(e)}")
                    traceback.print_exc()
                    failed_saves += 1
                    
            print(f"\nData collection summary:")
            print(f"Successfully saved: {successful_saves} parameter combinations")
            if failed_saves > 0:
                print(f"Failed to save: {failed_saves} parameter combinations")

    @handle_error
    def get_summary_statistics(self):
        """
        Calculate summary statistics for recollapse events.
        
        Returns:
            dict: Dictionary containing summary statistics including:
                - Total number of parameter combinations with recollapses
                - Distribution of number of recollapses per simulation
                - Mean time between recollapses
                - Correlations between parameters and recollapse times
        """
        stats = {
            'total_sims_with_recollapse': 0,
            'recollapse_counts': [],
            'mean_times': [],
            'parameter_correlations': {}
        }
        
        all_Z = []
        all_eta = []
        all_n = []
        all_logM = []
        all_first_recollapse_times = []
        
        # Collect data
        for Z in self.generator.parameter_space['Z']:
            for eta in self.generator.parameter_space['eta_sf']:
                for n in self.generator.parameter_space['n_cl']:
                    for logM in self.generator.parameter_space['logM']:
                        try:
                            data = self.data_manager.load_data_for_params(Z, eta, n, logM)
                            
                            if not data or 't_list_collapse' not in data['evolution']:
                                continue
                                
                            collapse_times = data['evolution']['t_list_collapse']
                            if collapse_times:
                                stats['total_sims_with_recollapse'] += 1
                                stats['recollapse_counts'].append(len(collapse_times))
                                
                                # Store first recollapse time and parameters for correlation
                                all_Z.append(Z)
                                all_eta.append(eta)
                                all_n.append(n)
                                all_logM.append(logM)
                                all_first_recollapse_times.append(collapse_times[0] / MYR_TO_SEC)  # Convert to Myr
                                
                                if len(collapse_times) > 1:
                                    mean_time = np.mean(np.diff(collapse_times)) / MYR_TO_SEC  # Convert to Myr
                                    stats['mean_times'].append(mean_time)
                                    
                        except Exception as e:
                            print(f"Error processing parameters Z={Z}, eta={eta}, n={n}, logM={logM}: {str(e)}")
                            continue
        
        # Convert to numpy arrays for statistics
        stats['recollapse_counts'] = np.array(stats['recollapse_counts'])
        stats['mean_times'] = np.array(stats['mean_times'])
        
        # Basic statistics
        if len(stats['recollapse_counts']) > 0:
            stats.update({
                'avg_recollapses_per_sim': float(np.mean(stats['recollapse_counts'])),
                'median_recollapses': float(np.median(stats['recollapse_counts'])),
                'max_recollapses': int(np.max(stats['recollapse_counts'])),
                'min_recollapses': int(np.min(stats['recollapse_counts']))
            })
        
        if len(stats['mean_times']) > 0:
            stats.update({
                'avg_time_between_recollapses_myr': float(np.mean(stats['mean_times'])),
                'median_time_between_recollapses_myr': float(np.median(stats['mean_times'])),
                'min_time_between_recollapses_myr': float(np.min(stats['mean_times'])),
                'max_time_between_recollapses_myr': float(np.max(stats['mean_times']))
            })
        
        # Calculate correlations if we have enough data points
        if len(all_first_recollapse_times) > 1:
            param_arrays = {
                'Z': np.array(all_Z),
                'eta_sf': np.array(all_eta),
                'n_cl': np.array(all_n),
                'logM': np.array(all_logM)
            }
            
            recollapse_times = np.array(all_first_recollapse_times)
            
            # Calculate correlations with first recollapse time
            correlations = {}
            for param_name, param_values in param_arrays.items():
                try:
                    correlation = np.corrcoef(param_values, recollapse_times)[0, 1]
                    correlations[param_name] = float(correlation)
                except Exception as e:
                    print(f"Error calculating correlation for {param_name}: {str(e)}")
                    correlations[param_name] = np.nan
            
            stats['parameter_correlations'] = correlations
        
        return stats


class LineInterpolantBuilder:
    """
    Builds interpolants for emission line luminosities.
    Handles both intrinsic and emergent luminosities for key diagnostic lines.
    """
    
    def __init__(self, data_manager):
        """
        Initialize the line interpolant builder.
        
        Args:
            data_manager: DataManager instance for accessing simulation data
        """
        self.data_manager = data_manager
        self.generator = data_manager.generator
        
        # Get parameter arrays from generator
        self.Z_arr = self.generator.parameter_space['Z']
        self.eta_arr = self.generator.parameter_space['eta_sf']
        self.n_arr = self.generator.parameter_space['n_cl']
        self.logM_arr = self.generator.parameter_space['logM']
        
        # Create line ID mapping
        self.line_map = self._create_line_map()

    def _normalize_wavelength(self, wavelength_value):
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

    def _create_line_map(self):
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
                wavelength = self._normalize_wavelength(line_info['wavelength'])
                
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

    def _process_luminosities(self, cloudy_data, time_point):
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
            (t[0], t[1], self._normalize_wavelength(t[2])): v 
            for t, v in intrinsic_orig.items()
        }
        emergent = {
            (t[0], t[1], self._normalize_wavelength(t[2])): v 
            for t, v in emergent_orig.items()
        }
        
        intr_lums = np.zeros(len(self.line_map))
        emer_lums = np.zeros(len(self.line_map))
        
        missing_lines = []
        for line_id, line_tuple in self.line_map.items():
            if line_tuple not in intrinsic or line_tuple not in emergent:
                missing_lines.append((line_id, line_tuple))
                
        if missing_lines:
            print("\nDetailed comparison for first missing line:")
            missing_id, missing_tuple = missing_lines[0]
            print(f"Missing line tuple: {missing_tuple}")
            print(f"Missing wavelength normalized: {self._normalize_wavelength(missing_tuple[2])}")
            print("Available tuples that look similar:")
            for available_tuple in list(intrinsic.keys())[:5]:
                print(f"  {available_tuple} (normalized wavelength: "
                      f"{self._normalize_wavelength(available_tuple[2])})")
                
        if missing_lines:
            raise ValueError(f"Missing line data for the following lines: {missing_lines}")
            
        # Fill arrays using normalized comparisons
        for line_id, line_tuple in self.line_map.items():
            intr_lums[line_id] = intrinsic[line_tuple]
            emer_lums[line_id] = emergent[line_tuple]
        
        return intr_lums, emer_lums

    def build(self):
        """
        Build interpolant for line luminosities and save line map.
        
        Returns:
            dict: Dictionary containing:
                - 'interpolant': RegularGridInterpolator for line luminosities
                - 'line_map': Mapping between numerical IDs and line tuples
                - 'line_info': Human-readable line information
        """
        # First pass to get timepoints
        timepoints = set()
        param_data = {}
        
        for Z in self.Z_arr:
            for eta in self.eta_arr:
                for n in self.n_arr:
                    for logM in self.logM_arr:
                        data = self.data_manager.load_data_for_params(Z, eta, n, logM)
                        if not data['cloudy']:
                            continue
                            
                        param_data[(Z, eta, n, logM)] = data
                        timepoints.update(data['cloudy'].keys())

        if not timepoints:
            raise ValueError("No valid timepoints found for line interpolant")

        # Convert to sorted array
        self.time_arr = np.sort(np.array(list(timepoints)))

        # Initialize luminosity array
        lum_array = np.zeros((
            len(self.line_map),    # line IDs
            2,                     # intrinsic/emergent
            len(self.time_arr),    # time points
            len(self.Z_arr),       # metallicities
            len(self.eta_arr),     # star formation efficiencies
            len(self.n_arr),       # number densities
            len(self.logM_arr)     # cloud masses
        ))
        
        # Fill array
        for i_Z, Z in enumerate(self.Z_arr):
            for i_eta, eta in enumerate(self.eta_arr):
                for i_n, n in enumerate(self.n_arr):
                    for i_m, logM in enumerate(self.logM_arr):
                        data = param_data.get((Z, eta, n, logM))
                        if not data:
                            continue
                            
                        # Process each time point
                        for i_t, t in enumerate(self.time_arr):
                            intr_lums, emer_lums = self._process_luminosities(
                                data['cloudy'], t
                            )
                            
                            if intr_lums is not None and emer_lums is not None:
                                # Store luminosities
                                lum_array[:, 0, i_t, i_Z, i_eta, i_n, i_m] = intr_lums
                                lum_array[:, 1, i_t, i_Z, i_eta, i_n, i_m] = emer_lums
        
        # Create interpolator points
        points = (
            np.arange(len(self.line_map)),  # line IDs
            np.array([0, 1]),               # intrinsic/emergent
            self.time_arr,                  # linear time
            np.log10(self.Z_arr),          # log Z
            self.eta_arr,                   # linear eta
            np.log10(self.n_arr),          # log n
            self.logM_arr                   # already log M
        )
        
        # Replace zeros and take log
        lum_array = np.clip(lum_array, 1e-99, np.inf)
        log_lum_array = np.log10(lum_array)
        
        # Create interpolant
        interpolant = RegularGridInterpolator(
            points=points,
            values=log_lum_array,
            bounds_error=True
        )
        
        # Get human-readable line info
        line_info = self.get_line_info()
        
        return {
            'interpolant': interpolant,
            'line_map': self.line_map,
            'line_info': line_info
        }

    def get_line_info(self):
        """
        Get information about mapped emission lines.
        
        Returns:
            dict: Mapping of line IDs to human-readable descriptions
        """
        line_info = {}
        for line_id, (element, ion, wavelength) in self.line_map.items():

            wave_str = f"{wavelength*1e6:.3f}μm"
                
            # Create description
            if element in {'CO', 'CN', 'CS', 'HCN', 'HNC', 'NH3'} or element.lower() == 'blnd':
                desc = f"{element} {wave_str}"
            else:
                desc = f"[{element} {ion}] {wave_str}"
                
            line_info[line_id] = desc

        return line_info


def save_interpolant(interpolant, filename):
    """Save interpolant to pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(interpolant, f)


def interpolant_exists(filepath):
    try:
        with open(filepath, 'rb') as f:
            pickle.load(f)
        return True
    except:
        return False


def main():
    parser = argparse.ArgumentParser(description='Generate TODDLERS interpolants')
    parser.add_argument('--evolution-dir', type=str, required=True,
                      help='Directory containing evolution simulation .dat files')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Directory to save interpolants')
    parser.add_argument('--dust-to-metal', type=float, default=1.0,
                      help='Dust-to-metal ratio relative to solar (default: 1.0)')

    args = parser.parse_args()

    evolution_dir = Path(args.evolution_dir)
    # Resolve to an absolute path: the Cloudy output handler chdir's into each model
    # directory while loading data and does not restore the cwd, so a relative output
    # path would no longer resolve by the time the interpolants are saved.
    output_dir = Path(args.output_dir).resolve()

    # Verify evolution directory exists and contains .dat files
    if not evolution_dir.exists():
        print(f"Error: Evolution directory {evolution_dir} does not exist")
        sys.exit(1)

    sim_files = list(evolution_dir.glob("*.dat"))
    if not sim_files:
        print(f"Error: No .dat files found in {evolution_dir}")
        sys.exit(1)

    # Derive model prefix from evolution directory structure
    evo_parts = evolution_dir.parts
    template = imf = star_type = None
    for part in evo_parts:
        if part.startswith('template_'): template = part[len('template_'):]
        elif part.startswith('imf_'): imf = part[len('imf_'):]
        elif part.startswith('star_type_'): star_type = part[len('star_type_'):]

    if not all([template, imf, star_type]):
        print("Warning: Could not extract template/imf/star_type from path, using generic names")
        model_prefix = "TODDLERS"
    else:
        model_prefix = f"{template}_{imf}_{star_type}"

    dtm = args.dust_to_metal
    dtm_suffix = dtm_label(dtm)

    print(f"\nFound {len(sim_files)} simulation files in {evolution_dir}")
    print(f"Model prefix: {model_prefix}")
    print(f"Dust-to-metal ratio: {dtm}")
    print("\nInitializing interpolant generator...")
    generator = TODDLERSInterpolantGenerator(evolution_dir, dust_to_metal=dtm)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Generate and save low-res SED interpolants
        # Naming follows config.py convention for SFR STAB pipeline
        print("\nGenerating low-resolution SED interpolants...")

        sed_low_dust = output_dir / f'TODDLERS_totSED_lr_{model_prefix}{dtm_suffix}.pkl'
        if not interpolant_exists(sed_low_dust):
            sed_interp = generator.generate_sed_interpolant(resolution='low', dust=True)
            save_interpolant(sed_interp, sed_low_dust)
        else:
            print("Low-res dust SED interpolant already exists, skipping...")

        sed_low_nodust = output_dir / f'TODDLERS_inciSED_lr_{model_prefix}{dtm_suffix}.pkl'
        if not interpolant_exists(sed_low_nodust):
            sed_interp = generator.generate_sed_interpolant(resolution='low', dust=False)
            save_interpolant(sed_interp, sed_low_nodust)
        else:
            print("Low-res no-dust SED interpolant already exists, skipping...")

        generator.data_manager.clear_cache()

        print("\nGenerating high-resolution SED interpolants...")

        sed_high_dust = output_dir / f'TODDLERS_tot_hr_{model_prefix}_lines_emergent=True{dtm_suffix}.pkl'
        if not interpolant_exists(sed_high_dust):
            sed_interp = generator.generate_sed_interpolant(resolution='high', dust=True)
            save_interpolant(sed_interp, sed_high_dust)
        else:
            print("High-res dust SED interpolant already exists, skipping...")

        sed_high_nodust = output_dir / f'TODDLERS_tot_hr_{model_prefix}_lines_emergent=False{dtm_suffix}.pkl'
        if not interpolant_exists(sed_high_nodust):
            sed_interp = generator.generate_sed_interpolant(resolution='high', dust=False)
            save_interpolant(sed_interp, sed_high_nodust)
        else:
            print("High-res no-dust SED interpolant already exists, skipping...")

        generator.data_manager.clear_cache()

        print("\nGenerating line luminosity interpolant...")
        line_interp = output_dir / 'line_luminosities_interp.pkl'
        line_map = output_dir / 'line_map.pkl'
        if not (interpolant_exists(line_interp) and interpolant_exists(line_map)):
            line_output = generator.generate_line_interpolant()
            save_interpolant(line_output['interpolant'], line_interp)
            save_interpolant({
                'line_map': line_output['line_map'],
                'line_info': line_output['line_info']
            }, line_map)
        else:
            print("Line interpolants already exist, skipping...")

        generator.data_manager.clear_cache()

        print("\nGenerating dissolution time interpolant...")
        diss_time = output_dir / 'dissolution_time_interp.pkl'
        if not interpolant_exists(diss_time):
            diss_interp = generator.generate_dissolution_interpolant()
            save_interpolant(diss_interp, diss_time)
        else:
            print("Dissolution time interpolant already exists, skipping...")

        generator.data_manager.clear_cache()

        print("\nSaving recollapse data...")
        recollapse_file = output_dir / 'recollapse_data.h5'
        if not recollapse_file.exists():
            generator.save_recollapse_data(recollapse_file)
        else:
            print("Recollapse data already exists, skipping...")

        generator.data_manager.clear_cache()

        # Generate and save property interpolants last
        print("\nGenerating property interpolants...")
        properties = [
            'velocity',
            'radius',
            'density',
            'temperature',
            'f_esc_i'
        ]

        for prop in properties:
            prop_file = output_dir / f'{prop}_interp.pkl'
            if not interpolant_exists(prop_file):
                print(f"  {prop}...")
                interp = generator.generate_property_interpolant(prop)
                save_interpolant(interp, prop_file)
            else:
                print(f"  {prop} interpolant already exists, skipping...")

        generator.data_manager.clear_cache()

        print("\nSuccessfully generated all interpolants!")
        print(f"Output files saved to: {output_dir}")
    except Exception as e:
        print(f"\nError generating interpolants: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()