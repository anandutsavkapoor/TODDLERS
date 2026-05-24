from .imports import np, os, re, sys, json, RegularGridInterpolator, interp1d
from .constants import *

class SpectralTableGenerator:
    def __init__(self, template, imf, star_type, max_age=None, wavelength_resolution_factor=1):
        """
        Initialize the SpectralTableGenerator.

        Args:
            template (str): The stellar evolution model template (e.g., 'BPASSv2.2.1', 'SB99', 'pySB99').
            imf (str): Initial Mass Function (e.g., 'chab100', 'kroupa100').
                      For pySB99: Must match a generated interpolant name.
            star_type (str): Type of stars ('binary' or 'single').
                           For pySB99: Always 'sin' (single stars only).
            max_age (float, optional): Maximum age in years. If None, use the default for the template.
            wavelength_resolution_factor (int, optional): Factor by which to degrade wavelength resolution.
        """
        self.template = template
        self.imf = imf
        self.star_type = star_type
        self.wavelength_resolution_factor = wavelength_resolution_factor
        
        if self.template.startswith('BPASS'):
            self.max_age = max_age if max_age is not None else 2e9 #yrs
            self.data = self._load_bpass_burst_data()
        elif self.template.startswith('SB99'):
            self.data = self._load_sb99_burst_data()
            self.max_age = np.max(self.data['ages']) # input overridden here => data is short
        elif self.template == 'pySB99':
            self.max_age = max_age if max_age is not None else 50e6  # 50 Myr default
            self.data = self._load_pysb99_burst_data()
        else:
            raise ValueError(f"Unsupported template: {self.template}")
        
        self._setup_interpolator()
        self.convolution_dt = 0.025 * MYR_TO_SEC

    def _get_burst_table_path(self):
        """Get the path to the burst table file."""
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if self.template.startswith('BPASS'):
            file_name = f"BPASSv2.2.1_imf_{self.imf}_burst_{'binary' if self.star_type == 'bin' else 'single'}.ascii"
            return os.path.join(base_path, "others", "Cloudy Spectra", file_name)
        
        elif self.template.startswith('SB99'):
            # SB99 files are organized differently, with separate files for each metallicity
            sb99_dir = os.path.join(base_path, "others", "SB99", self.imf)
            
            # return a dictionary of paths for each metallicity
            metallicities = [0.001, 0.004, 0.008, 0.020, 0.040]
            file_paths = {}
            
            for Z in metallicities:
                Z_str = f"{int(Z*1000):03d}"
                file_name = f"lim_100_Z_.{Z_str}.spectrum1"
                file_path = os.path.join(sb99_dir, f"Z_{Z_str}", file_name)
                file_paths[Z] = file_path
            
            return file_paths
        
        else:
            raise ValueError(f"Unsupported template: {self.template}")

    def _get_burst_spectrum(self, t, Z):
        """Get the spectrum for a given time and metallicity in burst mode using 2D interpolation in log-space."""
        log_Z = np.log10(Z)
        log_t = np.log10(max(t, 1e-10))  # Ensure we don't take log of 0
        log_spectrum = self.flux_interpolator((log_Z, log_t))
        return 10**log_spectrum

    def _load_bpass_burst_data(self):
        """
        Load the burst mode data from the BPASS file, treating age and metallicity as a grid.
        Ensures consistency between ages, metallicities, and fluxes.
        The ascii files were downloaded from Cloudy website, so units are consistent 
        with Cloudy.
        """
        file_path = self._get_burst_table_path()

        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Read header information
        n_models = int(lines[5].strip())
        n_wavelengths = int(lines[6].strip())
        wavelength_factor = float(lines[8].strip())
        flux_factor = float(lines[10].strip())

        # Read ages and metallicities
        ages_and_Z = []
        current_line = 11
        while len(ages_and_Z) < n_models:
            values = list(map(float, lines[current_line].split()))
            ages_and_Z.extend(zip(values[::2], values[1::2]))
            current_line += 1
        
        ages_and_Z = np.array(ages_and_Z)
        
        # Read wavelengths
        wavelengths = []
        while len(wavelengths) < n_wavelengths:
            wavelengths.extend(map(float, lines[current_line].split()))
            current_line += 1
        wavelengths = np.array(wavelengths)
        
        # Read fluxes
        fluxes = []
        for _ in range(n_models):
            flux_row = []
            while len(flux_row) < n_wavelengths:
                flux_row.extend(map(float, lines[current_line].split()))
                current_line += 1
            fluxes.append(flux_row)
        fluxes = np.array(fluxes)

        # Sort ages_and_Z and fluxes as a grid
        sorted_indices = np.lexsort((ages_and_Z[:, 1], ages_and_Z[:, 0]))
        ages_and_Z = ages_and_Z[sorted_indices]
        fluxes = fluxes[sorted_indices]

        # Extract unique ages and metallicities
        unique_ages = np.unique(ages_and_Z[:, 0])
        unique_metallicities = np.unique(10**ages_and_Z[:, 1]) # Convert from log10(Z) to Z

        # Create a grid of ages and metallicities
        n_ages = len(unique_ages)
        n_metallicities = len(unique_metallicities)

        # Reshape fluxes to match the grid structure (metallicities, ages, wavelengths)
        fluxes_grid = np.zeros((n_metallicities, n_ages, n_wavelengths))
        for i, (age, log_Z) in enumerate(ages_and_Z):
            age_index = np.where(unique_ages == age)[0][0]
            Z_index = np.where(unique_metallicities == 10**log_Z)[0][0]
            fluxes_grid[Z_index, age_index, :] = fluxes[i]

        # Create the burst_data dictionary
        burst_data = {
            'ages': unique_ages,
            'metallicities': unique_metallicities,
            'wavelengths': wavelengths * wavelength_factor,
            'fluxes': fluxes_grid * flux_factor
        }

        # Add t=0 entry (assuming it's the same as the spectrum at the earliest age)
        burst_data['ages'] = np.concatenate(([0], burst_data['ages']))
        burst_data['fluxes'] = np.concatenate((burst_data['fluxes'][:, :1, :], burst_data['fluxes']), axis=1)

        # Apply max_age filter if specified
        if self.max_age is not None:
            age_mask = burst_data['ages'] <= self.max_age
            burst_data['ages'] = burst_data['ages'][age_mask]
            burst_data['fluxes'] = burst_data['fluxes'][:, age_mask, :]

        # Apply wavelength resolution degradation
        burst_data['wavelengths'] = burst_data['wavelengths'][::self.wavelength_resolution_factor]
        burst_data['fluxes'] = burst_data['fluxes'][:, :, ::self.wavelength_resolution_factor]

        # Assign the burst_data to self.burst_data
        self.burst_data = burst_data

        return self.burst_data

    def _load_sb99_burst_data(self):
        """Load SB99 data from files downloaded from the SB99 website, unit conversion is required
        for use in Cloudy"""
        flux_fac = 4 * np.pi * (10 * PC_TO_CM)**2
        file_paths = self._get_burst_table_path()
        
        log_Z_list = []
        log_ages_list = []
        wavelengths_list = []
        log_fluxes_list = []

        for Z, file_path in file_paths.items():
            with open(file_path, 'r') as f:
                lines = f.readlines()[6:]  # Skip header lines
            
            ages = []
            wavelengths = []
            fluxes = []
            
            for line in lines:
                data = line.split()
                age = float(data[0])
                wavelength = float(data[1])
                flux = float(data[3]) - np.log10(flux_fac) # Stellar, not total, flux at 10pc 
                
                if len(ages) == 0 or age != ages[-1]:
                    if len(ages) > 0:
                        log_Z_list.append(np.log10(Z))
                        log_ages_list.append(np.log10(ages[-1]))
                        wavelengths_list.append(wavelengths)
                        log_fluxes_list.append(fluxes)
                    ages.append(age)
                    wavelengths = []
                    fluxes = []
                
                wavelengths.append(wavelength)
                fluxes.append(flux)
            
            # Append the last age group
            log_Z_list.append(np.log10(Z))
            log_ages_list.append(np.log10(ages[-1]))
            wavelengths_list.append(wavelengths)
            log_fluxes_list.append(fluxes)

        # Ensure all wavelength grids are the same
        assert all(np.array_equal(wavelengths_list[0], w) for w in wavelengths_list[1:])
        wavelengths = np.array(wavelengths_list[0])
        
        # Create unique sorted arrays for log_Z and log_ages
        log_Z = np.unique(log_Z_list)
        log_ages = np.unique(log_ages_list)

        # Create a 3D grid: (log_Z, log_age, wavelength)
        log_fluxes = np.array(log_fluxes_list).reshape(len(log_Z), len(log_ages), len(wavelengths))

        # Convert to linear scale for consistency with BPASS output
        burst_data = {
            'ages': 10**log_ages,
            'metallicities': 10**log_Z,
            'wavelengths': wavelengths,
            'fluxes': 10**log_fluxes
        }

        # Add t=0 entry (assuming it's the same as the spectrum at the earliest age)
        burst_data['ages'] = np.concatenate(([0], burst_data['ages']))
        burst_data['fluxes'] = np.concatenate((burst_data['fluxes'][:, :1, :], burst_data['fluxes']), axis=1)

        # Assign the burst_data to self.burst_data
        self.burst_data = burst_data

        return self.burst_data

    def _load_pysb99_burst_data(self):
        """
        Load burst mode data from pySB99 by running population synthesis.
        
        This method:
        1. Verifies that pySB99 interpolants exist for this IMF
        2. Loads metadata to get metallicities and spectral library
        3. Runs pySB99 for each metallicity used in interpolant generation
        4. Extracts wavelength-resolved spectral data
        5. Organizes into the standard burst_data format
        
        Returns:
            dict: Dictionary with keys 'ages', 'metallicities', 'wavelengths', 'fluxes'
        """
        from toddlers.pysb99.pysb99_core import StellarPopulationConfig, run_population_synthesis
        
        print(f"Generating pySB99 spectral data for IMF: {self.imf}")
        
        # Verify interpolants exist
        database_dir = os.path.join(src_dir, 'database')
        interpolant_file = os.path.join(database_dir, f'pySB99interpolation_{self.imf}.obj')
        metadata_file = os.path.join(database_dir, f'pySB99interpolation_{self.imf}_metadata.json')
        
        if not os.path.exists(interpolant_file):
            raise FileNotFoundError(
                f"pySB99 interpolant not found: {interpolant_file}\n"
                f"Generate interpolants first using generate_cloudy_spectral_tables.py"
            )
        
        # Load metadata to get metallicities and spectral library
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            metallicity_strings = metadata['metallicities']
            Z_values = metadata['Z_values']
            spectral_library = metadata['spectral_library']
            rotation = metadata.get('rotation', False)
            population_type = metadata.get('population_type', 'imf')
            
            # Get IMF or custom population parameters
            if population_type == 'custom':
                # JSON converts dict keys to strings - convert back to floats
                custom_star_numbers_raw = metadata.get('custom_star_numbers', None)
                if custom_star_numbers_raw is not None:
                    custom_star_numbers = {float(k): v for k, v in custom_star_numbers_raw.items()}
                else:
                    custom_star_numbers = None
                imf_exponents = None
                imf_mass_limits = None
            else:
                custom_star_numbers = None
                imf_exponents = metadata.get('imf_exponents', [1.3, 2.3])
                imf_mass_limits = tuple(metadata.get('imf_mass_limits', (0.1, 0.5, 100.0)))
            
            print(f"  Loaded metadata: {len(metallicity_strings)} metallicities, library={spectral_library}")
            print(f"  Population type: {population_type}")
        else:
            # Metadata is required - cannot safely assume defaults
            raise FileNotFoundError(
                f"Metadata file required but not found: {metadata_file}\n"
                f"Generate interpolants with metadata using:\n"
                f"  - generate_kroupa_like_interpolants() for IMF populations\n"
                f"  - generate_custom_population_interpolants() for custom populations\n"
                f"Or run execute_pysb99_examples.py which includes metadata generation."
            )
        
        metallicity_strings = list(metallicity_strings)
        Z_values = list(Z_values)
        
        print(f"  Metallicities: {metallicity_strings}")
        
        # Time grid
        n_ages = 50
        age_min = 1e4  # 10 kyr
        age_max = self.max_age
        time_points = np.logspace(np.log10(age_min), np.log10(age_max), n_ages)
        
        print(f"  Time: {age_min/1e6:.3f} - {age_max/1e6:.1f} Myr ({n_ages} points)")
        
        # Run pySB99 for each metallicity
        all_spectra = []
        wavelength_grid = None
        
        for met_string in metallicity_strings:
            print(f"  Running pySB99 for {met_string}...")
            
            config = StellarPopulationConfig(
                metallicity=met_string,
                time_start=time_points[0],
                time_end=time_points[-1],
                time_step=(time_points[-1] - time_points[0]) / n_ages,
                spectral_library=spectral_library,
                rotation=rotation,
                run_speed_mode='FAST'
            )
            
            # Set population parameters
            if custom_star_numbers is not None:
                # Custom stellar population
                config.custom_star_numbers = custom_star_numbers
                config.total_mass = None  # Not used with custom populations
            else:
                # IMF-based population
                config.total_mass = 1e6
                config.imf_exponents = imf_exponents
                config.imf_mass_limits = imf_mass_limits
            
            # Run synthesis
            results = run_population_synthesis(config)
            
            # Get actual stellar mass for normalization
            if custom_star_numbers is not None:
                # Custom population: use actual mass
                initial_masses = np.array(results.stellar_masses[0])
                actual_mass = np.sum(results.number_of_stars * initial_masses)
                normalization_factor = 1e6 / actual_mass
                print(f"    Custom population: {actual_mass:.2e} Msun --> normalizing to 1e6 Msun (x{normalization_factor:.3f})")
            else:
                # IMF population: already 1e6 Msun
                normalization_factor = 1.0
            
            # Extract wavelength grid
            if wavelength_grid is None:
                wavelength_grid = results.wavelength_grid
            
            # Interpolate spectra to desired time points
            met_spectra = []
            for t in time_points:
                log_t = np.log10(t)
                log_times = np.log10(results.times)
                
                spectrum = np.zeros(len(wavelength_grid))
                for i in range(len(wavelength_grid)):
                    safe_flux = np.maximum(results.flux_spectra[:, i], 1e-50)
                    log_flux = np.log10(safe_flux)
                    
                    interp_func = interp1d(
                        log_times, log_flux,
                        kind='linear',
                        bounds_error=False,
                        fill_value='extrapolate'
                    )
                    # pySB99 units (10^-20 erg/s/A), make it even smaller
                    # this doesnt matter as Cloudy renormalizes anyway
                    # but Cloudy complains if numbers are too large, adding a 1e-20 factor
                    spectrum[i] = 10**interp_func(log_t) * 1e-20 * normalization_factor
                
                met_spectra.append(spectrum)
            
            all_spectra.append(np.array(met_spectra))
        
        # Organize into standard format
        fluxes_grid = np.array(all_spectra)
        
        burst_data = {
            'ages': time_points,
            'metallicities': np.array(Z_values),
            'wavelengths': wavelength_grid,
            'fluxes': fluxes_grid
        }
        
        print(f"  Spectral data ready: {burst_data['fluxes'].shape}")
        
        # Apply wavelength resolution degradation if needed
        if self.wavelength_resolution_factor > 1:
            print(f"  Degrading wavelength resolution by factor {self.wavelength_resolution_factor}...")
            
            n_wav = len(burst_data['wavelengths'])
            new_n_wav = n_wav // self.wavelength_resolution_factor
            
            burst_data['wavelengths'] = burst_data['wavelengths'][::self.wavelength_resolution_factor][:new_n_wav]
            
            new_fluxes = np.zeros((fluxes_grid.shape[0], fluxes_grid.shape[1], new_n_wav))
            for i in range(new_n_wav):
                start = i * self.wavelength_resolution_factor
                end = min(start + self.wavelength_resolution_factor, n_wav)
                new_fluxes[:, :, i] = np.mean(fluxes_grid[:, :, start:end], axis=2)
            
            burst_data['fluxes'] = new_fluxes
            print(f"    New wavelength grid: {len(burst_data['wavelengths'])} points")
        
        self.burst_data = burst_data
        return self.burst_data

    def _setup_interpolator(self):
        """Set up 2D interpolator for age and metallicity in log-space."""
        log_metallicities = np.log10(self.burst_data['metallicities'])
        log_ages = np.log10(self.burst_data['ages'] + 1e-10)
        log_fluxes = np.log10(self.burst_data['fluxes'])

        self.flux_interpolator = RegularGridInterpolator(
            (log_metallicities, log_ages),
            log_fluxes,
            method='linear',
            bounds_error=False,
            fill_value=None
        )

    def _get_burst_spectrum(self, t, Z):
        """Get the spectrum for a given time and metallicity in burst mode using 2D interpolation in log-space."""
        log_Z = np.log10(Z)
        log_t = np.log10((max(t, 1e-10)))
        log_spectrum = self.flux_interpolator((log_Z, log_t))
        return 10**log_spectrum
    
    def _get_constant_sfr_spectrum(self, t, Z, formation_timescale):
        """Get the spectrum for a given time and metallicity in constant SFR mode.
        formation time-scale in years"""
        spectrum = np.zeros_like(self.burst_data['wavelengths'])
        dt = self.convolution_dt / YR_TO_SEC  # Convert to years
        
        for tau in np.arange(0, min(t, formation_timescale), dt):
            weight = dt / formation_timescale
            burst_spectrum = self._get_burst_spectrum(t - tau, Z)
            spectrum += weight * burst_spectrum

        return spectrum

    def _generate_output_filename(self, formation_timescale=None):
        """
        Generate an appropriate filename for the spectral table based on its characteristics.

        Args:
            formation_timescale (float, optional): Formation timescale in years for constant SFR mode.

        Returns:
            str: The generated filename.
        """

        if self.template.startswith('BPASS'): # Remove version number from BPASS
            template = re.sub(r'(BPASS)v\d+\.\d+\.\d+', r'\1', self.template)
        elif self.template.startswith('SB99'):
            template = 'SB99'  # using 'SB99' without any version
        elif self.template == 'pySB99':
            template = 'pySB99'
        else:
            template = self.template

        # Determine the mode (burst or constant SFR)
        mode = "burst" if formation_timescale is None else "constantSFR"

        # Create the base filename
        filename = f"{template}_{self.imf}_{self.star_type}_{mode}"

        # Add formation timescale for constant SFR mode
        if formation_timescale is not None:
            formation_timescale_myr = formation_timescale / 1e6
            filename += f"_t{formation_timescale_myr:.1e}myr"

        # Add wavelength resolution factor if it's not 1
        if self.wavelength_resolution_factor != 1:
            filename += f"_resFac{self.wavelength_resolution_factor}"

        # Add file extension
        filename += ".ascii"

        return filename

    def _write_spectral_table(self, time_points, Z_values, wavelengths, spectra, formation_timescale=None):
        """
        Write the spectral table in Cloudy-compatible format, exactly matching the input file format.
        If the file already exists, it skips the generation process.

        Args:
            time_points (array-like): Array of time points.
            Z_values (array-like): Array of metallicity values.
            wavelengths (array-like): Array of wavelengths.
            spectra (list): List of spectra for each metallicity and time point.
            formation_timescale (float, optional): Formation timescale for constant SFR mode.

        Returns:
            str: Path to the spectral table file.
        """
        # Generate the filename
        filename = self._generate_output_filename(formation_timescale)

        # Construct the full path with the cloudy_spectra directory
        full_path = os.path.join(CLOUDY_DATA_DIR, filename)

        # Check if the file already exists
        if os.path.exists(full_path):
            print(f"Spectral table '{filename}' already exists. Skipping generation.")
            return full_path

        # If the file doesn't exist, generate it
        with open(full_path, 'w') as f:
            f.write("20060612\n")
            f.write("2\n2\n")
            f.write("Age\nlog(Z)\n")
            f.write(f"{len(time_points) * len(Z_values)}\n")
            f.write(f"{len(wavelengths)}\n")
            f.write("lambda\n1.00000000e+00\n")
            f.write("F_lambda\n1.00000000e+00\n")

            # Write age-metallicity combinations
            for Z in Z_values:
                for t in time_points:
                    f.write(f"{t:.6e} {np.log10(Z):.6f}\n")

            # Write wavelengths
            for i, w in enumerate(wavelengths):
                f.write(f"{w:.6e}")
                if (i + 1) % 5 == 0 or i == len(wavelengths) - 1:
                    f.write("\n")
                else:
                    f.write("  ")

            # Write spectra
            for Z_spectra in spectra:
                for spectrum in Z_spectra:
                    for i, flux in enumerate(spectrum):
                        f.write(f"{flux:.6e}")
                        if (i + 1) % 5 == 0 or i == len(spectrum) - 1:
                            f.write("\n")
                        else:
                            f.write("  ")
        return full_path

    def generate_spectral_table(self, time_points, Z_values, formation_timescale=None):
        """
        Generate a spectral table for Cloudy, supporting both burst and constant SFR modes for multiple metallicities.

        Args:
            time_points (np.array): Array of time points (in years) for which to generate spectra.
            Z_values (list or np.array): List of metallicities in solar units.
            formation_timescale (float, optional): Formation timescale in years for constant SFR mode.
                                                   If None, burst mode is assumed.

        Returns:
            str: Path to the generated (or existing) spectral table file.
        """
        # Generate the filename
        filename = self._generate_output_filename(formation_timescale)
        full_path = os.path.join(CLOUDY_DATA_DIR, filename)

        # Check if the file already exists
        if os.path.exists(full_path):
            print(f"Spectral table '{filename}' already exists. Skipping generation.")
            return

        # If the file doesn't exist, generate it
        spectra = []
        for Z in Z_values:
            Z_spectra = []
            for t in time_points:
                if formation_timescale is None:
                    spectrum = self._get_burst_spectrum(t, Z)
                else:
                    spectrum = self._get_constant_sfr_spectrum(t, Z, formation_timescale)
                Z_spectra.append(spectrum)
            spectra.append(Z_spectra)

        self._write_spectral_table(time_points, Z_values, self.burst_data['wavelengths'], spectra, formation_timescale)      
        print(f"Spectral table '{filename}' has been generated and saved.")
        return