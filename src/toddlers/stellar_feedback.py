# stellar_feedback.py
from .imports import np, os, pickle, interp1d
from .constants import *


class StellarFeedback:
    """
    A class to calculate various stellar feedback quantities for different star formation scenarios.

    This class can handle single burst, constant star formation rate, and multiple generations
    of star formation. It uses interpolation of pre-computed stellar evolution models to
    calculate feedback quantities.

    Attributes:
        template (str): The stellar evolution model template to use.
        Z (float): Metallicity of the stellar population.
        M_cl_init (float): Initial cloud mass in solar masses.
        eta_sf (float): Star formation efficiency.
        t_list_collapse (list): List of collapse times for multiple generations.
        mode (str): Star formation mode ('burst' or 'constant_sfr').
        formation_timescale (float): Timescale for star formation in constant SFR mode.
        imf (str): Initial Mass Function to use. Options depend on the template.
        star_type (str): For BPASS models, specifies 'sin' (single) or 'bin' (binary) stars.
        
    The class maintains backward compatibility by using default values for new parameters.
    """

    def __init__(self, template, Z, M_cl_init, eta_sf, t_list_collapse, mode='burst', 
                 formation_timescale=None, imf=None, star_type=None):
        """
        Initialize the StellarFeedback object.

        Args:
            template (str): The stellar evolution model template to use.
                Options: 'BPASS', 'SB99', 'pySB99'
            Z (float): Metallicity of the stellar population.
            M_cl_init (float): Initial cloud mass in grams.
            eta_sf (float): Star formation efficiency.
            t_list_collapse (list): List of collapse times for multiple generations.
            mode (str, optional): Star formation mode ('burst' or 'constant_sfr'). Defaults to 'burst'.
            formation_timescale (float, optional): Timescale for star formation in constant SFR mode.
                Required if mode is 'constant_sfr'.
            imf (str, optional): Initial Mass Function to use. Options depend on the template.
                For BPASS: 'chab100' (default), 'chab300'.
                For SB99: 'kroupa100' (default), 'kroupa120', 'kroupa120_rot'.
                For pySB99: Any valid IMF name, can be standard IMF or custom stellar population.
                Examples: 'kroupa100', 'my_topheavy_imf', 'ob_assoc_v1', 'single_120msun'.
                For custom populations, the interpolants are normalized per M_sun, and the
                system automatically scales based on M_stellar = eta_sf * M_cl_init.
            star_type (str, optional): For BPASS models, specifies 'sin' (single) or 'bin' (binary) stars.
                Defaults to 'bin' for BPASS. Not used for other templates.

        Raises:
            ValueError: If an invalid mode, IMF, or star_type is provided, or if formation_timescale
                        is missing for constant SFR mode.
                        
        Note:
            For pySB99 custom populations: The 'imf' parameter can refer to either a standard IMF
            or a custom stellar population. In both cases, interpolants are normalized per solar mass,
            and feedback is scaled by M_stellar. This allows custom populations to work as scalable
            templates - define a reference mix, then scale it based on cloud mass and efficiency.
        """
        self.template = template
        self.Z = Z
        self.M_cl_init = M_cl_init / M_SUN
        self.eta_sf = eta_sf
        self.t_list_collapse = t_list_collapse
        self.mode = mode
        self.formation_timescale = formation_timescale
        self.generations = len(self.t_list_collapse) + 1
        self.n_collapse = self.generations - 1
        self.derivative_dt = 0.025 * MYR_TO_SEC
        self.convolution_dt = 0.025 * MYR_TO_SEC

        # Validate mode
        if mode not in ['burst', 'constant_sfr']:
            raise ValueError("Mode must be either 'burst' or 'constant_sfr'")
        if mode == 'constant_sfr' and formation_timescale is None:
            raise ValueError("Formation timescale must be provided for constant SFR mode")

        # Template-specific validation and setup
        if self.template.startswith('BPASS'):
            self._setup_bpass(imf, star_type)
        elif self.template.startswith('SB99'):
            self._setup_sb99(imf, star_type)
        elif self.template == 'pySB99':
            self._setup_pysb99(imf, star_type)
        else:
            raise ValueError(f"Unsupported template: {self.template}. "
                           f"Supported templates: 'BPASS', 'SB99', 'pySB99'")

        # Load interpolants and calculate stellar properties
        self.load_interpolants()
        self.calculate_stellar_mass()
        self.calculate_generation_start_times()
        self.precalculate_feedback_quantities()

    def _setup_bpass(self, imf, star_type):
        """
        Setup and validation for BPASS template.
        
        Args:
            imf (str): IMF specification
            star_type (str): Star type specification
        """
        self.imf = imf or "chab100"
        self.star_type = star_type or "bin"
        
        if self.star_type not in ['sin', 'bin']:
            raise ValueError("BPASS star_type must be either 'sin' or 'bin'")
        if self.imf not in ["chab100", "chab300"]:
            raise ValueError("Invalid IMF for BPASS. Supported options are 'chab100' and 'chab300'")
        
        self._validate_metallicity(Z_MIN_BPASS, Z_MAX_BPASS, Z_VALUES_BPASS)

    def _setup_sb99(self, imf, star_type):
        """
        Setup and validation for SB99 template.
        
        Args:
            imf (str): IMF specification
            star_type (str): Star type specification
        """
        self.imf = imf or "kroupa100"
        
        if star_type == 'bin':
            raise ValueError("SB99 does not support binary stars. Use star_type='sin' or None.")
        self.star_type = 'sin'  # Always 'sin' for SB99
        
        # Special handling for rotational model
        if self.imf == "kroupa120_rot":
            if not np.isclose(self.Z, Z_KROUPA120_ROT, rtol=1e-5):
                raise ValueError(f"For Kroupa 120 rotational model, only Z={Z_KROUPA120_ROT} is supported")
        elif self.imf not in ["kroupa100", "kroupa120"]:
            raise ValueError("Invalid IMF for SB99. Supported options are 'kroupa100', 'kroupa120', 'kroupa120_rot'")
        else:
            self._validate_metallicity(Z_MIN_SB99, Z_MAX_SB99, Z_VALUES_SB99)

    def _setup_pysb99(self, imf, star_type):
        """
        Setup and validation for pySB99 template.
        
        pySB99 allows flexible IMF naming and custom metallicity grids.
        Validation is done by checking if interpolant files exist.
        
        Args:
            imf (str): IMF specification (can be any valid string)
            star_type (str): Star type specification (must be 'sin' or None)
        """
        if imf is None:
            raise ValueError("IMF must be specified for pySB99 template. "
                           "This should match the IMF name used when generating interpolants.")
        
        self.imf = imf
        
        if star_type == 'bin':
            raise ValueError("pySB99 does not support binary stars. Use star_type='sin' or None.")
        self.star_type = 'sin'  # Always 'sin' for pySB99
        
        # For pySB99, we validate metallicity after loading interpolants
        # since the valid Z grid is defined by the interpolant generation
        # We'll do a preliminary check that Z is reasonable (non-negative, not too large)
        if self.Z < 0 or self.Z > 0.1:
            raise ValueError(f"Metallicity Z={self.Z} seems unreasonable. "
                           f"Expected range: 0 <= Z <= 0.1 (0 to ~5 Z_solar)")

    def _validate_metallicity(self, Z_min, Z_max, Z_values):
        """
        Validate metallicity against allowed range and discrete values.
        
        Args:
            Z_min (float): Minimum allowed metallicity
            Z_max (float): Maximum allowed metallicity
            Z_values (np.ndarray): Array of pre-computed metallicity values
        """
        if not (Z_min <= self.Z <= Z_max):
            raise ValueError(f"Metallicity (Z={self.Z}) is outside the allowed range [{Z_min}, {Z_max}]")
        if self.Z not in Z_values:
            closest_Z = Z_values[np.argmin(np.abs(Z_values - self.Z))]
            print(f"Warning: Specified Z={self.Z} is not in the pre-computed set. "
                  f"Using closest value: {closest_Z}")
            self.Z = closest_Z

    def _validate_pysb99_metallicity(self, interpolant):
        """
        Validate metallicity for pySB99 by probing the interpolant.
        
        This function tests the interpolant at the requested metallicity to ensure
        it's within the valid range of the interpolation grid.
        
        Args:
            interpolant: 2D interpolation function (time, Z)
            
        Raises:
            ValueError: If metallicity is outside interpolant's valid range
        """
        # Try to evaluate the interpolant at a test time point
        test_time = 1.0  # 1 Myr in log space
        
        try:
            # Attempt to evaluate at requested metallicity
            _ = interpolant(test_time, self.Z)
            print(f"pySB99 metallicity Z={self.Z} validated successfully")
            
        except ValueError as e:
            # Interpolant failed - metallicity likely out of bounds
            raise ValueError(
                f"Metallicity Z={self.Z} is outside the valid range for pySB99 interpolants "
                f"with IMF '{self.imf}'. The interpolants were generated for a specific "
                f"metallicity grid. Please check the metallicities used during interpolant "
                f"generation (typically in the range of available pySB99 metallicities: "
                f"IZw18=0.0004, SMC=0.002, LMC=0.006, MW=0.014, MWC=0.020)."
            ) from e

    def load_interpolants(self):
        """
        Load pre-computed interpolation functions for various feedback quantities.

        This method reads pickle files containing interpolation functions for different
        feedback quantities based on the chosen template, IMF, and star type (for BPASS).

        For pySB99, this method also validates that the metallicity is within the
        interpolant's valid range.

        The method constructs the appropriate file names based on the template, IMF, and star type,
        and loads the corresponding interpolation functions.

        Raises:
            FileNotFoundError: If the required interpolation files are not found.
            ValueError: If an unsupported template is provided or if pySB99 metallicity is invalid.
        """
        from ._paths import get_data_dir
        database_dir = get_data_dir()  # $TODDLERS_DATA or <pkg>/database

        # Construct file prefix based on template
        if self.template.startswith('BPASS'):
            file_prefix = f"BPASSinterpolation_{self.star_type}_{self.imf}"
        elif self.template.startswith('SB99'):
            file_prefix = f"SB99interpolation_{self.imf}"
        elif self.template == 'pySB99':
            file_prefix = f"pySB99interpolation_{self.imf}"
        else:
            raise ValueError(f"Unsupported template: {self.template}")

        main_file = os.path.join(database_dir, f"{file_prefix}.obj")
        lw_file = os.path.join(database_dir, f"{file_prefix}_LumLymanWerner.obj")

        try:
            # Load main interpolants
            with open(main_file, 'rb') as file:
                object_file = pickle.load(file)

            self.L_mech_interpolant = object_file[0]
            self.F_ram_interpolant = object_file[1]
            self.L_bolo_interpolant = object_file[2]
            self.L_ion_interpolant = object_file[3]
            self.Q_ion_interpolant = object_file[4]
            self.F_ram_dot_interpolant = object_file[5]
            self.L_mech_dot_interpolant = object_file[6]

            # Load Lyman-Werner data
            with open(lw_file, 'rb') as file:
                self.L_LyW_interpolant = pickle.load(file)

        except FileNotFoundError as e:
            # Provide helpful error message with file locations
            error_msg = (
                f"Required interpolation file not found: {e.filename}\n"
                f"Expected location: {database_dir}\n"
                f"File prefix: {file_prefix}\n"
            )
            
            if self.template == 'pySB99':
                error_msg += (
                    f"\nFor pySB99, please ensure you have generated interpolants for IMF '{self.imf}'.\n"
                    f"Use generate_pysb99_interpolants.py to create the required files.\n"
                    f"Expected files:\n"
                    f"  - {file_prefix}.obj\n"
                    f"  - {file_prefix}_LumLymanWerner.obj\n"
                    f"  - {file_prefix}_mean_ion_energy.obj (optional)\n"
                    f"  - {file_prefix}_others.obj (optional)"
                )
            
            raise FileNotFoundError(error_msg)

        # Validate metallicity for pySB99 by testing interpolant
        if self.template == 'pySB99':
            self._validate_pysb99_metallicity(self.Q_ion_interpolant)

        print(f"Successfully loaded interpolation data for {self.template} "
              f"(IMF: {self.imf}, Star Type: {self.star_type if self.template.startswith('BPASS') else 'sin'})")

    def calculate_stellar_mass(self):
        """
        Calculate the stellar mass for each generation and the final cloud mass.

        This method populates the M_stellar_list attribute with the stellar mass formed
        in each generation and calculates the final cloud mass after all star formation events.
        """
        self.M_stellar_list = np.empty(self.generations)
        for i in range(self.generations):
            M_cl_i = self.M_cl_init * ((1 - self.eta_sf)**i)
            self.M_stellar_list[i] = self.eta_sf * M_cl_i
        self.M_cl = ((1 - self.eta_sf)**self.generations) * self.M_cl_init
        self.M_stellar = np.sum(self.M_stellar_list) # Total stellar mass
        self.print_mass_distribution(is_recollapse=(self.generations > 1)) # Print the mass distribution after calculation

    def get_stellar_mass_cgs(self):
        """Return the total stellar mass in CGS units (grams)."""
        return self.M_stellar * M_SUN

    def get_cloud_mass_cgs(self):
        """Return the current cloud mass in CGS units (grams)."""
        return self.M_cl * M_SUN

    def calculate_generation_start_times(self):
        """
        Calculate the start times for each generation of star formation.

        This method creates a list of start times (t_start_list) for each generation,
        using 0 as the start time for the first generation and the provided collapse
        times for subsequent generations.
        """
        self.t_start_list = [0] + self.t_list_collapse

    def calculate_derivative(self, t, quantity_func):
        """
        Calculate the numerical derivative of a quantity at a given time.

        Args:
            t (float): The time at which to calculate the derivative.
            quantity_func (callable): A function that returns the quantity value given a time.

        Returns:
            float: The numerical derivative of the quantity at time t.
        """
        q1 = quantity_func(t)
        q2 = quantity_func(t + self.derivative_dt)
        return (q2 - q1) / self.derivative_dt

    def get_feedback_quantity(self, t, interpolant, generation=None, is_derivative=False):
        """
        Get a feedback quantity at a given time, optionally for a specific generation or as a derivative.

        Args:
            t (float): The time at which to calculate the feedback quantity.
            interpolant (callable): The interpolation function for the desired quantity.
            generation (int, optional): The specific generation to calculate for. If None, calculates for all generations.
            is_derivative (bool, optional): Whether to calculate the derivative of the quantity.

        Returns:
            float or np.ndarray: The feedback quantity (or its derivative) at time t.
            Returns a scalar float if the result is a single value, otherwise returns a numpy array.
        """
        if is_derivative:
            if generation is not None:
                raise ValueError("Derivative calculation is not supported for individual generations.")
            result = self.calculate_derivative(t, lambda x: self.get_feedback_quantity(x, interpolant, None, False))
        elif self.mode == 'burst':
            result = self._get_quantity_burst(t, interpolant, generation)
        elif self.mode == 'constant_sfr':
            result = self._get_quantity_constant_sfr(t, interpolant, generation)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # Sanitize the result
        if isinstance(result, (list, tuple, np.ndarray)):
            result = np.array(result)
            return result.item() if result.size == 1 else result
        elif isinstance(result, (int, float, np.number)):
            return float(result)
        else:
            raise ValueError(f"Unexpected result type: {type(result)}")

    def _get_quantity_burst(self, t, interpolant, generation=None):
        """
        Calculate a feedback quantity for the burst star formation mode.

        Args:
            t (float): The time at which to calculate the feedback quantity.
            interpolant (callable): The interpolation function for the desired quantity.
            generation (int, optional): The specific generation to calculate for. If None, calculates for all generations.

        Returns:
            float: The total feedback quantity at time t for the burst mode.
        """
        if generation is not None:
            # Calculate for a specific generation, t should be relative in this case
            t_myr = t / MYR_TO_SEC
            try:
                interp_result = interpolant(t_myr, self.Z)
                quantity = 10**interp_result * (self.M_stellar_list[generation] / M_LIB)
                return quantity
            except Exception as e:
                print(f"Error in burst calculation for generation {generation} at t={t_myr} Myr: {str(e)}")
                return 0
        else:
            # Calculate for all generations
            total_quantity = 0
            for i in range(self.generations):
                t_start = self.t_start_list[i]
                if t >= t_start:
                    t_myr = (t - t_start) / MYR_TO_SEC
                    try:
                        interp_result = interpolant(t_myr, self.Z)
                        quantity = 10**interp_result * (self.M_stellar_list[i] / M_LIB)
                        total_quantity += quantity
                    except Exception as e:
                        print(f"Error in burst calculation at t={t_myr} Myr: {str(e)}")
            return total_quantity


    def _get_quantity_constant_sfr(self, t, interpolant, generation=None):
        """
        Calculate a feedback quantity for the constant star formation rate mode.

        Args:
            t (float): The time at which to calculate the feedback quantity.
            interpolant (callable): The interpolation function for the desired quantity.
            generation (int, optional): The specific generation to calculate for. If None, calculates for all generations.

        Returns:
            float: The total feedback quantity at time t for the constant SFR mode.
        """
        if generation is not None:
            # Calculate for a specific generation, t should be relative in this case
            t_myr = t / MYR_TO_SEC
            formation_time_myr = self.formation_timescale / MYR_TO_SEC

            sfr = self.M_stellar_list[generation] / self.formation_timescale

            impulse_times = np.arange(0, min(t_myr, formation_time_myr), self.convolution_dt / MYR_TO_SEC)
            dm = sfr * self.convolution_dt

            responses = []
            for tau in impulse_times:
                delay = t_myr - tau
                try:
                    response = 10**interpolant(delay, self.Z) * dm
                    responses.append(response)
                except Exception as e:
                    print(f"Error calculating response for generation {generation} at delay {delay} Myr: {str(e)}")

            result = np.sum(responses) if responses else 0
            return result / M_LIB
        else:
            # Calculate for all generations
            total_quantity = 0
            for i in range(self.generations):
                t_start = self.t_start_list[i]
                t_rel = t - t_start
                
                if t_rel <= 0:
                    continue
                
                t_rel_myr = t_rel / MYR_TO_SEC
                formation_time_myr = self.formation_timescale / MYR_TO_SEC

                sfr = self.M_stellar_list[i] / self.formation_timescale

                impulse_times = np.arange(0, min(t_rel_myr, formation_time_myr), self.convolution_dt / MYR_TO_SEC)

                dm = sfr * self.convolution_dt

                responses = []
                for tau in impulse_times:
                    delay = t_rel_myr - tau
                    try:
                        response = 10**interpolant(delay, self.Z) * dm
                        responses.append(response)
                    except Exception as e:
                        print(f"Error calculating response at delay {delay} Myr: {str(e)}")

                result = np.sum(responses) if responses else 0
                total_quantity += result

            return total_quantity / M_LIB


    def precalculate_feedback_quantities(self):
        """
        Precalculate feedback quantities for efficiency.

        This method precalculates feedback quantities for a range of times, storing
        the results to avoid redundant calculations during runtime.
        """
        self.time_range = np.arange(0, 50 * MYR_TO_SEC, self.derivative_dt)
        
        # Create arrays to hold the calculated values
        L_mech_values = np.zeros_like(self.time_range)
        F_ram_values = np.zeros_like(self.time_range)
        L_bolo_values = np.zeros_like(self.time_range)
        L_ion_values = np.zeros_like(self.time_range)
        Q_ion_values = np.zeros_like(self.time_range)
        L_LyW_values = np.zeros_like(self.time_range)
        
        # Fill the arrays with values calculated using the feedback quantity methods
        for i, t in enumerate(self.time_range):
            L_mech_values[i] = self.get_feedback_quantity(t, self.L_mech_interpolant)
            F_ram_values[i] = self.get_feedback_quantity(t, self.F_ram_interpolant)
            L_bolo_values[i] = self.get_feedback_quantity(t, self.L_bolo_interpolant)
            L_ion_values[i] = self.get_feedback_quantity(t, self.L_ion_interpolant)
            Q_ion_values[i] = self.get_feedback_quantity(t, self.Q_ion_interpolant)
            L_LyW_values[i] = self.get_feedback_quantity(t, self.L_LyW_interpolant)
        
        # Apply floor to prevent cubic interpolation artifacts from producing negative values
        # Physical quantities must be non-negative
        FLOOR = 1e-50
        L_mech_values = np.maximum(L_mech_values, FLOOR)
        F_ram_values = np.maximum(F_ram_values, FLOOR)
        L_bolo_values = np.maximum(L_bolo_values, FLOOR)
        L_ion_values = np.maximum(L_ion_values, FLOOR)
        Q_ion_values = np.maximum(Q_ion_values, FLOOR)
        L_LyW_values = np.maximum(L_LyW_values, FLOOR)
        
        # Create interpolants for each quantity using the time range and calculated values
        # Use linear interpolation (matching SB99) to prevent cubic artifacts and negative values
        def make_positive_interp(values, name):
            """Create linear interpolant that never returns negative values."""
            base_interp = interp1d(self.time_range, values, kind='linear', fill_value='extrapolate')
            def safe_interp(t):
                val = base_interp(t)
                return np.maximum(val, 1e-50)  # Floor at 1e-50
            return safe_interp
        
        self.precalculated_quantities = {
            'L_mech': make_positive_interp(L_mech_values, 'L_mech'),
            'F_ram': make_positive_interp(F_ram_values, 'F_ram'),
            'L_bolo': make_positive_interp(L_bolo_values, 'L_bolo'),
            'L_ion': make_positive_interp(L_ion_values, 'L_ion'),
            'Q_ion': make_positive_interp(Q_ion_values, 'Q_ion'),
            'L_LyW': make_positive_interp(L_LyW_values, 'L_LyW')
        }

    def get_precalculated_quantity(self, t, quantity_name):
        """
        Retrieve a precalculated feedback quantity at a given time.

        Args:
            t (float): The time at which to retrieve the precalculated quantity.
            quantity_name (str): The name of the feedback quantity.

        Returns:
            float: The precalculated feedback quantity at time t.
        """
        return self.precalculated_quantities[quantity_name](t)

    def get_mechanical_luminosity(self, t):
        """
        Get the mechanical luminosity at a given time.

        Args:
            t (float): The time at which to calculate the mechanical luminosity.

        Returns:
            float: The mechanical luminosity in erg/s.
        """
        return self.get_precalculated_quantity(t, 'L_mech')

    def get_ram_force(self, t):
        """
        Get the ram force at a given time.

        Args:
            t (float): The time at which to calculate the ram force.

        Returns:
            float: The ram force in dynes.
        """
        return self.get_precalculated_quantity(t, 'F_ram')

    def get_bolometric_luminosity(self, t):
        """
        Get the bolometric luminosity at a given time.

        Args:
            t (float): The time at which to calculate the bolometric luminosity.

        Returns:
            float: The bolometric luminosity in erg/s.
        """
        return self.get_precalculated_quantity(t, 'L_bolo')

    def get_ionizing_luminosity(self, t):
        """
        Get the ionizing luminosity at a given time.

        Args:
            t (float): The time at which to calculate the ionizing luminosity.

        Returns:
            float: The ionizing luminosity in erg/s.
        """
        return self.get_precalculated_quantity(t, 'L_ion')

    def get_ionizing_photon_rate(self, t):
        """
        Get the ionizing photon rate at a given time.

        Args:
            t (float): The time at which to calculate the ionizing photon rate.

        Returns:
            float: The ionizing photon rate in photons/s.
        """
        return self.get_precalculated_quantity(t, 'Q_ion')

    def get_ram_force_derivative(self, t):
        """
        Get the derivative of the ram force at a given time.

        Args:
            t (float): The time at which to calculate the ram force derivative.

        Returns:
            float: The ram force derivative in dynes/s.
        """
        if self.template == 'SB99_mean':
            return 0
        return self.calculate_derivative(t, lambda x: self.get_feedback_quantity(x, self.F_ram_interpolant))

    def get_mechanical_luminosity_derivative(self, t):
        """
        Get the derivative of the mechanical luminosity at a given time.

        Args:
            t (float): The time at which to calculate the mechanical luminosity derivative.

        Returns:
            float: The mechanical luminosity derivative in erg/s^2.
        """
        if self.template == 'SB99_mean':
            return 0
        return self.calculate_derivative(t, lambda x: self.get_feedback_quantity(x, self.L_mech_interpolant))

    def get_lyman_werner_specific_luminosity(self, t):
        """
        Get the Lyman-Werner specific luminosity at a given time.

        Args:
            t (float): The time at which to calculate the Lyman-Werner specific luminosity.

        Returns:
            float: The Lyman-Werner specific luminosity in erg/s/Hz.
        """
        return self.get_precalculated_quantity(t, 'L_LyW')

    def get_feedback_data_per_generation(self, t):
        """Get feedback data for all generations at time t."""
        feedback_data = {
            'Q_i': [],
            'L_i': [],
            'L_n': [],
            'L_mech': [],
            'F_ram': [],
        }
        
        for gen in range(self.generations):
            t_start = self.t_start_list[gen]
            if t >= t_start:
                t_rel = t - t_start # very important, the methods being called below don't do this
                feedback_data['Q_i'].append(self.get_feedback_quantity(t_rel, self.Q_ion_interpolant, generation=gen))
                feedback_data['L_i'].append(self.get_feedback_quantity(t_rel, self.L_ion_interpolant, generation=gen))
                L_bolo = self.get_feedback_quantity(t_rel, self.L_bolo_interpolant, generation=gen)
                feedback_data['L_n'].append(L_bolo - feedback_data['L_i'][-1])
                feedback_data['L_mech'].append(self.get_feedback_quantity(t_rel, self.L_mech_interpolant, generation=gen))
                feedback_data['F_ram'].append(self.get_feedback_quantity(t_rel, self.F_ram_interpolant, generation=gen))
            else:
                for key in ['Q_i', 'L_i', 'L_n', 'L_mech', 'F_ram']:
                    feedback_data[key].append(0)

        return feedback_data

    def print_mass_distribution(self, is_recollapse=False):
        """
        Print the current mass distribution of the stellar generations and remaining cloud mass.
        
        Args:
            is_recollapse (bool): If True, indicates this is a recollapse event.
        """
        if is_recollapse:
            print("\nRecollapse occurred! Updated Stellar Mass Distribution:")
        else:
            print("\nInitial Stellar Mass Distribution:")
        
        print("=" * 60)
        print(f"Initial Cloud Mass: {self.M_cl_init:.2e} M☉")
        print(f"Star Formation Efficiency: {self.eta_sf:.2%}")
        print(f"Stellar Population Model: {self.template}")
        print(f"IMF: {self.imf}")
        if hasattr(self, 'star_type'):
            print(f"Star Type: {self.star_type}")
        print(f"Cluster Formation Mode: {self.mode}")
        if self.mode == 'constant_sfr':
            print(f"Formation Timescale: {self.formation_timescale/MYR_TO_SEC:.2f} Myr")
        print(f"Number of Generations: {len(self.M_stellar_list)}")
        print("-" * 60)
        for i, mass in enumerate(self.M_stellar_list, 1):
            fraction = mass / self.M_cl_init
            print(f"Generation {i:<2}: {mass:.2e} M☉ ({fraction:.2%} of initial cloud)")
        print("-" * 60)
        print(f"Total Stellar Mass: {self.M_stellar:.2e} M☉ ({self.M_stellar/self.M_cl_init:.2%} of initial cloud)")
        print(f"Remaining Cloud Mass: {self.M_cl:.2e} M☉ ({self.M_cl/self.M_cl_init:.2%} of initial cloud)")
        print("=" * 60)

