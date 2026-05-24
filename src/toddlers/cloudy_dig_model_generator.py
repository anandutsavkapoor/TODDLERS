from .imports import np
from .constants import *
from .cloudy_output_handler import CloudyOutputHandler

class DIGModelGenerator:
    """
    Generates Cloudy input files for the Diffuse Ionized Gas (DIG) model.

    This class is responsible for creating Cloudy input files for the DIG model,
    which includes components like direct stellar radiation, radiation from the
    inner model, and old stellar population background radiation.  The old
    stellar background is applied with 'illumination isotropic' so that Cloudy
    treats it as an isotropic bath rather than a beamed point source.

    Attributes:
        base (BaseCloudyInputGenerator): The base input generator instance.
        add_old_stellar_radiation (bool): Whether to include old stellar radiation.
        density (float): Hydrogen number density (n_H) for the DIG model in cm^-3.
        logU_background (float): Log of ionization parameter for background radiation.
        model_prefix (str): Prefix for the DIG model files.

    Methods:
        get_source_options(t, inner_model_prefix, inner_escape_fraction): Generates source options for the DIG model.
        get_old_stellar_source(): Gets the source specification for old stellar population.
        calculate_old_stellar_luminosity(t, inner_model_prefix, inner_escape_fraction):
            Calculates the luminosity of the old stellar population.
        get_inner_model_radius(t, inner_model_prefix): Gets the radius of the inner model.
        create_inner_sed_file(t, inner_model_prefix): Creates an SED file for the inner model.
        get_density(): Gets the density for the DIG model.
        get_density_options(): Generates density options.
        get_geometry_options(t, inner_model_prefix): Generates geometry options.
        get_stopping_criterion(): Generates stopping criteria for the Cloudy simulation.
        write_input_file(t, inner_model_prefix, inner_escape_fraction): Writes the complete Cloudy input file.
        calculate_total_Q_young(t, inner_escape_fraction): Calculates total ionizing photon rate from young stars.
    """

    def __init__(self, base_generator, add_old_stellar_radiation=False,
                density=None, logU_background=None, logger=None, include_grains=False,
                add_turbulent_dissipation=False, turbulent_velocity=None):
        """
        Initialize the DIGModelGenerator.

        Args:
            base_generator (BaseCloudyInputGenerator): The base input generator instance.
            add_old_stellar_radiation (bool, optional): Whether to include old stellar radiation. 
                Defaults to False.
            density (float, optional): Hydrogen number density (n_H) for the DIG model in cm^-3. Defaults to None.
            logU_background (float, optional): Log of ionization parameter for background radiation. 
                Defaults to None.
            include_grains (bool): Include grain options or not
            add_turbulent_dissipation (bool, optional): Include a uniform turbulent dissipation
                as heating or not. Defaults to False.
            turbulent_velocity (float, optional): User-specified turbulent velocity in km/s. If None,
                uses sound speed at T=10^4 K multiplied by default Mach number. Defaults to None.
        """
        self.logger = logger
        self.base = base_generator
        self.add_old_stellar_radiation = add_old_stellar_radiation
        self.include_grains = include_grains

        self.density = density

        if add_old_stellar_radiation:
            self.logU_background = logU_background if logU_background is not None else LOGU_BG_OLD_DEFAULT
        elif logU_background is not None:
            raise ValueError("logU_background should only be provided when add_old_stellar_radiation is True.")
        else:
            self.logU_background = None

        self.add_turbulent_dissipation = add_turbulent_dissipation
        self.turbulent_velocity = turbulent_velocity
        
        if self.add_turbulent_dissipation:
            # Set a large dissipation scale - making heating uniform in the model
            self.dissipation_scale = TURBULENT_DISSIPATION_SCALE  # set to 100 kpc
            self.logger.info("Turbulent dissipation heating is enabled")
            self.logger.info(f"Using large dissipation scale: {self.dissipation_scale/PC_TO_CM:.1f} pc")
            if self.turbulent_velocity is not None:
                self.logger.info(f"Using user-specified turbulent velocity: {self.turbulent_velocity:.1f} km/s")

        self.model_prefix = 'dig'

    def get_source_options(self, t, inner_model_prefix, inner_escape_fraction):

        covering_fraction = self.base.get_effective_covering_fraction(t) # 0 after dissolution

        source_options = []
        # 1. Direct stellar radiation if inner model is not fully covering, works with dissolved shells
        if covering_fraction < 1:
            direct_fraction = 1 - covering_fraction
            direct_source_options = self.base.get_primary_source_options(t, scaling=direct_fraction) # each of the generations, with same scaling
            source_options.extend(direct_source_options)
        
        # 2. Radiation from the leaky inner model (shell or unified)
        if inner_escape_fraction > ESC_FRAC_ZERO and covering_fraction > 0: # the second condition fails after dissolution
            inner_sed_file = self.create_inner_sed_file(t, inner_model_prefix)
            _, inner_Q, _  = self.calculate_total_Q_young(t, inner_escape_fraction) # includes cover * escape
            source_options.append(f"table SED \"{inner_sed_file}\"")
            source_options.append(f"luminosity={np.log10(inner_Q * E_LYC)}")

        # 3. Old stellar population (background ionizing source).
        #    Uses 'illumination isotropic' so the background field is
        #    treated as an isotropic bath rather than a beamed point source,
        #    which is more physical for a pervasive older population.
        if self.add_old_stellar_radiation:
            old_stellar_source = self.get_old_stellar_source()
            old_stellar_luminosity = self.calculate_old_stellar_luminosity(t, inner_model_prefix)
            source_options.append(f"{old_stellar_source}")
            source_options.append(f"luminosity {np.log10(old_stellar_luminosity)}")
            source_options.append("illumination isotropic")
        return source_options

    def get_old_stellar_source(self):
        """Get the source specification for the old stellar population."""
        return f'table star "BPASS_chab100_sin_burst_resFac10.ascii" age=1e9 logZ={np.log10(self.base.Z)}'

    def calculate_old_stellar_luminosity(self, t, inner_model_prefix):
        """Calculate the luminosity of the old stellar population to achieve the specified logU_background."""
        r_dig = self.get_inner_model_radius(t, inner_model_prefix) # Slightly larger than inner model radius
        n_H = self.get_density()
        U_background = 10**self.logU_background
        Q_background = 4 * np.pi * r_dig**2 * C * U_background * n_H
        return MEAN_E_FACTOR_OLD * E_LYC * Q_background # erg/s, the ionizing luminosity at 1e9 yr

    def get_inner_model_radius(self, t, inner_model_prefix):
        """
        Get the radius of the inner model, handling cases with and without dissolution.
        
        Args:
            t (float): Current time in seconds
            inner_model_prefix (str): Prefix for inner model files
            
        Returns:
            float: Radius in cm
        """
        if self.base.dissolution_time is None or t < self.base.dissolution_time:
            output_handler = CloudyOutputHandler(inner_model_prefix, t)
            radius_data = output_handler.get_radial_structure()
            return radius_data[0][-1]  # Return the last (outermost) radius
        else:
            # For dissolved case, use the radius at dissolution time
            return self.base.radius_interp(self.base.dissolution_time)  # extrapolates to last

    def create_inner_sed_file(self, t, inner_model_prefix):
        """Create SED file from inner model transmitted spectrum."""
        output_handler =  CloudyOutputHandler(inner_model_prefix, t)
        inner_cont = output_handler.get_continuum()
        wavelength = inner_cont['nu']
        transmitted = inner_cont['transmitted']
        
        sed_filename = f"{inner_model_prefix}_{t/MYR_TO_SEC:.2f}_transmitted.sed"
        self.logger.info(f"Creating SED file: {sed_filename}")
        
        threshold = 1e-4
        self.logger.debug(f"Using threshold value for zero fluxes: {threshold:.2e}")
        
        try:
            with open(sed_filename, 'w') as f:
                f.write("# Transmitted incident SED from inner model (col #3 from .cont file)\n")
                f.write("# wavelength (microns)  nu*f_nu\n")
                # Write the first line with units specification
                w = wavelength[0]
                tr = transmitted[0]
                nu_fnu = tr # luminosity units
                f.write(f"{w:.6e} {nu_fnu:.6e} nuFnu units micron\n")
                # Write the rest of the data
                for w, tr in zip(wavelength[1:], transmitted[1:]):
                    nu_fnu = max(tr, threshold)
                    f.write(f"{w:.6e} {nu_fnu:.6e}\n")
                f.write("***\n")
            self.logger.info(f"Successfully created SED file: {sed_filename}")
        except Exception as e:
            self.logger.error(f"Error creating SED file {sed_filename}: {str(e)}")
            raise
            
        return sed_filename

    def get_density(self, print_message=False):
        """Get the density of the DIG."""
        if self.density is not None:
            if print_message:
                self.logger.info(f"Using specified density: {self.density:.2e} cm^-3")
            return self.density
                
        self.logger.info(f"Using default DIG density: {N_ISM_CLOUDY:.2e} cm^-3")
        return N_ISM_CLOUDY

    def get_density_options(self):
        n_H = self.get_density(print_message=True)
        return [f'hden {np.log10(n_H)}']

    def get_geometry_options(self, t, inner_model_prefix):
        r_dig = self.get_inner_model_radius(t, inner_model_prefix,)
        return [
            'sphere',
            f'Radius inner = {r_dig / PC_TO_CM:.6e} parsec linear'
        ]

    def get_turbulence_options(self):
        """
        Create turbulence commands for the DIG model, with dissipation heating if requested.
        
        The method follows Cloudy's parameter ordering requirements (see Hazy 10.9.5):
        - First number: turbulent velocity in km/s
        - Second number: F parameter for turbulence isotropy
        - Third number: dissipation scale length in log cm
        
        If turbulent velocity is not specified, uses the sound speed in ionized gas 
        at T=10⁴ K multiplied by the default Mach number.
        
        Returns:
            list: Cloudy commands for turbulence settings. Empty list if dissipation not requested.
        
        Notes:
        - Uses F=3 for isotropic turbulent motions (Heiles & Troland 2005, eq 34)
        - Sets a large dissipation scale (~100 kpc) to achieve uniform heating
        - If turbulent_velocity is specified, uses that.
        """
        F = 3 
        if not self.add_turbulent_dissipation:
            return []
            
        if self.turbulent_velocity is not None:
            velocity = self.turbulent_velocity
        else:
            # Use sound speed in ionized gas at T=10^4 K in km/s
            c_s = np.sqrt((GAMMA * K_BOLTZMANN * T_ION) / MU_N) / KM_TO_CM
            velocity = MACH_DIG_DEF * c_s

        return [f'turbulence {velocity:.6f} km/sec {F} dissipate {np.log10(self.dissipation_scale)}']
        
    def get_stopping_criterion(self):
        return [
            "Stop temperature 100K",
            'stop ionization efrac 0.01',
        ]

    def write_input_file(self, t, inner_model_prefix):
        """
        Write the complete Cloudy input file for the DIG model.

        Args:
            t (float): Current time in seconds.
            inner_model_prefix (str): Prefix for the inner model ('shell' or 'unified').

        Returns:
            bool: True if input file was written, False if inner model failed.
        """
        if self.base.dissolution_time is not None and t >= self.base.dissolution_time:
            inner_escape_fraction = 1.0
            self.logger.info(f"t = {t/MYR_TO_SEC:.2f} Myr is past dissolution ==> " 
                            f"Using shell model with escape fraction: 1.0")
        else:
            inner_output_handler = CloudyOutputHandler(inner_model_prefix, t)
            if inner_output_handler.check_cloudy_success():
                inner_escape_fraction = inner_output_handler.calculate_escape_fraction()
                self.logger.info(f"t = {t/MYR_TO_SEC:.2f} Myr, " 
                                f"Using {inner_model_prefix} model with escape fraction: {inner_escape_fraction:.2f}")
            else:
                self.logger.error(f"Inner {inner_model_prefix} model is not delivering a valid Escape fraction, this model will not run")
                return False

        file_path = f"{self.model_prefix}_{t/MYR_TO_SEC:.2f}.in"
        inner_file = f"{inner_model_prefix}_{t/MYR_TO_SEC:.2f}.in"
        with open(file_path, 'w') as f:
            
            f.write(f"# DIG model at t = {t/MYR_TO_SEC:.2f} Myr\n")
            f.write(f"# Inner model file: {inner_file}\n")
            f.write(f"# Inner model escape fraction: {inner_escape_fraction:.2e}\n")
            f.write(f"# Inner model covering fraction: {self.base.get_effective_covering_fraction(t):.2e}\n")
            
            if self.add_old_stellar_radiation:
                f.write(f"# logU background (old stars): {self.logU_background:.2f}\n")
            
            for option in self.get_source_options(t, inner_model_prefix, inner_escape_fraction):
                f.write(f"{option}\n")

            for option in self.get_density_options():
                f.write(f"{option}\n")

            for option in self.get_turbulence_options():
                f.write(f"{option}\n")
            
            for option in self.get_geometry_options(t, inner_model_prefix):
                f.write(f"{option}\n")
            
            for option in self.base.get_abundance_options():
                f.write(f"{option}\n")
            
            if self.include_grains:
                for option in self.base.get_grain_options(skip_pah=True):
                    f.write(f"{option}\n")

            for option in self.base.get_cr_options(t, scale_with_Z=False):
                f.write(f"{option}\n")
            
            for option in self.base.get_other_physics_options():
                f.write(f"{option}\n")

            speedup_options = (self.base.get_speedup_options() if self.include_grains 
                             else self.base.get_speedup_options_no_grains())
            for option in speedup_options:
                f.write(f"{option}\n")
            
            for option in self.get_stopping_criterion():
                f.write(f"{option}\n")
            
            for option in self.base.get_output_options(grains=self.include_grains):
                f.write(f"{option}\n")
            
            return True

    def calculate_total_Q_young(self, t, inner_escape_fraction):
        """
        Calculate the total ionizing photon rate from young stellar populations.
        
        Args:
            t (float): Current time in seconds.
            inner_escape_fraction (float): Escape fraction from the inner model.
            
        Returns:
            tuple: (direct_Q, inner_Q, total_Q) where:
                - direct_Q: Direct ionizing photon rate from uncovered stars
                - inner_Q: Ionizing photon rate through the shell/unified model
                - total_Q: Total ionizing photon rate (direct + inner)
        """
        covering_fraction = self.base.get_effective_covering_fraction(t)
        
        # Get total Q from stellar feedback
        feedback_data = self.base.stellar_feedback.get_feedback_data_per_generation(t)
        stellar_Q = sum(feedback_data['Q_i'])  # Sum Q_i across all generations
        
        # Calculate components
        # 1. Direct stellar radiation
        direct_Q = stellar_Q * (1 - covering_fraction)

        # 2. Radiation from the leaky inner model (shell or unified)
        inner_Q = stellar_Q * inner_escape_fraction * covering_fraction
        
        return direct_Q, inner_Q, direct_Q + inner_Q