from .imports import np, interp1d
from .constants import *
from .cloud_density_profiles import create_density_profile
from .cloudy_output_handler import CloudyOutputHandler

class UnifiedModelGenerator:
    """
    Generates Cloudy input files for the unified model.

    This class creates Cloudy input files for the unified model, which combines
    the shell and the remaining cloud into a single structure.

    Attributes:
        model_prefix (str): Prefix for the unified model files.
        base (BaseCloudyInputGenerator): The base input generator instance.

    Methods:
        get_density_and_geometry_options(t): Generates density and geometry options for the unified model.
        _create_density_law(t): Creates a density law combining shell and cloud profiles.
        _resample_profile(r_values, n_H_values, split_radius, spacing_before, spacing_after):
            Resamples the density profile to avoid too closely spaced points.
        _write_density_law_file(filename, density_law): Writes the density law to a file.
        get_stopping_criteria(t): Generates stopping criteria for the Cloudy simulation.
        write_input_file(t): Writes the complete Cloudy input file for the unified model.
    """

    def __init__(self, base_generator, logger=None):
        """
        Initializes the UnifiedModelGenerator.

        Args:
            base_generator (BaseCloudyInputGenerator): The base input generator instance.
        """
        self.logger = logger
        self.model_prefix = "unified"
        self.base = base_generator

    def get_density_and_geometry_options(self, t):
        """
        Generate density options for the unified model of the shell + leftover cloud.
        """
        density_law, shell_inner_radius = self._create_density_law(t)
        density_law_filename = f"{self.model_prefix}_density_law_{t/MYR_TO_SEC:.2f}.ini"
        self._write_density_law_file(density_law_filename, density_law)
        
        return [
            f'init "{density_law_filename}"',
            self.base.get_turbulence_str(t),
            'sphere',
            f'Radius inner = {shell_inner_radius / PC_TO_CM:.6e} parsec linear'
        ]

    def _check_mass_sufficiency(self, r_values, n_H_values, expected_mass):
        """
        Check if the density profile provides enough mass (must exceed expected cloud mass).

        Args:
            r_values (np.ndarray): Radial coordinates in cm
            n_H_values (np.ndarray): Hydrogen number densities in cm^-3
            expected_mass (float): Expected cloud mass in g

        Returns:
            bool: True if mass is sufficient, False otherwise

        Notes:
            Density profile must provide more mass than expected cloud mass. Cloudy will automatically
            truncate at the correct mass during the simulation, but will raise a
            'requested radius outside range of dense_tabden' error if the profile contains
            insufficient mass requiring extrapolation.
        """
        integrand = 4 * np.pi * r_values**2 * n_H_values
        total_mass = np.trapz(integrand, r_values) * (MU_N / self.base.nuclei_to_h)
        
        self.logger.debug(f"Total mass in profile (should be > cloud mass): {total_mass/M_SUN:.2e} M_sun")
        self.logger.debug(f"Expected cloud mass: {expected_mass/M_SUN:.2e} M_sun")
        
        return total_mass > expected_mass

    def _create_density_law(self, t):
        """
        Create a density law for the unified model, combining shell and cloud profiles.
        
        Uses an extended outer radius to ensure _resample_profile works, while 
        the correct mass is maintained through Cloudy's stopping criteria.
        
        Args:
            t (float): Current time in seconds.
        
        Returns:
            tuple: (density_law, shell_inner_radius)
                - density_law: List of tuples (radius, n_H) representing the density law
                - shell_inner_radius: Inner radius of the shell
        """
        self.logger.debug(f"Creating density law at t={t/MYR_TO_SEC:.2f} Myr")

        output_handler = CloudyOutputHandler('shell', t)
        shell_density = output_handler.get_density_structure()
        shell_radii, shell_depth = output_handler.get_radial_structure()
        
        shell_inner_radius = shell_radii[0]
        shell_thickness = shell_depth[-1]

        self.logger.debug(f"t={t/MYR_TO_SEC:.2f} Myr: Shell inner radius: {shell_inner_radius/PC_TO_CM:.2f} pc")
        self.logger.debug(f"t={t/MYR_TO_SEC:.2f} Myr: Shell thickness: {shell_thickness/PC_TO_CM:.2f} pc")
        
        cloud_profile = create_density_profile(
            self.base.simulation_params['profile_type'],
            self.base.simulation_params['M_cl_init'] * M_SUN,
            self.base.simulation_params['n_cl']
        )

        self.logger.debug(f"Cloud radius is: {cloud_profile.R_cl/PC_TO_CM:.2f} pc")

        # Use an extended outer radius (e.g., 25x the cloud radius)
        outer_radius = 25 * cloud_profile.R_cl
        
        # Fixed dr of 0.1 pc in cm
        dr = 0.1 * PC_TO_CM 

        # Calculate number of points needed for this dr
        r_start = shell_inner_radius + TRANSITION_FACTOR_SHELL_CLOUD * shell_thickness
        num_points = int((outer_radius - r_start) / dr) + 1

        # Create radial grid: shell points + extended cloud points
        r_values = np.concatenate([
            shell_radii,
            np.linspace(r_start, outer_radius, num_points, endpoint=True)
        ])

        self.logger.debug(f"Unified profile radial grid details (pre-resampling for dlaw table):")
        self.logger.debug(f"  dr (cloud profile, shell profile is used as is) = {dr/PC_TO_CM:.2f} pc")
        self.logger.debug(f"  Number of shell points: {len(shell_radii)}")
        self.logger.debug(f"  Number of cloud points: {num_points}")
        self.logger.debug(f"  Total number of radial points: {len(r_values)}")

        # Calculate hydrogen number densities, offsetting the cloud profile by the shell thickness
        n_H_values = np.concatenate([
            shell_density,
            [cloud_profile.density(r - TRANSITION_FACTOR_SHELL_CLOUD * shell_thickness) * (self.base.nuclei_to_h / MU_N)
            for r in r_values[len(shell_radii):]]
        ])

        # Resample the profile to avoid too closely spaced points
        split_radius = shell_inner_radius + TRANSITION_FACTOR_SHELL_CLOUD * shell_thickness
        r_values, n_H_values = self._resample_profile(r_values, n_H_values, split_radius)

        # Check mass sufficiency
        expected_mass = self.base.cloud_mass_interp(t)
        if not self._check_mass_sufficiency(r_values, n_H_values, expected_mass):
            error_msg = (
                f"Unified model density profile at t={t/MYR_TO_SEC:.2f} Myr does not provide "
                "enough mass for the dlaw table to work correctly."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.debug(
            f"t={t/MYR_TO_SEC:.2f} Myr: Density law created successfully with outer radius at "
            f"{outer_radius/PC_TO_CM:.1f} pc "
            f"({outer_radius/cloud_profile.R_cl:.1f}x cloud radius)"
        )
        
        return list(zip(r_values, n_H_values)), shell_inner_radius

    def _resample_profile(self, r_values, n_H_values, split_radius, spacing_before=1e-5, spacing_after=1e-3):
        """
        Resample the density profile with different spacings before and after a specified radius.

        Args:
            r_values (np.array): Array of radii.
            n_H_values (np.array): Array of hydrogen number densities in cm^-3.
            split_radius (float): The radius where spacing changes.
            spacing_before (float): Relative spacing in log space before the split_radius.
            spacing_after (float): Relative spacing in log space after the split_radius.

        Returns:
            tuple: (r_values, n_H_values) arrays with resampled profile.
        """
        log_r = np.log10(r_values)
        log_n = np.log10(n_H_values)
        
        # Create interpolation function in log-log space
        interp_func = interp1d(log_r, log_n, kind='linear', bounds_error=False, fill_value=(log_n[0], log_n[-1]))
        
        # Split the log_r array into two parts: before and after the split_radius
        log_split_radius = np.log10(split_radius)
        
        # Resample before the split_radius with spacing_before and extend slightly
        log_r_before = log_r[log_r <= log_split_radius]
        frac_extend_lhs = (1 - spacing_before)
        num_points_before = int((log_r_before[-1] - log_r_before[0]) / spacing_before) + 1
        new_log_r_before = np.linspace(frac_extend_lhs * log_r_before[0], log_r_before[-1], num_points_before, endpoint=True)
        
        # Resample after the split_radius with spacing_after and extend slightly
        log_r_after = log_r[log_r > log_split_radius]
        frac_extend_rhs = (1 + spacing_after)
        num_points_after = int((log_r_after[-1] - log_r_after[0]) / spacing_after) + 1
        new_log_r_after = np.linspace(log_r_after[0], frac_extend_rhs * log_r_after[-1], num_points_after, endpoint=True)
        
        # Combine the two regions
        new_log_r = np.concatenate([new_log_r_before, new_log_r_after])
        
        # Interpolate to get new density values
        new_log_n = interp_func(new_log_r)
        
        # Convert back to linear space
        new_r = 10**new_log_r
        new_n = 10**new_log_n
        
        return new_r, new_n

    def _write_density_law_file(self, filename, density_law):
        """
        Write a dlaw table file for Cloudy.
        Col.1: log(r/cm)
        Col.2: log(n_H/cm-3)
        """
        self.logger.debug(f"Writing density law to file: {filename}")
        try:
            with open(filename, 'w') as f:
                f.write("dlaw table radius\n")
                for r, n in density_law:
                    f.write(f"{np.log10(r):.6e} {np.log10(n):.6e}\n")
                f.write("end of dlaw")
            self.logger.info(f"Successfully wrote density law to {filename}")
        except Exception as e:
            self.logger.error(f"Error writing density law file {filename}: {str(e)}")
            raise

    def get_stopping_criteria(self, t):
        M_cl = self.base.cloud_mass_interp(t)
        return [
            "Stop temperature off",
            f'Stop mass {np.log10(M_cl)}'
        ]

    def write_input_file(self, t):
        """
        Write the complete Cloudy input file for the unified model.
        
        Args:
            t (float): Current time in seconds.
        """
        file_path = f"{self.model_prefix}_{t/MYR_TO_SEC:.2f}.in"
        with open(file_path, 'w') as f:
            f.write(f"# Unified model at t = {t/MYR_TO_SEC:.2f} Myr\n")
            
            for option in self.base.get_primary_source_options(t):
                f.write(f"{option}\n")
            
            for option in self.get_density_and_geometry_options(t):
                f.write(f"{option}\n")
            
            for option in self.base.get_abundance_options():
                f.write(f"{option}\n")
            
            for option in self.base.get_grain_options():
                f.write(f"{option}\n")
            
            for option in self.base.get_cr_options(t):
                f.write(f"{option}\n")
            
            for option in self.base.get_other_physics_options(model_type='unified'):
                f.write(f"{option}\n")

            for option in self.base.get_speedup_options():
                f.write(f"{option}\n")
            
            for option in self.get_stopping_criteria(t):
                f.write(f"{option}\n")
            
            for option in self.base.get_output_options():
                f.write(f"{option}\n")