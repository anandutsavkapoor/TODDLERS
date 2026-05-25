# base_cloudy_input_generator.py

from .imports import np, abstractmethod, os
from .constants import *
from .cloudy_grains_generator import CloudyGrainsGenerator
from .cloudy_abundances_generator import CloudyAbundancesGenerator
from .stellar_feedback import StellarFeedback
from .utils import generate_spectral_table_filename


class BaseCloudyInputGenerator:
    """
    Base class for generating Cloudy input files.

    This class provides common functionality for creating Cloudy input files,
    including methods for generating various Cloudy commands related to stellar
    sources, abundances, grains, and other physical parameters.

    Attributes:
        simulation_params (dict): Parameters of the simulation.
        t_list_collapse (list): List of collapse times for each stellar generation.
        interpolants (tuple): Interpolation functions for various physical quantities.
        cloudy_run_dir (str): Directory for Cloudy run files.
        speedup (bool): Whether to use speedup options in Cloudy simulations.
        grains_generator (CloudyGrainsGenerator): Instance for generating grain-related commands.
        abundances_generator (CloudyAbundancesGenerator): Instance for generating abundance-related commands.

    Methods:
        initialize_parameters(): Sets up basic parameters for Cloudy input.
        get_primary_source_options(t, scaling=1): Generates source options for stellar radiation.
        get_spectral_table_filename(): Generates filename for spectral table.
        get_grain_options(): Generates grain-related Cloudy commands.
        get_abundance_options(): Generates abundance-related Cloudy commands.
        get_other_physics_options(): Generates commands for other physical processes.
        get_output_options(): Generates commands for Cloudy output options.
        get_cr_options(): Generates cosmic ray background options.
        get_turbulence_str(t): Generates turbulence-related commands.
        get_speedup_options(): Generates commands for speeding up Cloudy simulations.
    """

    def __init__(self, simulation_params, t_list_collapse, dissolution_time, interpolants, cloudy_run_dir, speedup=False, logger=None, dust_to_metal=1.0, small_to_large_ratio=None):
        """
        Initialize the BaseCloudyInputGenerator.

        Args:
            simulation_params (dict): Parameters of the simulation.
            t_list_collapse (list): List of collapse times for each generation.
            dissolution_time (float): Time of shell dissolution in seconds. After this, f_esc=1 and only DIG runs.
            interpolants (tuple): Interpolation functions for various physical quantities.
            cloudy_run_dir (str): Directory for Cloudy run files.
            speedup (bool, optional): Whether to use speedup options. Defaults to False.
        """
        self.logger = logger
        self.simulation_params = simulation_params
        self.t_list_collapse = t_list_collapse
        self.dissolution_time = dissolution_time
        self.radius_interp, self.velocity_interp, self.n_shell_in_interp,\
                self.shell_mass_interp, self.cloud_mass_interp, self.is_within_cloud_interp, self.covering_fraction_interp = interpolants
        self.cloudy_run_dir = cloudy_run_dir
        self.speedup = speedup
        self.dust_to_metal = dust_to_metal
        self._small_to_large_ratio_override = small_to_large_ratio

        # Initialize generators and parameters
        self.grains_generator = CloudyGrainsGenerator(CLOUDY_EXE, CLOUDY_DATA_DIR)
        self.abundances_generator = CloudyAbundancesGenerator()
        
        # Initialize stellar feedback
        self.stellar_feedback = StellarFeedback(
            template=simulation_params['template'],
            Z=simulation_params['Z'],
            M_cl_init=simulation_params['M_cl_init'] * M_SUN,
            eta_sf=simulation_params['eta_sf'],
            t_list_collapse=t_list_collapse,
            mode=simulation_params['cluster_formation_mode'],
            formation_timescale=simulation_params.get('formation_timescale'),
            imf=simulation_params['imf'],
            star_type=simulation_params['star_type']
        )
        
        self.initialize_parameters()

    def initialize_parameters(self):
        self.abund_set_name = 'GASS10'
        self.small_to_large_ratio = self._small_to_large_ratio_override if self._small_to_large_ratio_override is not None else SMALL_TO_LARGE_MASS_RATIO
        self.Z = self.simulation_params.get("Z")
        self.eta_sf = float(self.simulation_params.get("eta_sf"))
        self.n_cl = float(self.simulation_params.get("n_cl"))
        self.M_cl_init = float(self.simulation_params.get("M_cl_init")) * M_SUN
        self.template = self.simulation_params.get("template")
        self.imf = self.simulation_params.get("imf")
        self.cluster_formation_mode = self.simulation_params.get("cluster_formation_mode")
        self.star_type = self.simulation_params.get("star_type")
        self.add_cover_frac = self.simulation_params.get("add_cover_frac")
        self.post_sweep_covering_fraction = self.simulation_params.get("covering_fraction")
        self.dynamic_cloud_density = self.simulation_params.get("dynamic_cloud_density")
        if self.cluster_formation_mode == "constant_sfr":
            self.formation_time_yr = 1e6 * self.simulation_params.get("formation_timescale") / MYR_TO_SEC
        else:
            self.formation_time_yr = None

        # Z-dependent H/He conversion factors for the Cloudy interface.
        # He/H from the same relation used in the abundance generator
        # (Dopita et al. 2013, calibrated with Grevesse+2010 Z_sun=0.013).
        log_Z_ratio = np.log10(self.Z / Z_SOLAR_GASS10)
        He_H = 10.0**self.abundances_generator._calc_He(log_Z_ratio)
        self.nuclei_to_h = 1.0 / (1.0 + He_H) # n_H = n_nuclei * nuclei_to_h

    def get_effective_covering_fraction(self, t):
        """
        Get the effective covering fraction from evolution simulation results.
        
        Relies on the pre-computed covering fraction interpolant from evolution
        simulation. After dissolution time (if it exists), covering fraction is 
        always 0.
        
        Args:
            t (float): Current time in seconds.
                
        Returns:
            float: The effective covering fraction between 0 and 1.
                0.0 if after dissolution time (when dissolution time exists).
                Otherwise uses the value from evolution simulation.
        """
        if self.dissolution_time is not None and t >= self.dissolution_time:
            return 0.0
        return float(self.covering_fraction_interp(t))

    def get_primary_source_options(self, t, scaling=1, return_tot_Lbol=False):
        """
        Get primary source luminosity options using StellarFeedback.
        
        Handles multiple stellar generations by creating separate sources for each
        active generation, since their spectral shapes differ based on age.
        Enforces a minimum age threshold of 0.1 Myr.

        Args:
            t (float): Current time in seconds.
            scaling (float, optional): Scaling factor to apply to luminosities. Defaults to 1.
            return_tot_Lbol (bool, optional): Whether to return total bolometric luminosity. 
                                            Defaults to False.

        Returns:
            list or tuple: List of Cloudy source commands, optionally with total bolometric luminosity.
        """
        self.logger.debug(f"Getting source options for t = {t/MYR_TO_SEC:.2f} Myr")
        source_options = []
        tot_Lbol = 0

        # Get feedback data for each generation
        feedback_data = self.stellar_feedback.get_feedback_data_per_generation(t)
        
        min_age_yr = 1e6 * (T_INIT_TEMPLATE / MYR_TO_SEC) # Minimum age threshold in years
        
        for gen in range(len(self.stellar_feedback.t_list_collapse) + 1):
            # Get luminosities for this generation
            L_i = feedback_data['L_i'][gen]
            L_n = feedback_data['L_n'][gen]
            L_bol = L_i + L_n
            tot_Lbol += L_bol

            if L_bol > 0:  # Only add source if luminosity is positive
                # Calculate age of this generation
                if gen == 0:
                    age = t / MYR_TO_SEC
                else:
                    age = (t - self.stellar_feedback.t_list_collapse[gen-1]) / MYR_TO_SEC
                age_yr = age * 1e6

                # Apply minimum age threshold
                if age_yr < min_age_yr:
                    self.logger.info(
                        f"Generation {gen}: Age {age_yr:.2e} yr below minimum threshold. "
                        f"Using minimum age {min_age_yr:.2e} yr instead."
                    )
                    age_yr = min_age_yr

                # Add source options for this generation
                source_filename = self.get_spectral_table_filename()
                source_options.append(
                    f'table star "{source_filename}" age={age_yr:.2f} logZ={np.log10(self.Z)}'
                )
                source_options.append(f'luminosity total={np.log10(L_bol * scaling)}')
                
                self.logger.debug(
                    f"Generation {gen}: age={age_yr:.2e} yr, "
                    f"L_i={L_i * scaling:.2e}, L_n={L_n * scaling:.2e}"
                )

        if not source_options:
            self.logger.debug("No active stellar populations found")

        if return_tot_Lbol:
            return source_options, tot_Lbol * scaling
        else:
            return source_options

    def get_spectral_table_filename(self):
        # Dispatch with startswith for consistency with the rest of the codebase,
        # so template aliases like 'SB99_100' resolve correctly; raise on anything
        # genuinely unknown rather than failing later with an unbound variable.
        if self.template.startswith('BPASS'):
            res_fac = BPASS_REDUCE_RES_FAC
        elif self.template.startswith('SB99'):
            res_fac = SB99_REDUCE_RES_FAC
        elif self.template == 'pySB99':
            res_fac = 1
        else:
            raise ValueError(
                f"Unknown stellar template '{self.template}'. "
                "Expected one of 'SB99', 'BPASS', 'pySB99'."
            )
        return generate_spectral_table_filename(
            self.template, self.imf, self.star_type, 
            formation_timescale=self.formation_time_yr,
            wavelength_resolution_factor=res_fac
        )

    @abstractmethod
    def get_density_options(self, t):
        pass

    @abstractmethod
    def get_geometry_options(self, t):
        pass

    @abstractmethod
    def get_stopping_criteria(self):
        pass

    @abstractmethod
    def write_input_file(self, t, filename):
        pass

    def get_grain_options(self, skip_pah=False):
        """
        Generate grain-related Cloudy commands using CloudyGrainsGenerator.

        The dust_to_metal factor scales all grain abundances (silicate, graphite,
        PAH) relative to solar.

        Args:
            skip_pah (bool): If True, PAHs will not be included in the grain commands.
                            Default is False for backward compatibility.

        Returns:
            list: Grain commands for Cloudy input file including silicates, graphites,
                and optionally PAHs.
        """
        return self.grains_generator.generate_grain_commands(
            metallicity_scaling=self.dust_to_metal * self.Z / Z_SOLAR_GASS10,
            silicate_ratio=self.small_to_large_ratio,
            graphite_ratio=self.small_to_large_ratio,
            skip_pah=skip_pah
        )

    def get_abundance_options(self, abundance_set=None):
        """
        Generate abundance commands for Cloudy input using the abundance generator.
        
        This method generates the appropriate abundance commands for Cloudy using
        a specified abundance set and the model's metallicity.
        
        Returns:
            list: Cloudy commands for setting elemental abundances.
        
        Note:
            Uses GASS10 solar abundance scale (Z_☉ = 0.013).
        """
        # Use default abundance set if not specified
        abundance_set = abundance_set or self.abund_set_name
        
        # Calculate log metallicity relative to GASS10 solar value
        log_Z_ratio = np.log10(self.Z / Z_SOLAR_GASS10)
        
        # Generate abundance set using the abundance generator
        abundance_result = self.abundances_generator.generate_abundance_set(
            set_name=abundance_set,
            logZbyZsun=log_Z_ratio
        )
        
        return abundance_result['cloudy_commands']

    def get_other_physics_options(self, model_type='shell'):
        """
        Generate physics-related options for Cloudy input files.
        
        Args:
            model_type (str, optional): Type of model ('shell', 'unified', 'dig', 'dissolved'). 
                Defaults to 'shell'.
                - unified: Uses higher nend (4000) for unified models
                - other models: Use standard nend (2800)
        
        Returns:
            list: Cloudy commands for additional physics options.
        """
        options = []
        if not self.speedup:
            options.append('set trimming -10')
        
        # Base physics options
        options.extend([
            'set pres ioniz 3e4',
            'iterate to convergence max=10',
        ])
        
        # Model-specific nend value
        if model_type == 'unified':
            options.append('set nend 4000')
        else:
            options.append('set nend 2800')
        
        return options

    def get_output_options(self, grains=True, dissolved=False, save_elements=False, 
                        save_lines=True, save_overview=True, save_heating=True, 
                        save_cooling=True, save_opacities=False, line_list_version='v2'):
        """
        Generate Cloudy save commands for output files.
        
        Args:
            grains (bool): Whether to include grain-related save commands. Defaults to True.
                          When False, only intrinsic line intensities are saved.
            dissolved (bool): Whether this is a dissolved model. Defaults to False.
            save_elements (bool): Whether to save element-specific outputs. Defaults to False.
            save_lines (bool): Whether to save line luminosities. Defaults to True.
            save_overview (bool): Whether to save overview file. Defaults to True.
            save_heating (bool): Whether to save heating mechanisms. Defaults to True.
            save_cooling (bool): Whether to save cooling mechanisms. Defaults to True.
            save_opacities (bool): Whether to save optical depths. Defaults to False.
            line_list_version (str): Version of line list to use ('v1' or 'v2'). Defaults to 'v2'.
            
        Returns:
            list: Cloudy save commands for various physical quantities and diagnostic information.
            
        Raises:
            ValueError: If an invalid line_list_version is provided.
        """
        if dissolved:
            return ['save last continuum ".cont" units micron']
        
        if line_list_version not in ['v1', 'v2']:
            raise ValueError("line_list_version must be either 'v1' or 'v2'")
                
        # Essential commands that are always included
        commands = [
            'save last radius ".rad"',
            'save last continuum ".cont" units micron',
            'save last diffuse continuum unattenuated ".diffContUnatt" units micron',
            'save last physical conditions ".phy"'
        ]
        
        # Optional base commands
        if save_overview:
            commands.append('save last overview ".ovr"')
        if save_heating:
            commands.append('save last heating ".heat"')
        if save_cooling:
            commands.append('save last cooling ".cool"')
        if save_opacities:
            commands.append('save last optical depth ".opd"')
        
        # Element-specific commands - only added if save_elements is True
        if save_elements:
            element_commands = [
                'save last element hydrogen ".ele_H"',
                'save last element helium ".ele_He"',
                'save last element carbon ".ele_C"',
                'save last element nitrogen ".ele_N"',
                'save last element oxygen ".ele_O"',
                'save last element argon ".ele_Ar"',
                'save last element neon ".ele_Ne"',
                'save last element sulphur ".ele_S"',
                'save last element chlorin ".ele_Cl"',
                'save last element iron ".ele_Fe"',
                'save last element silicon ".ele_Si"'
            ]
            commands.extend(element_commands)
        
        # Line save commands - only added if save_lines is True
        if save_lines:
            src_dir = os.path.dirname(os.path.abspath(__file__))
            lines_file = os.path.join(src_dir, 'database', 'lines_list', f'cloudy_lines_TODDLERS_{line_list_version}.dat')
            
            try:
                with open(lines_file, 'r') as f:
                    # Filter out commented lines and empty lines
                    lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
                    
                # Intrinsic luminosities are always saved
                line_commands = [
                    'save last lines zone cumulative intrinsic ".cum"'
                ]
                line_commands.extend(lines)
                line_commands.append('end of lines')
                
                # Emergent luminosities only saved when grains=True
                if grains:
                    line_commands.extend([
                        'save last lines zone cumulative emergent ".cumEmer"'
                    ])
                    line_commands.extend(lines)
                    line_commands.append('end of lines')
                
                commands.extend(line_commands)
                
            except FileNotFoundError as exc:
                # Fail fast: silently dropping the line-save commands would produce
                # continuum-only Cloudy models, and the missing emission lines only
                # surface much later when the high-resolution SED interpolant is built.
                raise FileNotFoundError(
                    f"Line list not found: {lines_file}. The '{line_list_version}' line list "
                    f"must be present in database/lines_list/ to generate line-emitting models. "
                    f"Pass save_lines=False to deliberately build continuum-only models."
                ) from exc
            except Exception as e:
                self.logger.error(f"Error processing lines file: {str(e)}")
                raise
        
        # Grain commands - only added if grains is True
        if grains:
            grain_commands = [
                'save last grain temperature ".grTemp"',
                'save last grain abundances ".grAbund"',
                'save last grain D/G ratio ".grDGrat"',
                'save last grain continuum ".grCont" units micron'
            ]
            commands = grain_commands + commands
        
        return commands

    def get_cr_options(self, t, scale_with_Z=True, scale_inside_cloud_only=True):
        """
        Generate cosmic ray background commands for Cloudy input.

        The cosmic ray background can scale with metallicity in super-solar environments.
        By default, this scaling only occurs when the shell is inside the cloud and 
        the metallicity is super-solar. Outside the cloud, or for sub-solar metallicities,
        the cosmic ray background remains at the default value.

        Args:
            t (float): Model time in seconds.
            scale_with_Z (bool): Whether to enable metallicity scaling at all. 
                If False, uses default cosmic ray background everywhere regardless 
                of shell position or metallicity. Defaults to True.
            scale_inside_cloud_only (bool): Whether to restrict metallicity scaling 
                to when shell is inside cloud. When True (default), cosmic rays scale 
                with Z only inside cloud. When False, they scale with Z everywhere if 
                metallicity is super-solar.

        Scaling Logic:

        Cosmic rays scale with metallicity if ALL of these conditions are met:

        1. scale_with_Z is True
        2. metallicity is super-solar (Z > Z_sun)
        3. Either:

           - Shell is inside cloud (when scale_inside_cloud_only=True)
           - OR scale_inside_cloud_only is False (scale everywhere)

        Returns:
            list: Cloudy command for cosmic ray background. Will be one of:
                - ['cosmic rays background'] (no scaling)
                - ['cosmic rays background {Z/Z_☉}'] (with scaling)
        """
        if not scale_with_Z:
            return ['cosmic rays background']
            
        is_within = self.is_within_cloud_interp(t) > 0
        
        if (not scale_inside_cloud_only or is_within) and self.Z > Z_SOLAR:
            scale_factor = self.Z/Z_SOLAR
            return [f'cosmic rays background {scale_factor}']
        
        return ['cosmic rays background']

    def get_turbulence_str(self, t):
        """
        Generate basic turbulence command string for Cloudy input.
        
        Args:
            t (float): Current time in seconds.
            
        Returns:
            str: Cloudy command for turbulence settings with default fraction.
        """
        v_sh = self.velocity_interp(t)
        return f'turbulence {DEFAULT_TURB_FRAC * abs(v_sh)/KM_TO_CM:.6f} km/sec no pressure'

    def get_speedup_options(self):
        """
        Get speedup options.

        Returns:
            list: List of speedup commands for Cloudy input.
        """
        if not self.speedup:
            return []
        
        options = [
            'set trimming -8',  # Turn off elements with abundances less than 10^-7 of hydrogen
            'grains no qheat',  # Turn off quantum heating for grains
            'no molecules',  # Turn off the chemical network => no H_2 => high PAH abund
        ]
        
        return options

    def get_speedup_options_no_grains(self):
        """
        Get speedup options excluding grain-related commands.

        Returns:
            list: List of non-grain speedup commands for Cloudy input.
        """
        if not self.speedup:
            return []
        
        options = {
            'trimming': 'set trimming -8',  # Turn off elements with abundances less than 10^-7 of hydrogen
            'molecules': 'no molecules',     # Turn off the chemical network
        }
        
        return list(options.values())