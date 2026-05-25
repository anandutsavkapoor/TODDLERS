from .timeout_manager import TimeoutManager
from .imports import np, solve_ivp, interp1d, brentq, sys, os, contextmanager, warnings, traceback, pickle
from .constants import *
from .exceptions import ManualEventTermination, capture_output
from .evolution_logging import EvolutionLogger, NullEvolutionLogger
from .evolution_data import EvolutionState
from .initial_conditions import InitialConditions
from .stellar_feedback import StellarFeedback
from .cloud_density_profiles import create_density_profile, UniformDensityWithCavity
from .cloud_energy_injection import CloudEnergyInjection
from .shell_structure import ShellStructure
from .lyman_alpha import LymanAlpha
from .phase1 import Phase1
from .fragmentation import Fragmentation
from .phase2 import Phase2
from .dissolution import Dissolution
from .track_simulation import TrackSimulation
from .utils import format_value, func_tanh_extPressure_velocity, func_tanh_extPressure_fesc_ion, \
    func_tanh_sweep_coverfrac, func_tanh_infall_coverfrac, get_atols, calculate_cloud_escape_fraction, process_buffers, \
    get_shell_inner_face_density

class Evolution:
    """
    Evolution Module for TODDLERS (Time evolution of Observables including Dust Diagnostics and Line Emission from Regions containing young Stars)

    This module simulates the evolution of a stellar feedback-driven shell in a molecular cloud environment.

    The Evolution class handles the entire lifecycle of the shell, including different phases:
    - Phase1: Initial expansion through shocked mass loss.
    - Fragmentation: Shell becomes unstable and begins to break apart, loses pressure through rapid cooling.
    - Phase2: Continued expansion after fragmentation through ram force.
    - Dissolution: Shell becomes too diffuse and dissipates

    Key Features:
    - Handles multiple generations of star formation and shell evolution
    - Supports different star cluster formation modes (burst and constant star formation rate)
    - Incorporates various physical processes such as stellar feedback, radiation pressure, and gravitational forces
    - Dynamically updates cloud density profile (optional)
    - Possible to use variable ionized gas temperature in shell structure equations.
    - Calculates shell structure and Lyman-alpha radiation effects
    - Supports different stellar evolution models (BPASS and SB99) with various IMFs and star types

    **Classes:**
        Evolution: Main class for simulating shell evolution

    **Dependencies:**
        - numpy
        - scipy (solve_ivp, interp1d)
        - Various custom modules (constants, stellar_feedback, cloud_density_profiles, etc.)

    Usage::

        evolution = Evolution(Z, eta_sf, n_cl, M_cl_init, template,
                            cluster_formation_mode='burst',
                            formation_timescale=2*MYR_TO_SEC,
                            profile_type='uniform',
                            profile_params=None,
                            dynamic_cloud_density=False,
                            imf=None,
                            star_type=None,
                            dark_matter_fraction=0.0)
        results = evolution.run_simulation()

    Main Methods:

    - run_simulation(): Run the full simulation
    - integrated_evolution(t, x): Core method for evolution calculations

    This module is part of the TODDLERS project, aimed at modeling the evolution of 
    star-forming regions.
    """
    def __init__(self, Z, eta_sf, n_cl, M_cl_init, template, cluster_formation_mode='burst',
                formation_timescale=2*MYR_TO_SEC, profile_type='uniform', profile_params=None,
                dynamic_cloud_density=False, imf=None, star_type=None, add_cover_frac=False,
                post_sweep_covering_fraction=COVERING_FRACTION_DEF, dark_matter_fraction=0.0,
                use_Tion_model=False, skip_logger_init=False, init_with_fragmentation=False,
                include_gravity=True, include_lyman_alpha=True,
                include_external_pressure=True, dust_to_metal=1.0):
        """
        Initialize the Evolution object.

        Args:
            Z (float): Metallicity of the gas, in solar units.
            eta_sf (float): Star formation efficiency.
            n_cl (float): Mean cloud number density, in cm^-3.
            M_cl_init (float): Initial cloud mass, in grams.
            template (str): Stellar population synthesis model template ('BPASS_100', 'SB99_100', etc.).
            cluster_formation_mode (str, optional): Mode of star cluster formation ('burst' or 'constant_sfr'). 
                Defaults to 'burst'.
            formation_timescale (float, optional): Timescale for star formation in 'constant_sfr' mode, in seconds. 
                Defaults to 2 Myr.
            profile_type (str, optional): Type of density profile for the cloud. Defaults to 'uniform'.
            profile_params (dict, optional): Additional parameters for the density profile. Defaults to None.
            dynamic_cloud_density (bool, optional): Whether to dynamically update the cloud density. 
                Defaults to False.
            imf (str, optional): Initial Mass Function to use. Options depend on the template.
                For BPASS: 'chab100' (default), 'chab300'
                For SB99: 'kroupa100' (default), 'kroupa120', 'kroupa120_rot'
            star_type (str, optional): For BPASS models, specifies 'sin' (single) or 'bin' (binary) stars.
                Defaults to 'bin' for BPASS. Not used for other templates.
            add_cover_frac (bool, optional): If True, applies a covering fraction to radiation and ram forces 
                                            after the cloud is fully swept. Defaults to False.
            post_sweep_covering_fraction (float, optional): The target covering fraction to 
                transition to after cloud has been swept and when add_cover_frac is True. Must be 
                between 0 and 1. Defaults to 0.5.
            dark_matter_fraction (float, optional): Fraction of initial cloud mass in dark matter. This component
                only contributes to gravitational force and remains constant through generations. Defaults to 0.0
            use_Tion_model (bool): If True, uses metallicity, density, and ionization flux dependent
                temperature is used in ionized shell structure eqns instead of 1e4K. Defaults to False.
                This makes use of Cloudy single zone models with T_ion = f(n, Z, ionizing_flux).
            skip_logger_init (bool, optional): Whether to open a new log instance, useful for simulations needing
                multiple attempts.
            init_with_fragmentation (bool, optional): Whether to initialize the simulation directly
                in the fragmentation phase. Defaults to False.
            include_gravity (bool, optional): If False, disables gravitational force and
                gravitational potential energy. 
            include_lyman_alpha (bool, optional): If False, disables Lyman-alpha radiation
                pressure contribution. UV+IR radiation is unaffected.
            include_external_pressure (bool, optional): If False, disables the external
                cloud/ISM pressure force on the shell.
            dust_to_metal (float, optional): Dust-to-metal mass ratio normalized to solar.
                Scales sigma_dust, kappa_IR, and Lyman-alpha dust opacity. Defaults to 1.0.

        Raises:
            NotImplementedError: If dynamic cloud density is used with a non-uniform profile.
            ValueError: If an unsupported template or invalid star_type is provided.
        """                
        self.Z = Z
        self.eta_sf = eta_sf
        self.n_cl = n_cl
        self.M_cl_init = M_cl_init
        self.dark_matter_fraction = dark_matter_fraction
        self.M_dm = dark_matter_fraction * self.M_cl_init
        self.dust_to_metal = dust_to_metal
        self.template = template
        self.cluster_formation_mode = cluster_formation_mode
        self.formation_timescale = formation_timescale
        self.init_with_fragmentation = init_with_fragmentation
        self.include_gravity = include_gravity
        self.include_lyman_alpha = include_lyman_alpha
        self.include_external_pressure = include_external_pressure

        self.add_cover_frac = add_cover_frac
        if self.add_cover_frac:
            self.post_sweep_covering_fraction = post_sweep_covering_fraction
        else:
            self.post_sweep_covering_fraction = 1
        if self.add_cover_frac and not 0 <= self.post_sweep_covering_fraction <= 1:
            raise ValueError("post_sweep_covering_fraction must be between 0 and 1")
            
        # Set IMF and star_type based on template
        if self.template.startswith('BPASS'):
            self.imf = imf or "chab100"
            self.star_type = star_type or "bin"
            if self.star_type not in ['sin', 'bin']:
                raise ValueError("BPASS star_type must be either 'sin' or 'bin'")
        elif self.template.startswith('SB99'):
            self.imf = imf or "kroupa100"
            self.star_type = star_type or "sin" 
        elif self.template == 'pySB99':
            # pySB99 supports custom IMFs, no default
            if imf is None:
                raise ValueError("IMF must be specified for pySB99 template")
            self.imf = imf
            self.star_type = star_type or "sin"
        else:
            raise ValueError(f"Unsupported template: {self.template}")

        # set fix or variable temp in shell structure equations
        self.use_Tion_model = use_Tion_model
        if self.use_Tion_model:
            interpolator_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database', 
                                            'Hii_temperature_interpolator.obj')
            with open(interpolator_path, 'rb') as f:
                self.Hii_temperature_interpolator = pickle.load(f)
    
        self.profile_type = profile_type
        self.profile_params = profile_params if profile_params else {}
        self.dynamic_cloud_density = dynamic_cloud_density
        
        # once all parameters have been defined
        self.ensure_output_directories()
        self.output_path, self.log_path = self.get_output_paths()

        if os.path.exists(self.output_path):
            print(f"The simulation result file '{self.output_path}' already exists.")
            self._output_exists = True
            return

        if not skip_logger_init:
            self.logger = EvolutionLogger(self.log_path)
            self.logger.info(f"Initializing Evolution with Z={Z}, eta_sf={eta_sf}, n_cl={n_cl} cm^-3, M_cl_init={M_cl_init/M_SUN:.2e} M_sun")
        else:
            self.logger = NullEvolutionLogger()

        self.logger.info(f"Feedback template is: {self.template}")
        self.logger.info(f"IMF: {self.imf}")
        self.logger.info(f"Star type: {self.star_type}")
        self.logger.info(f"Cluster formation mode is: {self.cluster_formation_mode}")
        if self.cluster_formation_mode == 'constant_sfr':
            self.logger.info(f"Cluster formation timescale is: {self.formation_timescale/MYR_TO_SEC:.2f} Myr")
        self.logger.info(f"Covering fraction in acceleration calculations is {'enabled' if self.add_cover_frac else 'disabled'}")
        if self.add_cover_frac:
            self.logger.info(f"Post sweep value covering fraction value is: {self.post_sweep_covering_fraction}")
        if self.dark_matter_fraction > 0:
            self.logger.info(f"Dark matter component enabled with fraction: {self.dark_matter_fraction:.3f}")
            self.logger.info(f"Dark matter mass: {self.M_dm/M_SUN:.2e} M_sun")
        if self.use_Tion_model:
            self.logger.info(f"Variable ionized gas temperature for shell structure is enabled. Uses precalculated Cloudy interpolator.")
        else:
            self.logger.info(f"Variable ionized gas temperature for shell structure is disabled.")
        if not self.include_gravity:
            self.logger.info("Gravitational force is DISABLED")
        if not self.include_lyman_alpha:
            self.logger.info("Lyman-alpha radiation force is DISABLED")
        if not self.include_external_pressure:
            self.logger.info("External pressure force is DISABLED")
        if self.dust_to_metal != 1.0:
            self.logger.info(f"Dust-to-metal ratio (scaled by solar): {self.dust_to_metal:.2f}")
        if self.init_with_fragmentation:
            self.logger.info("SIMULATION CONFIGURED TO START DIRECTLY IN FRAGMENTATION PHASE")

        self.state = EvolutionState()
        self.phase1 = Phase1(self)
        self.fragmentation = Fragmentation(self)
        self.phase2 = Phase2(self)
        self.dissolution = Dissolution(self)

        self.stellar_feedback = StellarFeedback(
            template, Z, M_cl_init, eta_sf, self.state.t_list_collapse,
            mode=cluster_formation_mode, formation_timescale=formation_timescale,
            imf=self.imf, star_type=self.star_type
        )
        
        self.lyman_alpha = LymanAlpha(Z, T_NEUTRAL, dust_to_metal_relative_to_solar=self.dust_to_metal)
        self.initialize_parameters()
        self.initialize_others()
        self.initialize_density_profile()

        if self.dynamic_cloud_density and self.profile_type != 'uniform':
            raise NotImplementedError("Dynamic cloud density is only implemented for uniform density profiles.")
        else:
            self.dt_calc_dynamic_cloud = 1 / (ALPHA_B * self.density_profile.n_avg)
            self.kinetic_energy_efficiency = ETA_KE
            self.logger.info(f"Cloud dynamic density is : {self.dynamic_cloud_density}")

        self.t0 = T_INIT_TEMPLATE # first gen
        self.y0 = self.get_initial_conditions(self.t0, dt=T_INIT_TEMPLATE) # dt is the period of w77 expansion

        self.logger.log_profile_info(self.density_profile)
        self.shell_structure = None
        self.all_results = []
        self.tracker = TrackSimulation()
        self.initialize_shell_properties()
        self.check_early_fragmentation()
        
    def get_output_paths(self):
        """
        Generate output and log file paths based on simulation parameters.
        
        Returns:
            tuple: (output_path, log_path)
        """
        # Get the directory of evolution.py
        evolution_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Go up one level to the parent directory
        parent_dir = os.path.dirname(evolution_dir)
        
        # Create directory structure
        dir_path = os.path.join(
            parent_dir, 
            EVOLUTION_OUTPUT_DIR,  # Changed from 'results'
            f"template_{self.template}", 
            f"imf_{self.imf}", 
            f"star_type_{self.star_type}", 
            f"cluster_mode_{self.cluster_formation_mode}", 
            f"profile_type_{self.profile_type}"
        )
        
        # Create filename
        filename = f'sim_Z{self.Z:.3g}_eta{self.eta_sf:.3g}_n{self.n_cl:.1f}_logM{np.log10(self.M_cl_init/M_SUN):.2f}'
        
        # Add additional parameters to filename
        if self.dynamic_cloud_density:
            filename += '_dynDens'
        if self.add_cover_frac:
            filename += f'_cover{self.post_sweep_covering_fraction:.2f}'
        if self.cluster_formation_mode == 'constant_sfr':
            filename += f'_tform{self.formation_timescale/MYR_TO_SEC:.1f}'
        if self.dust_to_metal != 1.0:
            filename += f'_dtm{self.dust_to_metal:.2f}'

        filename += '.dat'
        
        # Create full path
        full_path = os.path.join(dir_path, filename)
        
        # Create log path - now using EVOLUTION_LOGS_DIR instead of 'logs'
        log_path = full_path.replace(EVOLUTION_OUTPUT_DIR, EVOLUTION_LOGS_DIR).replace('.dat', '.log')
        
        return full_path, log_path

    def ensure_output_directories(self):
        """
        Ensure that the directories for output and log files exist.
        """
        output_path, log_path = self.get_output_paths()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def initialize_shell_properties(self):
        """
        Initialize shell-related properties needed for fragmentation checks.
        Should be called after initial conditions are set but before
        fragmentation checks.
        """
        # Calculate initial pressure
        self.state.Rin = self.get_inner_radius()
        volume = self.state.get_volume()
        self.state.P_b = self.get_pressure(volume)
        
        # Set up shell structure
        self.update_shell_structure()
        
        # Now we can calculate initial beta
        self.state.alpha = (self.state.t / self.state.R) * self.state.V_sh
        self.state.beta = -(self.state.t / self.state.P_b) * self.get_pressure_change_rate()

    def initialize_parameters(self):
        """Initialize various parameters for the simulation."""
        self.rho_cl = self.n_cl * MU_N
        self.M_cl = self.M_cl_init * (1 - self.eta_sf)
        self.P_ext_ISM = (MU_N / MU_P) * N_ISM * K_BOLTZMANN * T_ION
        self.tau_recomb = 1 / (self.n_cl * ALPHA_B)
        self.sigma_dust = self.dust_to_metal * SIGMA_DUST_SOLAR * (self.Z / Z_SOLAR)
        self.kappa_IR = self.dust_to_metal * KAPPA_IR_SOLAR * (self.Z / Z_SOLAR)

    def initialize_others(self):
        self.f_esc_i_cloud = 0. 
        self.cover_frac = 1.
        self.t_fragmentation = None
        self.t_end_frag = None
        self.t_cloud_swept = None
        self.t_dissolution = None
        self.t_recollapse = None
        self.fragmentation_reason = None
        self.dissolution_reason = None
        self.t_fragmentation = None
        self.E_b_init_frag = None
        self.fragmentation_reason = None

    def get_initial_conditions(self, t0, dt):
        """
        Calculate the initial conditions for the simulation.

        Returns:
            numpy.ndarray: Initial conditions array [R0, V0, E0, M_sh, n_cloud_avg].
        """
        ic = InitialConditions(self.stellar_feedback, self.density_profile, t0, dt)
        R0, V0, E0, M_sh, Rin0 = ic.calculate()
        
        self.state.R = R0
        self.state.V_sh = V0
        self.state.M_sh = M_sh
        self.state.n_cloud_avg = self.n_cl
        self.state.Rin = Rin0
        self.state.E_b = E0
        volume = self.state.get_volume()
        self.state.P_b = (GAMMA - 1) * self.state.E_b / volume
        self.initial_radius = R0


        self.logger.info(
            f"Initial conditions: t0={format_value(t0 / MYR_TO_SEC, '.2f')} Myr, "
            f"R0={format_value(R0 / PC_TO_CM, '.2f')} pc, "
            f"V0={format_value(V0 / KM_TO_CM, '.2f')} km/s, "
            f"E0={format_value(E0, '.2e')} erg, "
            f"M_sh={format_value(self.state.M_sh / M_SUN, '.2e')} M_sun, "
            f"n_cloud_avg_0 (same as model average density)={format_value(self.state.n_cloud_avg, '.2e')} cm^-3"
        )

        return np.array([R0, V0, E0, self.state.M_sh, self.state.n_cloud_avg])

    def initialize_density_profile(self):
        """Initialize the cloud density profile.
            R_cl based on the new profile is also calculated """
        self.density_profile = create_density_profile(
            self.profile_type,
            self.M_cl,
            self.n_cl,
            **self.profile_params
        )
        self.R_cl = self.density_profile.R_cl
        self.R_cl_init = self.R_cl
        self.logger.info(f"Cloud radius is: R_cl={self.R_cl/PC_TO_CM}")

    def check_early_fragmentation(self):
        """
        Check and handle immediate fragmentation at initialization or after recollapse.
        
        This method checks if the system should start in fragmentation phase, either because:
        1. init_with_fragmentation=True was specified, or
        2. fragmentation conditions are met at initialization time
        
        If conditions are met, it:
        1. Sets the phase to fragmentation
        2. Uses Phase1's fragmentation_start() to initialize parameters
        3. Logs system state
        
        Returns:
            bool: True if system fragments immediately, False otherwise
        """
        # Start with fragmentation if explicitly requested
        if self.init_with_fragmentation:
            self.logger.log_initial_fragmentation()
            self.state.phase = 'fragmentation'
            self.phase1.fragmentation_start()
            self.logger.info(f"System initialized directly in fragmentation phase at t={self.state.t/MYR_TO_SEC:.6f} Myr")
            return True
        
        # Otherwise, check the standard fragmentation conditions
        elif self.phase1.check_fragmentation_condition():
            self.state.phase = 'fragmentation'
            self.phase1.fragmentation_start()
            self.logger.info(f"System fragmenting at initialization, t={self.state.t/MYR_TO_SEC:.6f} Myr")
            return True
        
        return False

    def simulation_attempt(self, solvers, rtol):
        """
        Run a single shell evolution simulation attempt with specified solvers and tolerance.
        This method manages the entire lifecycle of the shell, including different phases
        (Phase1, Fragmentation, Phase2, Dissolution) and transitions between these phases.
        It uses the solve_ivp function to integrate the evolution equations for each phase.

        The simulation continues until one of the following conditions is met:
        1. The shell dissolves
        2. The shell recollapses (leading to a new generation)
        3. The maximum simulation time is reached

        For each generation, the method:
        1. Solves the ODE system for Phase1
        2. If applicable, solves the ODE system for Fragmentation, Phase2 and Dissolution
        3. Processes and stores the results
        4. Checks for shell recollapse and starts a new generation if necessary

        Args:
            solvers (tuple): (phase1_solver, frag_solver, phase2_solver)
            rtol (float): Relative tolerance for this attempt
        
        Returns:
        -------------
        list of dict
            A list containing dictionaries of results for each generation.
            Each dictionary includes:
            - 'status': Final status of the generation ('dissolved', 'collapsed', 'fragmentation_ended', 'time_limit_reached')
            - 'phase_transitions': List of phase transitions that occurred
            - 'cluster_formation_mode': Mode of star cluster formation ('burst' or 'constant_sfr')
            - 'formation_timescale': Timescale for star formation in 'constant_sfr' mode
            - 'generation': Generation number
            - 'cloud_mass': Final cloud mass
            - 'cloud_radius': Final cloud radius
            - Various time series data (time, radius, velocity, mass, energy, etc.)

        Side Effects:
        -------------
        - Updates the state of the Evolution object
        - Logs information about the simulation progress
        - Writes output files with simulation results

        Raises:
        -------------
        Various exceptions may be raised and caught internally, including:
        - ManualEventTermination: For controlled termination of integration (phase-1, dissolution)
        TimeoutError: If simulation exceeds time limit
        Exception: For other simulation failures

        Notes:
        -------------
        - The method uses LSODA/BDF solver for different phases of the evolution.
        - It employs event detection to identify phase transitions and shell recollapse.
        - The results are processed and stored after each solve_ivp call.
        - The method handles multiple generations of shell evolution if recollapses occur.
        """

        global SOLVER_PH1, SOLVER_FRAG, SOLVER_PH2, RTOL_MAIN_PH1, RTOL_MAIN_FRAG, RTOL_MAIN_PH2

        # Store original parameters
        original_solvers = (SOLVER_PH1, SOLVER_FRAG, SOLVER_PH2)
        original_rtol_ph1 = RTOL_MAIN_PH1
        original_rtol_ratio_ph2 = RTOL_MAIN_PH2 / RTOL_MAIN_PH1  # Store ratio before changes

        # Set parameters for this attempt
        SOLVER_PH1, SOLVER_FRAG, SOLVER_PH2 = solvers

        # Set rtols using stored original ratio
        RTOL_MAIN_PH1 = rtol
        RTOL_MAIN_FRAG = rtol
        RTOL_MAIN_PH2 = rtol * original_rtol_ratio_ph2

        try:

            generation_results = {}
            while self.state.t_end <= MAX_SIMULATION_TIME:

                if self.state.phase == 'phase1':

                    if self.is_phase_timespan_negligible('phase1'):
                        if generation_results:
                            self.all_results.append(generation_results)
                        break
                    
                    t_span = (self.t0, MAX_SIMULATION_TIME)
                    atols = get_atols(self.y0[0], self.y0[2], phase=self.state.phase, dynamic_cloud_density=self.dynamic_cloud_density)
                    self.logger.debug(f"Solving IVP for Phase 1: t_span={t_span}, y0={self.y0}, atols={atols}, rtol={RTOL_MAIN_PH1}")
                    
                    termination_identifier = None
                    with capture_output() as get_buffers:
                        try:
                            sol_phase1 = solve_ivp(
                                self.integrated_evolution, 
                                t_span, 
                                self.y0,
                                method=SOLVER_PH1, 
                                rtol=RTOL_MAIN_PH1, 
                                max_step=MAX_STEP_PH1,
                                atol=atols,
                                dense_output=True
                            )
                        except ManualEventTermination as e:
                            termination_identifier = e.identifier
                            self.logger.info(f"Phase 1 manually terminated: {str(e)}")
                            sol_phase1 = self.create_partial_solution(e.t, e.y)
                            
                        stdout_output, stderr_output = process_buffers(get_buffers)

                    if termination_identifier != "MANUAL_TERMINATION":
                        if stdout_output:
                            print(stdout_output, file=sys.stdout)
                        if stderr_output:
                            print(stderr_output, file=sys.stderr)

                    self.logger.debug(f"Phase 1 IVP solved.")

                    if self.state.phase == 'fragmentation':
                        generation_results = self.process_and_concatenate_results(generation_results)
                    elif abs(sol_phase1.t[-1] - MAX_SIMULATION_TIME) <= RTOL_MAIN_PH1 * MAX_SIMULATION_TIME:
                        self.state.t_end = MAX_SIMULATION_TIME
                        self.state.t = MAX_SIMULATION_TIME
                        generation_results = self.process_and_concatenate_results(generation_results, append_to_all=True)
                        self.logger.info(f"Simulation reached maximum time during Phase 1 at t={self.state.t_end/MYR_TO_SEC:.2f} Myr")
                        break
                    else:
                        self.logger.error(f"Phase 1 ended unexpectedly at t={sol_phase1.t[-1]/MYR_TO_SEC:.2f} Myr")
                        self.logger.error(f"Solver status: {sol_phase1.status}")
                        self.logger.error(f"Solver message: {sol_phase1.message}")
                        self.logger.error(f"Time remaining: {(MAX_SIMULATION_TIME - sol_phase1.t[-1])/MYR_TO_SEC:.2f} Myr")
                        raise RuntimeError("Phase 1 failed: Neither transitioned to fragmentation nor reached maximum time")

                if self.state.phase == 'fragmentation':

                    if self.is_phase_timespan_negligible('fragmentation'):
                        if generation_results:
                            self.all_results.append(generation_results)
                        break

                    self.check_and_adjust_shell_mass()
                    t_frag_start = self.state.t
                    y0_frag = np.array([self.state.R, self.state.V_sh, self.state.E_b, self.state.M_sh, self.state.n_cloud_avg])
                    t_span_frag = (t_frag_start, MAX_SIMULATION_TIME)
                    atols_frag = get_atols(y0_frag[0], y0_frag[2], phase=self.state.phase, dynamic_cloud_density=self.dynamic_cloud_density)
                    self.logger.debug(f"Solving IVP for Fragmentation: t_span={t_span_frag}, y0={y0_frag}, atols={atols_frag}, rtol={RTOL_MAIN_FRAG}")

                    sol_frag = solve_ivp(
                        self.integrated_evolution, 
                        t_span_frag, 
                        y0_frag,
                        method=SOLVER_FRAG, 
                        rtol=RTOL_MAIN_FRAG, 
                        max_step=MAX_STEP_FRAG,
                        atol=atols_frag,
                        events=(self.terminate_fragmentation),
                        dense_output=True
                    )

                    self.logger.debug(f"Fragmentation IVP solved.")

                    if sol_frag.t_events[0].size > 0:  # Fragmentation ended
                        self.t_end_frag = self.state.t
                        generation_results = self.process_and_concatenate_results(generation_results) # append_to_all is not True here
                        self.state.phase = 'phase2'
                        self.logger.info(f"Initialized Phase 2 at t={self.state.t/MYR_TO_SEC:.2f} Myr")
                        self.logger.info(f"Phase 2 initial conditions: R={self.state.R/PC_TO_CM:.2f} pc, "
                                f"V={self.state.V_sh/KM_TO_CM:.2f} km/s, "
                                f"M={self.state.M_sh/M_SUN:.2e} M_sun,"
                                f"n_cloud={self.state.n_cloud_avg:.2e} cm-3")
                    elif abs(sol_frag.t[-1] - MAX_SIMULATION_TIME) <= RTOL_MAIN_FRAG * MAX_SIMULATION_TIME:
                        self.state.t_end = MAX_SIMULATION_TIME
                        self.state.t = MAX_SIMULATION_TIME
                        generation_results = self.process_and_concatenate_results(generation_results, append_to_all=True)
                        self.logger.info(f"Simulation ended during Fragmentation at t={self.state.t_end/MYR_TO_SEC:.2f} Myr")
                        break
                    else:
                        self.logger.error(f"Fragmentation ended unexpectedly at t={sol_frag.t[-1]/MYR_TO_SEC:.2f} Myr")
                        self.logger.error(f"Solver status: {sol_frag.status}")
                        self.logger.error(f"Solver message: {sol_frag.message}")
                        self.logger.error(f"Time remaining: {(MAX_SIMULATION_TIME - sol_frag.t[-1])/MYR_TO_SEC:.2f} Myr")
                        raise RuntimeError("Fragmentation failed: Neither transitioned to Phase-2 nor reached maximum time")

                if self.state.phase in ['phase2', 'dissolution']:

                    if self.is_phase_timespan_negligible(self.state.phase):
                        if generation_results:
                            self.all_results.append(generation_results)
                        break

                    self.check_and_adjust_shell_mass()
                    t_start_phase2 = self.state.t
                    t_span_phase2 = (t_start_phase2, MAX_SIMULATION_TIME)
                    y0_phase2 = np.array([self.state.R, self.state.V_sh, self.state.M_sh, self.state.n_cloud_avg])
                    atols_phase2 = get_atols(y0_phase2[0], phase=self.state.phase, dynamic_cloud_density=self.dynamic_cloud_density)
                    self.logger.debug(f"Solving IVP for phase-2/dissolution: t_span={t_span_phase2}, y0={y0_phase2}, atols={atols_phase2}, rtol={RTOL_MAIN_PH2}")      
                    
                    termination_identifier = None
                    with capture_output() as get_buffers:
                        try:
                            sol_phase2 = solve_ivp(
                                self.integrated_evolution, 
                                t_span_phase2, 
                                y0_phase2,
                                method=SOLVER_PH2,
                                rtol=RTOL_MAIN_PH2, 
                                max_step=MAX_STEP_PH2,
                                atol=atols_phase2, 
                                events=(self.terminate_shell_recollapse),
                                dense_output=True
                            )
                        except ManualEventTermination as e:
                            termination_identifier = e.identifier
                            self.logger.info(f"Dissolution manually terminated: {str(e)}")
                            sol_phase2 = self.create_partial_solution(e.t, e.y)
                            self.state.dissolved = True

                        stdout_output, stderr_output = process_buffers(get_buffers)

                    if termination_identifier != "MANUAL_TERMINATION":
                        if stdout_output:
                            print(stdout_output, file=sys.stdout)
                        if stderr_output:
                            print(stderr_output, file=sys.stderr)

                    self.logger.debug(f"Phase 2/Dissolution IVP solved.")

                    self.state.t_end = self.state.t
                    
                    # Process results of the second solve_ivp
                    if self.state.dissolved:
                        self.t_dissolution = self.state.t
                        self.logger.info(f"Shell dissolved at t={self.t_dissolution/MYR_TO_SEC:.2f} Myr")
                        generation_results = self.process_and_concatenate_results(generation_results, append_to_all=True)
                        break
                    elif sol_phase2.t_events[0].size > 0:  # Recollapse occurred
                        self.t_recollapse = self.state.t
                        self.logger.info(f"Shell recollapsed at t={self.t_recollapse/MYR_TO_SEC:.2f} Myr")
                        generation_results = self.process_and_concatenate_results(generation_results, append_to_all=True)
                        self.logger.info("Starting new generation")
                        self.add_new_generation()
                        generation_results = {}  # reset for the next gen
                    elif abs(sol_phase2.t[-1] - MAX_SIMULATION_TIME) <= RTOL_MAIN_PH2 * MAX_SIMULATION_TIME:
                        self.state.t_end = MAX_SIMULATION_TIME
                        self.state.t = MAX_SIMULATION_TIME
                        generation_results = self.process_and_concatenate_results(generation_results, append_to_all=True)
                        self.logger.info(f"Simulation reached maximum time at t={self.state.t_end/MYR_TO_SEC:.2f} Myr")
                        break
                    else:
                        self.logger.error(f"Phase-2 ended unexpectedly at t={sol_phase2.t[-1]/MYR_TO_SEC:.2f} Myr")
                        self.logger.error(f"Solver status: {sol_phase2.status}")
                        self.logger.error(f"Solver message: {sol_phase2.message}")
                        self.logger.error(f"Time remaining: {(MAX_SIMULATION_TIME - sol_phase2.t[-1])/MYR_TO_SEC:.2f} Myr")
                        raise RuntimeError("Phase-2 failed: did not dissolve, recollapse, or reach maximum time")
                        
            simulation_params = {
                "Z": self.Z,
                "eta_sf": self.eta_sf,
                "n_cl": self.n_cl,
                "M_cl_init": f"{self.M_cl_init/M_SUN:.2e} Msun",
                "template": self.template,
                "imf": self.imf,
                "cluster_formation_mode": self.cluster_formation_mode,
                "dark_matter_fraction": getattr(self, "dark_matter_fraction", 0.0),
                "dark_matter_mass": f"{getattr(self, 'M_dm', 0.0)/M_SUN:.2e} Msun",
                "dust_to_metal": self.dust_to_metal,
            }

            self.tracker.write_output_file(self.output_path, simulation_params, self.all_results, self.logger)
            self.logger.info(f"Simulation completed at t={self.state.t_end/MYR_TO_SEC:.2f} Myr")
                    
            return self.all_results
            
        finally:
            # Restore original parameters
            SOLVER_PH1, SOLVER_FRAG, SOLVER_PH2 = original_solvers
            RTOL_MAIN_PH1 = original_rtol_ph1
            RTOL_MAIN_FRAG = original_rtol_ph1
            RTOL_MAIN_PH2 = original_rtol_ph1 * original_rtol_ratio_ph2

    def run_simulation(self):
        """
        Run the full simulation with automatic retries using different parameters.
        
        The simulation is attempted up to 3 times:
        1. Standard parameters (default solvers, standard rtol)
        2. Alternative solvers (standard rtol)
        3. Original solvers with looser tolerance (default solvers, looser rtol)
        
        Returns:
            list: Simulation results
            
        Raises:
            RuntimeError: If all attempts fail
        """
        if getattr(self, "_output_exists", False):
            # Results were already computed and cached on disk (the constructor leaves
            # the object half-initialised in this case); load and return them rather
            # than re-running or failing.
            from .track_simulation import load_output_file
            print(f"Loading cached results from '{self.output_path}'.")
            _, results = load_output_file(self.output_path)
            return results

        with self._logging_context():
            self.logger.start_timer()
            self.logger.info("Starting simulation")
            # Define standard solvers
            standard_solvers = (SOLVER_PH1, SOLVER_FRAG, SOLVER_PH2)
            alternative_solvers = (ALT_SOLVER_PH1, ALT_SOLVER_FRAG, ALT_SOLVER_PH2)
            
            # Define attempts parameters: (solvers, rtol)
            attempts = [
                (standard_solvers, RTOL_MAIN_PH1),       # Standard attempt
                (alternative_solvers, RTOL_MAIN_PH1),    # Different solvers
                (standard_solvers, LOOSER_RTOL)          # Looser tolerance
            ]
            
            for attempt, (solvers, rtol) in enumerate(attempts, 1):
                self.logger.info(
                    f"\nStarting attempt {attempt}/3 with "
                    f"solvers=(ph1={solvers[0]}, frag={solvers[1]}, ph2={solvers[2]}), "
                    f"rtol={rtol}"
                )
                
                try:
                    with TimeoutManager(SIMULATION_TIMEOUT):
                        results = self.simulation_attempt(solvers, rtol)
                        self.logger.info(f"Attempt {attempt} succeeded")
                        self.logger.log_total_time()
                        self.logger.flush()
                        self.logger.close()
                        return results
                        
                except TimeoutError:
                    self.logger.warning(f"Attempt {attempt} timed out after {SIMULATION_TIMEOUT} seconds")
                except Exception as e:
                    tb_str = traceback.format_exc()
                    self.logger.warning(f"Attempt {attempt} failed with error: {str(e)}\n{tb_str}")
                
                # Clean up everything after failed attempt
                if attempt < len(attempts):
                    self.cleanup_after_attempt()
            
            # If we get here, all attempts failed
            self.logger.error("All attempts failed")
            self.logger.log_total_time()
            self.logger.flush()
            self.logger.error("Failed to complete simulation after all attempts")
            self.logger.close()
            return


    def integrated_evolution(self, t, x):
        """
        Core method for the evolution calculations.
        
        Updates state variables and tracks cumulative times for RT instability conditions.
        The primary condition is positive acceleration (dt_cum_posAcc). Only when
        accelerating are the additional conditions checked:
        - dt_cum_density: Time density contrast has exceeded threshold (if USE_DENSITY_CONTRAST)
        - dt_cum_beta: Time beta condition has been met (if USE_BETA_CONDITION)
        
        All cumulative times are reset to zero when acceleration becomes negative.
        Times are compared against the local sound crossing time (shell_thickness/c_s_shell_min)
        to determine if instabilities have persisted long enough to be physical.

        Args:
            t (float): Current time.
            x (numpy.ndarray): Current state vector [R, V_sh, E_b, M_sh, n_cloud_avg] for phase1/fragmentation,
                            or [R, V_sh, M_sh, n_cloud_avg] for phase2/dissolution.

        Returns:
            numpy.ndarray: Rate of change of the state vector. Length depends on phase:
                        - phase1/fragmentation: [dR/dt, dV/dt, dE/dt, dM/dt, dn_cloud_avg/dt]
                        - phase2/dissolution: [dR/dt, dV/dt, dM/dt, dn_cloud_avg/dt]
        """
        assert t >= max(self.state.t_list_collapse + [0]), "Current time is before last collapse time"
    
        if self.state.phase in ['phase1', 'fragmentation']:
            self.state.update(
                dt=t - self.state.t_old,
                t=t,
                R=x[0],
                V_sh=x[1],
                E_b=x[2],
                M_sh=x[3],
                n_cloud_avg=x[4]
                )
        else:  # phase2 or dissolution
            self.state.update(
                dt=t - self.state.t_old,
                t=t,
                R=x[0],
                V_sh=x[1],
                M_sh=x[2],
                n_cloud_avg=x[3]
            )

        if self.dynamic_cloud_density:
            self.R_cl = self.density_profile.calculate_radius() 
        
        if self.state.phase in ['phase1', 'fragmentation']:
            self.state.Rin = self.get_inner_radius()
            volume = (4/3) * np.pi * (self.state.R**3 - self.state.Rin**3)
            self.state.P_b = self.get_pressure(volume)

        if self.dynamic_cloud_density and self.M_cl - self.state.M_sh > 0 and self.state.n_cloud_avg > N_ISM:
            self.update_density_profile()

        self.cover_frac = self.get_covering_fraction()
        self.update_shell_structure()
        self.update_lyman_alpha()

        if self.state.phase == 'phase1':
            result = self.phase1.solve()
        elif self.state.phase == 'fragmentation':
            result = self.fragmentation.solve()
        elif self.state.phase == 'phase2':
            result = self.phase2.solve()
        elif self.state.phase == 'dissolution':
            self.dissolution.solve()
            raise ManualEventTermination(f"Termination condition met at t={t/MYR_TO_SEC:.6f} Myr", t, x)
        
        # Update cumulative time counters
        # RT inst
        acceleration = self.get_acceleration()
        if acceleration > 0:
            self.state.dt_cum_posAcc += self.state.dt
            
            # Only check these conditions if we're accelerating
            if USE_DENSITY_CONTRAST:
                n_external = self.density_profile.density(self.state.R) / MU_N
                density_contrast = self.state.n_shell_max / n_external
                if density_contrast > DENSITY_CONTRAST_RT:
                    self.state.dt_cum_density += self.state.dt
                else:
                    self.state.dt_cum_density = 0

            if USE_BETA_CONDITION:
                if self.state.beta <= 0:
                    self.state.dt_cum_beta += self.state.dt
                else:
                    self.state.dt_cum_beta = 0
        else:
            # Reset all RT-related timers if not accelerating
            self.state.dt_cum_posAcc = 0
            self.state.dt_cum_density = 0
            self.state.dt_cum_beta = 0

        # Dissolution
        if self.check_dissolution_density_condition():
            self.state.dt_cum_dissolution += self.state.dt
        else:
            self.state.dt_cum_dissolution = 0

        if abs(self.state.V_sh) < V_ZERO_STALL:
            self.state.dt_cum_stalled += self.state.dt
        else:
            self.state.dt_cum_stalled = 0

        # track solution, shell properties, feedback, and energetics
        if self.tracker.should_record(t):
            self.tracker.update_solution(t, x)

            self.tracker.update_shell_properties(t,
                n_shell_in = self.state.n_shell_in,
                n_shell_max = self.state.n_shell_max,
                n_shell_max_ionized = self.state.n_shell_max_ionized,
                M_shell = self.state.M_sh,
                f_esc_i = self.state.f_esc_i,
                f_esc_uv = self.state.f_esc_uv,
                columnDensity_H1 = self.state.columnDensity_H1,
                covering_fraction = self.cover_frac
            )

            self.tracker.update_cloud_properties(t,
                n_avg_cloud = self.state.n_cloud_avg, 
                f_esc_i_cloud = self.f_esc_i_cloud,
                R_cloud = self.R_cl
            )

            # stellar feedback tracking on a per-generation basis
            feedback_data = self.stellar_feedback.get_feedback_data_per_generation(t)
            # non-generation specific data
            feedback_data['Lya_multiplier'] = self.lyman_alpha.force_multiplier
            feedback_data['eta_rad'] = self.state.eta_rad
            self.tracker.update_feedback(t, **feedback_data)


            # overall force tracking
            force_data = self.get_forces()          
            self.tracker.update_forces(t, **force_data)
            
            # shell KE, PE tracking
            energetics = self.get_energetics()
            self.tracker.update_energetics(t, **energetics)

        self.state.t_old = t
        sanitized_result = self.sanitize_result(result)

        if self.logger.should_log(t, self.state.phase):
            self.logger.log_state(self.state, self)
            if self.state.phase == 'fragmentation':
                cooling_rate = self.fragmentation.get_cooling_rate_fragmentation()
                self.logger.log_fragmentation_state(
                    self.state.t,
                    self.t_fragmentation,
                    self.state.E_b,
                    self.E_b_init_frag,
                    self.t_sound_fragmentation,
                    cooling_rate
                )

        return sanitized_result

    def sanitize_result(self, result):
        """
        Sanitize the result to ensure it's a 1D array of scalar values.

        Args:
            result (list): The raw result from phase calculations.

        Returns:
            numpy.ndarray: Sanitized 1D array of 4 or 5 float values.

        Raises:
            ValueError: If the result cannot be sanitized to 4 or 5 float values.
        """
        sanitized = []
        for value in result:
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    sanitized.append(float(value.item()))
                else:
                    raise ValueError(f"Unexpected array with size > 1: {value}")
            elif isinstance(value, (int, float)):
                sanitized.append(float(value))
            else:
                raise ValueError(f"Unexpected type in result: {type(value)}")
        
        if len(sanitized) not in [4, 5]:
            raise ValueError(f"Expected 4 or 5 values, got {len(sanitized)}")
        
        return np.array(sanitized, dtype=float)

    def update_shell_structure(self):
        """
        Update the shell structure based on the current state.

        Raises:
            RuntimeWarning: If invalid values are encountered during calculation.
            ValueError: If an unknown phase is encountered.
        """

        if self.state.phase in ['phase1', 'fragmentation']:
            F = 4 * np.pi * self.state.R**2 * self.state.P_b
        elif self.state.phase in ['phase2', 'dissolution']:
            F = self.stellar_feedback.get_ram_force(self.state.t) # F_ram
        else:
            raise ValueError(f"Unknown phase: {self.state.phase}")
        
        # Calculate inner shell density => one of the initial conditions for the shell structure
        # use the interpolated ionized gas temperature if variable ionized temperature is enabled
        Hii_temperature_interpolator = self.Hii_temperature_interpolator if self.use_Tion_model else None
        Q_i = self.stellar_feedback.get_ionizing_photon_rate(self.state.t)
        
        self.state.n_shell_in, self.state.T_sh_ion = get_shell_inner_face_density(F, self.state.R, self.Z, Q_i, Hii_temperature_interpolator)

        self.shell_structure = ShellStructure(self.Z, self.state.n_shell_in, self.state.M_sh, \
                                              self.state.T_sh_ion, self.stellar_feedback, self.state.t, self.cover_frac)
        self.state.shell_thickness = self.shell_structure.solve_shell_structure(self.state.R)
        self.c_s_shell_weighted = self.shell_structure.calculate_weighted_sound_speed() 

        self.state.n_shell_max_ionized = np.max(self.shell_structure.shell_strctre_i.y[0])
        if self.shell_structure.shell_strctre_n is None:
            self.state.f_esc_i = self.shell_structure.shell_strctre_i.y[1][-1]
            tau_d = self.shell_structure.shell_strctre_i.y[2][-1]
            self.state.f_esc_uv = np.exp(-tau_d)
            self.state.n_shell_max = np.max(self.shell_structure.shell_strctre_i.y[0])
            self.c_s_shell_min = np.sqrt((GAMMA * K_BOLTZMANN * self.state.T_sh_ion) / MU_N)
        else:
            self.state.f_esc_i = 0
            tau_d = self.shell_structure.shell_strctre_n.y[1][-1]
            self.state.f_esc_uv = np.exp(-tau_d)
            self.state.n_shell_max = np.max(self.shell_structure.shell_strctre_n.y[0])
            self.c_s_shell_min = np.sqrt((GAMMA * K_BOLTZMANN * T_NEUTRAL) / MU_N)

        f_abs_i = 1 - self.state.f_esc_i
        f_abs_n = 1 - self.state.f_esc_uv
        tau_IR = (self.kappa_IR * MU_N * tau_d) / self.sigma_dust
        L_bol = self.stellar_feedback.get_bolometric_luminosity(self.state.t)
        Lum_i = self.stellar_feedback.get_ionizing_luminosity(self.state.t)
        Lum_n = L_bol - Lum_i
        self.state.eta_rad = ((f_abs_i * Lum_i + f_abs_n * Lum_n) / L_bol) * (1 + tau_IR)

    def update_lyman_alpha(self):
        """Update the Lyman-alpha related properties."""
        self.state.columnDensity_H1 = self.shell_structure.get_atomicHydrogenColumnDensity()

    def get_radiation_force(self):
        """
        Calculate the total radiation force on the shell.
        Assumes full covering of the shell

        Returns:
            float: Total radiation force in dynes.
        """
        F_rad_UV_IR = self.state.eta_rad * self.stellar_feedback.get_bolometric_luminosity(self.state.t) / C
        if self.include_lyman_alpha:
            Q_i = self.stellar_feedback.get_ionizing_photon_rate(self.state.t)
            F_rad_Lya = self.lyman_alpha.get_LyAlpha_radiationForce(self.state.columnDensity_H1, Q_i, self.shell_structure, self.state.V_sh)
        else:
            F_rad_Lya = 0.0
        return F_rad_UV_IR + F_rad_Lya

    def get_gravitational_force(self):
        """
        Calculate the gravitational force on the shell.
        Includes contributions from stellar mass, shell mass, and dark matter.

        Returns:
            float: Gravitational force in dynes.
        """
        if not self.include_gravity:
            return 0.0
        M_star = self.stellar_feedback.get_stellar_mass_cgs()
        M_grav = M_star + (self.state.M_sh / 2) + self.M_dm
        return G * self.state.M_sh * M_grav / self.state.R**2
    
    def calculate_gravitational_energy(self):
        """
        Calculate the gravitational potential energy of the system.

        Returns:
            float: Gravitational potential energy in ergs.
        """
        if not self.include_gravity:
            return 0.0
        M_star = self.stellar_feedback.get_stellar_mass_cgs()
        M_grav = M_star + (self.state.M_sh / 2) + self.M_dm
        E_grav = -G * self.state.M_sh * M_grav / self.state.R
        return E_grav

    def get_external_force(self):
        """
        Calculate the external force on the shell, accounting for the density profile and shell dynamics.
        Assumes full covering of the shell.

        Returns:
            float: External force on the shell in dynes.
        """
        if not self.include_external_pressure:
            return 0.0
        # Compute neutral pressure and ionized pressure
        P_neutral = max(self.density_profile.pressure(self.state.R + 1e-6 * PC_TO_CM), 
                            N_ISM * K_BOLTZMANN * T_NEUTRAL)
        P_ion = P_neutral * (MU_N / MU_P) * (T_ION / T_NEUTRAL)
        
        # Determine the base external pressure
        if self.state.M_sh < self.M_cl:
            # Inside the cloud, consider ionization effects
            if self.state.f_esc_i > 0:
                P_ext = func_tanh_extPressure_fesc_ion(
                    self.state.f_esc_i, P_ion=P_ion, P_neutral=P_neutral
                )
            else:
                P_ext = P_neutral
        else:
            # Outside the cloud, use interstellar medium pressure
            P_ext = self.P_ext_ISM
        
        return P_ext * 4 * np.pi * self.state.R**2 * func_tanh_extPressure_velocity(self.state.V_sh)

    def get_cooling_rate(self):
        """
        Calculate the cooling rate of the shell.

        Returns:
            float: Cooling rate in erg/s.
        """
        if self.state.phase == 'fragmentation':
            return self.fragmentation.get_cooling_rate_fragmentation()
        else:
            return 0

    def get_energy_change_rate(self):
        """
        Calculate the rate of energy change in the shell.

        Returns:
            float: Energy change rate in erg/s.
        """
        if self.state.phase == 'fragmentation':
            L_mech = 0
        else:
            L_mech = self.stellar_feedback.get_mechanical_luminosity(self.state.t)
        L_cool = self.get_cooling_rate()
        return L_mech - self.state.P_b * 4 * np.pi * self.state.R**2 * self.state.V_sh - L_cool

    def get_pressure(self, volume):
        """
        Calculate the pressure, handling potential NaN values.

        Args:
            volume (float): The volume of the shell.

        Returns:
            float: The calculated pressure.
        """
        thermal_pressure = (GAMMA - 1) * self.state.E_b / volume
        ram_pressure_R_IN_MAX = self.stellar_feedback.get_ram_force(self.state.t) / (4 * np.pi * (FRAC_R_IN_MAX * self.state.R)**2)
        
        if np.isnan(thermal_pressure) or np.isinf(thermal_pressure):
            return ram_pressure_R_IN_MAX
        else:
            return max(thermal_pressure, ram_pressure_R_IN_MAX)

    def get_acceleration(self):
        """
        Calculate the acceleration of the shell, 
        If turned on, takes into account a covering fraction after cloud fully swept.
        Post sweep value is a free parameter => see get_covering_fraction method
        Covering fraction impacts F_rad, F_ext, and F_ram

        Returns:
            float: Acceleration in cm/s^2.
        """

        F_th   = 4 * np.pi * self.state.R**2 * self.state.P_b
        F_rad  = self.get_radiation_force() * self.cover_frac
        F_grav = self.get_gravitational_force()
        F_ext  = self.get_external_force() * self.cover_frac
        
        if self.state.phase == 'phase2':
            F_th = 0
            F_ram = self.stellar_feedback.get_ram_force(self.state.t) * self.cover_frac
        else:
            F_ram = 0

        return (F_th + F_rad + F_ram - F_grav - F_ext - self.state.V_sh * self.get_mass_change_rate()) / self.state.M_sh
    
    def get_pressure_change_rate(self):
        """
        Calculate the rate of pressure change in the shell.

        Returns:
            float: Pressure change rate in dyne/cm^2/s.
        """
        dEdt = self.get_energy_change_rate()
        F_ram_rate = self.stellar_feedback.get_ram_force_derivative(self.state.t)
        F_ram = self.stellar_feedback.get_ram_force(self.state.t)
        a = 1.5 * (F_ram_rate / F_ram)
        c = 0.75 * (F_ram * self.state.Rin) 
        d = (self.state.R**3) - (self.state.Rin**3) 
        ratio = (1 - (c / (self.state.E_b + c)))
        rhs = (dEdt * (d * ratio)) - (3 * self.state.E_b * self.state.V_sh * (self.state.R**2) * ratio) + (a * (self.state.Rin**3) * (self.state.E_b**2) / (self.state.E_b + c))
        dPdt = rhs / (2 * np.pi * (d**2))
        return dPdt

    def get_mass_change_rate(self):
        """
        Calculate the rate of mass change of the shell using the density profile.
        """
        if self.state.M_sh < self.M_cl and self.state.V_sh > 0:
            rho = self.density_profile.density(self.state.R)
            return rho * 4 * np.pi * self.state.R**2 * self.state.V_sh
        else:
            return 0

    def check_and_adjust_shell_mass(self):
        """
        Check if the shell mass exceeds the cloud mass due to numerical issues.
        Adjust the shell mass if necessary and log the adjustment.
        """
        excess_mass = self.state.M_sh - self.M_cl
        if excess_mass > 0:
            self.state.M_sh = self.M_cl
            self.logger.warning(f"Tolerance issue detected: Shell mass exceeded cloud mass by {excess_mass/M_SUN:.2e} M_sun. Adjusted to match cloud mass.")
        else:
            pass

    def calculate_dn_cloud_avg_dt(self):
        if not self.dynamic_cloud_density:
            return 0  # Only calculate if dynamic cloud density is enabled
        
        if not isinstance(self.density_profile, UniformDensityWithCavity):
            raise RuntimeError("Dynamic cloud density is enabled but the density profile is not UniformDensityWithCavity.")

        if self.M_cl - self.state.M_sh <= 0 or self.state.n_cloud_avg <= N_ISM or self.state.f_esc_i == 0:
            return 0 # additional check
          
        # Calculate energy injection
        Q_i = self.stellar_feedback.get_ionizing_photon_rate(self.state.t)
        self.f_esc_i_cloud = calculate_cloud_escape_fraction(self.M_cl - self.state.M_sh, self.state.n_cloud_avg, self.Z, Q_i, self.state.R)
        E_inj = (1 - self.f_esc_i_cloud) * self.state.f_esc_i * self.kinetic_energy_efficiency * Q_i * E_LYC * self.dt_calc_dynamic_cloud

        # Create a new CloudEnergyInjection instance with current state
        current_profile = self.density_profile
        current_binding_energy = abs(current_profile.binding_energy())

        # Check if injection energy is greater than 90% of binding energy
        if E_inj > 0.9 * current_binding_energy:
            self.logger.warning(f"Injection energy ({E_inj:.2e} erg) exceeds 90% of binding energy ({current_binding_energy:.2e} erg). Setting dn_cloud_avg_dt to 0.")
            return 0.0

        cloud_energy = CloudEnergyInjection(current_profile)

        # Inject energy and get new profile
        cloud_energy.inject_energy(E_inj)
        new_profile = cloud_energy.get_current_cloud()

        # Calculate change in average density
        dn_avg_dt = - abs(new_profile.n_avg - current_profile.n_avg) / self.dt_calc_dynamic_cloud

        return dn_avg_dt
    
    def get_covering_fraction(self):
        """
        Calculate the covering fraction of the shell with smooth transitions.
        
        The covering fraction:
        - Starts at 1.0 in phase1 and fragmentation
        - Transitions to post_sweep_covering_fraction when cloud is fully swept
        - Transitions back to 1.0 if shell infalls to within INFALL_RADIUS_RATIO_COVERFRAC of cloud radius
        
        Returns:
            float: The calculated covering fraction between 0 and 1.
        """
        if not self.add_cover_frac:
            return 1.0

        if self.state.phase == 'phase1' or self.state.phase == 'fragmentation':
            return 1.0

        # Check for cloud sweep if not already marked
        if self.t_cloud_swept is None and np.isclose(self.state.M_sh, self.M_cl, rtol=1e-3):
            self.t_cloud_swept = self.state.t
            self.logger.info(f"Cloud fully swept at t={self.t_cloud_swept/MYR_TO_SEC:.2f} Myr. Starting covering fraction transition.")

        if self.t_cloud_swept is not None:
            # Get base covering fraction after sweep
            base_cf = func_tanh_sweep_coverfrac(self.state.t, self.t_cloud_swept, self.post_sweep_covering_fraction)
            
            # Check for infall condition
            if self.state.V_sh < 0:  # Infall
                radius_ratio = self.state.R / self.R_cl
                return func_tanh_infall_coverfrac(radius_ratio, base_cf)
            
            return base_cf # returns the base value till we reach R_shell close to R_cl / 2 

        return 1.0  # Cloud not yet fully swept in phase-2 or dissolution

    def get_inner_radius(self, use_implicit=True):
        """
        Calculate the inner radius of the shell. 
        Using the implicit equation helps the solver converge faster.

        If use_implicit is True, solves the implicit equation:
        R_1 = [(F_ram / (2 * E_b)) * (R_2^3 - R_1^3)]^(1/2)

        Where:
        R_1 is the inner radius
        R_2 is the outer radius (self.state.R)
        F_ram is the ram force
        E_b is the bubble energy

        If use_implicit is False, uses the original simplified calculation.

        Args:
            use_implicit (bool): Whether to use the implicit equation method. Defaults to True.

        Returns:
            float: Inner radius in cm.

        Raises:
            ValueError: If the root-finding method fails to converge when using the implicit method.
        """
        if self.state.phase in ['phase1', 'fragmentation']:
            F_ram = self.stellar_feedback.get_ram_force(self.state.t)
            R_2 = self.state.R

            if use_implicit:
                E_b = self.state.E_b

                def equation(R_1):
                    return R_1 - np.sqrt((F_ram / (2 * E_b)) * (R_2**3 - R_1**3))

                # Define the bracket for root-finding
                R_min = 1e-16 * R_2  # Very small, non-zero lower bound
                R_max = R_2  # Upper bound is the outer radius

                try:
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("default")
                        inner_radius = brentq(equation, R_min, R_max, xtol=XTOL_INNER_RADIUS, maxiter=MAX_ITER_INNER_RADIUS)
                        if len(w) > 0:
                            for warning in w:
                                self.logger.warning(f"Warning in inner radius calculation: {warning.message}")
                except ValueError as e:
                    self.logger.warning(f"Root-finding failed: {str(e)}. Falling back to simplified calculation.")
                    use_implicit = False  # Fall back to the original method
            
            if not use_implicit:
                # Original simplified calculation
                inner_radius_squared = F_ram / (4 * np.pi * self.state.P_b)
                inner_radius = np.sqrt(inner_radius_squared)

            # Sanity check
            if not 0 < inner_radius < R_2:
                self.logger.warning(f"Calculated inner radius ({inner_radius:.2e} cm) is outside expected range. Using {FRAC_R_IN_MAX:.3f} of outer radius.")
                inner_radius = FRAC_R_IN_MAX * R_2

            return inner_radius
        else:
            return self.state.R

    def check_dissolution_density_condition(self):
        """
        Check if the conditions for shell dissolution are met.

        Returns:
            bool: True if dissolution density conditions are met, False otherwise.
        """
        density_condition = self.state.n_shell_max <= DISSOLUTION_DENSITY
        SNE_condition = self.state.t > T_SN + self.state.t_end # latest SN explosion => shell reformation
        return density_condition and SNE_condition

    def update_density_profile(self):
        """Update density profile with new average density
        Current implementation only limited to uniform density
        To do: Add other profiles """
        self.density_profile = UniformDensityWithCavity(
            M_gas = self.M_cl - self.state.M_sh,
            n_avg = self.state.n_cloud_avg,
            R_cav = self.state.R)  

    def process_and_concatenate_results(self, generation_results, append_to_all=False):
        """
        Process current simulation state and concatenate with existing generation results.
        
        Args:
            generation_results (dict): Existing results for current generation, can be empty dict
            append_to_all (bool): Whether to append to self.all_results if generation complete
            
        Returns:
            dict: Updated generation_results including new processed data
            
        Note:
            If process_results returns None (no data points recorded), the original 
            generation_results is returned unchanged.
        """
        partial_results = self.process_results()
        if partial_results is None:
            self.logger.warning("No data points recorded in this integration step")
            return generation_results
            
        updated_results = self.concatenate_results(generation_results, partial_results)
        
        if append_to_all and updated_results:
            self.all_results.append(updated_results)
            self.logger.debug(f"Added generation {self.state.current_generation + 1} results to all_results")
            
        return updated_results

    def concatenate_results(self, current_results, new_results):
        """
        Concatenate new results with current results within a generation.

        Args:
            current_results (dict): Current results for the generation.
            new_results (dict): New results to be concatenated.

        Returns:
            dict: Concatenated results.
        """
        if not current_results:
            return new_results

        for key in new_results:
            if key in ['status', 'phase_transitions', 'cluster_formation_mode', 'formation_timescale', 'generation', 'cloud_mass', 'cloud_radius']:
                current_results[key] = new_results[key]
            elif isinstance(new_results[key], np.ndarray):
                current_results[key] = np.concatenate([current_results[key], new_results[key]])
            elif isinstance(new_results[key], dict):
                for subkey in new_results[key]:
                    if subkey in current_results[key]:
                        current_results[key][subkey] = np.concatenate([current_results[key][subkey], new_results[key][subkey]])
                    else:
                        current_results[key][subkey] = new_results[key][subkey]

        return current_results

    def process_results(self):
        """
        Process the results of the simulation for a single solve_ivp call.
        Skips processing if no data points were recorded.

        Returns:
            dict: Processed results including time series, energetics, and other properties.
                Returns None if no data points were recorded.
        """
        # Check if we have any recorded data points
        if not self.tracker.solution['t'] or not self.tracker.solution['y']:
            self.logger.warning("No data points recorded during this integration step, skipping processing")
            return None

        results = self.tracker.get_results()
        results.update({
            'status': self.get_final_status(),
            'phase_transitions': self.get_phase_transitions(),
            'cluster_formation_mode': self.cluster_formation_mode,
            'formation_timescale': self.formation_timescale,
            'star type': self.star_type,
            'generation': self.state.current_generation,
            'cloud_mass': self.M_cl,
            'cloud_radius': self.R_cl,
            'post_sweep_covering_fraction': self.post_sweep_covering_fraction if self.add_cover_frac else 1
        })
        self.tracker.reset()
        return results

    def get_final_status(self):
        """
        Determine the final status when a solve_ivp ends.

        Returns:
            str: Final status of the simulation ('dissolved', 'collapsed', 'fragmentation_ended', 'time_limit_reached').
        """
        if self.state.dissolved:
            return 'dissolved'
        elif self.state.t >= MAX_SIMULATION_TIME:
            return 'time_limit_reached'
        elif self.state.phase == 'phase2' and self.t_end_frag is not None and self.t_recollapse is None:
            return 'fragmentation_ended'
        elif self.state.phase == 'phase2' and self.t_end_frag is not None and self.t_recollapse is not None:
            return 'collapsed'

    def get_phase_transitions(self):
        """
        Get the list of phase transitions that occurred during the current generation of the simulation.

        Returns:
            list: List of tuples containing phase transition information (transition type, time, reason).
        """
        transitions = []

        if self.t_fragmentation is not None:
            transitions.append(('phase1_to_fragmentation', self.t_fragmentation, self.phase1.fragmentation_reason))
        if self.t_end_frag is not None:
            transitions.append(('fragmentation_to_phase2', self.t_end_frag, None))  # No specific reason for this transition
        if self.t_dissolution is not None:
            transitions.append(('phase2_to_dissolution', self.t_dissolution, self.dissolution_reason))
        elif self.t_recollapse is not None:
            transitions.append(('phase2_to_recollapse', self.t_recollapse, None))  # No specific reason for recollapse

        return transitions

    def get_energetics(self):
        """Helps the tracker class to track the shell energetics"""
        E_kinetic = 0.5 * self.state.M_sh * self.state.V_sh**2
        E_grav = self.calculate_gravitational_energy()
        return {
            'E_kinetic': E_kinetic,
            'E_potential': E_grav
        }

    def get_forces(self):
        """Helps the tracker class to track forces on the shell"""
        if self.include_lyman_alpha:
            Q_i = self.stellar_feedback.get_ionizing_photon_rate(self.state.t)
            F_rad_Lya = self.lyman_alpha.get_LyAlpha_radiationForce(self.state.columnDensity_H1, Q_i, \
                                                                     self.shell_structure, self.state.V_sh)
        else:
            F_rad_Lya = 0.0
        return {
            'F_rad_UV_IR': self.state.eta_rad * self.stellar_feedback.get_bolometric_luminosity(self.state.t) / C,
            'F_rad_Lya': F_rad_Lya,
            'F_grav': self.get_gravitational_force(),
            'F_ext': self.get_external_force(),
            'F_w_sn': self.stellar_feedback.get_ram_force(self.state.t) if self.state.phase in \
                                        ['phase2', 'dissolution'] else 4 * np.pi * self.state.R**2 * self.state.P_b,
            'F_mass_gain': self.state.V_sh * self.get_mass_change_rate()
            }

    def add_new_generation(self):
        """
        Set up a new generation after shell recollapse.

        This method updates the cloud mass, reinitializes the stellar feedback,
        and resets various state variables for the new generation.
        """

        if len(self.state.t_list_collapse) >= 1:
            assert self.state.t_end > max(self.state.t_list_collapse), "New generation starts before previous collapse"
        

        self.state.t_list_collapse.append(self.state.t_end)
        self.state.current_generation += 1
        
        # Update cloud mass
        self.M_cl *= (1 - self.eta_sf)
        
        # Reinitialize density profile with updated mass
        self.initialize_density_profile()
        self.logger.log_profile_info(self.density_profile)

        # Reinitialize StellarFeedback with updated parameters
        self.stellar_feedback = StellarFeedback(
            self.template, 
            self.Z, 
            self.M_cl_init,
            self.eta_sf, 
            self.state.t_list_collapse,
            mode=self.cluster_formation_mode, 
            formation_timescale=self.formation_timescale,
            imf=self.imf,
            star_type=self.star_type
        )
        
        # Reset or update other parameters as needed
        self.state.phase = 'phase1'
        self.state.dt_cum_dissolution = 0
        self.state.dt_cum_stalled = 0
        self.state.dt_cum_posAcc = 0
        self.f_esc_i_cloud = 0. 

        # Reset some parameters
        self.initialize_others()
        
        # Recalculate initial conditions for the new generation, absolute time coordinates
        self.t0 = self.state.t + T_INIT_TEMPLATE
        self.y0 = self.get_initial_conditions(self.t0, dt=T_INIT_TEMPLATE) # dt is the period of w77 expansion
        self.tracker.reset()
        # initialize the cavity density profile, after ICs
        if self.dynamic_cloud_density and self.profile_type == 'uniform':
            self.update_density_profile()
        # check for fragmentation
        self.initialize_shell_properties()
        self.check_early_fragmentation()


    def terminate_shell_recollapse(self, t, y):
        """
        Event function to detect shell recollapse.

        Args:
            t (float): Current time.
            y (numpy.ndarray): Current state vector.

        Returns:
            float: Difference between current radius and initial radius.
        """
        current_radius = y[0]
        recollapse_indicator = current_radius - self.initial_radius
        return recollapse_indicator

    terminate_shell_recollapse.terminal = True
    terminate_shell_recollapse.direction = -1

    def terminate_fragmentation(self, t, y):
        return y[2] - FRAC_EB_INIT * self.E_b_init_frag

    terminate_fragmentation.terminal = True
    terminate_fragmentation.direction = -1

    def create_partial_solution(self, t_end, y_end):
        class PartialSolution:
            pass
        
        sol = PartialSolution()
        
        # Get the tracked results
        results = self.tracker.get_results()
        
        # Ensure all arrays have the same length and include the final state
        t = np.append(results['time'], t_end)
        
        def interpolate_and_append(arr, final_value):
            if len(arr) == 0:
                return np.array([final_value])
            if len(arr) < len(results['time']):
                # Interpolate missing values
                interp = interp1d(results['time'][:len(arr)], arr, kind='linear', fill_value='extrapolate')
                arr = interp(results['time'])
            return np.append(arr, final_value)
        
        radius = interpolate_and_append(results['radius'], y_end[0])
        velocity = interpolate_and_append(results['velocity'], y_end[1])
        
        sol.t = t
        sol.y = np.column_stack((radius, velocity))
        
        if len(y_end) == 5:  # For phase1 and fragmentation
            energy = interpolate_and_append(results.get('energy', []), y_end[2])
            mass = interpolate_and_append(results['mass'], y_end[3])
            n_cloud_avg = interpolate_and_append(results.get('n_cloud_avg', []), y_end[4])
            sol.y = np.column_stack((sol.y, energy, mass, n_cloud_avg))
        else:  # For phase2 and dissolution
            mass = interpolate_and_append(results['mass'], y_end[2])
            n_cloud_avg = interpolate_and_append(results.get('n_cloud_avg', []), y_end[3])
            sol.y = np.column_stack((sol.y, mass, n_cloud_avg))
        
        sol.t_events = [np.array([t_end])]
        sol.y_events = [y_end]
        sol.status = 0  # Indicate successful termination
        sol.message = f"{self.state.phase.capitalize()} manually terminated at t={t_end/MYR_TO_SEC:.6f} Myr"
        
        return sol

    def cleanup_after_attempt(self):
        """Clean up all variables after a failed attempt to ensure fresh start."""
        # Store parameters that were passed to __init__
        init_params = {
            'Z': self.Z,
            'eta_sf': self.eta_sf,
            'n_cl': self.n_cl,
            'M_cl_init': self.M_cl_init,
            'template': self.template,
            'cluster_formation_mode': self.cluster_formation_mode,
            'formation_timescale': self.formation_timescale,
            'profile_type': self.profile_type,
            'profile_params': self.profile_params,
            'dynamic_cloud_density': self.dynamic_cloud_density,
            'imf': self.imf,
            'star_type': self.star_type,
            'add_cover_frac': self.add_cover_frac,
            'post_sweep_covering_fraction': self.post_sweep_covering_fraction,
            'dark_matter_fraction': self.dark_matter_fraction,
            'skip_logger_init': True,  # Skip logger initialization during cleanup
            'init_with_fragmentation': self.init_with_fragmentation 
        }
        
        # Store the old logger
        old_logger = self.logger

        # Re-run __init__ with stored parameters
        self.__init__(**init_params)
        
        # Restore the old logger
        self.logger = old_logger
        self.logger.last_log_time = 0

    def is_phase_timespan_negligible(self, phase):
        """
        Check if the current phase's starting time is too close to maximum simulation time.
        
        Args:
            phase (str): Current phase ('phase1', 'fragmentation', 'phase2', or 'dissolution')
        
        Returns:
            bool: True if simulation should end due to short timespan, False otherwise
        """
        # Select appropriate tolerance based on phase
        if phase == 'phase1':
            rtol = RTOL_MAIN_PH1
            t_start = self.t0
        elif phase == 'fragmentation':
            rtol = RTOL_MAIN_FRAG
            t_start = self.state.t
        elif phase in ['phase2', 'dissolution']:
            rtol = RTOL_MAIN_PH2
            t_start = self.state.t
        else:
            raise ValueError(f"Unknown phase: {phase}")
            
        if abs(t_start - MAX_SIMULATION_TIME) <= 2 * rtol * MAX_SIMULATION_TIME:
            self.logger.info(f"{phase.capitalize()} starting time ({t_start/MYR_TO_SEC:.2f} Myr) "
                            f"too close to maximum simulation time, ending simulation.")
            self.state.t_end = MAX_SIMULATION_TIME
            self.state.t = MAX_SIMULATION_TIME  # for registering in final status
            final_status = self.get_final_status()
            self.logger.info(f"Simulation ended with status: {final_status}")
        
            return True
            
        return False

    @contextmanager
    def _logging_context(self):
        try:
            yield
        finally:
            if hasattr(self, 'logger'):
                self.logger.close()    