from .imports import os, np, interp1d, subprocess, contextmanager
from .utils import dtm_label
from .cloudy_base_input_generator import BaseCloudyInputGenerator
from .cloudy_shell_model_generator import ShellModelGenerator
from .cloudy_unified_model_generator import UnifiedModelGenerator
from .cloudy_dig_model_generator import DIGModelGenerator
from .cloudy_dissolved_model_generator import DissolvedModelGenerator
from .cloudy_output_handler import CloudyOutputHandler
from .cloudy_timegrid_generator import TimeGridGenerator
from .cloudy_logging import CloudyLogger
from .track_simulation import load_output_file
from .constants import *

class CloudySimulationManager:
    """
    Manages the execution of Cloudy simulations for different models at various time points.

    This class coordinates the generation of Cloudy input files, runs Cloudy simulations,
    and processes the outputs. It supports different models (shell, unified, and DIG) and
    can handle both adaptive and uniform time sampling methods.

    Attributes:
        sim_out_file (str): Path to the simulation output file.
        cloudy_exec (str): Path to the Cloudy executable.
        method (str): Sampling method ('adaptive' or 'uniform').
        n_points (int): Number of time points for uniform sampling.
        add_DIG (bool): Whether to include DIG models.
        DIG_density (float): Density for the DIG model in cm-3 defaults to None.
        logU_background (float): Log of ionization parameter for background radiation.

    """

    def __init__(self, sim_out_file, method='toddlers_v1', n_points=None, add_DIG=True, DIG_density=None,
                logU_background=None, continue_after_dissolution=False, complete_init=True, add_logger=True,
                dust_to_metal=1.0, small_to_large_ratio=None):
        """
        Initializes the CloudySimulationManager.

        Args:
            sim_out_file (str): Path to the simulation output file.
            method (str, optional): Sampling method ('adaptive' or 'uniform'). Defaults to 'adaptive'.
            n_points (int, optional): Number of time points for uniform sampling. Defaults to None.
            add_DIG (bool, optional): Whether to include DIG. Defaults to True.
            DIG_density (float, optional): Density for the DIG model in cm-3. if None, a default value is used by DIG generator.
            logU_background (float, optional): Log of ionization parameter for background radiation. Defaults to None.
            continue_after_dissolution (bool): Whether to run Cloudy after dissolution time with DIG/dissolved runs.
            complete_init (bool): Whether to fully initialize, data processing is done and is slow.  
            add_logger (bool): Whether to add a logger. If false, log items are printed instead.
        """
        self.complete_init = complete_init
        self.sim_out_file = sim_out_file
        self.cloudy_exec = CLOUDY_EXE
        self.add_logger = add_logger
        self.dust_to_metal = dust_to_metal
        self.small_to_large_ratio = small_to_large_ratio

        self.simulation_params, self.all_results = load_output_file(self.sim_out_file)
        self.create_cloudy_directory()

        if self.complete_init:
            self.logger = self._setup_logger(self.add_logger)
            
            self.process_data() # process evolution data

            self.logger.log_simulation_parameters(self.simulation_params)
            self.logger.start_timer()
            
            if method not in ['adaptive', 'uniform', 'toddlers_v1']:
                raise ValueError("Method must be either 'adaptive' or 'uniform'")
            
            self.max_variations = {
                'n_H_shell': TRIGGER_FRAC_NSHELL,
                'r_shell': TRIGGER_FRAC_RSHELL,
                'M_shell': TRIGGER_FRAC_MSHELL, 
                'L_i': TRIGGER_FRAC_LION 
            }        

            self.add_DIG = add_DIG

            self.method = method
            self.continue_after_dissolution = continue_after_dissolution
            if self.continue_after_dissolution:
                self.logger.warning(f"WARNING: DIG/dissolved models are always added if continue_after_dissolution is True")    
                self.logger.info(f"Add DIG models: {self.add_DIG}")

            self.logger.info(f"Cloudy simulation temporal sampling is: {self.method}")
            self.n_points = n_points
            if self.method == 'adaptive':
                self.logger.info(f"Quantities and their variations that trigger a Cloudy run: {self.max_variations}")
                if self.continue_after_dissolution:
                    self.logger.info("After dissolution DIG/dissolved simulation will continue with a uniform time grid.")
                else:
                    self.logger.info("After dissolution simulation will Stop.")
                if n_points is not None:
                    self.logger.warning(f"n_points is ignored when method is 'adaptive'. Setting n_points to None.")
                self.n_points = None
            elif self.method == 'uniform':
                if n_points is None:
                    self.logger.warning(f"n_points is not defined, a uniform grid with a given time resolution will be used.")
                    self.n_points = None
            else:
                self.logger.warning(f"TODDLERS v1 grid will be used for this simulation")
                self.n_points = None

            if self.add_DIG:
                self.add_old_stellar_radiation = True
                self.logU_background = logU_background if logU_background is not None else LOGU_BG_OLD_DEFAULT
                self.logger.info(f"DIG heating by old stellar radiation is: {self.add_old_stellar_radiation}")
                self.logger.info(f"logU, if DIG heating by old stellar radiation is On:  {self.logU_background}")
                self.DIG_density = DIG_density
                self.logger.info(f"User defined dig density/cm-3 (if None, DIG Generator will use N_ISM):  {self.DIG_density}")
            else:
                self.add_old_stellar_radiation = False
  
            
            interpolants = [self.radius_interp, self.velocity_interp, self.n_shell_in_interp,
                self.shell_mass_interp, self.cloud_mass_interp, self.is_within_cloud_interp, self.covering_fraction_interp]

            self.base_generator = BaseCloudyInputGenerator(
                simulation_params=self.simulation_params,
                t_list_collapse=self.t_list_collapse,
                dissolution_time=self.dissolution_time,
                interpolants=interpolants,
                cloudy_run_dir=self.cloudy_run_dir, logger=self.logger,
                dust_to_metal=self.dust_to_metal,
                small_to_large_ratio=self.small_to_large_ratio
            )

    def create_cloudy_directory(self):

        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(src_dir)
        self.results_dir = os.path.join(project_root, CLOUDY_OUTPUT_DIR)

        params = self.simulation_params
        params_dir = (
            f"Z{params['Z']:.3g}_eta{params['eta_sf']:.3g}_n{params['n_cl']:.1f}"
            f"_logM{np.log10(params['M_cl_init']):.2f}"
        )

        if params.get('add_cover_frac', False):
            params_dir += f"_cf{params.get('post_sweep_covering_fraction', 1.0):.2f}"
        
        if params['cluster_formation_mode'] == 'constant_sfr' and 'formation_timescale' in params:
            params_dir += f"_tform{params['formation_timescale']/MYR_TO_SEC:.1f}"
        
        if params.get('dynamic_cloud_density', False):
            params_dir += "_dynDens"

        params_dir += dtm_label(self.dust_to_metal)

        self.cloudy_run_dir = os.path.join(
            self.results_dir, 
            f"template_{params['template']}",
            f"imf_{params['imf']}",
            f"star_type_{params['star_type']}",
            f"cluster_mode_{params['cluster_formation_mode']}",
            f"profile_type_{params.get('profile_type', 'uniform')}",
            params_dir
        )
        os.makedirs(self.cloudy_run_dir, exist_ok=True)

    def process_data(self):
        """ 
        Process the data from the evolution simulation and
        generates various interpolants needed for input file
        creation.

        Note:
        is_within_cloud uses a 1% tolerance for comparison.
        Thus, unified runs wont run if leftover cloud mass
        is within 1% of the total cloud mass.
        """

        self.dissolution_time = None
        self.t_list_collapse = []

        for i, gen_data in enumerate(self.all_results):
            for transition in gen_data.get('phase_transitions', []):
                if transition[0] == 'phase2_to_dissolution':
                    self.dissolution_time = transition[1]
                    self.logger.info(f"Found dissolution time: {self.dissolution_time/MYR_TO_SEC:.2f} Myr")
                    break
                elif transition[0] == 'phase2_to_recollapse':
                    self.t_list_collapse.append(transition[1])
                    self.logger.info(f"Found collapse time for generation {i}: {transition[1]/MYR_TO_SEC:.2f} Myr")

        if self.dissolution_time is None:
            self.logger.warning("No dissolution time found in simulation data. Will treat as non-dissolving shell.")

        if self.t_list_collapse:
            self.logger.info(f"Total number of collapses found: {len(self.t_list_collapse)}")
        else:
            self.logger.info("No collapse events found")

        self.time_list, self.radius_list, self.velocity_list = [], [], []
        self.shell_mass_list, self.n_shell_in_list, self.cloud_mass_list = [], [], []
        self.is_within_cloud_list, self.covering_fraction_list = [], []
        self.L_i_list, self.L_n_list = [], []
        self.generation_start_times = []

        total_time_points = sum(len(gen_data["time"]) for gen_data in self.all_results)
        num_generations = len(self.all_results)

        self.L_i_full = np.zeros((total_time_points, num_generations))
        self.L_n_full = np.zeros((total_time_points, num_generations))

        current_index = 0
        for _, gen_data in enumerate(self.all_results):
            time = gen_data["time"]
            self.time_list.append(time)
            self.radius_list.append(gen_data["radius"])
            self.velocity_list.append(gen_data["velocity"])
            self.shell_mass_list.append(gen_data["mass"])
            self.n_shell_in_list.append(gen_data["shell_properties"]["n_shell_in"])
            self.covering_fraction_list.append(gen_data["shell_properties"]["covering_fraction"])
            
            cloud_mass = gen_data["cloud_mass"]
            self.cloud_mass_list.append(np.full_like(time, cloud_mass))
            
            mass_ratio = gen_data["mass"] / cloud_mass
            is_within_cloud = mass_ratio < (1.0 - MASS_COMPAR_TOL)
            self.is_within_cloud_list.append(is_within_cloud)

            L_i = np.array(gen_data["stellar_feedback"]["L_i"])
            L_n = np.array(gen_data["stellar_feedback"]["L_n"])
            
            end_index = current_index + len(time)
            self.L_i_full[current_index:end_index, :L_i.shape[1]] = L_i
            self.L_n_full[current_index:end_index, :L_n.shape[1]] = L_n

            current_index = end_index

            self.generation_start_times.append(time[0])

        self.time = np.concatenate(self.time_list)
        self.radius = np.concatenate(self.radius_list)
        self.velocity = np.concatenate(self.velocity_list) 
        self.shell_mass = np.concatenate(self.shell_mass_list)
        self.n_shell_in = np.concatenate(self.n_shell_in_list)
        self.cloud_mass = np.concatenate(self.cloud_mass_list)
        self.is_within_cloud = np.concatenate(self.is_within_cloud_list)
        self.covering_fraction = np.concatenate(self.covering_fraction_list)
        self.interpolate_data()
        self.t_init = T_INIT_OBSERVABLES
        closest_index = np.argmin(np.abs(self.time - self.t_init))
        self.t_start = self.time[closest_index]

    def interpolate_data(self):
        """
        Create interpolation functions for all tracked quantities.
        
        For all quantities:
        - Uses linear interpolation between data points
        - Uses first and last points for extrapolation beyond data range
        
        This ensures physically meaningful behavior where properties maintain
        their boundary values when extrapolating beyond the data range.
        """
        # extrapolation with first/last points
        def create_interp(data):
            return interp1d(self.time, data, kind='linear', 
                        bounds_error=False, 
                        fill_value=(data[0], data[-1]))

        # All quantities use first/last point extrapolation
        self.radius_interp = create_interp(self.radius)
        self.velocity_interp = create_interp(self.velocity)
        self.n_shell_in_interp = create_interp(self.n_shell_in)
        self.shell_mass_interp = create_interp(self.shell_mass)
        self.cloud_mass_interp = create_interp(self.cloud_mass)
        self.is_within_cloud_interp = create_interp(self.is_within_cloud.astype(float))
        self.covering_fraction_interp = create_interp(self.covering_fraction)

    def get_model_generator(self, model_type):
        if model_type == 'shell':
            return ShellModelGenerator(self.base_generator, logger=self.logger)
        elif model_type == 'unified':
            return UnifiedModelGenerator(self.base_generator, logger=self.logger)
        elif model_type == 'dissolved':
            return DissolvedModelGenerator(self.base_generator, logger=self.logger)
        elif model_type == 'dig':
            return DIGModelGenerator(self.base_generator,
                                    add_old_stellar_radiation=self.add_old_stellar_radiation,
                                    density=self.DIG_density,
                                    logU_background=self.logU_background, logger=self.logger)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def determine_model_to_run(self, t):
        """
        Determine which set of models to run at a given time.
        
        Args:
            t (float): Current time in seconds.

        Returns:
            tuple: (list of models to run, status message)
            
        Models are determined based on:
        - Pre-dissolution: shell/unified based on cloud position
        - Post-dissolution: either dissolved model (if add_dig=False) or shell+dig (if add_dig=True)
        
        Uses Unicode squares for clear visual indication:
        [■] - Model will run
        [□] - Model will not run
        """
        models_to_run = []
        
        # Handle case where dissolution_time is None (no dissolution found)
        if self.dissolution_time is None:
            # If no dissolution time found, assume the shell never dissolves
            # and treat it like pre-dissolution behavior
            if self.is_within_cloud_interp(t):
                models_to_run.extend(['shell', 'unified'])
            else:
                models_to_run.append('shell')
            
            if self.add_DIG:
                models_to_run.append('dig')
        else:
            # Normal behavior with known dissolution time
            if t < self.dissolution_time:
                # Pre-dissolution behavior
                if self.is_within_cloud_interp(t):
                    models_to_run.extend(['shell', 'unified'])
                else:
                    models_to_run.append('shell')
                
                if self.add_DIG:
                    models_to_run.append('dig')
            else:
                # Post-dissolution behavior
                if self.add_DIG:
                    models_to_run.extend(['dig'])
                else:
                    models_to_run.append('dissolved')

        # Build status message with unicode squares
        all_possible_models = ['shell', 'unified', 'dig', 'dissolved']
        model_status = []
        
        for model in all_possible_models:
            if model in models_to_run:
                model_status.append(f"[■] {model}")
            else:
                model_status.append(f"[□] {model}")
                
        # Add explanatory note
        if not models_to_run:
            status_msg = "No models to run"
        elif self.dissolution_time is None:
            status_msg = f"Models to run: {', '.join(model_status)} (No dissolution time found)"
        elif t >= self.dissolution_time:
            if self.add_DIG:
                status_msg = f"Models to run: {', '.join(model_status)} (Post-dissolution with DIG)"
            else:
                status_msg = f"Models to run: {', '.join(model_status)} (Post-dissolution direct)"
        elif 'unified' not in models_to_run:
            status_msg = f"Models to run: {', '.join(model_status)} (Shell outside cloud)"
        else:
            status_msg = f"Models to run: {', '.join(model_status)}"

        return models_to_run, status_msg

    def get_time_points(self):
        """Get time points for Cloudy simulations."""
        self.grid_generator = TimeGridGenerator(
            t_start=self.t_start,
            dissolution_time=self.dissolution_time,
            interpolants={
                'radius_interp': self.radius_interp,
                'n_shell_in_interp': self.n_shell_in_interp,
                'shell_mass_interp': self.shell_mass_interp
            },
            logger=self.logger
        )
        
        return self.grid_generator.generate_grid(
            method=self.method,
            n_points=self.n_points,
            continue_after_dissolution=self.continue_after_dissolution,
            stellar_feedback=self.base_generator.stellar_feedback
        )
        
    def run_full_simulation(self):
        try:
            time_points = self.get_time_points()
            with self.change_dir(self.cloudy_run_dir):
                for t in time_points:
                    self.logger.log_timestep_demarcation(t)
                    models_to_run, models_to_run_msg = self.determine_model_to_run(t)
                    self.logger.info(models_to_run_msg)
                    self.logger.log_visual_break(char='-')

                    for model in models_to_run:
                        if model == 'shell':
                            model_generator = self.get_model_generator(model)
                            model_generator.write_input_file(t)
                            self.run_simulation(t, model)
                        elif model == 'unified':
                            model_generator = self.get_model_generator(model)
                            model_generator.write_input_file(t)
                            self.run_simulation(t, model)
                        elif model == 'dissolved':
                            model_generator = self.get_model_generator(model)
                            model_generator.write_input_file(t)
                            self.run_simulation(t, model)
                        elif model == 'dig':                           
                            if self.is_within_cloud_interp(t) > 0:  # True
                                inner_model_prefix = 'unified'
                            else:
                                inner_model_prefix = 'shell'
                            model_generator = self.get_model_generator(model)
                            run_flag = model_generator.write_input_file(t, inner_model_prefix)
                            if run_flag:
                                self.run_simulation(t, model)
                            else:
                                continue
            pass
        finally:
            self.logger.log_total_time()
            self.logger.close()

    def run_simulation(self, t, model):
        """Run a single Cloudy simulation for a specific time and model."""

        self.logger.log_model_start(t, model)

        output_handler = CloudyOutputHandler(model, t)
        input_file = output_handler.get_file_path("in")
        
        try:
            if output_handler.check_cloudy_success():
                self.logger.info(f"Simulation for {input_file} has already been run successfully. Skipping.")
                return

            if os.path.exists(input_file):
                if model == 'shell':
                    physical_conditions = {
                        'radius': (self.radius_interp(t) / PC_TO_CM, 'pc'),
                        'velocity': (self.velocity_interp(t) / KM_TO_CM , 'km/s'),
                        'shell_mass': (self.shell_mass_interp(t) / M_SUN, 'M☉'),
                        'n_shell_in': (self.n_shell_in_interp(t), 'cm^-3')
                    }
                    self.logger.log_physical_conditions(t, physical_conditions)

                env = os.environ.copy()
                if 'CLOUDY_DATA_PATH' in env:
                    del env['CLOUDY_DATA_PATH'] # Run Cloudy without 'CLOUDY_DATA_PATH'
                subprocess.run([self.cloudy_exec, input_file], env=env, check=True, cwd=self.cloudy_run_dir)

                self.logger.info(f"Cloudy run for {input_file} ended.")
                return
            else:
                self.logger.warning(f"Input file {input_file} does not exist. Skipping simulation.")
                return

        except Exception as e:
            self.logger.log_error(t, model, str(e))
            raise

    def _setup_logger(self, add_logger=True):
        """
        Set up logging functionality.
        
        Args:
            add_logger (bool): Whether to use CloudyLogger (True) or print statements (False).
                            Defaults to True.
        
        Returns:
            Logger or PrintWrapper: Logger instance or print wrapper object.
        """
        if add_logger:
            return CloudyLogger(self.cloudy_run_dir)
        else:
            # Create a simple wrapper that mimics logger interface but just prints
            class PrintWrapper:
                @staticmethod
                def info(msg): print(f"INFO: {msg}")
                
                @staticmethod
                def debug(msg): print(f"DEBUG: {msg}")
                
                @staticmethod
                def warning(msg): print(f"WARNING: {msg}")
                
                @staticmethod
                def error(msg): print(f"ERROR: {msg}")
                
                @staticmethod
                def log_error(t, model_type, error_message): 
                    print(f"ERROR: In {model_type} model at t = {t/MYR_TO_SEC:.2f} Myr:")
                    print(f"ERROR: {error_message}")
                
                @staticmethod
                def flush(): pass
                
                @staticmethod
                def close(): pass
                
                @staticmethod
                def log_simulation_parameters(params):
                    print("\nSimulation Parameters:")
                    for key, value in params.items():
                        print(f"{key}: {value}")
                
                @staticmethod
                def start_timer(): pass
                
                @staticmethod
                def log_timestep_demarcation(t):
                    print(f"\nTIMESTEP: t = {t/MYR_TO_SEC:.2f} Myr")
                
                @staticmethod
                def log_model_start(t, model_type):
                    print(f"Starting {model_type} model at t = {t/MYR_TO_SEC:.2f} Myr")
                
                @staticmethod
                def log_physical_conditions(t, conditions):
                    print(f"\nPhysical conditions at t = {t/MYR_TO_SEC:.2f} Myr:")
                    for key, (value, unit) in conditions.items():
                        print(f"{key}: {value:.2e} {unit}")
                
                @staticmethod
                def log_total_time(): pass
                
                @staticmethod
                def log_visual_break(char='='): print(char * 80)

            return PrintWrapper()

    @contextmanager
    def change_dir(self, new_dir):
        """Context manager for changing the current working directory."""
        prev_dir = os.getcwd()
        os.chdir(new_dir)
        try:
            yield
        finally:
            os.chdir(prev_dir)