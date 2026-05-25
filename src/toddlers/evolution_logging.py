from .imports import multiprocessing, logging, os, datetime, uuid, time
from .constants import *
from .utils import add_banner_to_log, format_row, format_value

class NullEvolutionLogger:
    """A no-op stand-in for EvolutionLogger.

    Used when ``skip_logger_init=True`` (e.g. for large batches of parallel
    Evolution instances where opening a log file per run is undesirable). Every
    attribute access returns a callable that accepts any arguments and does
    nothing, so the unconditional ``self.logger.*`` calls in Evolution are safe.
    """
    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None
        return _noop

class EvolutionLogger:
    def __init__(self, base_log_path):

        # Create a unique log filename based on the simulation parameters and current timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_id = uuid.uuid4().hex[:8]  # Use first 8 characters of a unique id

        # Modify the log file name to include UID and timestamp
        log_dir = os.path.dirname(base_log_path)
        log_filename = os.path.basename(base_log_path)
        log_name, log_ext = os.path.splitext(log_filename)
        self.log_filepath = os.path.join(log_dir, f"{log_name}_uid{unique_id}_ts{timestamp}{log_ext}")
        os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)

        # Create a multiprocessing lock
        self.log_lock = multiprocessing.Lock()

        # Set up the logger
        self.logger = logging.getLogger(f"{__name__}.{os.getpid()}")
        self.logger.setLevel(logging.DEBUG)

        # Create a file with line buffering
        self.log_file = open(self.log_filepath, 'a', buffering=1)

        # Create a StreamHandler that writes to our line-buffered file
        file_handler = logging.StreamHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)

        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(file_handler)

        # Log the start of the simulation
        add_banner_to_log(self.log_filepath)
        self.logger.info("TODDLERS Evolution simulation started")    
        self.log_solver_info()    
        self.log_fudge_factors()
        
        self.log_intervals = {
            'phase1': 0.1 * MYR_TO_SEC, 
            'fragmentation': 0.05 * MYR_TO_SEC, 
            'phase2': 0.1 * MYR_TO_SEC,
            'dissolution': 1000 * MYR_TO_SEC
        }
        self.last_log_time = 0
        self.start_time = None

        # Flush the initial logs
        self.flush()

    def log_solver_info(self):
        """Log information about the numerical solvers being used for different phases."""
        platform_name = platform.system()
        self.logger.info("Numerical Solver Configuration:")
        self.logger.info(f"Platform detected: {platform_name}")
        self.logger.info(f"Phase 1 solver: {SOLVER_PH1}")
        self.logger.info(f"Fragmentation solver: {SOLVER_FRAG}")
        self.logger.info(f"Phase 2 solver: {SOLVER_PH2}")
        if platform_name.lower() == 'darwin':
            self.logger.info("Note: Using BDF solver for all phases on MacOS to avoid semaphore leaks")
        self.logger.info("Solver tolerances:")
        self.logger.info(f"  Phase 1 rtol: {RTOL_MAIN_PH1}")
        self.logger.info(f"  Fragmentation rtol: {RTOL_MAIN_FRAG}")
        self.logger.info(f"  Phase 2 rtol: {RTOL_MAIN_PH2}")

    def log_fudge_factors(self):
        """Log all fudge factors defined in the constants module."""
        self.logger.info("Fudge Factors:")
        self.logger.info(f"LYA_FF (Lyman-alpha fudge factor): {LYA_FF}")
        self.logger.info(f"Kinetic energy efficiency factor (From Freyer+2003, if dynamic density is True): {ETA_KE}")

    def should_log(self, current_time, phase):
        time_since_last_log = current_time - self.last_log_time
        current_interval = self.log_intervals.get(phase, self.log_intervals['phase1'])
        
        if time_since_last_log >= current_interval:
            self.last_log_time = current_time
            return True
        return False

    def log_profile_info(self, profile):
        """Log information about the density profile."""
        self.logger.info(f"Density Profile Type: {type(profile).__name__}")
        self.logger.info(f"Cloud Mass: {profile.M_cl/M_SUN:.2e} M_sun")
        self.logger.info(f"Average Density: {profile.n_avg:.2e} cm^-3")
        self.logger.info(f"Cloud Radius: {profile.R_cl/PC_TO_CM:.2f} pc")
        
        # Log profile-specific parameters
        if hasattr(profile, 'alpha'):
            self.logger.info(f"Power-law index (alpha): {profile.alpha:.2f}")
        elif hasattr(profile, 'xi_max'):
            self.logger.info(f"Bonnor-Ebert sphere xi_max: {profile.xi_max:.2f}")
        
        # Log density at a few characteristic radii
        radii = [0.01, 0.1, 0.5, 1.0]  # in units of R_cl
        for r in radii:
            density = profile.density(r * profile.R_cl)
            self.logger.info(f"Density at {r:.2f}R_cl: {density:.2e} g/cm^3")
        self.flush()

    def log_initial_fragmentation(self):
        """Log a banner notice that the simulation starts in fragmentation phase."""
        notice = [
            "=" * 80,
            "SIMULATION STARTING IN FRAGMENTATION PHASE",
            "Bypassing Phase 1 and immediately beginning with fragmentation cooling",
            "=" * 80,
        ]
        self.info("\n".join(notice))

    def log_fragmentation_acceleration_time_details(self, t_sound_shell, min_acceleration_time, actual_acceleration_time, 
                                                    fac_t_sound, is_acceleration_triggered=False):
        """
        Log detailed information about acceleration time thresholds for shell fragmentation.
        
        This method creates a formatted log section showing the shell sound crossing time,
        the minimum required acceleration time, and the actual time the shell has been 
        accelerating. It also explains which threshold was used (fixed minimum or sound-crossing
        time based).
        
        Only logs details if fragmentation was actually triggered by acceleration-related
        instability (e.g., Rayleigh-Taylor).
        
        Args:
            t_sound_shell (float): Sound crossing time in the shell in seconds
            min_acceleration_time (float): Required minimum acceleration time threshold in seconds
            actual_acceleration_time (float): Actual time shell has been accelerating in seconds
            fac_t_sound (float): Factor applied to sound crossing time for dynamic threshold
            is_acceleration_triggered (bool): Whether fragmentation was triggered by acceleration
        """
        if not is_acceleration_triggered:
            return
            
        self.info("Fragmentation acceleration time details:")
        self.info(f"  -> Shell sound crossing time: {t_sound_shell/MYR_TO_SEC:.3f} Myr")
        self.info(f"  -> Required acceleration time: {min_acceleration_time/MYR_TO_SEC:.3f} Myr")
        self.info(f"  -> Actual acceleration time: {actual_acceleration_time/MYR_TO_SEC:.3f} Myr")

        # Add information about which threshold was used
        if np.isclose(min_acceleration_time, MIN_ACCEL_TIME):
            self.info(f"  -> Used fixed minimum acceleration time: {MIN_ACCEL_TIME/MYR_TO_SEC:.3f} Myr")
        elif np.isclose(min_acceleration_time, MAX_ACCEL_TIME):
            self.info(f"  -> Used maximum acceleration time cap: {MAX_ACCEL_TIME/MYR_TO_SEC:.3f} Myr")
        else:
            self.info(f"  -> Used dynamic threshold: {fac_t_sound:.1f} x t_sound = {t_sound_shell*fac_t_sound/MYR_TO_SEC:.3f} Myr")

    def log_fragmentation_start(self, t_fragmentation, E_b_init_frag, t_sound_fragmentation, fragmentation_reason):
        self.logger.info(f"Fragmentation started at t={format_value(t_fragmentation / MYR_TO_SEC, '.2f')} Myr")
        self.logger.info(f"Initial bubble energy: {format_value(E_b_init_frag, '.2e')} erg")
        self.logger.info(f"Bubble sound crossing time: {format_value(t_sound_fragmentation / MYR_TO_SEC, '.2f')} Myr")
        self.logger.info(f"Fragmentation reason: {fragmentation_reason}")
        self.flush()

    def log_fragmentation_state(self, t, t_fragmentation, E_b, E_b_init_frag, t_sound_fragmentation, cooling_rate=None):
        elapsed_time = t - t_fragmentation
        energy_ratio = E_b / E_b_init_frag
        time_ratio = elapsed_time / t_sound_fragmentation

        log_message = (
            f"Fragmentation state:\n"
            f"  Elaspsed time: {format_value(elapsed_time / MYR_TO_SEC, '.2f')} Myr\n"
            f"  Bubble energy: {format_value(E_b, '.2e')} erg\n"
            f"  Energy ratio (E_b / E_b_init): {format_value(energy_ratio, '.2e')}\n"
            f"  Time ratio (Elapsed time / sound crossing time): {format_value(time_ratio, '.2e')}"
        )

        if cooling_rate is not None:
            log_message += f"\n  Cooling rate: {format_value(cooling_rate, '.2e')} erg/s"
        self.logger.debug(log_message)
        self.flush()

    def log_fragmentation_end(self, t_end_frag, final_E_b):
        self.logger.info(f"Fragmentation ended at t={format_value(t_end_frag / MYR_TO_SEC, '.2f')} Myr")
        self.logger.info(f"Final bubble energy: {format_value(final_E_b, '.2e')} erg")
        self.flush()

    def log_state(self, state, evolution):
        time_str = f" t (absolute) = {state.t/MYR_TO_SEC:.2f} Myr, {state.phase}"
        separator = "-" * 40 + time_str + "-" * 40
        self.logger.debug("\n" + separator + "\n")

        header = "+" + "-" * 40 + "+" + "-" * 17 + "+" + "-" * 17 + "+"
        self.logger.debug(header)
        self.logger.debug(format_row("Parameter", "Value", "Unit"))
        self.logger.debug(header)

        # Basic state information (always relevant)
        self.logger.debug(format_row("Current time", f"{state.t/MYR_TO_SEC:.2f}", "Myr"))
        self.logger.debug(format_row("Radius", f"{state.R/PC_TO_CM:.2f}", "pc"))
        self.logger.debug(format_row("Velocity", f"{state.V_sh/KM_TO_CM:.2f}", "km/s"))
        self.logger.debug(format_row("Shell Mass", f"{state.M_sh/M_SUN:.2e}", "M_sun"))


        if state.phase in ['phase1', 'fragmentation']:
            self.logger.debug(format_row("Shock radius", format_value(state.Rin/PC_TO_CM, '.2f'), "pc"))
            self.logger.debug(format_row("Pressure",  format_value(state.P_b, "2e"), "dyne/cm^2"))
        if state.phase in ['phase2', 'dissolution']: 
            self.logger.debug(format_row("Covering fraction",  format_value(evolution.cover_frac, ".3f"), ""))
        
        self.logger.debug(header)

        # Energetics        
        if state.phase in ['phase1', 'fragmentation']:
            E_thermal = (state.P_b * state.get_volume()) / (GAMMA - 1)
            self.logger.debug(format_row("Thermal Energy", format_value(E_thermal, '.2e'), "erg"))

        E_kinetic = 0.5 * state.M_sh * state.V_sh**2
        E_potential = evolution.calculate_gravitational_energy()
        self.logger.debug(format_row("Kinetic Energy", format_value(E_kinetic, '.2e'), "erg"))
        self.logger.debug(format_row("Potential Energy", format_value(E_potential, '.2e'), "erg"))
        self.logger.debug(header)

        # Forces (log relevant forces based on phase)
        F_rad = evolution.get_radiation_force()
        F_grav = evolution.get_gravitational_force()
        F_ext = evolution.get_external_force()

        if state.phase in ['phase1', 'fragmentation']:
            F_th = 4 * np.pi * state.R**2 * state.P_b
            self.logger.debug(format_row("Thermal Force", format_value(F_th, '.2e'), "dyne"))
        elif state.phase in ['phase2', 'dissolution']:
            F_ram = evolution.stellar_feedback.get_ram_force(state.t)
            self.logger.debug(format_row("Ram Force", format_value(F_ram, '.2e'), "dyne"))

        self.logger.debug(format_row("Radiation Force", format_value(F_rad, '.2e'), "dyne"))
        self.logger.debug(format_row("Gravitational Force", format_value(F_grav, '.2e'), "dyne"))
        self.logger.debug(format_row("External Force", format_value(F_ext, '.2e'), "dyne"))
        self.logger.debug(header)

        # Radiation coupling parameters
        L_radiation = evolution.stellar_feedback.get_bolometric_luminosity(state.t)
        Q_i = evolution.stellar_feedback.get_ionizing_photon_rate(state.t)
        F_rad_UV_IR = state.eta_rad * L_radiation / C
        if evolution.include_lyman_alpha:
            F_rad_Lya = evolution.lyman_alpha.get_LyAlpha_radiationForce(state.columnDensity_H1, Q_i, evolution.shell_structure, state.V_sh)
        else:
            F_rad_Lya = 0.0

        F_rad_ratio = F_rad_Lya / F_rad_UV_IR if F_rad_UV_IR != 0 else float('inf')

        self.logger.debug(format_row("Radiation coupling eta", format_value(state.eta_rad, '.2e'), ""))
        self.logger.debug(format_row("Direct rad. force (UV+IR)", format_value(F_rad_UV_IR, '.2e'), "dyne"))
        self.logger.debug(format_row("Lyman-alpha force", format_value(F_rad_Lya, '.2e'), "dyne"))
        self.logger.debug(format_row("Lya/Direct force ratio", format_value(F_rad_ratio, '.2e'), ""))
        self.logger.debug(format_row("Lya force multiplier", format_value(evolution.lyman_alpha.force_multiplier, '.2e'), ""))
        self.logger.debug(format_row("L_Lya w-w/o dust ratio", format_value(evolution.lyman_alpha.Lyalpha_Luminosity_ratio, '.2e'), ""))
        self.logger.debug(header)

        # Phase-specific information
        if state.phase == 'phase1':  # Only for phase1, not fragmentation
            # Calculate sound crossing time using shell thickness and weighted sound speed
            t_sound = evolution.state.shell_thickness / evolution.c_s_shell_weighted
            
            # Log sound speed components if shell has both ionized and neutral parts
            if evolution.shell_structure.shell_strctre_n is not None:
                r_ionized = evolution.shell_structure.shell_strctre_i.t
                ionized_thickness = r_ionized[-1] - r_ionized[0]
                r_neutral = evolution.shell_structure.shell_strctre_n.t
                neutral_thickness = r_neutral[-1] - r_neutral[0]
                total_thickness = ionized_thickness + neutral_thickness
                
                w_ion = ionized_thickness / total_thickness
                w_neutral = neutral_thickness / total_thickness
                
                self.logger.debug(format_row("Ionized thickness frac.", format_value(w_ion, '.2f'), ""))
                self.logger.debug(format_row("Neutral thickness frac", format_value(w_neutral, '.2f'), ""))
            
            # Base RT condition
            acceleration = evolution.get_acceleration()
            RT_accelerating = acceleration > 0
            min_R_condition = evolution.state.R >= FAC_FRAG_R_CL_INIT * evolution.R_cl_init

            # Calculate acceleration time threshold with maximum cap
            sound_based_time = t_sound * FAC_T_SOUND
            acceleration_time_threshold = min(MAX_ACCEL_TIME, max(MIN_ACCEL_TIME, sound_based_time))
            RT_unstable = evolution.state.dt_cum_posAcc > acceleration_time_threshold

            self.logger.debug(format_row("Sound cross. time (shell)", format_value(t_sound/MYR_TO_SEC, '.2e'), "Myr"))
            self.logger.debug(format_row("Shell accelerating", "True" if RT_accelerating else "False", ""))
            self.logger.debug(format_row("Min R condition", "True" if min_R_condition else "False", ""))
            self.logger.debug(format_row("Acceleration time", format_value(evolution.state.dt_cum_posAcc/MYR_TO_SEC, '.2f'), "Myr"))
            self.logger.debug(format_row("Accel. time threshold", format_value(acceleration_time_threshold/MYR_TO_SEC, '.2f'), "Myr"))

            if RT_accelerating:
                # Beta condition if enabled
                if USE_BETA_CONDITION:
                    beta_condition = evolution.state.beta <= 0
                    beta_stable = evolution.state.dt_cum_beta > acceleration_time_threshold
                    RT_unstable = RT_unstable and beta_stable
                    self.logger.debug(format_row("  Beta condn.", "True" if beta_condition else "False", ""))
                    self.logger.debug(format_row("  Beta condn. dt_cum", format_value(evolution.state.dt_cum_beta/MYR_TO_SEC, '.2f'), "Myr"))
                    self.logger.debug(format_row("  Beta", format_value(state.beta, '.2f'), ""))

                # Density contrast if enabled
                if USE_DENSITY_CONTRAST:
                    n_external = evolution.density_profile.density(evolution.state.R) / MU_N
                    density_contrast = evolution.state.n_shell_max / n_external
                    density_condition = evolution.state.dt_cum_density > acceleration_time_threshold
                    RT_unstable = RT_unstable and density_condition
                    self.logger.debug(format_row("  Density contrast condn.", "True" if density_condition else "False", ""))
                    self.logger.debug(format_row("  Density contrast", format_value(density_contrast, '.2e'), ""))
                    self.logger.debug(format_row("  Density condn. dt_cum", format_value(evolution.state.dt_cum_density/MYR_TO_SEC, '.2f'), "Myr"))

            self.logger.debug(format_row("RT unstable (combined)", "True" if RT_unstable else "False", ""))

            # Calculate gravitational instability (relevant for all phases)
            grav_crit_value = 0.67 * ((3 * G * evolution.state.M_sh) / 
                                (4 * np.pi * evolution.state.V_sh * evolution.state.R * evolution.c_s_shell_min))
            grav_unstable = grav_crit_value > 1

            self.logger.debug(format_row("Gravitationally unstable", "True" if grav_unstable else "False", ""))
            self.logger.debug(format_row("Grav. instab. param", format_value(grav_crit_value, '.2f'), ""))
            
        cloud_swept = np.isclose(evolution.state.M_sh, evolution.M_cl, rtol=1e-2)            
        self.logger.debug(format_row("Cloud swept", "True" if cloud_swept else "False", ""))
        self.logger.debug(format_row("dt_cum_dissolution_dens", format_value(state.dt_cum_dissolution / MYR_TO_SEC, '.2f'), " Myr"))
        self.logger.debug(format_row("dt_cum_dissolution_stall", format_value(state.dt_cum_stalled / MYR_TO_SEC, '.2f'), " Myr"))
        self.logger.debug(header)

        # Shell properties (always relevant)
        if evolution.use_Tion_model:
            self.logger.debug(format_row("Shell ion T", f"{state.T_sh_ion/1e4:.2f}", "10^4 K"))
            self.logger.debug(format_row("n_shell_in (if T=1e4)", format_value(evolution.state.n_shell_in * (state.T_sh_ion / T_ION), '.2e'), "cm^-3"))
        self.logger.debug(format_row("n_shell_in", format_value(evolution.state.n_shell_in, '.2e'), "cm^-3"))
        self.logger.debug(format_row("n_shell_max", format_value(evolution.state.n_shell_max, '.2e'), "cm^-3"))
        self.logger.debug(format_row("n_shell_max_ionized", format_value(evolution.state.n_shell_max_ionized, '.2e'), "cm^-3"))
        self.logger.debug(format_row("H1 column density", format_value(evolution.state.columnDensity_H1, '.2e'), "cm^-2"))
        self.logger.debug(format_row("Escape fraction Ion.", format_value(evolution.state.f_esc_i, '.2e'), ""))
        self.logger.debug(format_row("Escape fraction UV", format_value(evolution.state.f_esc_uv, '.2e'), ""))
        self.logger.debug(header)

        if evolution.dynamic_cloud_density == True:
            self.logger.debug(format_row("n_avg_cloud", format_value(evolution.state.n_cloud_avg, '.2e'), "cm^-3"))
            self.logger.debug(format_row("f_esc_i_cloud", format_value(evolution.f_esc_i_cloud, '.2e'), ""))
            self.logger.debug(format_row("Cloud Radius", format_value(evolution.R_cl / PC_TO_CM, '.2f'), "pc"))
            self.logger.debug(header)

        # Stellar feedback properties (always relevant)
        n_gens = len(evolution.stellar_feedback.M_stellar_list)
        L_mech = evolution.stellar_feedback.get_mechanical_luminosity(state.t)
        self.logger.debug(format_row("No. of generations", n_gens, ""))
        self.logger.debug(format_row("Mechanical Luminosity", format_value(L_mech, '.2e'), "erg/s"))
        self.logger.debug(format_row("Ionizing photon rate", format_value(Q_i, '.2e'), "s^-1"))
        self.logger.debug(format_row("log(phi(H))", format_value(np.log10(Q_i / (4 * np.pi * evolution.state.R**2)), '.2f'), "log[cm^-2 s^-1]"))
        self.logger.debug(format_row("Radiation Luminosity", format_value(L_radiation, '.2e'), "erg/s"))
        
        if state.phase in ['phase2', 'dissolution']:
            F_ram = evolution.stellar_feedback.get_ram_force(state.t)
            self.logger.debug(format_row("Ram force", format_value(F_ram, '.2e'), "dyne"))
        
        self.logger.debug(header + "\n")
        self.flush()

    def info(self, message):
        self.safe_log('INFO', message)

    def debug(self, message):
        self.safe_log('DEBUG', message)

    def warning(self, message):
        self.safe_log('WARNING', message)

    def error(self, message):
        self.safe_log('ERROR', message)

    def start_timer(self):
        self.start_time = time.time()
        self.logger.info("Simulation timer started.")

    def log_total_time(self):
        if self.start_time is None:
            self.logger.warning("Cannot calculate total time: start time was not recorded.")
            return

        end_time = time.time()
        total_time = end_time - self.start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        time_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
        self.logger.info(f"Total simulation time: {time_str}")

    def safe_log(self, level, message):
        with self.log_lock:
            if level == 'DEBUG':
                self.logger.debug(message)
            elif level == 'INFO':
                self.logger.info(message)
            elif level == 'WARNING':
                self.logger.warning(message)
            elif level == 'ERROR':
                self.logger.error(message)
            self.flush()

    def flush(self):
        for handler in self.logger.handlers:
            handler.flush()

    def close(self):
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)