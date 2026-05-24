from .imports import np
from .constants import *
from .exceptions import ManualEventTermination

class Phase1:
    """
    Represents the first phase of shell evolution.

    This phase represents the initial expansion of the shell.
    During this phase mass, momentum and energy equations are 
    solved simulataneously.

    Attributes:
        evolution (Evolution): The parent Evolution object.
    """

    def __init__(self, evolution):
        """
        Initialize the Phase1 object.

        Args:
            evolution (Evolution): The parent Evolution object.
        """
        self.evolution = evolution

    def solve(self):
        """
        Solve the evolution equations for Phase 1.

        This method calculates the rates of change for shell radius,
        velocity, energy, mass, and average cloud density. Also checks for 
        transition to the fragmentation phase.

        Returns:
            list: Rates of change for [R, V, E, M, n_cloud_avg].
        """
        dRdt = self.evolution.state.V_sh
        dMdt = self.evolution.get_mass_change_rate()
        dEdt = self.evolution.get_energy_change_rate()
        dVdt = self.evolution.get_acceleration()
        dn_cloud_avg_dt = self.evolution.calculate_dn_cloud_avg_dt()

        # Calculate alpha and beta using energy instead of pressure
        self.evolution.state.alpha = (self.evolution.state.t / self.evolution.state.R) * (self.evolution.state.V_sh)
        self.evolution.state.beta = -(self.evolution.state.t / self.evolution.state.P_b) * self.evolution.get_pressure_change_rate()

        if self.check_fragmentation_condition():
            self.evolution.state.phase = 'fragmentation'
            self.fragmentation_start()
            raise ManualEventTermination(f"Fragmentation conditions met at t={self.evolution.state.t/MYR_TO_SEC:.6f} Myr", 
                                         self.evolution.state.t, 
                                         [self.evolution.state.R, self.evolution.state.V_sh, self.evolution.state.E_b, 
                                          self.evolution.state.M_sh, self.evolution.state.n_cloud_avg])


        return [dRdt, dVdt, dEdt, dMdt, dn_cloud_avg_dt]

    def check_fragmentation_condition(self, fixed_acc_time=False):
        """
        Check if conditions for transitioning to fragmentation phase are met.
        
        Uses density contrast criterion for RT instability: the shell fragments when
        its density exceeds the external density by a sufficient factor and has been
        accelerating for long enough. Time conditions are evaluated against the maximum
        of MIN_ACCEL_TIME and local sound crossing time calculated using a thickness-
        weighted sound speed that accounts for both ionized and neutral shell components.
        
        Args:
            fixed_acc_time (bool, optional): If True, use only MIN_ACCEL_TIME for acceleration
                                        time threshold instead of the dynamic calculation.
                                        Defaults to False.
        
        Returns:
            bool: True if fragmentation conditions are met, False otherwise.
        """
        # Basic conditions that always apply
        min_R_condition = self.evolution.state.R >= FAC_FRAG_R_CL_INIT * self.evolution.R_cl_init
        
        # Calculate sound crossing time using shell thickness and weighted sound speed
        t_sound = self.evolution.state.shell_thickness / self.evolution.c_s_shell_weighted
        
        # Use fixed time or dynamic time based on parameter
        if fixed_acc_time:
            min_acceleration_time = MIN_ACCEL_TIME
        else:
            # Default behavior: use maximum of MIN_ACCEL_TIME and t_sound * FAC_T_SOUND
            # but capped by MAX_ACCEL_TIME
            sound_based_time = t_sound * FAC_T_SOUND
            min_acceleration_time = min(MAX_ACCEL_TIME, max(MIN_ACCEL_TIME, sound_based_time))
        
        # Initialize RT instability flag and message
        RT_unstable = self.evolution.state.dt_cum_posAcc > min_acceleration_time
        rt_message_parts = []

        # Check density contrast if enabled
        if USE_DENSITY_CONTRAST:
            n_external = self.evolution.density_profile.density(self.evolution.state.R) / MU_N
            density_contrast = self.evolution.state.n_shell_max / n_external
            density_condition = self.evolution.state.dt_cum_density > min_acceleration_time
            RT_unstable = RT_unstable and density_condition
            rt_message_parts.append(f"density contrast = {density_contrast:.1f}, time = {self.evolution.state.dt_cum_density/t_sound:.1f}t_sound")

        # Check beta condition if enabled
        if USE_BETA_CONDITION:
            beta_condition = self.evolution.state.dt_cum_beta > min_acceleration_time
            RT_unstable = RT_unstable and beta_condition
            rt_message_parts.append(f"beta = {self.evolution.state.beta:.2f}, time = {self.evolution.state.dt_cum_beta/t_sound:.1f}t_sound")

        # Build RT message
        rt_message = "Rayleigh-Taylor instability"
        if rt_message_parts:
            rt_message += f" ({', '.join(rt_message_parts)})"
        
        # Add information about the acceleration time method used
        if fixed_acc_time:
            rt_message += f" [fixed acc time: {MIN_ACCEL_TIME/MYR_TO_SEC:.3f} Myr]"

        # Check other instabilities
        grav_unstable = self.check_gravitational_instability()
        cloud_swept = np.isclose(self.evolution.state.M_sh, self.evolution.M_cl, rtol=1e-2)
        
        if RT_unstable and min_R_condition:
            self.fragmentation_reason = rt_message
            return True
        elif grav_unstable and min_R_condition:
            self.fragmentation_reason = "Gravitational instability"
            return True
        elif cloud_swept and min_R_condition:
            self.fragmentation_reason = "Cloud fully swept up"
            return True
        
        return False

    def check_gravitational_instability(self):
        """
        Check if the shell is gravitationally unstable.

        Returns:
            bool: True if the shell is gravitationally unstable, False otherwise.
        """
        grav_crit_value = 0.67 * ((3 * G * self.evolution.state.M_sh) / 
                                  (4 * np.pi * self.evolution.state.V_sh * self.evolution.state.R * self.evolution.c_s_shell_min))
        return grav_crit_value > 1

    def fragmentation_start(self):
        """
        Initialize parameters for the start of the fragmentation phase.
        """
        self.evolution.E_b_init_frag = self.evolution.state.E_b
        self.evolution.t_fragmentation = self.evolution.state.t
        self.evolution.c_sound_last = np.sqrt((GAMMA * K_BOLTZMANN * MEAN_T_BUBBBLE) / MU_P)
        self.evolution.t_sound_fragmentation = self.evolution.state.R / self.evolution.c_sound_last
        self.fragmentation_reason = getattr(self, 'fragmentation_reason', "Initialized with fragmentation")
        
        self.evolution.logger.log_fragmentation_start(
            self.evolution.t_fragmentation,
            self.evolution.E_b_init_frag,
            self.evolution.t_sound_fragmentation,
            self.fragmentation_reason
        )

        # Additional detailed logging of time thresholds
        t_sound_shell = self.evolution.state.shell_thickness / self.evolution.c_s_shell_weighted
        
        # Determine if fixed acceleration time was used based on the fragmentation reason
        fixed_acc_time = "[fixed acc time:" in self.fragmentation_reason
        
        # Calculate the acceleration time threshold that was used
        if fixed_acc_time:
            min_acceleration_time = MIN_ACCEL_TIME
        else:
            sound_based_time = t_sound_shell * FAC_T_SOUND
            min_acceleration_time = min(MAX_ACCEL_TIME, max(MIN_ACCEL_TIME, sound_based_time))
            
        is_acceleration_triggered = "Rayleigh-Taylor instability" in self.fragmentation_reason
        
        self.evolution.logger.log_fragmentation_acceleration_time_details(
            t_sound_shell,
            min_acceleration_time, 
            self.evolution.state.dt_cum_posAcc,
            FAC_T_SOUND,
            is_acceleration_triggered
        )