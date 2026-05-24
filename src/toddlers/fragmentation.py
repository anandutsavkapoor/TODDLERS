from .constants import *
from .utils import func_tanh_fragmentation_cooling_up, func_tanh_fragmentation_cooling_down

class Fragmentation:
    """
    Represents the fragmentation phase of shell evolution.

    This phase occurs when the shell becomes unstable and begins to break apart,
    typically due to cooling or gravitational instabilities.

    Attributes:
        evolution (Evolution): The parent Evolution object.
    """

    def __init__(self, evolution):
        """
        Initialize the Fragmentation object.

        Args:
            evolution (Evolution): The parent Evolution object.
        """
        self.evolution = evolution

    def solve(self):
        """
        Solve the evolution equations for the Fragmentation phase.

        This method calculates the rates of change for shell radius,
        velocity, energy, mass, and cloud average density. Also checks for
        transition to the second phase.

        Returns:
            list: Rates of change for [R, V, E, M, n_cloud_avg].
        """
        dRdt = self.evolution.state.V_sh
        dMdt = self.evolution.get_mass_change_rate()
        dEdt = self.evolution.get_energy_change_rate()
        dVdt = self.evolution.get_acceleration()
        dn_cloud_avg_dt = self.evolution.calculate_dn_cloud_avg_dt()

        return [dRdt, dVdt, dEdt, dMdt, dn_cloud_avg_dt]

    def get_cooling_rate_fragmentation(self, use_tanh=True):
        """
        Calculate the cooling rate during the fragmentation phase.
        To avoid a discontinuity, it uses a smoothing function to
        ease into to the max cooling and lowers it down smoothly.
        Returns:
            float: Cooling rate in erg/s.
        """
        base_cooling_rate = self.evolution.E_b_init_frag / self.evolution.t_sound_fragmentation
        if use_tanh:    
            t_since_frag_start = self.evolution.state.t - self.evolution.t_fragmentation
            multiplier  = func_tanh_fragmentation_cooling_up(t_since_frag_start, self.evolution.t_sound_fragmentation)
        else:
            multiplier  = 1 
        return base_cooling_rate * multiplier

    def phase2_start(self):
        """
        Initialize parameters for the start of the second phase.
        """
        self.evolution.t_end_frag = self.evolution.state.t
        self.evolution.dEdt_end_frag = self.evolution.get_energy_change_rate()
        self.evolution.logger.log_fragmentation_end(self.evolution.t_end_frag, self.evolution.state.E_b)