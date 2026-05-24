from .constants import *

class Phase2:
    """
    This phase occurs after fragmentation as the shell
    continues to evolve. This phase requires the solution of 
    mass and momentum equations of the shell. As no buildup of
    energy is assumed, the energy equation drops out.

    Attributes:
        evolution (Evolution): The parent Evolution object.
    """

    def __init__(self, evolution):
        """
        Initialize the Phase2 object.

        Args:
            evolution (Evolution): The parent Evolution object.
        """
        self.evolution = evolution

    def solve(self):
        """
        Solve the evolution equations for Phase 2.

        This method calculates the rates of change for shell radius,
        velocity, mass, average cloud density. Also checks for transition to the dissolution phase.
        dEdt = 0 for this phase.
        Returns:
            list: Rates of change for [R, V, M, n_cloud_avg].
        """
        dRdt = self.evolution.state.V_sh
        dMdt = self.evolution.get_mass_change_rate()
        dVdt = self.evolution.get_acceleration()
        dn_cloud_avg_dt = self.evolution.calculate_dn_cloud_avg_dt()
        dissolution, reason = self.check_dissolution()
        if dissolution:
            self.evolution.state.phase = 'dissolution'
            self.dissolution_start(reason)

        return [dRdt, dVdt, dMdt, dn_cloud_avg_dt]

    def check_dissolution(self):
        """
        Check if conditions for transitioning to dissolution phase are met.
        Apart from the shell being too large, too thin, or stalled, we impose
        another condition, i.e., the shell only dissolves if the cloud has been fully swept or
        the cloud has expanded to the ISM density with either sufficient mass swept up or shell stalling.

        Returns:
            tuple: (bool, str) - (True if dissolution conditions are met, reason for dissolution)
        """

        shell_dissolved = (self.evolution.check_dissolution_density_condition() and 
                        self.evolution.state.dt_cum_dissolution >= DISSOLUTION_PERIOD) 
        shell_stalled = self.evolution.state.dt_cum_stalled >= DISSOLUTION_PERIOD
        shell_too_large = self.evolution.state.R >= DISSOLUTION_RADIUS

        if self.evolution.dynamic_cloud_density:
            cloud_swept = self.is_cloud_fully_swept()
            cloud_expanded = self.has_cloud_expanded_to_ism_density()
            cloud_condition = cloud_swept or cloud_expanded
        else:
            cloud_condition = self.is_cloud_fully_swept()

        dissolution_condition = (shell_dissolved or shell_stalled or shell_too_large) and cloud_condition

        if dissolution_condition:
            if shell_dissolved:
                reason = "Low shell density post-supernova"
            elif shell_too_large:
                reason = "Shell radius exceeds dissolution limit"
            elif shell_stalled:
                reason = "Shell stalled for extended period"
            else:
                reason = "Unknown dissolution condition"

            if self.evolution.dynamic_cloud_density:
                if cloud_swept:
                    reason += ", cloud fully swept"
                elif cloud_expanded:
                    # Check which condition triggered the cloud_expanded flag
                    mass_condition = self.evolution.state.M_sh >= FRAC_M_CL * self.evolution.M_cl
                    stalled_condition = self.evolution.state.dt_cum_stalled >= DISSOLUTION_PERIOD
                    
                    reason += f", cloud expanded to ISM density (n ≤ {N_ISM} cm⁻³)"
                    
                    if mass_condition and stalled_condition:
                        reason += f" with M_sh ≥ {FRAC_M_CL:.2f}M_cl and shell stalled"
                    elif mass_condition:
                        reason += f" with M_sh ≥ {FRAC_M_CL:.2f}M_cl"
                    elif stalled_condition:
                        reason += " with shell stalled"
            else:
                reason += ", cloud fully swept"

        else:
            reason = "N/A"

        return dissolution_condition, reason

    def is_cloud_fully_swept(self):
        """
        Check if the cloud is considered fully swept, including cases where the shell mass
        slightly exceeds or is very close to the cloud mass due to numerical issues.
        """
        relative_difference = (self.evolution.state.M_sh - self.evolution.M_cl) / self.evolution.M_cl
        return relative_difference >= -1e-3  # True if M_sh >= 0.999 * M_cl

    def has_cloud_expanded_to_ism_density(self):
        """
        Check if the cloud has expanded to ISM density and either:
        1. The shell has swept up a significant fraction of the cloud mass, OR
        2. The shell has stalled for a sufficient period
        
        Used only when dynamic_cloud_density is True.
        
        Returns:
            bool: True if the cloud has expanded to ISM density and either the mass
                condition or stalling condition is met, False otherwise.
        """
        density_condition = self.evolution.state.n_cloud_avg <= N_ISM
        mass_condition = self.evolution.state.M_sh >= FRAC_M_CL * self.evolution.M_cl
        stalled_condition = self.evolution.state.dt_cum_stalled >= DISSOLUTION_PERIOD
        
        return density_condition and (mass_condition or stalled_condition)
    
    def dissolution_start(self, reason):
        """
        Initialize parameters for the start of the dissolution phase.
        """
        self.evolution.t_dissolution = self.evolution.state.t
        self.evolution.dissolution_reason = reason
        self.evolution.logger.info(f"Dissolution at t={self.evolution.state.t/MYR_TO_SEC:.2f} Myr. Reason: {reason}")