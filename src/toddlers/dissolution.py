from .constants import *

class Dissolution:
    """
    Represents the dissolution phase of shell evolution.

    This phase occurs when the shell becomes too diffuse or meets other
    dissolution criteria, leading to its dissipation into the ambient medium.
    This is a trivial class in its present form.

    Attributes:
        evolution (Evolution): The parent Evolution object.
    """

    def __init__(self, evolution):
        """
        Initialize the Dissolution object.

        Args:
            evolution (Evolution): The parent Evolution object.
        """
        self.evolution = evolution

    def solve(self):
        """
        Solve the evolution equations for the Dissolution phase.

        This method calculates the rates of change for shell radius,
        velocity, mass, and average cloud density during the dissolution process. The dEdt
        is assumed to be zero in this phase. dn_cloud_avg_dt is obtained from
        calculate_dn_cloud_avg_dt() (nonzero only when the dynamic cloud-density model is active).

        Returns:
            list: Rates of change for [R, V, M, n_cloud_avg].
        """
        dRdt = self.evolution.state.V_sh
        dMdt = self.evolution.get_mass_change_rate()
        dVdt = self.evolution.get_acceleration()
        dn_cloud_avg_dt = self.evolution.calculate_dn_cloud_avg_dt()

        return [dRdt, dVdt, dMdt, dn_cloud_avg_dt]
