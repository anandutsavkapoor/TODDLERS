# initial_conditions.py

from .imports import np, root_scalar, fsolve
from .constants import *

class InitialConditions:
    """
    A class to calculate initial conditions for stellar feedback-driven shell evolution.

    This class handles the calculation of initial conditions for both uniform and non-uniform
    density profiles, using the Weaver et al. (1977) solution as a basis.

    Attributes:
        stellar_feedback (StellarFeedback): Object containing stellar feedback properties.
        density_profile (DensityProfile): Object representing the density profile of the cloud.
        t0 (float): Initial time in seconds.
        L_38 (float): Mechanical luminosity in units of 10^38 erg/s.
        dt_7 (float): Initial period in units of 10^7 years.
    """

    def __init__(self, stellar_feedback, density_profile, t0, dt):
        """
        Initialize the InitialConditions object.

        Args:
            stellar_feedback (StellarFeedback): Object containing stellar feedback properties.
            density_profile (DensityProfile): Object representing the density profile of the cloud.
            t0 (float): Initial time in seconds.
            dt (float): initial period of w77 in seconds
        """
        self.stellar_feedback = stellar_feedback
        self.density_profile = density_profile
        self.profile_type = self.density_profile.get_parameters()["type"]
        self.t0 = t0
        self.L_38 = self.stellar_feedback.get_mechanical_luminosity(t0) / 1e38 
        self.dt = dt
        self.dt_7 = self.dt / (10 * MYR_TO_SEC)

    def calculate(self):
        """
        Calculate initial conditions based on the density profile type.

        Returns:
            tuple: (R0, V0, E0, M_sh, Rin0) where:
                R0 (float): Initial radius in cm.
                V0 (float): Initial velocity in cm/s.
                E0 (float): Initial energy in erg.
                M_sh (float): Initial shell mass in g.
                Rin0 (float): Initial inner radius in cm.

        Raises:
            ValueError: If the profile type is not recognized.
        """
        try:
            if self.profile_type == 'uniform':
                R0, V0, E0, M_sh, Rin0 = self._calculate_uniform()
            else:
                R0, V0, E0, M_sh, Rin0 = self._calculate_non_uniform()
            
            return R0, V0, E0, M_sh, Rin0
        except Exception as e:
            raise ValueError(f"Error calculating initial conditions: {str(e)}")

    def _calculate_uniform(self):
        """Calculate initial conditions for a uniform density profile."""
        n_cl = self.density_profile.n_avg
        R0 = 267 * ((self.L_38 * (self.dt_7)**3) / n_cl)**(1/5) * PC_TO_CM 
        V0 = self._calculate_initial_velocity(n_cl)
        E0 = self._calculate_initial_energy()
        M_sh = self.density_profile.mass_enclosed(R0)
        Rin0 = self._calculate_inner_radius(R0, E0)
        return R0, V0, E0, M_sh, Rin0

    def _calculate_non_uniform(self):
        """Calculate initial conditions for a non-uniform density profile."""
        R0 = self._find_initial_radius()
        M_sh = self.density_profile.mass_enclosed(R0)
        volume_enclosed = (4/3) * np.pi * R0**3 
        n_eff = (M_sh / volume_enclosed) / MU_N
        V0 = self._calculate_initial_velocity(n_eff)
        E0 = self._calculate_initial_energy()
        Rin0 = self._calculate_inner_radius(R0, E0)
        return R0, V0, E0, M_sh, Rin0
    
    def _find_initial_radius(self):
        """Find the initial radius for non-uniform density profiles."""
        mass_enc = lambda r: self.density_profile.mass_enclosed(r)
        radius_eq = lambda r: self._weaver77_radius(r, mass_enc) - r
        try:
            result = root_scalar(radius_eq, bracket=[1e-5*PC_TO_CM, 100*PC_TO_CM], method='brentq')
            return result.root
        except ValueError as e:
            raise ValueError(f"Failed to find initial radius: {str(e)}")

    def _weaver77_radius(self, r, mass_enclosed):
        """Calculate radius based on Weaver et al. (1977) solution."""
        M = mass_enclosed(r)
        volume_enclosed = (4/3) * np.pi * r**3
        n_eff = (M / volume_enclosed) / MU_N 
        return 267 * ((self.L_38 * (self.dt_7)**3) / n_eff)**(1/5) * PC_TO_CM

    def _calculate_initial_velocity(self, n):
        """Calculate initial energy of the bubble."""
        V = 15.7 * ((self.L_38 / n)**(1/5)) * (self.dt_7)**(-2/5) * KM_TO_CM
        return V

    def _calculate_initial_energy(self):
        """Calculate initial velocity of the shell."""
        E_thermal = (5/11) * self.L_38 * 1e38 * self.dt
        return E_thermal 

    def _calculate_inner_radius(self, R0, E0):
        """Calculate the initial inner radius of the shell."""
        F_ram = self.stellar_feedback.get_ram_force(self.t0)

        def equation_inner_radius(R_in_guess):
            if F_ram <= 0 or E0 <= 0 or R0 <= R_in_guess:
                return np.inf
            return np.sqrt(max(0, (F_ram / (2 * E0)) * (R0**3 - R_in_guess**3))) - R_in_guess

        Rin_guess = R0 / 10
        try:
            Rin = fsolve(equation_inner_radius, Rin_guess, full_output=True)[0][0]
            if Rin >= R0 or Rin <= 0:
                raise ValueError(f"Invalid inner radius calculated: {Rin}")
            return Rin
        except Exception as e:
            raise ValueError(f"Failed to calculate inner radius: {str(e)}")