from .imports import np, quad, warnings, IntegrationWarning
from .constants import *

class WithCavity:
    """
    A modifier class that adds a cavity to any CloudDensityProfile.
    This class wraps an existing profile and modifies its density and mass calculations
    to account for a central cavity.
    """
    def __init__(self, base_profile, R_cav):
        """
        Initialize the cavity-modified profile.
        
        Args:
            base_profile: Any CloudDensityProfile instance
            R_cav (float): Cavity radius in cm
        """
        self.base_profile = base_profile
        self.R_cav = R_cav
        
        # Verify cavity radius is valid
        if R_cav >= base_profile.R_cl:
            raise ValueError(f"Cavity radius ({R_cav}) must be smaller than cloud radius ({base_profile.R_cl})")
        
        # Calculate cavity mass for later use
        self.M_cavity = self._calculate_cavity_mass()
        
        # Update total mass
        self.M_cl = base_profile.M_cl - self.M_cavity
        
        # Inherit other properties from base profile
        self.R_cl = base_profile.R_cl
        self.n_avg = base_profile.n_avg
        self.rho_avg = base_profile.rho_avg
        
    def _calculate_cavity_mass(self):
        """Calculate the mass that would have been inside the cavity."""
        return self.base_profile.mass_enclosed(self.R_cav)
    
    def density(self, r):
        """Modified density function that accounts for the cavity."""
        if r <= self.R_cav:
            return 0.0
        return self.base_profile.density(r)
    
    def mass_enclosed(self, r):
        """Calculate mass enclosed within radius r, accounting for the cavity."""
        if r <= self.R_cav:
            return 0.0
        return max(0.0, self.base_profile.mass_enclosed(r) - self.M_cavity)
    
    def _numerical_binding_energy(self):
        """
        Calculate binding energy numerically, accounting for the cavity.
        Uses the same method as CloudDensityProfile but integrates from R_cav.
        """
        def integrand(r):
            return (4 * np.pi * r**2 * self.density(r) * self.mass_enclosed(r)) / r
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", IntegrationWarning)
                U, abserr = quad(integrand, self.R_cav, self.R_cl, 
                               epsabs=1e-10, epsrel=1e-9)
            
            if not np.isfinite(U):
                raise ValueError("Integration result is not finite")
            
            binding_energy = -G * U
            
            if binding_energy >= 0:
                raise ValueError("Binding energy calculation resulted in a non-negative value")
            
            return binding_energy
            
        except (IntegrationWarning, ValueError) as e:
            raise ValueError(f"Binding energy calculation failed: {str(e)}")
    
    def binding_energy(self):
        """Calculate total binding energy of the cavity-modified profile."""
        return self._numerical_binding_energy()
    
    def get_parameters(self):
        """Get parameters of the cavity-modified profile."""
        params = self.base_profile.get_parameters()
        params.update({
            "cavity_radius": self.R_cav,
            "cavity_mass": self.M_cavity,
            "modified_mass": self.M_cl
        })
        return params