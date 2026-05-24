from .imports import np, brentq, newton, Dict, List, warnings
from .add_cavity_to_density_profiles import WithCavity
from .cloud_density_profiles import CloudDensityProfile, create_density_profile
from .constants import *

class GeneralizedCloudEnergyInjection:
    def __init__(self, initial_cloud):
        """
        Initialize energy injection handler for any cloud profile, with or without cavity.
        
        Args:
            initial_cloud: Either a CloudDensityProfile instance or a WithCavity instance
        """
        self.initial_cloud = initial_cloud
        self.current_cloud = initial_cloud
        self.U_init = initial_cloud.binding_energy()
        self.energy_history = []
        self.energy_threshold = 1e-10
        
        # Determine if this is a cavity-modified profile
        self.has_cavity = hasattr(initial_cloud, 'base_profile')
        if self.has_cavity:
            self.base_profile_type = type(initial_cloud.base_profile)
            self.R_cav = initial_cloud.R_cav
        else:
            self.base_profile_type = type(initial_cloud)

    def inject_energy(self, E_inj: float) -> None:
        """
        Inject energy into the cloud, causing expansion while maintaining profile shape.
        
        Args:
            E_inj (float): Energy to inject in ergs
        """
        if abs(E_inj) > 0.99 * abs(self.U_init):
            raise ValueError("Injection energy exceeds 99% of initial binding energy.")
        
        # Check if the injected energy is negligible
        if abs(E_inj) < abs(self.U_init) * self.energy_threshold:
            warnings.warn(f"Injected energy ({E_inj:.2e} erg) is below threshold. No changes made.", 
                        RuntimeWarning)
            self.energy_history.append({"E_inj": E_inj, "U_final": self.U_init})
            return
        
        U_fin = self.U_init + E_inj
        new_radius = self._calculate_new_radius(U_fin)
        
        # Check that density has decreased (radius increased)
        if new_radius <= self.current_cloud.R_cl:
            raise ValueError("Energy injection should increase radius (decrease density).")
        
        self.current_cloud = self._create_new_cloud_profile(new_radius)
        self.energy_history.append({"E_inj": E_inj, "U_final": U_fin})

    def _calculate_new_radius(self, U_fin: float) -> float:
        """
        Calculate the new cloud radius after energy injection.
        Uses numerical root finding to solve for radius that gives desired binding energy.
        
        Args:
            U_fin (float): Target final binding energy
            
        Returns:
            float: New cloud radius in cm
        """
        def binding_energy_difference(R):
            temp_cloud = self._create_new_cloud_profile(R)
            return temp_cloud.binding_energy() - U_fin

        R_min = self.current_cloud.R_cl
        R_max = self.current_cloud.R_cl * 100

        # Try multiple root-finding methods for robustness
        try:
            result = brentq(binding_energy_difference, R_min, R_max)
        except ValueError:
            try:
                result = newton(binding_energy_difference, self.current_cloud.R_cl * 1.5)
            except RuntimeError:
                raise ValueError("Failed to find new radius numerically.")

        # Verify the result
        if not R_min < result < R_max:
            raise ValueError("Calculated radius out of expected range.")

        return result

    def _create_new_cloud_profile(self, new_R_cl: float) -> CloudDensityProfile:
        """
        Create a new cloud profile with the expanded radius.
        Maintains the same profile type and cavity if present.
        
        Args:
            new_R_cl (float): New cloud radius in cm
            
        Returns:
            CloudDensityProfile or WithCavity: New expanded cloud profile
        """
        # Get current parameters
        old_params = self.current_cloud.get_parameters()
        
        # Calculate new average density based on new radius
        if self.has_cavity:
            V_new = (4/3) * np.pi * (new_R_cl**3 - self.R_cav**3)
            new_n_avg = (old_params["modified_mass"] / (V_new * MU_N))
        else:
            new_n_avg = (old_params["M_cl"] / ((4/3) * np.pi * new_R_cl**3)) / MU_N

        # Create base profile with new parameters
        if self.has_cavity:
            base_profile = create_density_profile(
                old_params["type"].lower(),
                old_params["modified_mass"],
                new_n_avg,
                **{k: v for k, v in old_params.items() 
                   if k not in ["type", "M_cl", "n_avg", "R_cl", "cavity_radius", 
                              "cavity_mass", "modified_mass"]}
            )
            # Wrap with cavity
            return WithCavity(base_profile, self.R_cav)
        else:
            return create_density_profile(
                old_params["type"].lower(),
                old_params["M_cl"],
                new_n_avg,
                **{k: v for k, v in old_params.items() 
                   if k not in ["type", "M_cl", "n_avg", "R_cl"]}
            )

    def get_current_cloud(self):
        """Get the current state of the cloud."""
        return self.current_cloud

    def get_energy_history(self) -> List[Dict[str, float]]:
        """Get the history of energy injections and resulting binding energies."""
        return self.energy_history

    def verify_binding_energy(self, tolerance=0.01):
        """
        Verify the binding energy calculation using numerical integration.
        
        Args:
            tolerance (float): Maximum allowed relative error
            
        Returns:
            tuple: (analytical binding energy, numerical binding energy, relative error)
        """
        U_analytical = self.current_cloud.binding_energy()
        U_numerical = self.current_cloud._numerical_binding_energy()
        relative_error = abs((U_analytical - U_numerical) / U_analytical)
        
        if relative_error > tolerance:
            raise ValueError(f"Binding energy calculation may be inaccurate. Relative error: {relative_error}")
            
        return U_analytical, U_numerical, relative_error