from .imports import np, brentq, newton, Dict, List, root_scalar, warnings
from .cloud_density_profiles import CloudDensityProfile, UniformDensityWithCavity
from .constants import *

class CloudEnergyInjection:
    """
    A class to handle energy injection into molecular clouds.
    
    This class manages the evolution of a molecular cloud's properties when energy is injected into it.
    Currently only supports clouds with uniform density profiles containing a central cavity.
    
    Attributes:
        initial_cloud (CloudDensityProfile): The initial state of the cloud
        current_cloud (CloudDensityProfile): The current state of the cloud after energy injections
        U_init (float): Initial binding energy of the cloud in ergs
        energy_history (List[Dict[str, float]]): History of energy injections and resulting binding energies
        energy_threshold (float): Minimum relative energy injection to consider, as fraction of binding energy
    """

    def __init__(self, initial_cloud: CloudDensityProfile):
        """
        Initialize the CloudEnergyInjection instance.

        Args:
            initial_cloud (CloudDensityProfile): Initial cloud density profile.
                Currently only UniformDensityWithCavity profiles are supported.

        Raises:
            TypeError: If initial_cloud is not a supported CloudDensityProfile type.
        """
        if not isinstance(initial_cloud, UniformDensityWithCavity):
            raise TypeError("Currently only UniformDensityWithCavity profiles are supported")
            
        self.initial_cloud = initial_cloud
        self.current_cloud = initial_cloud
        self.U_init = initial_cloud.binding_energy()
        self.energy_history = []
        self.energy_threshold = 1e-10  # Threshold for negligible energy injection

    def inject_energy(self, E_inj: float) -> None:
        """
        Inject energy into the cloud and calculate its new properties.

        This method updates the cloud's properties (radius, density) based on the injected energy.
        The cloud expands while maintaining its mass and cavity radius.

        Args:
            E_inj (float): Energy to inject in ergs. Must be less than 99% of the binding energy.

        Raises:
            ValueError: If injection energy exceeds 99% of binding energy or if resulting radius would decrease.
            RuntimeWarning: If injected energy is below the threshold.
        """
        if abs(E_inj) > 0.99 * abs(self.U_init):
            raise ValueError("Injection energy exceeds 99% of initial binding energy.")
        
        if abs(E_inj) < abs(self.U_init) * self.energy_threshold:
            warnings.warn(f"Injected energy ({E_inj:.2e} erg) is below the threshold. No changes made to the system.", RuntimeWarning)
            self.energy_history.append({"E_inj": E_inj, "U_final": self.U_init})
            return
        
        U_fin = self.U_init + E_inj
        new_radius = self._calculate_new_radius(U_fin)
        
        if new_radius <= self.current_cloud.R_cl:
            raise ValueError("Energy injection should increase radius (decrease density).")
        
        self.current_cloud = self._create_new_cloud_profile(new_radius)
        self.energy_history.append({"E_inj": E_inj, "U_final": U_fin})

    def _calculate_new_radius(self, U_fin: float) -> float:
        """
        Calculate the new cloud radius after energy injection.

        Args:
            U_fin (float): Final binding energy in ergs

        Returns:
            float: New cloud radius in cm

        Raises:
            NotImplementedError: If cloud profile is not UniformDensityWithCavity
        """
        if isinstance(self.current_cloud, UniformDensityWithCavity):
            return self._calculate_new_radius_uniform_with_cavity(U_fin)
        else:
            raise NotImplementedError(f"Radius calculation not implemented for {type(self.current_cloud)}")

    def _calculate_new_radius_uniform_with_cavity(self, U_fin: float) -> float:
        """
        Calculate new radius for a uniform density cloud with cavity.

        Uses root finding to determine the radius that gives the desired final binding energy.

        Args:
            U_fin (float): Target final binding energy in ergs

        Returns:
            float: New cloud radius in cm

        Raises:
            ValueError: If root finding fails to converge
        """
        M_gas = self.current_cloud.M_gas
        R_cav = self.current_cloud.R_cav

        def equation(R_cl):
            V_gas = (4/3) * np.pi * (R_cl**3 - R_cav**3)
            n_avg = M_gas / (V_gas * MU_N)
            cloud = UniformDensityWithCavity(M_gas, n_avg, R_cav)
            return cloud.binding_energy() - U_fin

        R_cl_old = self.current_cloud.R_cl
        sol = root_scalar(equation, bracket=[R_cl_old, R_cl_old*100], method='brentq')
        
        if not sol.converged:
            raise ValueError("Failed to converge on a new radius")

        return sol.root

    def _create_new_cloud_profile(self, new_R_cl: float) -> CloudDensityProfile:
        """
        Create a new cloud profile with the calculated radius.

        Args:
            new_R_cl (float): New cloud radius in cm

        Returns:
            CloudDensityProfile: New cloud profile with updated properties

        Raises:
            NotImplementedError: If cloud profile is not UniformDensityWithCavity
        """
        if isinstance(self.current_cloud, UniformDensityWithCavity):
            M_gas = self.current_cloud.M_gas
            R_cav = self.current_cloud.R_cav
            V_gas = (4/3) * np.pi * (new_R_cl**3 - R_cav**3)
            new_n_avg = M_gas / (V_gas * MU_N)
            return UniformDensityWithCavity(M_gas, new_n_avg, R_cav)
        else:
            raise NotImplementedError(f"Profile creation not implemented for {type(self.current_cloud)}")

    def get_current_cloud(self) -> CloudDensityProfile:
        """
        Get the current state of the cloud.

        Returns:
            CloudDensityProfile: Current cloud profile
        """
        return self.current_cloud

    def get_energy_history(self) -> List[Dict[str, float]]:
        """
        Get the history of energy injections and resulting binding energies.

        Returns:
            List[Dict[str, float]]: List of dictionaries containing:
                - 'E_inj': Injected energy in ergs
                - 'U_final': Final binding energy in ergs
        """
        return self.energy_history

    def verify_binding_energy(self, tolerance=0.01):
        """
        Verify the binding energy calculation using numerical integration.

        Compares analytical and numerical calculations of binding energy.

        Args:
            tolerance (float, optional): Maximum allowed relative error. Defaults to 0.01.

        Returns:
            tuple: (analytical_energy, numerical_energy, relative_error)

        Raises:
            ValueError: If relative error exceeds tolerance
        """
        U_analytical = self.current_cloud.binding_energy()
        print("U_anal: ", U_analytical)
        U_numerical = self.current_cloud._numerical_binding_energy()
        print("U_num: ", U_numerical)
        relative_error = abs((U_analytical - U_numerical) / U_analytical)
        if relative_error > tolerance:
            raise ValueError(f"Binding energy calculation may be inaccurate. Relative error: {relative_error}")
        return U_analytical, U_numerical, relative_error