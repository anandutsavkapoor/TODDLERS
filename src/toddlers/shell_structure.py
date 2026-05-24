from .imports import np, solve_ivp
from .constants import *
from .utils import get_fmol_Krum2013

class ShellStructure:
    """
    A class to model the hydrostatic structure of a shell around a 
    young stellar population. See Draine 2011 for more details.
    
    This class solves for the density, ionization, and dust optical depth structure 
    of a shell that is driven by radiation pressure from a central stellar population (SP).
    The shell can have both an ionized and a neutral component.

    Physical Model:
    ---------------
    The model assumes quasi-hydrostatic equilibrium where the thermal pressure gradient
    force balances the force due to radiation pressure gradient. This simplified treatment,
    which neglects turbulent and magnetic pressures, determines the shell's density and 
    ionization structure.

    Attributes:
        Z (float): Metallicity
        n_shell_in (float): Inner edge (closest to the SP) shell number density in cm^-3
        M_sh (float): Total shell mass in g
        T_ion (float): Ionized gas temperature in K, can be varied based on photon flux, Z, and density.
        stellar_feedback: Object handling stellar population properties and feedback
        age (float): Age of the stellar population in seconds
        cover_frac (float): Covering fraction of the shell (0-1)
        sigma_dust (float): Dust cross section per H nucleon
        kappa_IR (float): IR opacity in cm^2/g
        Q_i (float): Ionizing photon rate in s^-1
        Lum_i (float): Ionizing luminosity in erg/s
        Lum_n (float): Non-ionizing luminosity in erg/s
        R (float): Current shell radius in cm
    
    """

    def __init__(self, Z, n_shell_in, M_sh, T_ion, stellar_feedback, age, cover_frac):
        """
        Initialize the ShellStructure instance.

        Args:
            Z (float): Metallicity
            n_shell_in (float): Inner shell number density in cm^-3
            M_sh (float): Total shell mass in g
            T_ion (float): Ionized gas temperature in K
            stellar_feedback: Object with methods for stellar feedback properties
            age (float): Age of the stellar population in seconds
            cover_frac (float): Covering fraction of the shell (0-1)
        """
        self.Z = Z
        self.n_shell_in = n_shell_in
        self.M_sh = M_sh
        self.T_ion = T_ion
        self.stellar_feedback = stellar_feedback
        self.age = age
        self.sigma_dust = SIGMA_DUST_SOLAR * (Z / Z_SOLAR)
        self.kappa_IR = KAPPA_IR_SOLAR * (Z / Z_SOLAR)
        self.cover_frac = cover_frac
        
        # Get feedback parameters
        self.Q_i = self.stellar_feedback.get_ionizing_photon_rate(self.age)
        self.Lum_i = self.stellar_feedback.get_ionizing_luminosity(self.age)
        self.Lum_n = self.stellar_feedback.get_bolometric_luminosity(self.age) - self.Lum_i

        # Define ODE solver tolerances
        self.rtol_i = 1e-6
        self.atol_i = [1e-8, 1e-8, 1e-8, 1e-10]  # for n_sh, phi, tau_d, m_sh respectively
        
        self.rtol_n = 1e-6
        self.atol_n = [1e-8, 1e-8, 1e-10]  # for n_sh, tau_d, m_sh respectively

        # Define mass tolerance based on ODE solver tolerance
        self.mass_tolerance = 2 * self.rtol_i
        
    def shellStructureEqns_ionized(self, r, x):
        """
        System of ODEs describing the ionized shell structure.
        
        Solves for density, ionization fraction, dust optical depth, and mass
        profiles in the ionized region.

        Args:
            r (float): Radius in cm
            x (list): Current values [n_sh, phi, tau_d, m_sh] where:
                n_sh: Shell number density in cm^-3
                phi: Ionizing photon survival probability
                tau_d: Dust optical depth
                m_sh: Cumulative shell mass in g

        Returns:
            list: Derivatives [dn_dr, dphi_dr, dtau_dr, dm_dr]
        """
        n_sh, phi, tau_d, _ = x
        
        A1 = 4 * np.pi * (r**2)
        dm_dr = A1 * MU_N * n_sh * self.cover_frac
        dtau_dr = n_sh * self.sigma_dust
        dphi_dr = -((A1 * ALPHA_B * (n_sh**2)) / self.Q_i) - (n_sh * self.sigma_dust * phi)
        
        A2 = (MU_N / MU_P) * K_BOLTZMANN * self.T_ion
        dn_dr = ((1 / (A1 * C)) * (-self.Lum_i * dphi_dr + self.Lum_n * np.exp(-tau_d) * dtau_dr)) / A2
        
        return [dn_dr, dphi_dr, dtau_dr, dm_dr]

    def shellStructureEqns_neutral(self, r, x):
        """
        System of ODEs describing the neutral shell structure.
        
        Solves for density, dust optical depth, and mass profiles in the
        neutral region where ionizing radiation has been absorbed.

        Args:
            r (float): Radius in cm
            x (list): Current values [n_sh, tau_d, m_sh] where:
                n_sh: Shell number density in cm^-3
                tau_d: Dust optical depth
                m_sh: Cumulative shell mass in g

        Returns:
            list: Derivatives [dn_dr, dtau_dr, dm_dr]
        """
        n_sh, tau_d, _ = x
        
        A1 = 4 * np.pi * (r**2)
        dm_dr = A1 * MU_N * n_sh * self.cover_frac
        dtau_dr = n_sh * self.sigma_dust
        
        A2 = K_BOLTZMANN * T_NEUTRAL
        dn_dr = ((1 / (A1 * C)) * (self.Lum_n * np.exp(-tau_d) * dtau_dr)) / A2
        
        return [dn_dr, dtau_dr, dm_dr]

    def terminate_ionized_phi(self, r, x):
        """Event function to stop integration when ionizing photons are absorbed."""
        return x[1]  # phi
    terminate_ionized_phi.terminal = True
    terminate_ionized_phi.direction = -1

    def terminate_ionized_mass(self, r, x):
        """Event function to stop integration when full shell mass is reached."""
        return self.M_sh - x[3]  # m_sh
    terminate_ionized_mass.terminal = True
    terminate_ionized_mass.direction = -1

    def terminate_neutral(self, r, x):
        """Event function to stop neutral region integration at full mass."""
        return self.M_sh - x[2]  # m_sh
    terminate_neutral.terminal = True
    terminate_neutral.direction = -1

    def solve_shell_structure(self, R):
        """
        Solve for the shell structure at a given radius.
        
        First solves the ionized region equations until either all ionizing photons
        are absorbed or the full shell mass is reached. If needed, continues with
        neutral region equations until the full shell mass is incorporated.

        Args:
            R (float): Shell radius in cm
            
        Note:
            Results are stored in self.shell_strctre_i and self.shell_strctre_n
            for the ionized and neutral regions respectively.
        """
        self.R = R
        R_max_shell = 1e5 * PC_TO_CM  # Very large radius to ensure stopping criteria are used
        
        # Initial conditions for ionized shell
        IC_i = [float(self.n_shell_in), 1.0, 0.0, 0.0]
        
        # Solve for ionized shell structure
        self.shell_strctre_i = solve_ivp(
            self.shellStructureEqns_ionized, 
            (self.R, R_max_shell), 
            IC_i,
            method='BDF', 
            vectorized=True, 
            rtol=self.rtol_i, 
            atol=self.atol_i,
            events=(self.terminate_ionized_phi, self.terminate_ionized_mass)
        )

        # Get thickness of ionized part
        r_ionized = self.shell_strctre_i.t
        ionized_thickness = r_ionized[-1] - r_ionized[0]

        if (self.M_sh - self.shell_strctre_i.y[3][-1]) / self.M_sh > self.mass_tolerance:
            # If the relative difference in mass is greater than the tolerance,
            # we consider that the ionized region doesn't account for the full shell mass
            IC_n = [
                float(self.shell_strctre_i.y[0][-1]), 
                float(self.shell_strctre_i.y[2][-1]), 
                float(self.shell_strctre_i.y[3][-1])
            ]
            
            # Solve for neutral shell structure
            self.shell_strctre_n = solve_ivp(
                self.shellStructureEqns_neutral, 
                (self.shell_strctre_i.t[-1], R_max_shell), 
                IC_n,
                method='BDF', 
                vectorized=True, 
                rtol=self.rtol_n, 
                atol=self.atol_n,
                events=self.terminate_neutral
            )
            # Get thickness of neutral part
            r_neutral = self.shell_strctre_n.t
            neutral_thickness = r_neutral[-1] - r_neutral[0]

            # Total thickness is sum of both parts
            self.shell_thickness = ionized_thickness + neutral_thickness
        else:
            # The ionized region accounts for (almost) all of the shell mass
            self.shell_strctre_n = None
            self.shell_thickness = ionized_thickness
            
        return self.shell_thickness

    def get_atomicHydrogenColumnDensity(self):
        """
        Calculate the neutral atomic hydrogen column density through the shell.
        
        This method determines the column density of HI by:
        1. Finding regions where the molecular fraction (from Krumholz+ 2013) is negligible
        2. Converting total dust optical depth to total hydrogen column using sigma_dust
        3. Subtracting the ionized hydrogen column to get neutral atomic component

        Returns:
            float: Column density of neutral atomic hydrogen (HI) in cm^-2.
                  Returns 0 if shell is fully ionized.
                  
        Note:
            Uses the Krumholz+ 2013 model to determine molecular fractions based on
            local Lyman-Werner flux, gas density, and dust shielding.
        """
        if self.shell_strctre_n is None:
            return 0
        else:
            # Calculate molecular fraction using Krumholz+ 2013 model
            r_n = self.shell_strctre_n.t  # Radial positions in neutral shell
            flux_LyW = self.stellar_feedback.get_lyman_werner_specific_luminosity(self.age) / (4 * np.pi * r_n**2)
            gas_density = self.shell_strctre_n.y[0]  # Number density
            tau_dust = self.shell_strctre_n.y[1]  # dust optical depth => cumulative
            f_mol = get_fmol_Krum2013(flux_LyW, gas_density, tau_dust)
            
            # Find atomic regions (negligible molecular fraction)
            idx_atomic = f_mol < 1e-6
            if np.any(idx_atomic):
                # HII + neutral atomic column density 
                column_H_all = tau_dust[idx_atomic][-1] / self.sigma_dust
                
                # ionized column
                column_ion = self.shell_strctre_i.y[2][-1] / self.sigma_dust
                
                # Subtract ionized column to get atomic column
                column_H1 = column_H_all - column_ion
                return column_H1
            else:
                return 0

    def calculate_weighted_sound_speed(self):
        """
        Calculate sound speed weighted by ionized and neutral shell thicknesses.
        
        Uses sound speed c_s = sqrt(gamma*k*T/mu) for each region:
        - Ionized: T = 10^4 K, mu = mu_p (mean mass per particle)
        - Neutral: T = 100 K, mu = mu_n (mean mass per nucleus)
        
        Weight factors are determined by the thickness of each region relative
        to total shell thickness.
        
        Returns:
            float: Thickness-weighted sound speed in cm/s
        """
        # Get thicknesses
        if self.shell_strctre_n is None:
            # Fully ionized shell
            return np.sqrt((GAMMA * K_BOLTZMANN * self.T_ion) / MU_P)
        else:
            # Get thicknesses from shell structure solution
            r_ionized = self.shell_strctre_i.t
            ionized_thickness = r_ionized[-1] - r_ionized[0]
            
            r_neutral = self.shell_strctre_n.t
            neutral_thickness = r_neutral[-1] - r_neutral[0]
            
            total_thickness = ionized_thickness + neutral_thickness
            
            # Calculate sound speeds
            c_s_ion = np.sqrt((GAMMA * K_BOLTZMANN * self.T_ion) / MU_P)
            c_s_neutral = np.sqrt((GAMMA * K_BOLTZMANN * T_NEUTRAL) / MU_N)
            
            # Weight by thicknesses
            w_ion = ionized_thickness / total_thickness
            w_neutral = neutral_thickness / total_thickness
            
            return w_ion * c_s_ion + w_neutral * c_s_neutral   
                 
    # to do: Add intrinsic and emergent H_alpha : phi, density to H_alpha conversion
    # to do: Add Hii region temperature based on the spectral hardness, metallicity, and logU => 3d single zone cloudy tables
    # to do: Use Yang et al to add ionized region line predictions directly