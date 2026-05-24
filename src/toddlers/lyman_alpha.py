from .imports import np, interp1d
from .constants import *
from .utils import interpolate_with_tanh, linear_interpolation

"""
Lyman Alpha Radiation Pressure Module
===================================

This module implements the Lyman alpha radiation pressure calculations based on:
Kimm et al. (2018) - "Impact of Lyα pressure on metal-poor dwarf galaxies"
DOI: 10.1093/mnras/sty126

The module calculates the momentum boost from resonantly scattered Lyman α photons. 
The implementation includes dust attenuation and velocity dependence.
"""

class LymanAlpha:
    """
    Class to calculate Lyman alpha radiation pressure effects.
    
    Attributes:
        Z (float): Metallicity in solar units
        T_sh_atm (float): Temperature in K
        f_dm (float): Dust-to-metal mass ratio normalized to solar value
        sigma_dust (float): Dust absorption cross-section [cm^2/H]
        sigma_dust_ext (float): Dust extinction cross-section [cm^2/H]
        sigma_Lya (float): Temperature-dependent Lyman alpha cross-section [cm^2]
        dMf_dtau0_interpolant (callable): Interpolation function for force multiplier derivative
        Lyalpha_radiationPressure_fudgeFactor (float): Overall calibration factor
    """
    
    def __init__(self, Z, T_sh_atm, dust_to_metal_relative_to_solar=1):
        """
        Initialize Lyman alpha radiation pressure calculations.
        Dust cross-section have been scaled by Z/Z_solar in init itself,
        so the form could look somewhat different from Kimm+ 2018.
        We do not vary f_dm with Z.
        
        Args:
            Z (float): Gas metallicity in solar units 
            T_sh_atm (float): Gas temperature in K
            dust_to_metal_relative_to_solar (float, optional): Dust-to-metal mass ratio 
                normalized to solar. Defaults to 1.
        """
        self.Z = Z
        self.T_sh_atm = T_sh_atm
        self.f_dm = dust_to_metal_relative_to_solar
        self.sigma_dust = SIGMA_DUST_SOLAR * (Z / Z_SOLAR) # absorption cross section, not extinction
        self.sigma_dust_ext = self.sigma_dust / (1 - DUST_ALBEDO) # already scaled with Z, modify kimm's eqn accordingly
        self.sigma_Lya = 5.88e-14 / np.sqrt(self.T_sh_atm / 1e4)
        self.dMf_dtau0_interpolant = self.get_interp_derivative_dusty_forceMultiplier_wrt_tau0()
        self.Lyalpha_radiationPressure_fudgeFactor = LYA_FF
        self.force_multiplier = 0.0
        self.Lyalpha_Luminosity_ratio = 0.0
            
    def get_LyAlpha_radiationForce(self, columnDensity_H1, Q_i, shell_structure, V_sh):
        """
        Calculate total radiation force from Lyman alpha photons.

        Args:
            columnDensity_H1 (float): Neutral hydrogen column density [cm^-2]
            Q_i (float): Ionizing photon rate [s^-1]
            shell_structure (object): Shell structure containing ionization properties
            V_sh (float): Shell velocity [cm/s]

        Returns:
            float: Total radiation force [dyne]
        """
        self.shell_strctre_i = shell_structure.shell_strctre_i
        self.shell_strctre_n = shell_structure.shell_strctre_n

        velocityFactor = self.get_Lyman_alpha_veloFactor(V_sh)
        
        # Calculate tau_d_fore using cumulative sum
        tau_d = self.shell_strctre_i.y[2]
        tau_d_fore = tau_d[-1] - np.cumsum(np.diff(tau_d))
        atten_fore = np.exp(-tau_d_fore)

        r_ion = self.shell_strctre_i.t
        n_sh_ion = self.shell_strctre_i.y[0]
        dr_ion = np.diff(r_ion)

        # Ensure all arrays have the same length
        atten_fore = atten_fore[:len(dr_ion)]
        n_sh_ion = n_sh_ion[:len(dr_ion)]
        r_ion = r_ion[:len(dr_ion)]

        L_Lyalpha_noDust = ION_TO_LYALPHA * Q_i * E_LYALPHA
        L_Lyalpha_dusty = np.sum(atten_fore * ION_TO_LYALPHA * ALPHA_B * (n_sh_ion**2) * (4*np.pi*(r_ion**2)*dr_ion) * E_LYALPHA)
        self.Lyalpha_Luminosity_ratio = L_Lyalpha_dusty / L_Lyalpha_noDust
        self.force_multiplier = velocityFactor * self.Lyalpha_radiationPressure_fudgeFactor * self.get_dusty_forceMultiplier(columnDensity_H1)
        return self.force_multiplier * (L_Lyalpha_dusty / C)

    def get_Lyman_alpha_veloFactor(self, V_sh):
        """
        Calculate velocity-dependent reduction of radiation force.
        
        Reduces force for high velocities where doppler shifts reduce 
        resonant scattering efficiency.

        Args:
            V_sh (float): Shell velocity [cm/s]

        Returns:
            float: Velocity reduction factor [dimensionless]
        """
        if abs(V_sh) <= LYMAN_ALPHA_VELOCITY_THRESHOLD:
            return linear_interpolation(0, 1, LYMAN_ALPHA_VELOCITY_THRESHOLD, 0.1, abs(V_sh))
        else:
            return 10**interpolate_with_tanh(LYMAN_ALPHA_VELOCITY_THRESHOLD, -1, LYMAN_ALPHA_VELOCITY_THRESHOLD + 5, -6, abs(V_sh))

    def get_dusty_forceMultiplier(self, columnDensity_H1):
        """
        Calculate force multiplier including dust effects.
        
        Combines dust-free force multiplier with escape fraction
        reduction from dust absorption.

        Args:
            columnDensity_H1 (float): Neutral hydrogen column density [cm^-2]

        Returns:
            float: Force multiplier with dust attenuation [dimensionless]
        """
        T4 = self.T_sh_atm / 1e4
        if columnDensity_H1 == 0:
            return 0
        else:
            tau_0_peak, columnDensity_at_peak = self.get_tau_0_peak()
            tau_0 = columnDensity_H1 * self.sigma_Lya
            arg_tau = np.minimum(tau_0, tau_0_peak)
            arg_columnDensity = np.minimum(columnDensity_H1, columnDensity_at_peak)
            dustFree_forceMultiplier_at_arg = self.get_dustFree_forceMultiplier(arg_tau)
            f_esc_Lya_at_arg = self.get_f_esc_Lya(arg_columnDensity)
            return dustFree_forceMultiplier_at_arg * f_esc_Lya_at_arg

    def get_tau_0_peak(self):
        """
        Calculate peak optical depth and corresponding column density.
        
        Uses metallicity and temperature dependent scaling from
        Kimm et al. (2018) Eq. 10.

        Returns:
            tuple: Peak optical depth [dimensionless] and column density [cm^-2]
        """
        T4 = self.T_sh_atm / 1e4
        tau_0_peak = 4.06e6 * (T4**-0.25) * (((self.f_dm) * (self.sigma_dust_ext / 3e-21))**-0.75) # kimm+ eqn 10
        columnDensity_at_peak = tau_0_peak / self.sigma_Lya
        return tau_0_peak, columnDensity_at_peak

    def get_f_esc_Lya(self, columnDensity_H1):
        """
        Calculate Lyman alpha escape fraction through dusty medium.
        
        Accounts for dust absorption using the modified escape fraction
        formula from Kimm et al. (2018).

        Args:
            columnDensity_H1 (float): Neutral hydrogen column density [cm^-2]

        Returns:
            float: Escape fraction [dimensionless]
        """
        T4 = self.T_sh_atm / 1e4
        dust_albedo = 0.46
        tau_da = columnDensity_H1 * self.f_dm * self.sigma_dust_ext * (1 - dust_albedo) # absorption
        zeta_fit = 1.78 # replaces zeta (0.58) to fit MCRT results in Kimm+ 2018
        a_v = 4.7e-4 / np.sqrt(T4)
        tau_0 = columnDensity_H1 * self.sigma_Lya
        a = (np.sqrt(3)) / (zeta_fit * (np.pi)**(5/12))
        b = np.sqrt(np.cbrt(a_v * tau_0) * tau_da)
        return 1 / np.cosh(a * b)

    def get_dustFree_forceMultiplier(self, tau_0):
        """
        Calculate force multiplier for dust-free medium.
        
        Uses different power-law relations for low and high optical depths
        based on Monte Carlo radiative transfer results.

        Args:
            tau_0 (float): Optical depth at line center

        Returns:
            float: Dust-free force multiplier [dimensionless]
        """
        mask = tau_0 < 1e6
        dustFree_forceMultiplier = np.ones_like(tau_0)
        x = np.log10(tau_0[mask])
        dustFree_forceMultiplier[mask] = 10**(-0.433 + 0.874*x - 0.173*(x**2) + 0.0133*(x**3))
        y = tau_0[~mask]/1e6
        dustFree_forceMultiplier[~mask] = 29 * (y**0.29)
        dustFree_forceMultiplier[tau_0 == 0] = 0
        return dustFree_forceMultiplier

    def get_interp_derivative_dusty_forceMultiplier_wrt_tau0(self):
        """
        Create interpolation function for force multiplier derivative.
        
        Uses Savitzky-Golay filter to smooth numerical derivative and
        ensures smooth transition at peak optical depth.

        Returns:
            callable: Interpolation function for dM_F/dτ_0
        """
        from scipy.signal import savgol_filter as sg_filt
        tau_0 = np.logspace(-16, 10, 1000, endpoint=True)
        tau_0_peak, _ = self.get_tau_0_peak()
        Mf_list = [self.get_dusty_forceMultiplier(i / self.sigma_Lya) for i in tau_0]
        Mf_list = np.array(Mf_list)
        dMf_dtau0 = np.gradient(Mf_list, tau_0)
        dMf_dtau0 = sg_filt(dMf_dtau0, 21, 3)
        idx_peak = np.abs((tau_0-tau_0_peak)).argmin()
        dMf_peak = dMf_dtau0[idx_peak]
        dMf_dtau0[tau_0>tau_0_peak] = interpolate_with_tanh(tau_0_peak, dMf_peak, tau_0_peak + 1e-3, dMf_peak/1e32, tau_0[tau_0>tau_0_peak])
        interp_dMf_dtau0 = interp1d(tau_0, dMf_dtau0, fill_value=0, bounds_error=False)
        return interp_dMf_dtau0