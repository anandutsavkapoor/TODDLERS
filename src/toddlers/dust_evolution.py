# dust_evolution.py
"""
Two-size dust evolution model based on Hirashita (2015).

Tracks evolution of small (a < 0.03 micron) and large (a > 0.03 micron) dust grains
through stellar production, supernova destruction, accretion, coagulation,
and shattering.

References:
    Hirashita, H. (2015), MNRAS, 447, 2937
"""

import numpy as np
try:
    from .constants import (
        Z_SOLAR, MYR_TO_SEC, PC_TO_CM, M_P, K_BOLTZMANN,
        SIGMA_DUST_SOLAR
    )
except ImportError:
    from constants import (
        Z_SOLAR, MYR_TO_SEC, PC_TO_CM, M_P, K_BOLTZMANN,
        SIGMA_DUST_SOLAR
    )


class DustBin:
    """
    Container for dust properties in a spatial bin.
    
    Attributes:
        Ds (float): Small grain dust-to-gas ratio [dimensionless]
        Dl (float): Large grain dust-to-gas ratio [dimensionless]
        n (float): Number density [cm^-3]
        T (float): Temperature [K]
        Z (float): Metallicity [dimensionless, Z_solar = 0.02]
        r_inner (float): Inner radius [cm]
        r_outer (float): Outer radius [cm]
        M_gas (float): Gas mass in bin [g]
    """
    
    def __init__(self, Ds, Dl, n, T, Z, r_inner, r_outer, M_gas):
        self.Ds = Ds
        self.Dl = Dl
        self.n = n
        self.T = T
        self.Z = Z
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.M_gas = M_gas
        
    @property
    def D_total(self):
        """Total dust-to-gas ratio."""
        return self.Ds + self.Dl
    
    @property
    def small_to_large_ratio(self):
        """Ratio of small to large grain abundance."""
        return self.Ds / self.Dl if self.Dl > 0 else 0.0
    
    def evolve_full(self, dZ, fin=0.1, beta_SN=9.65, alpha=1.0, 
                    tau_sh_0=1e8, tau_co_0=1e7, tau_SF=5e9, tau_cl=2e9,
                    YZ=0.013, RZ=0.0):
        """
        Evolve dust with full galactic equations (Hirashita 2015 eqs 17-18).
        
        Includes stellar production, SN destruction, accretion, coagulation, shattering.
        
        Parameters:
            dZ (float): Metallicity change
            fin (float): Dust condensation efficiency
            beta_SN (float): SN destruction parameter
            alpha (float): Small/large destruction ratio
            tau_sh_0 (float): Shattering timescale [yr]
            tau_co_0 (float): Coagulation timescale [yr]
            tau_SF (float): Star formation timescale [yr]
            tau_cl (float): Cloud timescale [yr]
            YZ (float): Metal yield
            RZ (float): Recycling rate
        
        Returns:
            None (modifies bin in place)
        """
        # Compute beta parameters with capping for numerical stability
        tau_acc = timescale_accretion(self.n, self.T, self.Z, self.D_total) / (365.25 * 86400)  # to yr
        beta_acc = min((tau_SF / tau_cl) * (tau_cl / tau_acc), 1e5) if tau_acc < np.inf else 0.0
        
        if self.Ds > 1e-20:
            tau_co = tau_co_0 * (DMW_S / self.Ds)
            beta_co = min(tau_SF / tau_co, 1e5)
        else:
            beta_co = 0.0
        
        if self.Dl > 1e-20:
            tau_sh = tau_sh_0 * (DMW_L / self.Dl)
            beta_sh = min(tau_SF / tau_sh, 1e5)
        else:
            beta_sh = 0.0
        
        stellar_source = RZ * self.Z + YZ
        
        # Hirashita (2015) equations 17-18
        # Eq 17: dDl/dZ = (1/YZ)[fin(RZ*Z+YZ) + βco*Ds - (βSN+βsh+R)*Dl]
        dDl_dZ = (1.0 / YZ) * (
            fin * stellar_source +
            beta_co * self.Ds -
            (beta_SN + beta_sh + RZ) * self.Dl
        )
        
        # Eq 18: dDs/dZ = (1/YZ)[βsh*Dl - (βSN/α+βco+R-βacc)*Ds]
        dDs_dZ = (1.0 / YZ) * (
            beta_sh * self.Dl -
            ((beta_SN / alpha) + beta_co + RZ - beta_acc) * self.Ds
        )
        
        # Update with stability check
        Ds_new = self.Ds + dDs_dZ * dZ
        Dl_new = self.Dl + dDl_dZ * dZ
        
        # Prevent runaway growth
        if Ds_new > 10 * self.Z or Dl_new > 10 * self.Z:
            # Limit dust to reasonable fraction of metals
            scale = min(1.0, 0.9 * self.Z / (Ds_new + Dl_new))
            Ds_new *= scale
            Dl_new *= scale
        
        self.Ds = max(0.0, Ds_new)
        self.Dl = max(0.0, Dl_new)
        self.Z += dZ


# Hirashita (2015) parameters
DMW_S = 0.0030  # Milky Way small grain dust-to-gas ratio
DMW_L = 0.0070  # Milky Way large grain dust-to-gas ratio
A_SMALL = 0.005  # Characteristic small grain radius [micron]
A_LARGE = 0.1    # Characteristic large grain radius [micron]
FIN_DEFAULT = 0.1  # Dust condensation efficiency in stellar ejecta


def timescale_accretion(n, T, Z, D, a0=0.1, S=0.3):
    """
    Accretion timescale for grain growth in dense clouds.
    
    From Hirashita & Kuo (2011) eq. 23, applicable for silicate grains.
    
    Parameters:
        n (float): Number density [cm^-3]
        T (float): Gas temperature [K]
        Z (float): Metallicity [dimensionless]
        D (float): Total dust-to-gas ratio
        a0 (float): Normalization grain radius [μm]
        S (float): Sticking probability [dimensionless]
    
    Returns:
        float: Accretion timescale [seconds]
        
    Notes:
        - Only relevant in cold, dense gas (T < 300 K, n > 10 cm^-3)
        - Returns np.inf if T > 300 K or gas-phase metals depleted
    """
    if T > 300.0:  # No accretion in warm gas
        return np.inf
    
    xi = (Z - D) / Z  # Fraction of metals in gas phase
    if xi < 0.01:  # Metals nearly depleted
        return np.inf
    
    # Convert to cgs
    a0_cm = a0 * 1e-4
    
    # Hirashita & Kuo (2011) eq. 23
    tau_yr = 6.3e7 * (Z / Z_SOLAR)**(-1) * (a0 / 0.1) * (n / 1e3)**(-1) * \
             (T / 50.0)**(-0.5) * (S / 0.3)**(-1) * xi**(-1)
    
    return tau_yr * 365.25 * 86400  # Convert to seconds


def timescale_coagulation(Ds, n, tau_co_0=1e7, n_MW=1.0):
    """
    Coagulation timescale for small->large grain growth.
    
    From Hirashita (2015) eq. 8.
    
    Parameters:
        Ds (float): Small grain dust-to-gas ratio
        n (float): Number density [cm^-3]
        tau_co_0 (float): Timescale at MW conditions [yr]
        n_MW (float): MW average density [cm^-3]
    
    Returns:
        float: Coagulation timescale [seconds]
    """
    if Ds < 1e-10:
        return np.inf
    
    tau_yr = tau_co_0 * (Ds / DMW_S)**(-1) * (n / n_MW)**(-1)
    return tau_yr * 365.25 * 86400


def timescale_shattering(Dl, n, tau_sh_0=1e8, n_MW=1.0):
    """
    Shattering timescale for large->small grain disruption.
    
    From Hirashita (2015) eq. 7.
    
    Parameters:
        Dl (float): Large grain dust-to-gas ratio
        n (float): Number density [cm^-3]
        tau_sh_0 (float): Timescale at MW conditions [yr]
        n_MW (float): MW average density [cm^-3]
    
    Returns:
        float: Shattering timescale [seconds]
    """
    if Dl < 1e-10:
        return np.inf
    
    tau_yr = tau_sh_0 * (Dl / DMW_L)**(-1) * (n / n_MW)**(-1)
    return tau_yr * 365.25 * 86400


def sn_destruction_efficiency(a, v_shock, material='silicate'):
    """
    Survival fraction after supernova shock passage.
    
    Based on Nozawa et al. (2006, 2007) thermal + nonthermal sputtering.
    
    Parameters:
        a (float): Grain radius [μm]
        v_shock (float): Shock velocity [km/s]
        material (str): 'silicate' or 'carbon'
    
    Returns:
        float: Survival fraction (0 = fully destroyed, 1 = no destruction)
        
    Notes:
        - Simple parametrization; full calculation requires integrating
          through shock structure
        - Small grains more efficiently destroyed
    """
    # Simplified model: f_surv = exp(-v_shock / v_char) * (a / a_char)^0.5
    # Calibrated roughly to Nozawa+2007 Fig 8
    
    if material == 'silicate':
        v_char = 200.0  # km/s
        a_char = 0.1    # μm
    else:  # carbon
        v_char = 150.0
        a_char = 0.1
    
    f_surv = np.exp(-v_shock / v_char) * (a / a_char)**0.5
    return np.clip(f_surv, 0.0, 1.0)


def apply_sn_destruction(Ds, Dl, v_shock, f_silicate=0.54):
    """
    Apply SN shock destruction to dust.
    
    Parameters:
        Ds (float): Small grain dust-to-gas ratio (before shock)
        Dl (float): Large grain dust-to-gas ratio (before shock)
        v_shock (float): Shock velocity [km/s]
        f_silicate (float): Mass fraction of silicate vs carbon
    
    Returns:
        tuple: (Ds_new, Dl_new) after destruction
    """
    # Weighted average survival
    f_surv_small = (f_silicate * sn_destruction_efficiency(A_SMALL, v_shock, 'silicate') +
                    (1 - f_silicate) * sn_destruction_efficiency(A_SMALL, v_shock, 'carbon'))
    
    f_surv_large = (f_silicate * sn_destruction_efficiency(A_LARGE, v_shock, 'silicate') +
                    (1 - f_silicate) * sn_destruction_efficiency(A_LARGE, v_shock, 'carbon'))
    
    return Ds * f_surv_small, Dl * f_surv_large


def dust_odes(Ds, Dl, n, T, Z, include_shattering=True):
    """
    Right-hand side of dust evolution ODEs.
    
    Parameters:
        Ds (float): Small grain dust-to-gas ratio
        Dl (float): Large grain dust-to-gas ratio
        n (float): Number density [cm^-3]
        T (float): Temperature [K]
        Z (float): Metallicity
        include_shattering (bool): Include shattering process
    
    Returns:
        tuple: (dDs_dt, dDl_dt) in [1/s]
    """
    D_total = Ds + Dl
    
    # Timescales
    tau_acc = timescale_accretion(n, T, Z, D_total)
    tau_co = timescale_coagulation(Ds, n)
    tau_sh = timescale_shattering(Dl, n) if include_shattering else np.inf
    
    # Source/sink terms
    S_acc = Ds / tau_acc if tau_acc < np.inf else 0.0
    S_co = Ds / tau_co if tau_co < np.inf else 0.0
    S_sh = Dl / tau_sh if tau_sh < np.inf else 0.0
    
    # Evolution equations (Hirashita 2015, eqs 3-4, neglecting stellar/SN terms)
    dDs_dt = S_sh - S_co + S_acc
    dDl_dt = -S_sh + S_co
    
    return dDs_dt, dDl_dt


def initialize_dust_from_stellar(Z, fin=FIN_DEFAULT, Ds_fraction=0.1):
    """
    Initialize dust from stellar production only.
    
    Parameters:
        Z (float): Metallicity
        fin (float): Dust condensation efficiency
        Ds_fraction (float): Fraction in small grains
    
    Returns:
        tuple: (Ds, Dl)
    """
    D_total = fin * Z
    Ds = D_total * Ds_fraction
    Dl = D_total * (1 - Ds_fraction)
    return Ds, Dl


def compute_effective_cross_section(Ds, Dl):
    """
    Compute effective dust cross-section from two-size distribution.
    
    Parameters:
        Ds (float): Small grain dust-to-gas ratio
        Dl (float): Large grain dust-to-gas ratio
    
    Returns:
        float: Effective cross-section [cm^2]
        
    Notes:
        - Small grains dominate UV absorption (high surface/volume)
        - Assumes lognormal size distribution within each population
    """
    # Simple approximation: weight by surface area
    # More accurate: integrate lognormal distributions (see Hirashita 2015 eq. 30)
    
    rho_grain = 3.0  # g/cm^3, average
    
    # Surface-to-mass ratios [cm^2/g]
    # For lognormal with a0, Ïƒ: <S/M> ~ 3/<a>_volume
    surf_to_mass_small = 3.0 / (A_SMALL * 1e-4)  # 1/cm
    surf_to_mass_large = 3.0 / (A_LARGE * 1e-4)
    
    # Total cross-section per H
    D_total = Ds + Dl
    if D_total < 1e-20:
        return 0.0
    
    # Weight by small grain fraction (they have more surface)
    f_small_mass = Ds / D_total
    f_small_surface = (Ds * surf_to_mass_small) / \
                      (Ds * surf_to_mass_small + Dl * surf_to_mass_large)
    
    # Scale solar cross-section
    sigma_eff = SIGMA_DUST_SOLAR * D_total / 0.01 * (f_small_surface / 0.3)
    
    return sigma_eff


def compute_IR_opacity(Dl):
    """
    Compute IR opacity from large grain abundance.
    
    Parameters:
        Dl (float): Large grain dust-to-gas ratio
    
    Returns:
        float: IR opacity [cm^2/g]
        
    Notes:
        - Large grains dominate IR emission/absorption
    """
    from .constants import KAPPA_IR_SOLAR
    
    # Scale with large grain abundance
    kappa_IR = KAPPA_IR_SOLAR * (Dl / DMW_L)
    return kappa_IR


# ============================================================================
# Galactic Dust Evolution (Hirashita 2015 equations 17-18)
# ============================================================================

class GalacticDustEvolution:
    """
    Full galactic dust evolution following Hirashita (2015) equations 17-18.
    
    Integrates dust evolution with metallicity including:
    - Stellar dust production (condensation in ejecta)
    - SN shock destruction
    - Accretion (grain growth in dense clouds)
    - Coagulation (small → large)
    - Shattering (large → small)
    
    References:
        Hirashita (2015), MNRAS 447, 2937, equations 17-18
        Hirashita & Kuo (2011), MNRAS 416, 1340 (accretion timescales)
    """
    
    def __init__(self, 
                 fin=0.1,
                 beta_SN=9.65,
                 alpha=1.0,
                 tau_sh_0=1e8,
                 tau_co_0=1e7,
                 tau_SF=5e9,
                 tau_cl=2e9,
                 n_ISM=1.0,
                 T_cloud=50.0,
                 n_cloud=1e3):
        """
        Initialize galactic dust evolution model.
        
        Parameters:
            fin (float): Dust condensation efficiency in stellar ejecta [0.01-0.1]
                        Fiducial: 0.1 (Table 2)
            beta_SN (float): SN destruction parameter [4.83-19.3]
                            β_SN = τ_SF × R_SN × M_swept / M_ISM
                            Fiducial: 9.65 (Table 2)
            alpha (float): Relative destruction of small vs large grains [0.1-1]
                          α = 1: equal destruction
                          α < 1: small grains destroyed more efficiently
                          Fiducial: 1.0 (Table 2)
            tau_sh_0 (float): Shattering timescale at MW conditions [yr]
                             Fiducial: 10^8 yr (Table 2)
            tau_co_0 (float): Coagulation timescale at MW conditions [yr]
                             Fiducial: 10^7 yr (Table 2)
            tau_SF (float): Star formation timescale [yr]
                           Fiducial: 5×10^9 yr (Table 2)
            tau_cl (float): Cloud timescale for accretion [yr]
                           Fiducial: 2×10^9 yr (eq. 12)
            n_ISM (float): ISM-averaged number density [cm^-3]
            T_cloud (float): Temperature in dense clouds [K]
            n_cloud (float): Number density in dense clouds [cm^-3]
        
        Notes:
            Default parameters correspond to fiducial model in Hirashita (2015).
        """
        self.fin = fin
        self.beta_SN = beta_SN
        self.alpha = alpha
        self.tau_sh_0 = tau_sh_0
        self.tau_co_0 = tau_co_0
        self.tau_SF = tau_SF
        self.tau_cl = tau_cl
        self.n_ISM = n_ISM
        self.T_cloud = T_cloud
        self.n_cloud = n_cloud
        
        # Galactic chemical evolution parameters (Hirashita 2015, Section 2.3)
        self.YZ = 0.013  # Metal yield per stellar generation
        self.RZ = 0.0    # Instantaneous recycling rate (set to 0 for simplicity)
        
    def tau_accretion(self, Z, D):
        """
        Accretion timescale in dense clouds.
        
        From Hirashita & Kuo (2011) eq. 23:
            τ_acc = 6.3×10^7 yr (Z/Z_⊙)^-1 (n/10^3 cm^-3)^-1 (T/50K)^-0.5 
                    (S/0.3)^-1 ξ^-1
        
        where ξ = (Z - D)/Z is the gas-phase metal fraction.
        
        Parameters:
            Z (float): Metallicity
            D (float): Total dust-to-gas ratio
        
        Returns:
            float: Accretion timescale [yr]
        """
        if Z < 1e-15:
            return np.inf
        
        xi = (Z - D) / Z  # Gas-phase metal fraction
        if xi < 0.001:
            return np.inf  # Metals depleted
        
        # Hirashita & Kuo (2011) eq. 23
        tau_yr = 6.3e7 * (Z / Z_SOLAR)**(-1) * (self.n_cloud / 1e3)**(-1) * \
                 (self.T_cloud / 50.0)**(-0.5) * (0.3 / 0.3)**(-1) * xi**(-1)
        
        return tau_yr
    
    def beta_accretion(self, Z, D, Ds):
        """
        Accretion efficiency parameter β_acc.
        
        From Hirashita (2015) equations 10-11:
            B(Z) = (τ_cl / τ_acc) × y(Z)
            β_acc = (τ_SF / τ_cl) × B
        
        where y(Z) accounts for saturation of accretion.
        
        Parameters:
            Z (float): Metallicity
            D (float): Total dust-to-gas ratio
            Ds (float): Small grain dust-to-gas ratio
        
        Returns:
            float: β_acc
        """
        tau_acc = self.tau_accretion(Z, D)
        
        if tau_acc == np.inf:
            return 0.0
        
        # Saturation factor y(Z) from eq. 10
        # y → ∞ (most efficient) when all metals are accreted onto small grains
        # Simplified: use B = τ_cl / τ_acc directly
        B = self.tau_cl / tau_acc
        
        # β_acc from eq. 11
        beta_acc = (self.tau_SF / self.tau_cl) * B
        
        return beta_acc
    
    def beta_coagulation(self, Ds):
        """
        Coagulation efficiency parameter β_co.
        
        From Hirashita (2015) equation 8:
            τ_co = τ_co,0 (D_MW,s / D_s) (n_MW / n)
            β_co = τ_SF / τ_co
        
        Parameters:
            Ds (float): Small grain dust-to-gas ratio
        
        Returns:
            float: β_co
        """
        if Ds < 1e-20:
            return 0.0
        
        tau_co = self.tau_co_0 * (DMW_S / Ds) * (1.0 / self.n_ISM)
        beta_co = self.tau_SF / tau_co
        
        return beta_co
    
    def beta_shattering(self, Dl):
        """
        Shattering efficiency parameter β_sh.
        
        From Hirashita (2015) equation 7:
            τ_sh = τ_sh,0 (D_MW,l / D_l) (n_MW / n)
            β_sh = τ_SF / τ_sh
        
        Parameters:
            Dl (float): Large grain dust-to-gas ratio
        
        Returns:
            float: β_sh
        """
        if Dl < 1e-20:
            return 0.0
        
        tau_sh = self.tau_sh_0 * (DMW_L / Dl) * (1.0 / self.n_ISM)
        beta_sh = self.tau_SF / tau_sh
        
        return beta_sh
    
    def evolution_equations(self, Z, y):
        """
        Evolution ODEs: dy/dZ where y = [Ds, Dl].
        
        Hirashita (2015) equations 17-18:
        
        dD_l/dZ = (1/Y_Z) [(1-f_in)(R_Z Z + Y_Z) + β_co D_s 
                           - (β_sh + β_SN) D_l]
        
        dD_s/dZ = (1/Y_Z) [f_in(R_Z Z + Y_Z) + β_sh D_l 
                           - (α β_SN + β_co - β_acc) D_s]
        
        Parameters:
            Z (float): Metallicity
            y (array): [Ds, Dl] dust-to-gas ratios
        
        Returns:
            array: [dDs/dZ, dDl/dZ]
        """
        Ds, Dl = y
        Ds = max(Ds, 0.0)  # Ensure non-negative
        Dl = max(Dl, 0.0)
        D = Ds + Dl
        
        # Efficiency parameters
        beta_acc = self.beta_accretion(Z, D, Ds)
        beta_co = self.beta_coagulation(Ds)
        beta_sh = self.beta_shattering(Dl)
        
        # Stellar source term (eq. 17-18)
        stellar_source = self.RZ * Z + self.YZ
        
        # Large grain evolution (eq. 17)
        # YZ dDl/dZ = fin(RZ*Z + YZ) + βco*Ds - (βSN + βsh + R)*Dl
        dDl_dZ = (1.0 / self.YZ) * (
            self.fin * stellar_source +  # ALL stellar dust → large grains
            beta_co * Ds -
            (beta_sh + self.beta_SN + self.RZ) * Dl  # Include R term
        )
        
        # Small grain evolution (eq. 18)
        # YZ dDs/dZ = βsh*Dl - (βSN/α + βco + R - βacc)*Ds
        dDs_dZ = (1.0 / self.YZ) * (
            beta_sh * Dl -  # Small grains from shattering only, NO stellar source
            ((self.beta_SN / self.alpha) + beta_co + self.RZ - beta_acc) * Ds  # βSN/α not α*βSN!
        )
        
        return np.array([dDs_dZ, dDl_dZ])
    
    def solve_evolution(self, Z_min=1e-3, Z_max=1.5, N_points=1000):
        """
        Solve dust evolution from Z_min to Z_max (in Z_solar units).
        
        Parameters:
            Z_min (float): Minimum metallicity [Z_solar]
            Z_max (float): Maximum metallicity [Z_solar]
            N_points (int): Number of evaluation points
        
        Returns:
            tuple: (Z_arr, Ds_arr, Dl_arr)
                Z_arr: Metallicity array
                Ds_arr: Small grain dust-to-gas ratios
                Dl_arr: Large grain dust-to-gas ratios
        """
        from scipy.integrate import solve_ivp
        
        Z_span = (Z_min * Z_SOLAR, Z_max * Z_SOLAR)
        Z_eval = np.logspace(np.log10(Z_min), np.log10(Z_max), N_points) * Z_SOLAR
        
        # Initial conditions: stellar dust only at low Z
        D0 = self.fin * Z_span[0]
        Ds0 = 0.1 * D0  # Initially mostly large grains from stars
        Dl0 = 0.9 * D0
        y0 = np.array([Ds0, Dl0])
        
        # Solve with adaptive ODE solver
        sol = solve_ivp(
            self.evolution_equations,
            Z_span,
            y0,
            t_eval=Z_eval,
            method='LSODA',  # Handles stiff equations
            rtol=1e-8,
            atol=1e-15
        )
        
        if not sol.success:
            print(f"Warning: ODE integration failed: {sol.message}")
        
        Z_arr = sol.t
        Ds_arr = np.maximum(sol.y[0], 0.0)
        Dl_arr = np.maximum(sol.y[1], 0.0)
        
        return Z_arr, Ds_arr, Dl_arr
    
    def compute_observables(self, Z_arr, Ds_arr, Dl_arr):
        """
        Compute observable quantities from evolution.
        
        Parameters:
            Z_arr (array): Metallicity array
            Ds_arr (array): Small grain dust-to-gas ratios
            Dl_arr (array): Large grain dust-to-gas ratios
        
        Returns:
            dict: Dictionary with:
                - D_total: Total dust-to-gas ratio
                - Ds_Dl_ratio: Small-to-large grain ratio
                - D_Z_ratio: Dust-to-metal ratio
                - Z_crit: Critical metallicity [Z_solar]
                - beta_acc_arr: Accretion efficiency vs Z
                - beta_co_arr: Coagulation efficiency vs Z
                - beta_sh_arr: Shattering efficiency vs Z
        """
        D_total = Ds_arr + Dl_arr
        Ds_Dl_ratio = Ds_arr / (Dl_arr + 1e-30)
        D_Z_ratio = D_total / Z_arr
        
        # Compute beta parameters
        beta_acc_arr = np.array([self.beta_accretion(Z, D, Ds) 
                                 for Z, D, Ds in zip(Z_arr, D_total, Ds_arr)])
        beta_co_arr = np.array([self.beta_coagulation(Ds) for Ds in Ds_arr])
        beta_sh_arr = np.array([self.beta_shattering(Dl) for Dl in Dl_arr])
        
        # Find critical metallicity where β_acc / 2 ≈ β_SN (Hirashita 2015, Section 4.3)
        idx_crit = np.argmin(np.abs(beta_acc_arr / 2.0 - self.beta_SN))
        Z_crit = Z_arr[idx_crit] / Z_SOLAR
        
        return {
            'D_total': D_total,
            'Ds_Dl_ratio': Ds_Dl_ratio,
            'D_Z_ratio': D_Z_ratio,
            'Z_crit': Z_crit,
            'beta_acc_arr': beta_acc_arr,
            'beta_co_arr': beta_co_arr,
            'beta_sh_arr': beta_sh_arr
        }


# ============================================================================
# Integration with shell/cloud structure
# ============================================================================

def create_cloud_dust_bins(r_cl, n_profile, Z, N_bins=10, fin=FIN_DEFAULT):
    """
    Initialize radial dust bins in cloud.
    
    Parameters:
        r_cl (float): Cloud radius [cm]
        n_profile (callable): n(r) function [cm^-3]
        Z (float): Metallicity
        N_bins (int): Number of radial bins
        fin (float): Dust condensation efficiency
    
    Returns:
        list: List of DustBin objects
    """
    bins = []
    r_edges = np.linspace(0, r_cl, N_bins + 1)
    
    for i in range(N_bins):
        r_in = r_edges[i]
        r_out = r_edges[i + 1]
        r_mid = 0.5 * (r_in + r_out)
        
        n_avg = n_profile(r_mid)
        V_bin = (4/3) * np.pi * (r_out**3 - r_in**3)
        M_gas = n_avg * V_bin * (14/11) * M_P  # Mean molecular weight
        
        Ds, Dl = initialize_dust_from_stellar(Z, fin)
        
        bins.append(DustBin(Ds, Dl, n_avg, 100.0, Z, r_in, r_out, M_gas))
    
    return bins


def evolve_dust_bin(bin, dt, include_shattering=True):
    """
    Evolve dust in a single bin forward in time.
    
    Parameters:
        bin (DustBin): Dust bin to evolve
        dt (float): Timestep [s]
        include_shattering (bool): Include shattering
    
    Returns:
        None (modifies bin in place)
        
    Notes:
        - Uses simple forward Euler (could upgrade to RK4)
        - Assumes n, T, Z constant over timestep
    """
    dDs_dt, dDl_dt = dust_odes(bin.Ds, bin.Dl, bin.n, bin.T, bin.Z, include_shattering)
    
    # Forward Euler
    bin.Ds += dDs_dt * dt
    bin.Dl += dDl_dt * dt
    
    # Ensure non-negative
    bin.Ds = max(0.0, bin.Ds)
    bin.Dl = max(0.0, bin.Dl)