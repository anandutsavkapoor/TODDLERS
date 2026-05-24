"""
Cloud Density Profile Module
============================

This module implements various density profiles for molecular clouds, providing
a framework for modeling cloud density structures in TODDLERS simulations.

The module includes:
- Uniform density
- Uniform density with cavity
- Bonnor-Ebert sphere
- Modified Bonnor-Ebert profile
- Plummer profile
- Gaussian profile
- Smoothed Modified Bonnor-Ebert profile

Each profile maintains mass conservation and implements methods for:
- Density calculation
- Mass enclosed calculation
- Binding energy computation
- Pressure estimation
"""

from .imports import np, odeint, cumtrapz, quad, brentq, interp1d, warnings, IntegrationWarning
from .constants import *

class CloudDensityProfile:
    """
    Base class for molecular cloud density profiles.
    
    This abstract base class defines the interface and common functionality
    for all cloud density profile implementations. Each profile maintains
    mass conservation and provides methods for key physical quantities.

    Parameters
    ----------
    M_cl : float
        Total cloud mass in grams
    n_avg : float
        Average number density in cm^-3

    Attributes
    ----------
    M_cl : float
        Total cloud mass in grams
    n_avg : float
        Average number density in cm^-3
    rho_avg : float
        Average mass density in g/cm^3
    R_cl : float
        Cloud radius in cm
    R_cl_min : float
        Minimum radius (to avoid singularity) in cm
    is_uniform : bool
        Whether the profile has constant density
    """

    def __init__(self, M_cl, n_avg):
        """
        Initialize base cloud density profile.

        Parameters
        ----------
        M_cl : float
            Total cloud mass in grams
        n_avg : float
            Average number density in cm^-3
        """
        self.M_cl = M_cl
        self.n_avg = n_avg
        self.rho_avg = n_avg * MU_N
        self.R_cl = self.calculate_radius()
        self.R_cl_min = 1e-16 * self.R_cl
        self.is_uniform = False

    def calculate_radius(self):
        """
        Calculate cloud radius based on mass and average density.

        Returns
        -------
        float
            Cloud radius in cm

        Notes
        -----
        Uses the relation R = (3M/4πρ)^(1/3) for spherical clouds.
        """
        return (3 * self.M_cl / (4 * np.pi * self.rho_avg))**(1/3)

    def density(self, r):
        """
        Calculate density at given radius.

        Parameters
        ----------
        r : float or array-like
            Radius in cm

        Returns
        -------
        float or array-like
            Density in g/cm^3

        Raises
        ------
        NotImplementedError
            If not implemented by derived class
        """
        raise NotImplementedError("Subclasses must implement this method")

    def mass_enclosed(self, r):
        """
        Calculate mass enclosed within radius r.

        Parameters
        ----------
        r : float
            Radius in cm

        Returns
        -------
        float
            Mass enclosed in grams

        Raises
        ------
        NotImplementedError
            If not implemented by derived class
        """
        raise NotImplementedError("Subclasses must implement this method")

    def pressure(self, r):
        """
        Calculate thermal pressure at given radius.

        Assumes neutral gas at temperature T_NEUTRAL.

        Parameters
        ----------
        r : float
            Radius in cm

        Returns
        -------
        float
            Pressure in dyne/cm^2
        """
        rho = self.density(r)
        n = rho / MU_N
        return n * K_BOLTZMANN * T_NEUTRAL

    def _numerical_binding_energy(self):
        """
        Calculate gravitational binding energy numerically.

        Uses integration of U = -∫[0 to R_cl] (4πr²ρ(r)GM(r)/r)dr

        Returns
        -------
        float
            Binding energy in ergs

        Raises
        ------
        ValueError
            If integration fails or result is unphysical
        """
        def integrand(r):
            return (4 * np.pi * r**2 * self.density(r) * self.mass_enclosed(r)) / r

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", IntegrationWarning)
                U, abserr = quad(integrand, self.R_cl_min, self.R_cl, 
                               epsabs=1e-10, epsrel=1e-9)
            
            if not np.isfinite(U):
                raise ValueError("Integration result is not finite")
            
            binding_energy = -G * U
            
            if binding_energy >= 0:
                raise ValueError("Binding energy calculation resulted in a non-negative value")
            
            return binding_energy

        except (IntegrationWarning, ValueError) as e:
            raise ValueError(f"Binding energy calculation failed: {str(e)}")

    def get_parameters(self):
        """
        Get basic profile parameters.

        Returns
        -------
        dict
            Dictionary containing basic profile parameters
        """
        return {
            "M_cl": self.M_cl,
            "n_avg": self.n_avg,
            "R_cl": self.R_cl
        }

class UniformDensity(CloudDensityProfile):
    """
    Uniform density profile for molecular clouds.
    
    Implements a constant density profile where ρ(r) = ρ_avg for all r ≤ R_cl.
    
    Parameters
    ----------
    M_cl : float
        Total cloud mass in grams
    n_avg : float
        Average number density in cm^-3
        
    Notes
    -----
    Key properties:
    - Density is constant: ρ(r) = ρ_avg
    - Mass enclosed scales as M(r) ∝ r³
    - Binding energy is analytical: U = -3GM²/(5R)
    """

    def density(self, r):
        """
        Calculate density at given radius.

        Parameters
        ----------
        r : float or array-like
            Radius in cm

        Returns
        -------
        float or array-like
            Constant density ρ_avg in g/cm^3
        """
        return self.rho_avg

    def mass_enclosed(self, r):
        """
        Calculate mass enclosed within radius r.

        Parameters
        ----------
        r : float
            Radius in cm

        Returns
        -------
        float
            Mass enclosed in grams
        """
        return self.M_cl * (r / self.R_cl)**3
    
    def binding_energy(self):
        """
        Calculate gravitational binding energy analytically.

        Returns
        -------
        float
            Binding energy in ergs
        """
        return -3 * G * self.M_cl**2 / (5 * self.R_cl)

    def get_parameters(self):
        """
        Get profile parameters.

        Returns
        -------
        dict
            Dictionary containing profile parameters
        """
        params = super().get_parameters()
        params["type"] = "uniform"
        return params

class UniformDensityWithCavity(CloudDensityProfile):
    """
    Uniform density profile with central cavity.
    
    Implements a constant density profile with a central spherical cavity:
    ρ(r) = 0 for r < R_cav
    ρ(r) = ρ_gas for R_cav ≤ r ≤ R_cl
    
    Parameters
    ----------
    M_gas : float
        Mass of gas (excluding cavity) in grams
    n_avg : float
        Average number density in gas region in cm^-3
    R_cav : float
        Cavity radius in cm
        
    Attributes
    ----------
    M_gas : float
        Mass of gas in grams
    rho_gas : float
        Gas density in g/cm^3
    R_cav : float
        Cavity radius in cm
    V_gas : float
        Volume of gas region in cm^3
        
    Notes
    -----
    - Must have R_cav < R_cl
    - Density jumps discontinuously at R_cav
    - Binding energy accounts for cavity
    """

    def __init__(self, M_gas, n_avg, R_cav):
        """
        Initialize uniform density profile with cavity.

        Parameters
        ----------
        M_gas : float
            Mass of gas in grams
        n_avg : float
            Average number density in cm^-3
        R_cav : float
            Cavity radius in cm

        Raises
        ------
        AssertionError
            If cavity radius exceeds cloud radius
        """
        self.M_gas = M_gas
        self.n_avg = n_avg
        self.R_cav = R_cav
        self.rho_gas = n_avg * MU_N

        self.R_cl = self.calculate_radius()

        assert R_cav < self.R_cl, \
            f"Cavity radius ({R_cav}) must be smaller than cloud radius ({self.R_cl})"

        self.V_gas = (4/3) * np.pi * (self.R_cl**3 - R_cav**3)

    def calculate_radius(self):
        """
        Calculate cloud radius accounting for cavity.

        Returns
        -------
        float
            Cloud radius in cm
        """
        V_gas = self.M_gas / self.rho_gas
        return ((3 * V_gas / (4 * np.pi) + self.R_cav**3)**(1/3))

    def density(self, r):
        """
        Calculate density at given radius.

        Parameters
        ----------
        r : float or array-like
            Radius in cm

        Returns
        -------
        float or array-like
            Density in g/cm^3
        """
        return 0 if r < self.R_cav else self.rho_gas

    def mass_enclosed(self, r):
        """
        Calculate mass enclosed within radius r.

        Parameters
        ----------
        r : float
            Radius in cm

        Returns
        -------
        float
            Mass enclosed in grams
        """
        if r <= self.R_cav:
            return 0
        elif r >= self.R_cl:
            return self.M_gas
        else:
            return max(0, self.rho_gas * (4/3) * np.pi * (r**3 - self.R_cav**3))

    def binding_energy(self):
        """
        Calculate gravitational binding energy for a uniform density 
        sphere with a central cavity.
        
        Returns:
            float: Binding energy in ergs
        """
        # For a uniform sphere with a cavity, the binding energy is:
        # U = -(3/5)GM^2/R * [1 - (r_cav/R)^5]
        cavity_term = (self.R_cav / self.R_cl)**5
        U = -(3.0/5.0) * G * (self.M_gas**2) / self.R_cl * (1.0 - cavity_term)
        
        return U

    def get_parameters(self):
        """
        Get profile parameters.

        Returns
        -------
        dict
            Dictionary containing profile parameters
        """
        return {
            "type": "uniform_with_cavity",
            "M_gas": self.M_gas,
            "n_avg": self.n_avg,
            "R_cav": self.R_cav,
            "R_cl": self.R_cl,
            "rho_gas": self.rho_gas
        }

class BonnorEbertSphere(CloudDensityProfile):
    """
    Bonnor-Ebert sphere density profile.
    The density profile is determined by solving Lane-Emden equation with n=∞.
    
    Parameters
    ----------
    M_cl : float
        Total cloud mass in grams
    n_avg : float
        Average number density in cm^-3
    xi_max : float, optional
        Dimensionless outer radius, default 6.5
        
    Attributes
    ----------
    xi_max : float
        Dimensionless outer radius
    rho_c : float
        Central density in g/cm^3
    density_interp : callable
        Interpolation function for density profile
    mass_interp : callable
        Interpolation function for enclosed mass
        
    Notes
    -----
    - Profile normalized to match given mass and radius
    - Uses numerical solution of Lane-Emden equation
    """

    def __init__(self, M_cl, n_avg, xi_max=6.5):
        """
        Initialize Bonnor-Ebert sphere profile.

        Parameters
        ----------
        M_cl : float
            Total cloud mass in grams
        n_avg : float
            Average number density in cm^-3
        xi_max : float, optional
            Dimensionless outer radius, default 6.5
        """
        super().__init__(M_cl, n_avg)
        self.xi_max = xi_max
        self.solve_be_equation()

    def solve_be_equation(self):
        """
        Solve Lane-Emden equation for isothermal sphere.

        Creates interpolation functions for density and mass profiles.
        Normalizes solution to match total mass.
        """
        def be_ode(y, xi):
            phi, dphi_dxi = y
            if xi == 0:
                return [dphi_dxi, 0]
            return [dphi_dxi, (np.exp(-phi) - 2 * dphi_dxi / xi)]

        xi = np.logspace(-16, np.log10(self.xi_max), 10000, endpoint=True)
        solution = odeint(be_ode, [0, 0], xi)
        phi = solution[:, 0]
        
        density_profile = np.exp(-phi)
        mass_profile = cumtrapz(4 * np.pi * xi**2 * density_profile, xi, initial=0)
        
        self.rho_c = self.M_cl / (self.R_cl**3 * mass_profile[-1] / self.xi_max**3)
        
        self.density_interp = interp1d(xi, density_profile, kind='cubic', 
                                     bounds_error=False,
                                     fill_value=(density_profile[0], density_profile[-1]))
        self.mass_interp = interp1d(xi, mass_profile, kind='cubic',
                                   bounds_error=False,
                                   fill_value=(mass_profile[0], mass_profile[-1]))

    def density(self, r):
        """
        Calculate density at given radius.

        Parameters
        ----------
        r : float or array-like
            Radius in cm

        Returns
        -------
        float or array-like
            Density in g/cm^3
        """
        xi = r * self.xi_max / self.R_cl
        return self.rho_c * self.density_interp(xi)

    def mass_enclosed(self, r):
        """
        Calculate mass enclosed within radius r.

        Parameters
        ----------
        r : float
            Radius in cm

        Returns
        -------
        float
            Mass enclosed in grams
        """
        xi = r * self.xi_max / self.R_cl
        return self.M_cl * self.mass_interp(xi) / self.mass_interp(self.xi_max)

    def binding_energy(self):
        """
        Calculate gravitational binding energy numerically.

        Returns
        -------
        float
            Binding energy in ergs
        """
        return self._numerical_binding_energy()

    def get_parameters(self):
        """
        Get profile parameters.

        Returns
        -------
        dict
            Dictionary containing profile parameters
        """
        params = super().get_parameters()
        params.update({
            "type": "bonnor_ebert",
            "xi_max": self.xi_max
        })
        return params

class ModifiedBonnorEbertProfile(CloudDensityProfile):
    """
    Modified Bonnor-Ebert density profile.
    
    Implements a piecewise profile with constant central density and
    power-law decay in outer regions:
    ρ(r) = ρ_0 for r ≤ R_0
    ρ(r) = ρ_0 (r/R_0)^(-α) for R_0 < r ≤ R_cl
    
    Parameters
    ----------
    M_cl : float
        Total cloud mass in grams
    n_avg : float
        Average number density in cm^-3
    alpha : float
        Power-law index for outer region
    rho_0 : float, optional
        Central density in g/cm^3. If None, calculated to be 3*rho_avg
    rho_amb : float, optional
        Ambient density in g/cm^3, default 1e-23
        
    Attributes
    ----------
    alpha : float
        Power-law index
    rho_0 : float
        Central density in g/cm^3
    rho_amb : float
        Ambient density in g/cm^3
    R_0 : float
        Transition radius in cm where density profile changes
        
    Notes
    -----
    - Transition radius R_0 calculated to ensure mass conservation
    - Profile continuous but not differentiable at R_0
    """
    
    def __init__(self, M_cl, n_avg, alpha, rho_0=None, rho_amb=1e-23):
        super().__init__(M_cl, n_avg)
        self.alpha = alpha
        self.rho_0 = rho_0 if rho_0 is not None else (3 * self.rho_avg if alpha != 0 else self.rho_avg)
        self.rho_amb = rho_amb
        self.R_0 = self.calculate_R_0()

    def calculate_R_0(self):
        """
        Calculate transition radius ensuring mass conservation.
        
        Uses numerical root finding to determine R_0 such that the total
        mass equals M_cl.

        Returns
        -------
        float
            Transition radius in cm
            
        Raises
        ------
        ValueError
            If root finding fails to converge
        """
        def mass_integral(R_0):
            def integrand(R):
                return 4 * np.pi * R**2 * self.density_function(R, R_0)
            
            total_mass, _ = quad(integrand, 0, self.R_cl)
            return total_mass - self.M_cl

        return brentq(mass_integral, self.R_cl_min, self.R_cl)

    def density_function(self, R, R_0):
        """
        Calculate density at radius R given transition radius R_0.
        
        Parameters
        ----------
        R : float or array-like
            Radius in cm
        R_0 : float
            Transition radius in cm
            
        Returns
        -------
        float or array-like
            Density in g/cm^3
        """
        if R <= R_0:
            return self.rho_0
        elif R_0 < R <= self.R_cl:
            return self.rho_0 * (R / R_0)**(-self.alpha)
        else:
            return self.rho_amb

    def density(self, R):
        """
        Calculate density at given radius.

        Parameters
        ----------
        R : float or array-like
            Radius in cm

        Returns
        -------
        float or array-like
            Density in g/cm^3
        """
        return self.density_function(R, self.R_0)

    def mass_enclosed(self, R):
        """
        Calculate mass enclosed within radius R using numerical integration.

        Parameters
        ----------
        R : float
            Radius in cm

        Returns
        -------
        float
            Mass enclosed in grams
        """
        def integrand(r):
            return 4 * np.pi * r**2 * self.density(r)
        
        mass, _ = quad(integrand, 0, R)
        return mass

    def binding_energy(self):
        """
        Calculate binding energy numerically.

        Returns
        -------
        float
            Binding energy in ergs
        """
        return self._numerical_binding_energy()

    def get_parameters(self):
        """
        Get profile parameters.

        Returns
        -------
        dict
            Dictionary containing profile parameters
        """
        params = super().get_parameters()
        params.update({
            "type": "modified_bonnor_ebert",
            "alpha": self.alpha,
            "rho_0": self.rho_0,
            "rho_amb": self.rho_amb,
            "R_0": self.R_0
        })
        return params

class SmoothedModifiedBonnorEbertProfile(CloudDensityProfile):
    """
    Smoothed version of Modified Bonnor-Ebert profile.
    
    Uses hyperbolic tangent functions to smooth transitions between regions:
    - Core to power-law transition at R_0
    - Power-law to ambient transition at R_cl
    
    Parameters
    ----------
    M_cl : float
        Total cloud mass in grams
    n_avg : float
        Average number density in cm^-3
    alpha : float
        Power-law index for outer region
    rho_0 : float, optional
        Central density in g/cm^3
    rho_amb : float, optional
        Ambient density in g/cm^3, default 1e-23
    smoothing_factor : float, optional
        Controls width of transition regions, default 0.1
        
    Notes
    -----
    - Continuous and differentiable at all radii
    - Preserves total mass
    """

    def __init__(self, M_cl, n_avg, alpha, rho_0=None, rho_amb=1e-23, smoothing_factor=0.1):
        super().__init__(M_cl, n_avg)
        self.alpha = alpha
        self.rho_0 = rho_0 if rho_0 is not None else (3 * self.rho_avg if alpha != 0 else self.rho_avg)
        self.rho_amb = rho_amb
        self.smoothing_factor = smoothing_factor
        self.R_0 = self.calculate_R_0()

    def smooth_step(self, x, edge, width):
        """
        Create smooth transition using hyperbolic tangent.

        Parameters
        ----------
        x : float or array-like
            Input value
        edge : float
            Location of transition
        width : float
            Width of transition region

        Returns
        -------
        float or array-like
            Smoothed step function value between 0 and 1
        """
        return 0.5 * (1 + np.tanh((x - edge) / (width * edge)))

    def calculate_R_0(self):
        """
        Calculate transition radius ensuring mass conservation.

        Returns
        -------
        float
            Transition radius in cm

        Notes
        -----
        Uses numerical root finding with the smoothed density profile.
        """
        def mass_integral(R_0):
            self.R_0 = R_0
            total_mass, _ = quad(lambda r: 4 * np.pi * r**2 * self.density(r), 0, self.R_cl)
            return total_mass - self.M_cl

        return brentq(mass_integral, self.R_cl_min, self.R_cl)

    def density(self, R):
        """
        Calculate smoothed density profile.

        Parameters
        ----------
        R : float or array-like
            Radius in cm

        Returns
        -------
        float or array-like
            Density in g/cm^3

        Notes
        -----
        Uses smooth transitions between:
        1. Core (constant density)
        2. Power-law region
        3. Ambient medium
        """
        if R < self.R_cl_min:
            return self.rho_0

        core_to_power = self.smooth_step(R, self.R_0, self.smoothing_factor)
        power_to_ambient = self.smooth_step(R, self.R_cl, self.smoothing_factor)
        
        rho_core = self.rho_0
        rho_power = self.rho_0 * (R / self.R_0)**(-self.alpha)
        
        rho = (1 - core_to_power) * rho_core + \
              core_to_power * (1 - power_to_ambient) * rho_power + \
              power_to_ambient * self.rho_amb
        
        return rho

    def binding_energy(self):
        """
        Calculate binding energy numerically.

        Returns
        -------
        float
            Binding energy in ergs
        """
        return self._numerical_binding_energy()

    def mass_enclosed(self, R):
        """
        Calculate mass enclosed within radius R.

        Parameters
        ----------
        R : float
            Radius in cm

        Returns
        -------
        float
            Mass enclosed in grams
        """
        mass, _ = quad(lambda r: 4 * np.pi * r**2 * self.density(r), 0, R)
        return mass

    def get_parameters(self):
        """
        Get profile parameters.

        Returns
        -------
        dict
            Dictionary containing profile parameters
        """
        params = super().get_parameters()
        params.update({
            "type": "smoothed_modified_bonnor_ebert",
            "alpha": self.alpha,
            "rho_0": self.rho_0,
            "rho_amb": self.rho_amb,
            "R_0": self.R_0,
            "smoothing_factor": self.smoothing_factor
        })
        return params


class PlummerProfile(CloudDensityProfile):
    """
    Modified Plummer profile that ensures all mass is contained within R_cl.
    
    The profile is truncated at R_cl and renormalized to ensure mass conservation.
    Uses the form:

    - rho(r) = rho_0 * (1 + (r/a)^2)^(-5/2) for r <= R_cl
    - rho(r) = 0 for r > R_cl

    Attributes:
        a (float): Scale radius, defined as a fraction of R_cl
        rho_0 (float): Central density, calculated to ensure mass conservation
    """
    
    def __init__(self, M_cl, n_avg, a_fraction=0.1):
        """
        Initialize Plummer profile.
        
        Args:
            M_cl (float): Total cloud mass in g
            n_avg (float): Average number density in cm^-3
            a_fraction (float): Scale radius as fraction of R_cl
        """
        super().__init__(M_cl, n_avg)
        self.a = a_fraction * self.R_cl
        self.rho_0 = self._calculate_central_density()
        
    def _calculate_central_density(self):
        """
        Calculate central density to ensure mass conservation within R_cl.
        
        Returns:
            float: Central density in g/cm^3
        """
        # Mass within R_cl for unnormalized profile with rho_0 = 1
        def mass_integrand(r):
            return 4 * np.pi * r**2 / (1 + (r/self.a)**2)**(5/2)
        
        unnorm_mass, _ = quad(mass_integrand, 0, self.R_cl)
        
        # Calculate rho_0 needed for total mass = M_cl
        return self.M_cl / unnorm_mass
        
    def density(self, r):
        """Calculate density at radius r."""
        if isinstance(r, (list, np.ndarray)):
            r = np.asarray(r)
            result = MU_N * N_ISM * np.ones_like(r, dtype=float)
            mask = r <= self.R_cl
            result[mask] = self.rho_0 / (1 + (r[mask]/self.a)**2)**(5/2)
            return result
        else:
            return self.rho_0 / (1 + (r/self.a)**2)**(5/2) if r <= self.R_cl else MU_N * N_ISM
    
    def mass_enclosed(self, r):
        """Calculate mass enclosed within radius r."""
        if r >= self.R_cl:
            return self.M_cl
            
        def mass_integrand(x):
            return 4 * np.pi * x**2 * self.density(x)
            
        mass, _ = quad(mass_integrand, 0, min(r, self.R_cl))
        return mass

    def binding_energy(self):
        """
        Calculate binding energy numerically.

        Returns
        -------
        float
            Binding energy in ergs
        """
        return self._numerical_binding_energy()
    
    def get_parameters(self):
        """Get profile parameters."""
        params = super().get_parameters()
        params.update({
            "type": "plummer",
            "a": self.a,
            "rho_0": self.rho_0
        })
        return params


class GaussianProfile(CloudDensityProfile):
    """
    Modified Gaussian profile that ensures all mass is contained within R_cl.
    
    The profile is truncated at R_cl and renormalized to ensure mass conservation.
    Uses the form:

    - rho(r) = rho_0 * exp(-r^2/(2*sigma^2)) for r <= R_cl
    - rho(r) = 0 for r > R_cl

    Attributes:
        sigma (float): Gaussian width, defined as fraction of R_cl
        rho_0 (float): Central density, calculated to ensure mass conservation
    """
    
    def __init__(self, M_cl, n_avg, sigma_fraction=0.2):
        """
        Initialize Gaussian profile.
        
        Args:
            M_cl (float): Total cloud mass in g
            n_avg (float): Average number density in cm^-3
            sigma_fraction (float): Gaussian width as fraction of R_cl
        """
        super().__init__(M_cl, n_avg)
        self.sigma = sigma_fraction * self.R_cl
        self.rho_0 = self._calculate_central_density()
        
    def _calculate_central_density(self):
        """
        Calculate central density to ensure mass conservation within R_cl.
        
        Returns:
            float: Central density in g/cm^3
        """
        # Mass within R_cl for unnormalized profile with rho_0 = 1
        def mass_integrand(r):
            return 4 * np.pi * r**2 * np.exp(-r**2/(2*self.sigma**2))
            
        unnorm_mass, _ = quad(mass_integrand, 0, self.R_cl)
        
        # Calculate rho_0 needed for total mass = M_cl
        return self.M_cl / unnorm_mass
        
    def density(self, r):
        """Calculate density at radius r."""
        if isinstance(r, (list, np.ndarray)):
            r = np.asarray(r)
            result = MU_N * N_ISM * np.ones_like(r, dtype=float)
            mask = r <= self.R_cl
            result[mask] = self.rho_0 * np.exp(-r[mask]**2/(2*self.sigma**2))
            return result
        else:
            return self.rho_0 * np.exp(-r**2/(2*self.sigma**2)) if r <= self.R_cl else MU_N * N_ISM
    
    def mass_enclosed(self, r):
        """Calculate mass enclosed within radius r."""
        if r >= self.R_cl:
            return self.M_cl
            
        def mass_integrand(x):
            return 4 * np.pi * x**2 * self.density(x)
            
        mass, _ = quad(mass_integrand, 0, min(r, self.R_cl))
        return mass

    def binding_energy(self):
        """
        Calculate binding energy numerically.

        Returns
        -------
        float
            Binding energy in ergs
        """
        return self._numerical_binding_energy()

    def get_parameters(self):
        """Get profile parameters."""
        params = super().get_parameters()
        params.update({
            "type": "gaussian",
            "sigma": self.sigma,
            "rho_0": self.rho_0
        })
        return params

def create_density_profile(profile_type, M_cl, n_avg, **kwargs):
    """
    Factory function to create density profile objects.
    
    Parameters
    ----------
    profile_type : str
        Type of density profile to create. Options:
        - 'uniform'
        - 'uniform_with_cavity' 
        - 'bonnor_ebert'
        - 'modified_bonnor_ebert'
        - 'plummer'
        - 'gaussian'
        - 'smooth_modified_bonnor_ebert'
    M_cl : float
        Total cloud mass in grams
    n_avg : float
        Average number density in cm^-3
    **kwargs : dict
        Additional parameters specific to each profile type
        
    Returns
    -------
    CloudDensityProfile
        Instance of requested density profile class
        
    Raises
    ------
    ValueError
        If profile_type is not recognized or required parameters missing
    """
    if profile_type == 'uniform':
        return UniformDensity(M_cl, n_avg)
    elif profile_type == 'uniform_with_cavity':
        R_cav = kwargs.get('R_cav')
        if R_cav is None:
            raise ValueError("R_cav must be provided for uniform_with_cavity profile")
        return UniformDensityWithCavity(M_cl, n_avg, R_cav)
    elif profile_type == 'bonnor_ebert':
        return BonnorEbertSphere(M_cl, n_avg, kwargs.get('xi_max', 6.5))
    elif profile_type == 'modified_bonnor_ebert':
        return ModifiedBonnorEbertProfile(
            M_cl, n_avg,
            alpha=kwargs.get('alpha', 2),
            rho_0=kwargs.get('rho_0', 1.67e-19),
            rho_amb=kwargs.get('rho_amb', 1e-23)
        )
    elif profile_type == 'plummer':
        return PlummerProfile(M_cl, n_avg, kwargs.get('a_fraction', 0.1))
    elif profile_type == 'gaussian':
        return GaussianProfile(M_cl, n_avg, kwargs.get('sigma_fraction', 0.2))
    elif profile_type == 'smooth_modified_bonnor_ebert':
        return SmoothedModifiedBonnorEbertProfile(
            M_cl, n_avg,
            alpha=kwargs.get('alpha', 2),
            rho_0=kwargs.get('rho_0', 1.67e-19),
            rho_amb=kwargs.get('rho_amb', 1e-23),
            smoothing_factor=kwargs.get('smoothing_factor', 0.1)
        )
    else:
        raise ValueError(f"Unknown profile type: {profile_type}")