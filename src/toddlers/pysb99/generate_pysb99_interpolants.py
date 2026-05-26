#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_pysb99_interpolants.py
================================

Generate stellar population synthesis interpolants using pySB99 for custom IMFs
and custom stellar mixes. This replaces the previous workflow that required 
downloading files from the SB99 website.

Key Features:
- Support for custom IMF parameters (mass limits and exponents)
- Support for custom stellar distributions (individual star counts)
- Direct generation of interpolants from pysb99
- Compatible with existing TODDLERS stellar_feedback infrastructure

Author: Anand Utsav KAPOOR
Created: 2025
"""

import os
import sys
import json
import numpy as np
import pickle
from datetime import datetime
from .._interp_compat import Interp2DLinear
from typing import Dict, List, Optional, Tuple
import warnings


def _make_interp2d(time_grid, Z_values, data, kind='linear', fill_value=None):
    """Build a linear 2-D (time, Z) interpolant, handling the single-metallicity case.

    Returns an :class:`Interp2DLinear` (the future-proof, picklable replacement for the
    removed ``scipy.interpolate.interp2d``; same ``(x=time, y=Z, z)`` layout and call
    signature). A linear interpolant needs at least 2 points in the Z direction, so when
    only one metallicity is requested we pad with a duplicate row offset by a negligible
    epsilon; the interpolant then returns identical values for any Z. ``kind`` is accepted
    for backward compatibility but only 'linear' is supported.
    """
    if kind != 'linear':
        raise ValueError(f"Interp2DLinear supports only kind='linear', got {kind!r}")
    Z_arr = list(Z_values)
    dat = np.asarray(data)
    if len(Z_arr) == 1:
        Z_arr = [Z_arr[0], Z_arr[0] + 1e-10]
        dat = np.vstack([dat, dat])
    return Interp2DLinear(time_grid, Z_arr, dat, fill_value=fill_value)

# Add pysb99 to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), 'pysb99'))

from .pysb99_core import (
    StellarPopulationConfig,
    run_population_synthesis,
    StellarPopulationResults
)

# Metallicity mapping from pysb99 string identifiers to numerical Z values
# Based on Table 1 from the stellar evolutionary models paper
METALLICITY_MAPPING = {
    'MWC': 0.020,   # GalC: Z = 0.02 (Yus+22)
    'MW': 0.014,    # MW: Z = 0.014 (Eks+12)
    'LMC': 0.006,   # LMC: Z = 0.006 (Egg+21)
    'SMC': 0.002,   # SMC: Z = 0.002 (Geo+13)
    'IZw18': 0.0004, # IZw18: Z = 0.0004 (Gro+19)
    'Z0': 0.0       # Z0: Z = 0.0 (Mur+21)
}

# Inverse mapping for Z to metallicity string
Z_TO_METALLICITY = {v: k for k, v in METALLICITY_MAPPING.items()}

# Default metallicities to use for interpolation
DEFAULT_METALLICITIES = ['IZw18', 'SMC', 'LMC', 'MW', 'MWC']
DEFAULT_Z_VALUES = [METALLICITY_MAPPING[m] for m in DEFAULT_METALLICITIES]


class PySB99InterpolantGenerator:
    """
    Generate interpolants for stellar population synthesis using pySB99.
    
    This class provides functionality to:
    1. Run pySB99 for different metallicities and IMF configurations
    2. Extract relevant physical quantities
    3. Create 2D interpolation functions (time, metallicity)
    4. Save interpolants in format compatible with TODDLERS
    """
    
    def __init__(self, 
                 metallicities: Optional[List[str]] = None,
                 time_start_myr: float = 0.01,
                 time_end_myr: float = 31.0,
                 time_step_myr: float = 0.1,
                 spectral_library: str = 'FW',
                 rotation: bool = False,
                 run_speed_mode: str = 'DEFAULT'):
        """
        Initialize the interpolant generator.
        
        Parameters
        ----------
        metallicities : list of str, optional
            List of metallicity identifiers ('MWC', 'MW', 'LMC', 'SMC', 'IZw18', 'Z0')
            Default: ['IZw18', 'SMC', 'LMC', 'MW', 'MWC']
        time_start_myr : float
            Start time in Myr (default: 0.01)
        time_end_myr : float
            End time in Myr (default: 31.0)
        time_step_myr : float
            Time step in Myr (default: 0.1)
        spectral_library : str
            'FW' (Fastwind) or 'WM' (WMbasic)
        rotation : bool
            Include stellar rotation
        run_speed_mode : str
            'FAST', 'DEFAULT', or 'HIGH_RES'
        """
        self.metallicities = metallicities or DEFAULT_METALLICITIES
        self.Z_values = [METALLICITY_MAPPING[m] for m in self.metallicities]
        self.time_start_myr = time_start_myr
        self.time_end_myr = time_end_myr
        self.time_step_myr = time_step_myr
        self.spectral_library = spectral_library
        self.rotation = rotation
        self.run_speed_mode = run_speed_mode
        
        # Convert time to years for pysb99
        self.time_start_yr = time_start_myr * 1e6
        self.time_end_yr = time_end_myr * 1e6
        self.time_step_yr = time_step_myr * 1e6
        
        # Storage for results
        self.results_dict = {}
        self.time_grid = None
        
    def run_for_metallicity(self,
                           metallicity: str,
                           total_mass: float = 1e6,
                           imf_exponents: Optional[List[float]] = None,
                           imf_mass_limits: Optional[Tuple[float, ...]] = None,
                           custom_star_numbers: Optional[Dict[float, int]] = None,
                           auto_extend_imf: bool = True,
                           verbose: bool = True) -> StellarPopulationResults:
        """
        Run pySB99 for a single metallicity.

        Parameters
        ----------
        metallicity : str
            Metallicity identifier
        total_mass : float
            Total stellar mass in solar masses (ignored if custom_star_numbers provided)
        imf_exponents : list of float, optional
            IMF power law exponents (e.g., [1.3, 2.3] for Kroupa)
        imf_mass_limits : tuple of float, optional
            Mass limits for IMF segments (e.g., (0.1, 0.5, 120.0))
        custom_star_numbers : dict, optional
            Custom star distribution {mass: count}
        auto_extend_imf : bool
            If True (default), extend IMF upper limit to the track grid maximum.
            Set to False to strictly respect the user-specified upper mass limit.
        verbose : bool
            Print progress

        Returns
        -------
        StellarPopulationResults
            Results from pySB99
        """
        if verbose:
            Z_val = METALLICITY_MAPPING[metallicity]
            print(f"Running pySB99 for {metallicity} (Z={Z_val:.4f})...")

        # Set up configuration
        config = StellarPopulationConfig(
            total_mass=total_mass,
            metallicity=metallicity,
            spectral_library=self.spectral_library,
            rotation=self.rotation,
            time_start=self.time_start_yr,
            time_end=self.time_end_yr,
            time_step=self.time_step_yr,
            run_speed_mode=self.run_speed_mode,
            custom_star_numbers=custom_star_numbers,
            auto_extend_imf=auto_extend_imf
        )

        # Set IMF parameters if provided
        if imf_exponents is not None:
            config.imf_exponents = imf_exponents
        if imf_mass_limits is not None:
            config.imf_mass_limits = imf_mass_limits
        
        # Run synthesis
        results = run_population_synthesis(config)
        
        # Store results
        self.results_dict[metallicity] = results
        
        # Set time grid from first run
        if self.time_grid is None:
            self.time_grid = results.times / 1e6  # Convert to Myr
        
        return results
    
    def run_all_metallicities(self,
                             total_mass: float = 1e6,
                             imf_exponents: Optional[List[float]] = None,
                             imf_mass_limits: Optional[Tuple[float, ...]] = None,
                             custom_star_numbers: Optional[Dict[float, int]] = None,
                             auto_extend_imf: bool = True,
                             verbose: bool = True):
        """
        Run pySB99 for all metallicities.

        Parameters
        ----------
        total_mass : float
            Total stellar mass in solar masses
        imf_exponents : list of float, optional
            IMF power law exponents
        imf_mass_limits : tuple of float, optional
            Mass limits for IMF segments
        custom_star_numbers : dict, optional
            Custom star distribution
        auto_extend_imf : bool
            If True (default), extend IMF upper limit to the track grid maximum.
            Set to False to strictly respect the user-specified upper mass limit.
        verbose : bool
            Print progress
        """

        self._requested_total_mass = total_mass
        self._using_custom_distribution = (custom_star_numbers is not None)

        for metallicity in self.metallicities:
            self.run_for_metallicity(
                metallicity=metallicity,
                total_mass=total_mass,
                imf_exponents=imf_exponents,
                imf_mass_limits=imf_mass_limits,
                custom_star_numbers=custom_star_numbers,
                auto_extend_imf=auto_extend_imf,
                verbose=verbose
            )
    
    def create_interpolants(self, verbose: bool = True) -> Dict:
        """
        Create 2D interpolation functions (time, Z) for all quantities.
        
        This method matches the approach in generate_SB99_interpolants.py:
        - Main quantities stored in log space
        - Derivatives computed by oversampling, converting to linear, then using np.gradient
        - Derivative interpolants in linear space (dynes/Myr and erg/s/Myr)
        
        Returns
        -------
        dict
            Dictionary containing interpolation functions for:
            - L_mech: Mechanical luminosity (wind power) [log space]
            - F_ram: Ram pressure (wind momentum) [log space]
            - L_bolo: Bolometric luminosity [log space]
            - L_ion: Ionizing luminosity [log space]
            - Q_ion: Ionizing photon rate [log space]
            - L_LyW: Lyman-Werner luminosity [log space]
            - Q_HeI_to_HI: HeI/HI ionizing photon ratio [log space]
            - Q_HeII_to_HI: HeII/HI ionizing photon ratio [log space]
            - mean_ionizing_photon_energy: Mean energy of ionizing photons [log space]
            - f_wind_power_rate: dL_mech/dt [linear space: erg/s/Myr]
            - f_wind_momentum_rate: dF_ram/dt [linear space: dyne/Myr]
        """
        if not self.results_dict:
            raise ValueError("No results available. Run simulations first.")
        
        if verbose:
            print("Creating interpolation functions...")
        
        # Physical constants
        Msun = 1.989e33  # g
        yr2s = 365 * 24 * 60 * 60  # s
        
        # Initialize arrays for each quantity
        n_time = len(self.time_grid)
        n_Z = len(self.Z_values)
        
        # Wind properties (per solar mass of stars formed)
        all_wind_power = np.zeros((n_Z, n_time))
        all_wind_momentum = np.zeros((n_Z, n_time))
        
        # Luminosities
        all_L_bolo = np.zeros((n_Z, n_time))
        all_L_ion_HI = np.zeros((n_Z, n_time))
        all_L_ion_HeI = np.zeros((n_Z, n_time))
        all_L_ion_HeII = np.zeros((n_Z, n_time))
        
        # Ionizing fluxes
        all_Q_HI = np.zeros((n_Z, n_time))
        all_Q_HeI = np.zeros((n_Z, n_time))
        all_Q_HeII = np.zeros((n_Z, n_time))
        
        # Spectral hardness ratios
        all_QratHe1H = np.zeros((n_Z, n_time))
        all_QratHe2H = np.zeros((n_Z, n_time))
        
        # Lyman-Werner flux (at 1000 Angstroms)
        all_LyW_flux = np.zeros((n_Z, n_time))
        
        # Mean ionizing photon energy
        all_mean_ion_energy = np.zeros((n_Z, n_time))
        
        # Extract data for each metallicity
        for i, (metallicity, Z_val) in enumerate(zip(self.metallicities, self.Z_values)):
            results = self.results_dict[metallicity]
            
            # Get initial stellar masses (at first timestep, before any evolution)
            initial_masses = np.array(results.stellar_masses[0])
            
            # but we normalize by the *requested* mass for correct feedback per 1e6 Msun
            actual_mass = np.sum(results.number_of_stars * initial_masses)
            
            # For custom distributions, use actual mass; otherwise use requested mass
            if hasattr(self, '_using_custom_distribution') and self._using_custom_distribution:
                total_mass = actual_mass
                if verbose:
                    print(f"  {metallicity}: Custom distribution, total mass = {total_mass:.2e} Msun")
            else:
                total_mass = self._requested_total_mass
                if verbose:
                    print(f"  {metallicity}: Actual mass = {actual_mass:.2e} Msun, "
                          f"Normalizing by requested mass = {total_mass:.2e} Msun")

            # =======================================================================       
            # Understanding pySB99 Units Convention
            # ========================================================================
            #
            # UNITS IN pysb99_core.py CALCULATIONS:
            # 
            # 1. Intermediate calculations use scaled units (from physical formulas):
            #    - Wind power: (Msun/yr) × (km/s)² × 3.155 ≈ value × 10^-35 erg/s
            #    - Wind momentum: (Msun/yr) × (km/s) × 3.155e-5 ≈ value × 10^-35 dyne
            #    - Luminosity: integrated flux in units ≈ value × 10^-20 erg/s
            #    - Photon rates: integrated photon flux ≈ value × 10^-20 photons/s
            #
            # 2. Conversion to log10(CGS) via offset addition:
            #    stored_value = log10(scaled_value) + offset
            #    
            #    For wind power (offset = 35):
            #      wpower = log10(wind_power_sum) + 35
            #             = log10(value × 10^-35) + 35
            #             = log10(value) + log10(10^-35) + 35
            #             = log10(value) - 35 + 35
            #             = log10(value)  [in CGS: erg/s]
            #    
            #    For luminosity/photon rates (offset = 20):
            #      L_bolo = log10(full_lum) + 20
            #             = log10(value × 10^-20) + 20
            #             = log10(value) - 20 + 20
            #             = log10(value)  [in CGS: erg/s or photons/s]
            #
            # 3. RESULT: Values in StellarPopulationResults are already log10(CGS):
            #    - results.wind_power        → log10(erg/s)
            #    - results.wind_momentum     → log10(dyne)
            #    - results.bolometric_luminosity → log10(erg/s)
            #    - results.hi_ionizing_flux  → log10(photons/s)
            #    (and similarly for hei_ionizing_flux, heii_ionizing_flux)
            #
            #    EXCEPTION - flux_spectra (SED):
            #    - results.flux_spectra is LINEAR (not log), units 10^-20 erg/s/Angstrom
            #    - Stored without log conversion (see pysb99_core.py line 1455: / 1e20)
            #    - To convert to log10(CGS): log10(flux_spectra) + 20
            #
            # NORMALIZATION TO 10^6 MSUN:
            # The downstream code (TODDLERS) expects values normalized to 10^6 Msun.
            # Since results are already in log10(CGS), we normalize in log space:
            #   log10(Q per 10^6 Msun) = log10(Q_total) - log10(M_total) + log10(10^6)
            #                          = log10(Q_total) - log10(M_total/10^6)
            #   where Q_total is the total quantity and M_total is the actual stellar mass
            #
            # See pysb99_core.py for implementation:
            #   - Lines 1730-1756: wind power/momentum calculation
            #   - Lines 1465-1491: ionizing flux calculation
            #   - Line 1455: flux_spectra stored in LINEAR 10^-20 units
            # ========================================================================
            
            # Normalize to 10^6 Msun reference
            # log10(Q per 10^6 Msun) = log10(Q) - log10(M/10^6)
            log_total_mass = np.log10(total_mass)
            log_normalization = log_total_mass - 6.0  # Equivalent to log10(M/1e6)
            
            # Wind properties (already log10 of values in CGS)
            all_wind_power[i, :] = results.wind_power - log_normalization  # log10(erg/s per 10^6 Msun)
            all_wind_momentum[i, :] = results.wind_momentum - log_normalization  # log10(dyne per 10^6 Msun)
            
            # Bolometric luminosity (already log10 of values in CGS)
            all_L_bolo[i, :] = results.bolometric_luminosity - log_normalization  # log10(erg/s per 10^6 Msun)
            
            # Ionizing luminosities (already log10 of values in CGS)
            # These are TRUE energy fluxes from spectrum integration, not approximated
            all_L_ion_HI[i, :] = results.hi_ionizing_luminosity - log_normalization  # log10(erg/s per 10^6 Msun)
            all_L_ion_HeI[i, :] = results.hei_ionizing_luminosity - log_normalization  # log10(erg/s per 10^6 Msun)
            all_L_ion_HeII[i, :] = results.heii_ionizing_luminosity - log_normalization  # log10(erg/s per 10^6 Msun)
            
            # Ionizing photon rates (already log10 of values in CGS)
            all_Q_HI[i, :] = results.hi_ionizing_flux - log_normalization  # log10(photons/s per 10^6 Msun)
            all_Q_HeI[i, :] = results.hei_ionizing_flux - log_normalization  # log10(photons/s per 10^6 Msun)
            all_Q_HeII[i, :] = results.heii_ionizing_flux - log_normalization  # log10(photons/s per 10^6 Msun)
            
            # Spectral hardness ratios (need linear values for ratios)
            # Since all_Q values are in log space, convert to get ratios
            # Q_HeI/Q_HI = 10^log(Q_HeI/M) / 10^log(Q_HI/M) = 10^(log_HeI - log_HI)
            with np.errstate(divide='ignore', invalid='ignore'):
                # Compute in log space then convert
                log_ratio_He1H = all_Q_HeI[i, :] - all_Q_HI[i, :]
                log_ratio_He2H = all_Q_HeII[i, :] - all_Q_HI[i, :]
                
                all_QratHe1H[i, :] = np.where(
                    np.isfinite(log_ratio_He1H),
                    10**log_ratio_He1H,
                    0
                )
                all_QratHe2H[i, :] = np.where(
                    np.isfinite(log_ratio_He2H),
                    10**log_ratio_He2H,
                    0
                )
            
            # Extract Lyman-Werner flux (around 1000 Angstroms)
            # NOTE: flux_spectra is stored with 10^-20 offset (see pysb99_core.py line 1455: / 1e20)
            # Units: 10^-20 erg/s/Angstrom (need to add 20 in log space to get erg/s/Angstrom)
            wavelength = results.wavelength_grid
            waves_idx_LyW = np.logical_and(wavelength >= 995., wavelength <= 1005.)
            
            if np.any(waves_idx_LyW):
                # Average flux in Lyman-Werner band (linear scale, units: 10^-20 erg/s/A)
                LyW_flux = np.mean(results.flux_spectra[:, waves_idx_LyW], axis=1)
                # Normalize to 10^6 Msun and convert to log space
                # Add 20 to convert from log10(10^-20 erg/s/A) to log10(erg/s/A)
                all_LyW_flux[i, :] = np.log10(LyW_flux / total_mass * 1e6 + 1e-50) + 20  # log10(erg/s/A per 10^6 Msun)
            else:
                warnings.warn(f"Lyman-Werner band not found in wavelength grid for {metallicity}")
            
            # Calculate mean ionizing photon energy using TRUE ionizing luminosities
            # Convert from log space (all_L_ion values are already normalized to 10^6 Msun)
            linear_L_HI = 10**all_L_ion_HI[i, :]
            linear_L_HeI = 10**all_L_ion_HeI[i, :]
            linear_L_HeII = 10**all_L_ion_HeII[i, :]
            
            # Convert photon rates from log space
            linear_Q_HI = 10**all_Q_HI[i, :]
            linear_Q_HeI = 10**all_Q_HeI[i, :]
            linear_Q_HeII = 10**all_Q_HeII[i, :]
            
            # Total ionizing luminosity and photon rate
            total_L = linear_L_HI + linear_L_HeI + linear_L_HeII
            total_Q = linear_Q_HI + linear_Q_HeI + linear_Q_HeII
            
            # Mean ionizing photon energy: <E> = L_total / Q_total
            # This uses TRUE luminosities from spectrum integration, not approximations
            eV_to_erg = 1.60218e-12
            with np.errstate(divide='ignore', invalid='ignore'):
                all_mean_ion_energy[i, :] = np.where(
                    total_Q > 1e-50,
                    (total_L / total_Q) / eV_to_erg,  # Convert erg to eV
                    13.6  # Default to HI ionization threshold
                )
        
        # Create interpolation functions
        # NOTE: Wind power, momentum, luminosities, and photon rates are normalized to 10^6 Msun
        # (we normalized them in log space above)
        # Only spectral hardness ratios and mean energy need logging
        interpolants = {
            # Wind properties (in log space: log10(value per 10^6 Msun))
            'f_wind_power': _make_interp2d(self.time_grid, self.Z_values,
                                           all_wind_power, kind='linear'),
            'f_wind_momentum': _make_interp2d(self.time_grid, self.Z_values,
                                              all_wind_momentum, kind='linear'),

            # Luminosities (in log space: log10(value per 10^6 Msun))
            'f_L_bolo': _make_interp2d(self.time_grid, self.Z_values,
                                       all_L_bolo, kind='linear'),
            'f_L_bol': _make_interp2d(self.time_grid, self.Z_values,
                                      all_L_bolo, kind='linear'),  # Alias
            'f_L_ion': _make_interp2d(self.time_grid, self.Z_values,
                                      np.log10(10**all_L_ion_HI + 10**all_L_ion_HeI + 10**all_L_ion_HeII + 1e-99),
                                      kind='linear'),  # Total ionizing luminosity

            # Ionizing photon rates (in log space: log10(value per 10^6 Msun))
            'f_Q_ion': _make_interp2d(self.time_grid, self.Z_values,
                                      all_Q_HI, kind='linear'),

            # Spectral hardness (linear values, need to log)
            'f_QratHe1H': _make_interp2d(self.time_grid, self.Z_values,
                                         np.log10(all_QratHe1H + 1e-50), kind='linear'),
            'f_QratHe2H': _make_interp2d(self.time_grid, self.Z_values,
                                         np.log10(all_QratHe2H + 1e-50), kind='linear'),

            # Lyman-Werner (in log space: log10(value per 10^6 Msun))
            'f_LyW': _make_interp2d(self.time_grid, self.Z_values,
                                    all_LyW_flux, kind='linear'),

            # Mean ionizing photon energy (linear value in eV, need to log)
            'f_mean_ion_energy': _make_interp2d(self.time_grid, self.Z_values,
                                                np.log10(all_mean_ion_energy), kind='linear'),
        }
        
        # ========================================================================
        # Calculate time derivatives (matching SB99 approach)
        # ========================================================================
        # Oversample the log interpolants to find the gradients
        # Convert to linear, compute gradient, then create new interpolants
        # This matches the approach in generate_SB99_interpolants.py lines 298-310
        
        if verbose:
            print("  Computing time derivatives (oversampling method)...")
        
        # Oversample time grid
        t_oversampled = np.linspace(self.time_grid[0], self.time_grid[-1], num=10000, endpoint=True)
        
        # Initialize arrays for derivatives
        all_wind_power_rate = np.zeros((n_Z, len(t_oversampled)))
        all_wind_momentum_rate = np.zeros((n_Z, len(t_oversampled)))
        
        for i, Z_val in enumerate(self.Z_values):
            # Evaluate interpolants on oversampled grid and convert to linear
            wind_power_oversampled = 10**interpolants['f_wind_power'](t_oversampled, Z_val)
            wind_momentum_oversampled = 10**interpolants['f_wind_momentum'](t_oversampled, Z_val)
            
            # Compute gradients in linear space (units: erg/s/Myr and dyne/Myr)
            all_wind_power_rate[i, :] = np.gradient(wind_power_oversampled.flatten(), t_oversampled)
            all_wind_momentum_rate[i, :] = np.gradient(wind_momentum_oversampled.flatten(), t_oversampled)
        
        # Create derivative interpolants (in linear space, NOT log space)
        interpolants['f_wind_power_rate'] = _make_interp2d(t_oversampled, self.Z_values,
                                                           all_wind_power_rate, kind='linear', fill_value=0)
        interpolants['f_wind_momentum_rate'] = _make_interp2d(t_oversampled, self.Z_values,
                                                              all_wind_momentum_rate, kind='linear', fill_value=0)
        
        if verbose:
            print("  Computed time derivatives via oversampling")
        
        
        # Store raw data for reference
        interpolants['_metadata'] = {
            'time_grid_myr': self.time_grid,
            'Z_values': self.Z_values,
            'metallicities': self.metallicities,
            'spectral_library': self.spectral_library,
            'rotation': self.rotation,
        }
        
        if verbose:
            print(f"Created {len(interpolants)-1} interpolation functions")
        
        return interpolants
    
    def save_interpolants(self, 
                         output_dir: str = './database',
                         imf_name: str = 'custom',
                         overwrite: bool = False):
        """
        Save interpolants to pickle files compatible with TODDLERS.
        
        Parameters
        ----------
        output_dir : str
            Directory to save interpolants
        imf_name : str
            Name identifier for the IMF configuration
        overwrite : bool
            Whether to overwrite existing files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create interpolants
        interpolants = self.create_interpolants()
        
        # Generate filename
        rotation_suffix = '_rot' if self.rotation else ''
        filename_main = f"pySB99interpolation_{imf_name}{rotation_suffix}.obj"
        filename_lw = f"pySB99interpolation_{imf_name}{rotation_suffix}_LumLymanWerner.obj"
        filename_mean_energy = f"pySB99interpolation_{imf_name}{rotation_suffix}_mean_ionizing_photon_energy.obj"
        filename_others = f"pySB99interpolation_{imf_name}{rotation_suffix}_hardness.obj"
        
        filepath_main = os.path.join(output_dir, filename_main)
        filepath_lw = os.path.join(output_dir, filename_lw)
        filepath_mean_energy = os.path.join(output_dir, filename_mean_energy)
        filepath_others = os.path.join(output_dir, filename_others)
        
        # Check if files exist
        if not overwrite:
            for filepath in [filepath_main, filepath_lw, filepath_mean_energy]:
                if os.path.exists(filepath):
                    raise FileExistsError(
                        f"File {filepath} already exists. Set overwrite=True to replace."
                    )
        
        # Save main interpolants (compatible with stellar_feedback.py format)
        # Format matches SB99: [L_mech, F_ram, L_bolo, L_ion, Q_ion, F_ram_rate, L_mech_rate]
        # where rates are derivatives in LINEAR space (dynes/Myr and erg/s/Myr)
        main_interpolants = [
            interpolants['f_wind_power'],          # L_mech (log space)
            interpolants['f_wind_momentum'],       # F_ram (log space)
            interpolants['f_L_bolo'],              # L_bolo (log space)
            interpolants['f_L_ion'],               # L_ion (log space)
            interpolants['f_Q_ion'],               # Q_ion (log space)
            interpolants['f_wind_momentum_rate'],  # F_ram_rate (linear space: dyne/Myr)
            interpolants['f_wind_power_rate'],     # L_mech_rate (linear space: erg/s/Myr)
        ]
        
        with open(filepath_main, 'wb') as f:
            pickle.dump(main_interpolants, f)
        print(f"Saved main interpolants to {filepath_main}")
        
        # Save Lyman-Werner interpolant
        with open(filepath_lw, 'wb') as f:
            pickle.dump(interpolants['f_LyW'], f)
        print(f"Saved Lyman-Werner interpolant to {filepath_lw}")
        
        # Save mean ionizing photon energy
        with open(filepath_mean_energy, 'wb') as f:
            pickle.dump(interpolants['f_mean_ion_energy'], f)
        print(f"Saved mean ionizing photon energy to {filepath_mean_energy}")
        # Save spectral hardness ratios
        # For compatibility with generate_SB99_interpolants format
        # [f_QratHe1H, f_QratHe2H]
        other_interpolants = [
            interpolants['f_QratHe1H'],         # HeI/HI ratio
            interpolants['f_QratHe2H'],         # HeII/HI ratio
        ]
        
        with open(filepath_others, 'wb') as f:
            pickle.dump(other_interpolants, f)
        print(f"Saved hardness interpolants to {filepath_others}")        
        
        print(f"\nAll interpolants saved successfully!")
        print(f"To use with TODDLERS, ensure these files are in the database directory")


def _save_population_metadata(generator, output_dir, imf_name, *, population_type,
                              custom_star_numbers=None, imf_exponents=None,
                              imf_mass_limits=None, time_start_myr=0.01,
                              time_end_myr=31.0, time_step_myr=0.1):
    """Write ``pySB99interpolation_<imf_name>_metadata.json`` next to the interpolant.

    SpectralTableGenerator reads this JSON to recover the metallicities, spectral library
    and population definition it needs to regenerate the incident spectra, so the
    deterministic pySB99 -> Cloudy path works directly after interpolant generation (no
    separate metadata step). Written without a rotation suffix to match how
    SpectralTableGenerator resolves the file.
    """
    metadata = {
        'population_name': imf_name,
        'metallicities': list(generator.metallicities),
        'Z_values': list(generator.Z_values),
        'spectral_library': generator.spectral_library,
        'rotation': generator.rotation,
        'time_start_myr': time_start_myr,
        'time_end_myr': time_end_myr,
        'time_step_myr': time_step_myr,
        'generation_date': datetime.now().isoformat(),
        'population_type': population_type,
    }
    if population_type == 'custom':
        metadata['custom_star_numbers'] = {str(m): n for m, n in custom_star_numbers.items()}
    else:
        metadata['imf_exponents'] = list(imf_exponents)
        metadata['imf_mass_limits'] = list(imf_mass_limits)

    path = os.path.join(output_dir, f'pySB99interpolation_{imf_name}_metadata.json')
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)
    return path


def generate_kroupa_like_interpolants(
        imf_exponents: List[float] = [1.3, 2.3],
        imf_mass_limits: Tuple[float, ...] = (0.1, 0.5, 100.0),
        output_dir: str = './database',
        imf_name: str = 'kroupa100',
        metallicities: Optional[List[str]] = None,
        rotation: bool = False,
        time_start_myr: float = 0.01,
        time_end_myr: float = 31.0,
        time_step_myr: float = 0.1,
        auto_extend_imf: bool = True,
        verbose: bool = True):
    """
    Generate interpolants for a Kroupa-like IMF.

    Parameters
    ----------
    imf_exponents : list of float
        IMF power law exponents (default: [1.3, 2.3] for Kroupa)
    imf_mass_limits : tuple
        Mass limits in solar masses (default: (0.1, 0.5, 100.0))
    output_dir : str
        Directory to save interpolants
    imf_name : str
        Name identifier for the IMF
    metallicities : list of str, optional
        List of metallicities to use
    rotation : bool
        Include stellar rotation
    auto_extend_imf : bool
        If True (default), extend IMF upper limit to the track grid maximum.
        Set to False to strictly respect the user-specified upper mass limit.
    verbose : bool
        Print progress

    Returns
    -------
    PySB99InterpolantGenerator
        The generator object with results
    """
    generator = PySB99InterpolantGenerator(
        metallicities=metallicities,
        rotation=rotation,
        time_start_myr=time_start_myr,
        time_end_myr=time_end_myr,
        time_step_myr=time_step_myr
    )

    generator.run_all_metallicities(
        total_mass=1e6,
        imf_exponents=imf_exponents,
        imf_mass_limits=imf_mass_limits,
        auto_extend_imf=auto_extend_imf,
        verbose=verbose
    )
    
    generator.save_interpolants(
        output_dir=output_dir,
        imf_name=imf_name,
        overwrite=True
    )

    _save_population_metadata(generator, output_dir, imf_name, population_type='imf',
                              imf_exponents=imf_exponents, imf_mass_limits=imf_mass_limits,
                              time_start_myr=time_start_myr, time_end_myr=time_end_myr,
                              time_step_myr=time_step_myr)

    return generator


def generate_custom_population_interpolants(
        custom_star_numbers: Dict[float, int],
        output_dir: str = './database',
        imf_name: str = 'custom_population',
        metallicities: Optional[List[str]] = None,
        rotation: bool = False,
        time_start_myr: float = 0.01,
        time_end_myr: float = 31.0,
        time_step_myr: float = 0.1,
        verbose: bool = True):
    """
    Generate interpolants for a custom stellar population.
    
    Parameters
    ----------
    custom_star_numbers : dict
        Dictionary mapping stellar mass to number of stars
        Example: {20.0: 100, 40.0: 50, 85.0: 10}
    output_dir : str
        Directory to save interpolants
    imf_name : str
        Name identifier
    metallicities : list of str, optional
        List of metallicities to use
    rotation : bool
        Include stellar rotation
    verbose : bool
        Print progress
        
    Returns
    -------
    PySB99InterpolantGenerator
        The generator object with results
    """
    generator = PySB99InterpolantGenerator(
        metallicities=metallicities,
        rotation=rotation,
        time_start_myr=time_start_myr,
        time_end_myr=time_end_myr,
        time_step_myr=time_step_myr
    )
    
    generator.run_all_metallicities(
        custom_star_numbers=custom_star_numbers,
        verbose=verbose
    )

    generator.save_interpolants(
        output_dir=output_dir,
        imf_name=imf_name,
        overwrite=True
    )

    _save_population_metadata(generator, output_dir, imf_name, population_type='custom',
                              custom_star_numbers=custom_star_numbers,
                              time_start_myr=time_start_myr, time_end_myr=time_end_myr,
                              time_step_myr=time_step_myr)

    return generator


if __name__ == "__main__":
    """
    Example usage of the pySB99 interpolant generator.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate stellar population synthesis interpolants using pySB99'
    )
    parser.add_argument('--imf-name', type=str, default='kroupa100',
                       help='Name identifier for IMF (default: kroupa100)')
    parser.add_argument('--imf-exponents', type=float, nargs='+', default=[1.3, 2.3],
                       help='IMF power law exponents (default: 1.3 2.3)')
    parser.add_argument('--imf-limits', type=float, nargs='+', default=[0.1, 0.5, 120.0],
                       help='IMF mass limits in Msun (default: 0.1 0.5 100.0)')
    parser.add_argument('--rotation', action='store_true',
                       help='Include stellar rotation')
    parser.add_argument('--custom-stars', type=str,
                       help='Custom star distribution: "mass1:count1,mass2:count2,..."')
    parser.add_argument('--output-dir', type=str, default='./database',
                       help='Output directory (default: ./database)')
    parser.add_argument('--metallicities', type=str, nargs='+',
                       choices=['MWC', 'MW', 'LMC', 'SMC', 'IZw18', 'Z0'],
                       help='Metallicities to use (default: all)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Parse custom star distribution if provided
    custom_stars = None
    if args.custom_stars:
        custom_stars = {}
        for pair in args.custom_stars.split(','):
            mass, count = pair.split(':')
            custom_stars[float(mass)] = int(count)
        
        if args.verbose:
            print("Custom star distribution:")
            for mass, count in sorted(custom_stars.items()):
                print(f"  {mass:.1f} Msun: {count} stars")
    
    # Generate interpolants
    if custom_stars:
        generator = generate_custom_population_interpolants(
            custom_star_numbers=custom_stars,
            output_dir=args.output_dir,
            imf_name=args.imf_name,
            metallicities=args.metallicities,
            rotation=args.rotation,
            verbose=args.verbose
        )
    else:
        generator = generate_kroupa_like_interpolants(
            imf_exponents=args.imf_exponents,
            imf_mass_limits=tuple(args.imf_limits),
            output_dir=args.output_dir,
            imf_name=args.imf_name,
            metallicities=args.metallicities,
            rotation=args.rotation,
            verbose=args.verbose
        )
    
    print("\nInterpolant generation complete!")
    print(f"Files saved to {args.output_dir}")
    print("\nTo use with TODDLERS:")
    print(f"  1. Copy files to your TODDLERS database directory")
    print(f"  2. Update stellar_feedback.py to use imf='{args.imf_name}'")