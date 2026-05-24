#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stochastic/database.py
======================

Generate and query HDF5 database of single-star feedback tracks for stochastic
stellar population synthesis.

This module creates a database where each entry is a SINGLE STAR of a given mass
at a given metallicity, evolved over time. All quantities are stored exactly as
pySB99 provides them - no normalization, just pure CGS (in log10).

Units Convention:
-----------------
All feedback quantities from pySB99 are ALREADY in log10(CGS):

From pysb99_core.py calculations:
  - Physical formulas -> scaled values (e.g., value * 10^-35 erg/s)
  - Offset addition -> log10(scaled) + offset = log10(CGS)
  
Database stores (for single star):
  - wind_power: log10(erg/s)
  - wind_momentum: log10(dyne)
  - bolometric_luminosity: log10(erg/s)
  - hi/hei/heii_ionizing_flux: log10(photons/s)
  - L_LyW: log10(erg/s/Angstrom) - CONVERTED from flux_spectra
  - flux_spectra: LINEAR (NOT log), units 10^-20 erg/s/Angstrom - NEEDS +20 to get log10(CGS)
  - wavelength_grid: Angstrom

IMPORTANT: 
  - flux_spectra is the ONLY quantity stored in LINEAR (not log) form
  - To convert flux_spectra to log10(CGS): log10(flux_spectra) + 20
  - L_LyW is already converted: log10(mean(flux_spectra[995-1005Angstrom])) + 20

NO normalization applied - values stored directly from pySB99.
Normalization happens later when building stochastic interpolants.

See _extract_feedback() docstring for detailed units explanation.

Author: Anand Utsav Kapoor
Created: 2025
"""

import os
import h5py
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple
import sys

# Handle imports for both standalone and package usage
try:
    # Try package-relative import first (when used as part of stochastic package)
    from pysb99_core import (
        StellarPopulationConfig,
        run_population_synthesis,
        StellarPopulationResults,
        StellarDataLoader
    )
except ImportError:
    # Fall back to adding parent directory to path (when run as standalone script)
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from pysb99_core import (
        StellarPopulationConfig,
        run_population_synthesis,
        StellarPopulationResults,
        StellarDataLoader
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

# Standard metallicities to include in database
DEFAULT_METALLICITIES = ['IZw18', 'SMC', 'LMC', 'MW', 'MWC']
DEFAULT_Z_VALUES = [METALLICITY_MAPPING[m] for m in DEFAULT_METALLICITIES]


def get_available_mass_grid(metallicity: str, rotation: bool = False) -> np.ndarray:
    """
    Get the available stellar masses for a given metallicity and rotation.
    
    This scans the actual pySB99 stellar evolution tracks to find what masses
    truly exist, rather than relying on hardcoded lists.
    
    Parameters
    ----------
    metallicity : str
        Metallicity identifier ('MWC', 'MW', 'LMC', 'SMC', 'IZw18', 'Z0')
    rotation : bool
        Whether to use rotating stellar models
        
    Returns
    -------
    np.ndarray
        Array of available stellar masses in solar masses, sorted descending
        
    Notes
    -----
    This function loads the tracks .npy file and extracts unique initial masses
    from the data, ensuring we only use masses that actually have tracks.
    """
    # Get file path for this metallicity
    import os
    
    # Find pySB99_files directory relative to this module
    module_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(module_dir)  # Up one level to pysb99/
    
    metallicity_paths = {
        'MWC': 'pySB99_files/Z020_pySB99_files/',
        'MW': 'pySB99_files/Z014_pySB99_files/',
        'LMC': 'pySB99_files/Z006_pySB99_files/',
        'SMC': 'pySB99_files/Z002_pySB99_files/',
        'IZw18': 'pySB99_files/Z0004_pySB99_files/',
        'Z0': 'pySB99_files/Z00_pySB99_files/'
    }
    
    z_strings = {
        'MWC': '020',
        'MW': '014',
        'LMC': '006',
        'SMC': '002',
        'IZw18': '0004',
        'Z0': '00'
    }
    
    file_dir = os.path.join(parent_dir, metallicity_paths[metallicity])
    z_string = z_strings[metallicity]
    rotation_suffix = 'v40' if rotation else 'v00'
    
    # Determine filename (VMS tracks for high-mass metallicities, regular for low-mass)
    if metallicity in ['MWC', 'MW', 'LMC', 'Z0']:
        filename = f'Z{z_string}{rotation_suffix}_VMS_tracks.npy'
    else:
        filename = f'Z{z_string}{rotation_suffix}_tracks.npy'
    
    full_path = os.path.join(file_dir, filename)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Tracks file not found: {full_path}")
    
    # Load tracks array
    # Structure: columns are [id, age, mass, luminosity, temperature, ...]
    tracks = np.load(full_path)
    
    # Extract initial masses (at age ~ 0) from column 2
    # The tracks are structured with all timesteps for mass1, then all for mass2, etc.
    # So we need to find unique initial masses
    
    masses_col = tracks[:, 2]  # Column 2 contains masses
    ages_col = tracks[:, 1]    # Column 1 contains ages
    
    # Find initial timestep for each track (age ~ 0)
    # Group by finding where age resets to small value
    initial_mass_indices = []
    for i in range(len(ages_col) - 1):
        if i == 0:
            initial_mass_indices.append(i)
        elif ages_col[i+1] < ages_col[i]:  # Age decreased, new track starting
            initial_mass_indices.append(i+1)
    
    # Get initial masses
    initial_masses = masses_col[initial_mass_indices]
    
    # Get unique masses and sort descending (high to low)
    unique_masses = np.unique(initial_masses)[::-1]
    
    return unique_masses


def generate_single_star_database(
        output_file: str,
        metallicities: Optional[List[str]] = None,
        time_max_myr: float = 100.0,
        time_resolution_myr: float = 0.1,
        spectral_library: str = 'FW',
        rotation: bool = False,
        max_mass: Optional[float] = None,
        overwrite: bool = False,
        verbose: bool = True) -> None:
    """
    Generate HDF5 database of single-star feedback tracks.
    
    For each (metallicity, mass) combination, runs pySB99 with custom_star_numbers={mass: 1}
    to get the evolution of a SINGLE star. Stores all quantities exactly as pySB99 provides.
    
    Parameters
    ----------
    output_file : str
        Path to output HDF5 file
    metallicities : list of str, optional
        Metallicity identifiers. Default: ['IZw18', 'SMC', 'LMC', 'MW', 'MWC']
    time_max_myr : float
        Maximum stellar age in Myr (default: 100.0)
    time_resolution_myr : float
        Time resolution in Myr (default: 0.1)
    spectral_library : str
        'FW' (Fastwind) or 'WM' (WMbasic)
    rotation : bool
        Include stellar rotation
    overwrite : bool
        Overwrite existing file
    verbose : bool
        Print progress
        
    Notes
    -----
    HDF5 structure::

        /metadata/
            time_grid              [Myr] - uniform time grid
            description
            spectral_library
            rotation

        /Z_<value>/               (e.g., /Z_0.0140/)
            metallicity_string    (e.g., 'MW')
            Z_value               (e.g., 0.014)
            mass_grid             [Msun] - available masses
            wavelength_grid       [Angstrom] - common for all masses

            /mass_<value>/        (e.g., /mass_25.0/)
                wind_power        [log10(erg/s)] - single star
                wind_momentum     [log10(dyne)] - single star
                L_bol             [log10(erg/s)] - single star
                Q_HI              [log10(photons/s)] - single star
                Q_HeI             [log10(photons/s)] - single star
                Q_HeII            [log10(photons/s)] - single star
                L_ion_HI          [log10(erg/s)] - HI ionizing luminosity
                L_ion_HeI         [log10(erg/s)] - HeI ionizing luminosity
                L_ion_HeII        [log10(erg/s)] - HeII ionizing luminosity
                L_LyW             [log10(erg/s/Angstrom)] - Lyman-Werner band (995-1005 Angstrom)
                                  Converted from flux_spectra: log10(mean_flux) + 20
                flux_spectra      [10^-20 erg/s/Angstrom, LINEAR not log] - single star
                                  To convert to log10(CGS): log10(flux_spectra) + 20
                                  Shape: (n_times, n_wavelengths)
    """
    if metallicities is None:
        metallicities = DEFAULT_METALLICITIES
    
    if os.path.exists(output_file) and not overwrite:
        raise FileExistsError(
            f"Database already exists: {output_file}\n"
            f"Use overwrite=True to regenerate."
        )
        
    # Create the directory if it doesn't exist
    directory = os.path.dirname(output_file)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    if verbose:
        print("="*80)
        print("GENERATING SINGLE-STAR FEEDBACK DATABASE")
        print("="*80)
        print(f"Output: {output_file}")
        print(f"Metallicities: {metallicities}")
        print(f"Time: 0.01 - {time_max_myr} Myr (resolution: {time_resolution_myr} Myr)")
        print(f"Library: {spectral_library}, Rotation: {rotation}")
        print()
    
    # Create uniform time grid
    n_timesteps = int(time_max_myr / time_resolution_myr) + 1
    time_grid = np.linspace(1e-6, time_max_myr, n_timesteps)
    
    if verbose:
        print(f"Time grid: {len(time_grid)} points")
        print()
    
    with h5py.File(output_file, 'w') as f:
        
        # === Metadata ===
        meta = f.create_group('metadata')
        meta.create_dataset('time_grid', data=time_grid)
        meta.attrs['time_unit'] = 'Myr'
        meta.attrs['description'] = 'pySB99 single-star tracks for stochastic populations'
        meta.attrs['spectral_library'] = spectral_library
        meta.attrs['rotation'] = rotation
        meta.attrs['n_metallicities'] = len(metallicities)
        meta.attrs['creation_date'] = np.string_(str(np.datetime64('now')))
        
        # Units documentation
        meta.attrs['wind_power_units'] = 'log10(erg/s)'
        meta.attrs['wind_momentum_units'] = 'log10(dyne)'
        meta.attrs['L_bol_units'] = 'log10(erg/s)'
        meta.attrs['Q_units'] = 'log10(photons/s)'
        meta.attrs['L_ion_units'] = 'log10(erg/s) - ionizing luminosity from pySB99'
        meta.attrs['L_LyW_units'] = 'log10(erg/s/Angstrom) - Lyman-Werner band (995-1005 Angstrom), converted from flux_spectra'
        meta.attrs['flux_spectra_units'] = '10^-20 erg/s/Angstrom (LINEAR, not log) - to convert to log10(CGS): log10(flux_spectra) + 20'
        meta.attrs['wavelength_units'] = 'Angstrom'
        meta.attrs['IMPORTANT'] = 'flux_spectra is the ONLY quantity stored in LINEAR (not log) form'
        
        # === Process each metallicity ===
        for i_met, metallicity in enumerate(metallicities):
            Z_value = METALLICITY_MAPPING[metallicity]
            
            if verbose:
                print(f"[{i_met+1}/{len(metallicities)}] {metallicity} (Z={Z_value:.4f})")
            
            # Get available masses
            mass_grid = get_available_mass_grid(metallicity, rotation)
            
            # Filter mass grid to respect max_mass limit
            if max_mass is not None:
                available_masses = mass_grid[mass_grid <= max_mass]
                if verbose:
                    n_skipped = len(mass_grid) - len(available_masses)
                    if n_skipped > 0:
                        print(f"  Filtered {n_skipped} masses > {max_mass:.1f} Msun from mass grid")
                mass_grid = available_masses
            
            if verbose:
                print(f"  Mass grid: {len(mass_grid)} stars, {mass_grid[-1]:.1f} - {mass_grid[0]:.1f} Msun")
            
            # Create metallicity group
            Z_group = f.create_group(f'Z_{Z_value:.4f}')
            Z_group.attrs['metallicity_string'] = metallicity
            Z_group.attrs['Z_value'] = Z_value
            Z_group.create_dataset('mass_grid', data=mass_grid)
            
            # Wavelength grid (common to all masses at this Z)
            wavelength_grid = None
            
            # === Process each mass ===
            for i_mass, mass in enumerate(mass_grid):
                # Safety check - this should not trigger since mass_grid is already filtered
                if max_mass is not None and mass > max_mass:
                    if verbose:
                        print(f"  WARNING: Unexpected mass {mass:.1f} Msun > max_mass {max_mass:.1f} Msun in filtered grid!")
                    continue
                
                if verbose:
                    print(f"    [{i_mass+1}/{len(mass_grid)}] {mass:.1f} Msun", end=' ... ')
                
                try:
                    # Run pySB99 for single star
                    results = _run_single_star(
                        mass=mass,
                        metallicity=metallicity,
                        time_grid_myr=time_grid,
                        spectral_library=spectral_library,
                        rotation=rotation
                    )
                    
                    # Initialize mass_to_save to requested mass (default)
                    mass_to_save = mass
                    
                    # Verify actual mass simulated
                    if hasattr(results, 'number_of_stars') and hasattr(results, 'stellar_masses'):
                        try:
                            # Find which mass bin has non-zero stars
                            star_counts = results.number_of_stars[0]  # First timestep
                            nonzero_idx = np.nonzero(star_counts)[0]
                            
                            if len(nonzero_idx) >= 1:  # At least one star
                                # Use the first (or only) non-zero mass
                                # Convert to scalar to avoid array comparison issues
                                actual_mass = float(results.stellar_masses[nonzero_idx[0]])
                                
                                # Check for significant discrepancy
                                if abs(actual_mass - mass) / mass > 0.1:  # More than 10% difference
                                    warnings.warn(
                                        f"Skipping mass {mass:.1f} Msun - pySB99 mapped to {actual_mass:.1f} Msun. "
                                        f"Not adding to database to avoid duplicate entries."
                                    )
                                    if verbose:
                                        print("  SKIPPED - mass not in tracks")
                                    continue  # Skip this mass entirely
                        except (IndexError, TypeError) as e:
                            # If anything goes wrong, use requested mass
                            pass
                    
                    # Extract wavelength grid (same for all masses)
                    if wavelength_grid is None:
                        wavelength_grid = results.wavelength_grid
                        Z_group.create_dataset('wavelength_grid', data=wavelength_grid)
                    
                    # Extract and regrid feedback
                    feedback = _extract_feedback(results, time_grid)
                    
                    # Create mass group with ACTUAL simulated mass
                    mass_group = Z_group.create_group(f'mass_{mass_to_save:.1f}')
                    mass_group.attrs['mass_Msun'] = mass_to_save
                    mass_group.attrs['requested_mass_Msun'] = mass  # Store original request for reference
                    
                    # Store feedback quantities
                    for key, value in feedback.items():
                        mass_group.create_dataset(key, data=value)
                    
                    if verbose:
                        print("*")
                
                except Exception as e:
                    if verbose:
                        print(f"X {str(e)}")
                    warnings.warn(f"Failed {metallicity} {mass:.1f} Msun: {str(e)}")
            
            if verbose:
                print()
    
    if verbose:
        file_size_mb = os.path.getsize(output_file) / (1024**2)
        print("="*80)
        print(f"Complete: {output_file} ({file_size_mb:.1f} MB)")
        print("="*80)


def _run_single_star(
        mass: float,
        metallicity: str,
        time_grid_myr: np.ndarray,
        spectral_library: str = 'FW',
        rotation: bool = False) -> StellarPopulationResults:
    """
    Run pySB99 for a single star.
    
    Uses custom_star_numbers={mass: 1} to simulate exactly one star.
    """
    time_start_yr = time_grid_myr[0] * 1e6
    time_end_yr = time_grid_myr[-1] * 1e6
    time_step_yr = (time_grid_myr[1] - time_grid_myr[0]) * 1e6
    
    config = StellarPopulationConfig(
        total_mass=mass,  # Ignored when custom_star_numbers is set
        metallicity=metallicity,
        spectral_library=spectral_library,
        rotation=rotation,
        time_start=time_start_yr,
        time_end=time_end_yr,
        time_step=time_step_yr,
        run_speed_mode='DEFAULT',
        custom_star_numbers={mass: 1}  # SINGLE STAR
    )
    
    results = run_population_synthesis(config)
    return results


def _extract_feedback(
        results: StellarPopulationResults,
        time_grid_myr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract feedback quantities from pySB99 results and regrid to uniform time.
    
    Stores exactly what pySB99 gives - no normalization.
    
    Parameters
    ----------
    results : StellarPopulationResults
        Results from single-star run
    time_grid_myr : np.ndarray
        Target uniform time grid [Myr]
        
    Returns
    -------
    dict
        Feedback quantities on uniform grid
        
    Notes
    -----
    UNITS FROM pySB99:
    
    All feedback quantities in StellarPopulationResults are already in log10(CGS).
    This happens through the following conversion in pysb99_core.py:
    
    1. Physical calculations produce scaled values:
       - Wind power: (Msun/yr) * (km/s)^2 * 3.155 ~ value * 10^-35 erg/s
       - Wind momentum: (Msun/yr) * (km/s) * 3.155e-5 ~ value * 10^-35 dyne
       - Bolometric luminosity: integrated flux ~ value * 10^-20 erg/s
       - Ionizing photon rates: integrated flux ~ value * 10^-20 photons/s
    
    2. Conversion to log10(CGS) via offset:
       stored = log10(scaled_value) + offset
       
       Example for wind power (offset = 35):
         wpower = log10(wind_power_sum) + 35
                = log10(value * 10^-35) + 35
                = log10(value) - 35 + 35
                = log10(value in CGS units)
       
       Example for luminosity (offset = 20):
         L_bolo = log10(full_lum) + 20
                = log10(value * 10^-20) + 20
                = log10(value) - 20 + 20
                = log10(value in CGS units)
    
    3. RESULT - what we store in database:
       - results.wind_power        -> log10(erg/s) for single star
       - results.wind_momentum     -> log10(dyne) for single star
       - results.bolometric_luminosity -> log10(erg/s) for single star
       - results.hi_ionizing_flux  -> log10(photons/s) for single star
       - results.hei_ionizing_flux -> log10(photons/s) for single star
       - results.heii_ionizing_flux -> log10(photons/s) for single star
       
       EXCEPTION - flux_spectra (SED):
       - results.flux_spectra is LINEAR (not log), units 10^-20 erg/s/Angstrom
       - To convert to log10(CGS): log10(flux_spectra) + 20
       - We do this conversion for L_LyW but store flux_spectra as-is
    
    NO NORMALIZATION is applied here - we store the single star values directly.
    Normalization happens later when building stochastic population interpolants.
    
    See pysb99_core.py:
      - Lines 1730-1756: wind power/momentum calculation
      - Lines 1465-1491: ionizing flux calculation
      - Line 1455: flux_spectra scaled by 1e20 (hence LINEAR in 10^-20 units)
    """
    pysb99_times_myr = results.times / 1e6
    
    feedback = {}
    
    # ========================================================================
    # Extensive quantities - already in log10(CGS), no conversion needed
    # ========================================================================
    # These are for the single star from custom_star_numbers={mass: 1}
    
    feedback['wind_power'] = np.interp(
        time_grid_myr, pysb99_times_myr, results.wind_power
    )
    
    feedback['wind_momentum'] = np.interp(
        time_grid_myr, pysb99_times_myr, results.wind_momentum
    )
    
    feedback['L_bol'] = np.interp(
        time_grid_myr, pysb99_times_myr, results.bolometric_luminosity
    )
    
    feedback['Q_HI'] = np.interp(
        time_grid_myr, pysb99_times_myr, results.hi_ionizing_flux
    )
    
    feedback['Q_HeI'] = np.interp(
        time_grid_myr, pysb99_times_myr, results.hei_ionizing_flux
    )
    
    feedback['Q_HeII'] = np.interp(
        time_grid_myr, pysb99_times_myr, results.heii_ionizing_flux
    )
    
    # Ionizing luminosities (energy flux in ionizing photons, erg/s)
    # These are the ACTUAL ionizing luminosities from pySB99, not derived
    feedback['L_ion_HI'] = np.interp(
        time_grid_myr, pysb99_times_myr, results.hi_ionizing_luminosity
    )
    
    feedback['L_ion_HeI'] = np.interp(
        time_grid_myr, pysb99_times_myr, results.hei_ionizing_luminosity
    )
    
    feedback['L_ion_HeII'] = np.interp(
        time_grid_myr, pysb99_times_myr, results.heii_ionizing_luminosity
    )
    
    # === Lyman-Werner band luminosity ===
    # Average flux in 995-1005 Angstrom band
    # NOTE: flux_spectra is LINEAR, units 10^-20 erg/s/Angstrom (NOT log)
    # We convert to log10(CGS) here, unlike flux_spectra which we store as-is
    wavelength = results.wavelength_grid
    waves_idx_LyW = np.logical_and(wavelength >= 995., wavelength <= 1005.)
    
    if np.any(waves_idx_LyW):
        # Average flux in Lyman-Werner band for each timestep
        # Units: 10^-20 erg/s/Angstrom (LINEAR, for this single star)
        LyW_flux_raw = np.mean(results.flux_spectra[:, waves_idx_LyW], axis=1)
        
        # Convert from LINEAR 10^-20 units to log10(CGS):
        # log10(flux in 10^-20 erg/s/Angstrom) + 20 = log10(erg/s/Angstrom)
        # NO normalization - this is for the single star
        LyW_log_cgs = np.log10(LyW_flux_raw + 1e-50) + 20.0
        
        # Interpolate to uniform time grid
        feedback['L_LyW'] = np.interp(time_grid_myr, pysb99_times_myr, LyW_log_cgs)
    else:
        # If wavelength band not available, store zeros
        warnings.warn("Lyman-Werner band (995-1005 Angstrom) not in wavelength grid")
        feedback['L_LyW'] = np.zeros(len(time_grid_myr))
    
    # === Spectra (LINEAR, units 10^-20 erg/s/Angstrom) ===
    # Stored as-is in LINEAR form - conversion to log10(CGS) done by user
    # To convert: log10(flux_spectra) + 20 = log10(erg/s/Angstrom)
    # Interpolate each wavelength separately
    # flux_spectra shape: (n_pysb99_times, n_wavelengths)
    
    n_wavelengths = results.flux_spectra.shape[1]
    flux_regridded = np.zeros((len(time_grid_myr), n_wavelengths))
    
    for i_wave in range(n_wavelengths):
        flux_regridded[:, i_wave] = np.interp(
            time_grid_myr, 
            pysb99_times_myr, 
            results.flux_spectra[:, i_wave]
        )
    
    feedback['flux_spectra'] = flux_regridded
    
    return feedback


def _query_single(
        hdf5_file: str,
        mass: float,
        age_myr: float,
        metallicity_Z: float,
        quantity: str = 'Q_HI',
        strict_mass_limits: bool = False,
        suppress_warnings: bool = False) -> float:
    """
    Query database for a specific feedback quantity (single age, single quantity).
    
    Uses nearest neighbor in mass, linear interpolation in age.
    Optimized for discrete sampling where masses are exactly on grid.
    
    Parameters
    ----------
    hdf5_file : str
        Path to HDF5 database
    mass : float
        Stellar mass [Msun]
    age_myr : float
        Stellar age [Myr]
    metallicity_Z : float
        Metallicity (numerical, e.g., 0.014)
    quantity : str
        Quantity to retrieve:
        - 'wind_power': log10(erg/s)
        - 'wind_momentum': log10(dyne)
        - 'L_bol': log10(erg/s)
        - 'Q_HI', 'Q_HeI', 'Q_HeII': log10(photons/s)
        - 'L_LyW': log10(erg/s/Angstrom) - Lyman-Werner band
        - 'L_ion_HI', 'L_ion_HeI', 'L_ion_HeII': log10(erg/s) - true ionizing luminosities
    strict_mass_limits : bool
        If True, raise error when mass is outside database range.
        If False, cap mass at database limits with warning (default).
    suppress_warnings : bool
        If True, suppress warnings about mass capping (useful for batch operations)
        
    Returns
    -------
    float
        Interpolated value
        
    Raises
    ------
    ValueError
        If strict_mass_limits=True and mass is outside database range
        
    Notes
    -----
    - Caps mass/age at database limits (or raises error if strict_mass_limits=True)
    - Nearest neighbor in mass (optimized for discrete sampling on grid)
    - Linear interpolation in age
    """
    with h5py.File(hdf5_file, 'r') as f:
        
        # Find nearest metallicity group
        Z_group_name = _find_nearest_Z_group(f, metallicity_Z)
        Z_group = f[Z_group_name]
        
        # Get grids
        mass_grid = Z_group['mass_grid'][:]
        time_grid = f['metadata/time_grid'][:]
        
        # Validate mass range
        mass_min = np.min(mass_grid)
        mass_max = np.max(mass_grid)
        
        if strict_mass_limits:
            # Strict mode: raise error if out of range
            if mass > mass_max:
                raise ValueError(
                    f"Requested mass {mass:.1f} Msun exceeds database maximum {mass_max:.1f} Msun. "
                    f"This would assign incorrect feedback data. "
                    f"Check your population sampling or regenerate database with higher max_mass."
                )
            if mass < mass_min:
                raise ValueError(
                    f"Requested mass {mass:.1f} Msun below database minimum {mass_min:.1f} Msun. "
                    f"This would assign incorrect feedback data."
                )
        else:
            # Permissive mode: cap and warn
            if mass > mass_max:
                if not suppress_warnings:
                    warnings.warn(f"Mass {mass:.1f} > max {mass_max:.1f}, capping")
                mass = mass_max
            if mass < mass_min:
                if not suppress_warnings:
                    warnings.warn(f"Mass {mass:.1f} < min {mass_min:.1f}, capping")
                mass = mass_min
        
        # Cap age at limits (always permissive for age)
        if age_myr > time_grid[-1]:
            if not suppress_warnings:
                warnings.warn(f"Age {age_myr:.2f} > max, using {time_grid[-1]:.1f} Myr")
            age_myr = time_grid[-1]
        
        # Use nearest neighbor for mass (since discrete sampling ensures 
        # all masses are on grid - no interpolation needed)
        idx_nearest = np.argmin(np.abs(mass_grid - mass))
        mass_nearest = mass_grid[idx_nearest]
        
        # Get data for nearest mass
        mass_group = Z_group[f'mass_{mass_nearest:.1f}']
        data = mass_group[quantity][:]
        
        # Linear interpolation in age only
        return np.interp(age_myr, time_grid, data)


def query_database(
        database_path: str,
        metallicity: str,
        mass: float,
        ages_myr: np.ndarray,
        rotation: bool = False,
        quantities: List[str] = None,
        strict_mass_limits: bool = False,
        suppress_warnings: bool = True) -> Dict[str, np.ndarray]:
    """
    Batch query database for multiple feedback quantities across multiple ages.
    
    This is a convenience wrapper around _query_single() that:
    - Accepts string metallicity (converted to numeric)
    - Handles arrays of ages
    - Returns multiple quantities at once
    - Uses interface expected by stochastic interpolants code
    
    Parameters
    ----------
    database_path : str
        Path to HDF5 database
    metallicity : str
        Metallicity identifier ('MW', 'LMC', 'SMC', etc.)
    mass : float
        Stellar mass [Msun]
    ages_myr : np.ndarray
        Array of stellar ages [Myr], shape (n_ages,)
    rotation : bool
        Use rotating models (currently ignored, uses database as-is)
    quantities : List[str]
        List of quantities to retrieve. If None, returns all standard quantities.
        Options: 'wind_power', 'wind_momentum', 'L_bol',
        'Q_HI', 'Q_HeI', 'Q_HeII', 'L_LyW',
        'L_ion_HI', 'L_ion_HeI', 'L_ion_HeII'
    strict_mass_limits : bool
        If True, raise error when mass is outside database range
    suppress_warnings : bool
        If True, suppress mass/age capping warnings (default: True for batch operations)
        
    Returns
    -------
    dict
        Dictionary mapping quantity names to arrays of values:
        Each array has shape (n_ages,) with values in log10(CGS units)
        
    Examples
    --------
    >>> feedback = query_database(
    ...     database_path='database.h5',
    ...     metallicity='MW',
    ...     mass=25.0,
    ...     ages_myr=np.array([1.0, 2.0, 5.0, 10.0]),
    ...     quantities=['Q_HI', 'wind_power']
    ... )
    >>> print(feedback['Q_HI'])  # Array of length 4
    """
    # Convert metallicity string to numeric value
    metallicity_Z = METALLICITY_MAPPING.get(metallicity)
    if metallicity_Z is None:
        raise ValueError(f"Unknown metallicity: {metallicity}. Available: {list(METALLICITY_MAPPING.keys())}")
    
    # Default quantities
    if quantities is None:
        quantities = ['wind_power', 'wind_momentum', 'L_bol',
                     'Q_HI', 'Q_HeI', 'Q_HeII', 'L_LyW',
                     'L_ion_HI', 'L_ion_HeI', 'L_ion_HeII']
    
    # Ensure ages_myr is array
    ages_myr = np.atleast_1d(ages_myr)
    n_ages = len(ages_myr)
    
    # Initialize output dict
    results = {q: np.zeros(n_ages) for q in quantities}
    
    # Query each age for each quantity
    for i_age, age in enumerate(ages_myr):
        for quantity in quantities:
            results[quantity][i_age] = _query_single(
                hdf5_file=database_path,
                mass=mass,
                age_myr=age,
                metallicity_Z=metallicity_Z,
                quantity=quantity,
                strict_mass_limits=strict_mass_limits,
                suppress_warnings=suppress_warnings
            )
    
    return results


def _query_spectrum_single(
        hdf5_file: str,
        mass: float,
        age_myr: float,
        metallicity_Z: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Query database for spectrum of a single star at single age.
    
    Parameters
    ----------
    hdf5_file : str
        Path to HDF5 database
    mass : float
        Stellar mass [Msun]
    age_myr : float
        Stellar age [Myr]
    metallicity_Z : float
        Metallicity
        
    Returns
    -------
    wavelength : np.ndarray
        Wavelength grid [Angstrom]
    flux : np.ndarray
        Flux density [10^-20 erg/s/Angstrom, LINEAR]
    """
    with h5py.File(hdf5_file, 'r') as f:
        
        Z_group_name = _find_nearest_Z_group(f, metallicity_Z)
        Z_group = f[Z_group_name]
        
        wavelength = Z_group['wavelength_grid'][:]
        mass_grid = Z_group['mass_grid'][:]
        time_grid = f['metadata/time_grid'][:]
        
        # Cap limits
        if age_myr > time_grid[-1]:
            age_myr = time_grid[-1]
        if mass > mass_grid[0]:
            mass = mass_grid[0]
        if mass < mass_grid[-1]:
            mass = mass_grid[-1]
        
        # Simple case: use nearest mass
        idx_mass = np.argmin(np.abs(mass_grid - mass))
        closest_mass = mass_grid[idx_mass]
        
        # Get spectra: shape (n_times, n_wavelengths)
        spectra_data = Z_group[f'mass_{closest_mass:.1f}/flux_spectra'][:]
        
        # Interpolate in time for each wavelength
        flux = np.zeros(len(wavelength))
        for i_wave in range(len(wavelength)):
            flux[i_wave] = np.interp(age_myr, time_grid, spectra_data[:, i_wave])
        
        return wavelength, flux


def query_spectrum(
        database_path: str,
        metallicity: str,
        mass: float,
        ages_myr: np.ndarray,
        rotation: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch query database for spectra across multiple ages.
    
    This is a convenience wrapper around _query_spectrum_single() that:
    - Accepts string metallicity (converted to numeric)
    - Handles arrays of ages
    - Uses interface expected by stochastic interpolants code
    
    Parameters
    ----------
    database_path : str
        Path to HDF5 database
    metallicity : str
        Metallicity identifier ('MW', 'LMC', 'SMC', etc.)
    mass : float
        Stellar mass [Msun]
    ages_myr : np.ndarray
        Array of stellar ages [Myr], shape (n_ages,)
    rotation : bool
        Use rotating models (currently ignored, uses database as-is)
        
    Returns
    -------
    wavelength : np.ndarray
        Wavelength grid [Angstrom], shape (n_wavelengths,)
    flux_spectra : np.ndarray
        Flux density [10^-20 erg/s/Angstrom, LINEAR]
        Shape: (n_ages, n_wavelengths)
        
    Examples
    --------
    >>> wavelengths, spectra = query_spectrum(
    ...     database_path='database.h5',
    ...     metallicity='MW',
    ...     mass=25.0,
    ...     ages_myr=np.array([1.0, 2.0, 5.0, 10.0])
    ... )
    >>> print(spectra.shape)  # (4, n_wavelengths)
    """
    # Convert metallicity string to numeric value
    metallicity_Z = METALLICITY_MAPPING.get(metallicity)
    if metallicity_Z is None:
        raise ValueError(f"Unknown metallicity: {metallicity}. Available: {list(METALLICITY_MAPPING.keys())}")
    
    # Ensure ages_myr is array
    ages_myr = np.atleast_1d(ages_myr)
    n_ages = len(ages_myr)
    
    # Query first age to get wavelength grid and initialize output
    wavelength, flux_first = _query_spectrum_single(
        hdf5_file=database_path,
        mass=mass,
        age_myr=ages_myr[0],
        metallicity_Z=metallicity_Z
    )
    
    n_wavelengths = len(wavelength)
    flux_spectra = np.zeros((n_ages, n_wavelengths))
    flux_spectra[0, :] = flux_first
    
    # Query remaining ages
    for i_age in range(1, n_ages):
        _, flux_spectra[i_age, :] = _query_spectrum_single(
            hdf5_file=database_path,
            mass=mass,
            age_myr=ages_myr[i_age],
            metallicity_Z=metallicity_Z
        )
    
    return wavelength, flux_spectra


def _find_nearest_Z_group(f: h5py.File, Z_target: float) -> str:
    """Find HDF5 group name for nearest metallicity."""
    Z_groups = [key for key in f.keys() if key.startswith('Z_')]
    
    Z_values = np.array([f[g].attrs['Z_value'] for g in Z_groups])
    idx = np.argmin(np.abs(Z_values - Z_target))
    
    return Z_groups[idx]


def get_database_info(hdf5_file: str, verbose: bool = True) -> Dict:
    """
    Get database metadata and structure.
    
    Parameters
    ----------
    hdf5_file : str
        Path to database
    verbose : bool
        Print to screen
        
    Returns
    -------
    dict
        Database information
    """
    with h5py.File(hdf5_file, 'r') as f:
        
        meta = f['metadata']
        
        info = {
            'time_grid': meta['time_grid'][:],
            'spectral_library': meta.attrs['spectral_library'],
            'rotation': meta.attrs['rotation'],
            'metallicities': [],
            'mass_grids': {},
            'units': {
                'wind_power': meta.attrs['wind_power_units'],
                'wind_momentum': meta.attrs['wind_momentum_units'],
                'L_bol': meta.attrs['L_bol_units'],
                'Q': meta.attrs['Q_units'],
                'L_LyW': meta.attrs['L_LyW_units'],
                'flux_spectra': meta.attrs['flux_spectra_units'],
                'IMPORTANT': meta.attrs['IMPORTANT'],
            }
        }
        
        for group_name in f.keys():
            if group_name.startswith('Z_'):
                Z_group = f[group_name]
                met_string = Z_group.attrs['metallicity_string']
                Z_val = Z_group.attrs['Z_value']
                mass_grid = Z_group['mass_grid'][:]
                
                info['metallicities'].append((met_string, Z_val))
                info['mass_grids'][met_string] = mass_grid
        
        if verbose:
            print("="*80)
            print("SINGLE-STAR DATABASE INFO")
            print("="*80)
            print(f"File: {hdf5_file}")
            print(f"Library: {info['spectral_library']}, Rotation: {info['rotation']}")
            print(f"Time: {len(info['time_grid'])} points, "
                  f"{info['time_grid'][0]:.3f} - {info['time_grid'][-1]:.1f} Myr")
            print()
            print("Metallicities:")
            for met, Z in info['metallicities']:
                masses = info['mass_grids'][met]
                print(f"  {met:6s} (Z={Z:.4f}): {len(masses)} masses, "
                      f"{masses[-1]:.1f} - {masses[0]:.1f} Msun")
            print()
            print("Units:")
            for key, val in info['units'].items():
                print(f"  {key}: {val}")
            print("="*80)
        
        return info


if __name__ == "__main__":
    """
    Example: generate and query database.
    """
    
    # Generate
    print("Generating database...")
    generate_single_star_database(
        output_file='./database/single_star_tracks.h5',
        time_max_myr=100.0,
        time_resolution_myr=0.1,
        verbose=True
    )
    
    # Batch query - multiple quantities, multiple ages
    print("\nBatch querying database...")
    feedback = query_database(
        database_path='./database/single_star_tracks.h5',
        metallicity='MW',
        mass=25.0,
        ages_myr=np.array([1.0, 5.0, 10.0]),
        quantities=['Q_HI', 'wind_power']
    )
    print(f"25 Msun (MW):")
    print(f"  Ages: [1.0, 5.0, 10.0] Myr")
    print(f"  Q_HI: {feedback['Q_HI']}")
    print(f"  Wind power: {feedback['wind_power']}")
    
    # Spectrum query - multiple ages
    wave, flux_spectra = query_spectrum(
        database_path='./database/single_star_tracks.h5',
        metallicity='MW',
        mass=25.0,
        ages_myr=np.array([1.0, 5.0])
    )
    print(f"\nSpectra: {len(wave)} wavelengths, {flux_spectra.shape[0]} ages")
    print(f"Units: 10^-20 erg/s/Angstrom (LINEAR)")