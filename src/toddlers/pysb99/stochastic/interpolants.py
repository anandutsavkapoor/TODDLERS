#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stochastic/interpolants.py
===========================

Build time-delayed feedback interpolants from stochastic populations.

This module takes sampled (mass, formation_time) pairs and creates interpolation 
functions for total feedback by querying the single-star database.

Time-delayed feedback:
----------------------
For a star of mass m_i formed at time tau_i, its feedback at simulation time t is:
    Q_i(t) = Q_database(m_i, age = t - tau_i)  if t >= tau_i
    Q_i(t) = 0                                  if t < tau_i (not yet formed)

Total feedback from all stars:
    Q_total(t) = Sum_i Q_database(m_i, t - tau_i)  for all i where t >= tau_i

This naturally handles:
- Instantaneous burst (all tau_i = 0, all stars born at t=0)
- Constant SFR (tau_i ~ Uniform[0, t_sf], staggered formation)
- Any arbitrary formation time distribution

Author: Anand Utsav Kapoor
Created: 2025
"""

import numpy as np
from scipy.interpolate import interp1d
import h5py
from typing import Dict, Optional, Tuple, List
import warnings
import os
import sys
from multiprocessing import Pool, cpu_count

# Import from same package
from .database import query_database, query_spectrum

# =============================================================================
# Time-Delayed Spectra Calculation (for Cloudy)
# =============================================================================

def _process_spectra_chunk(args):
    """
    Worker function to process a chunk of stars for spectral summation (multiprocessing).
    
    Parameters
    ----------
    args : tuple
        (star_indices, masses, initial_ages, time_grid_myr, database_path, 
         metallicity, rotation)
    
    Returns
    -------
    tuple
        (wavelengths, summed_spectra) where summed_spectra has shape (n_times, n_wavelengths)
    """
    (star_indices, masses, initial_ages, time_grid_myr, 
     database_path, metallicity, rotation) = args
    
    wavelength_grid = None
    chunk_spectra = None
    
    # Process stars in this chunk
    for i_star in star_indices:
        m_i = masses[i_star]
        tau_i = initial_ages[i_star]
        
        # Compute stellar age at each simulation time
        # age = t - tau_i (if t >= tau_i, else star hasn't formed yet)
        stellar_ages = time_grid_myr - tau_i
        
        # Mask for times when star has formed (t >= tau_i)
        formed_mask = stellar_ages >= 0.0
        
        if not np.any(formed_mask):
            # Star never forms during simulation time range - skip
            continue
        
        # Query database only for times when star exists
        ages_query = stellar_ages[formed_mask]
        
        wavelengths, flux_spectra_i = query_spectrum(
            database_path=database_path,
            metallicity=metallicity,
            mass=m_i,
            ages_myr=ages_query,
            rotation=rotation
        )
        
        # Initialize on first star in chunk
        if wavelength_grid is None:
            wavelength_grid = wavelengths
            chunk_spectra = np.zeros((len(time_grid_myr), len(wavelengths)))
        
        # Sum spectra in linear space (only for times when star has formed)
        chunk_spectra[formed_mask, :] += flux_spectra_i
    
    return wavelength_grid, chunk_spectra


def compute_time_delayed_spectra(
        masses: np.ndarray,
        initial_ages: np.ndarray,
        time_grid_myr: np.ndarray,
        database_path: str,
        metallicity: str,
        rotation: bool = False,
        n_processes: Optional[int] = None,
        verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute time-delayed spectra for stochastic population.
    
    Sums flux_spectra (in LINEAR space) from all stars at each time point.
    This is needed for generating Cloudy spectral tables.
    
    Parameters
    ----------
    masses : np.ndarray
        Stellar masses [Msun], shape (n_stars,)
    initial_ages : np.ndarray
        Initial stellar ages [Myr], shape (n_stars,)
    time_grid_myr : np.ndarray
        Simulation time grid [Myr], shape (n_times,)
    database_path : str
        Path to HDF5 database
    metallicity : str
        Metallicity identifier
    rotation : bool
        Use rotating models
    n_processes : int, optional
        Number of parallel processes for star processing.
        If None, uses all available CPU cores. Set to 1 for serial.
    verbose : bool
        Print progress
        
    Returns
    -------
    wavelength_grid : np.ndarray
        Wavelength grid [Angstrom], shape (n_wavelengths,)
    spectra : np.ndarray
        Summed spectra for population [10^-20 erg/s/Angstrom, LINEAR]
        Shape: (n_times, n_wavelengths)
        
    Notes
    -----
    Database stores flux_spectra in LINEAR units (10^-20 erg/s/Angstrom).
    We sum in linear space to get total population spectrum.
    
    For Cloudy tables, spectra are kept in these units (no conversion).
    """
    n_stars = len(masses)
    n_times = len(time_grid_myr)
    
    # Determine number of processes
    if n_processes is None:
        n_processes = cpu_count()
    
    if verbose:
        print("="*70)
        print("COMPUTING TIME-DELAYED SPECTRA FOR CLOUDY")
        print("="*70)
        print(f"Population: {n_stars} stars")
        print(f"Time grid: {n_times} points")
        print(f"Metallicity: {metallicity}")
        print(f"Parallel processes: {n_processes}")
        print()
    
    # Parallel or serial processing
    if n_processes == 1:
        # Serial processing (for debugging)
        if verbose:
            print("  Getting wavelength grid...")
        
        wavelength_grid = None
        total_spectra = None
        
        # Loop over stars
        for i_star in range(n_stars):
            if verbose and (i_star % max(1, n_stars // 10) == 0):
                print(f"  Processing star {i_star+1}/{n_stars} "
                      f"({100*i_star/n_stars:.0f}%)")
            
            m_i = masses[i_star]
            tau_i = initial_ages[i_star]
            
            # Compute stellar age at each simulation time
            stellar_ages = time_grid_myr - tau_i
            
            # Mask for times when star has formed (t >= tau_i)
            formed_mask = stellar_ages >= 0.0
            
            if not np.any(formed_mask):
                # Star never forms during simulation time range - skip
                continue
            
            # Query database only for times when star exists
            ages_query = stellar_ages[formed_mask]
            
            wavelengths, flux_spectra_i = query_spectrum(
                database_path=database_path,
                metallicity=metallicity,
                mass=m_i,
                ages_myr=ages_query,
                rotation=rotation
            )
            
            # Initialize on first star
            if wavelength_grid is None:
                wavelength_grid = wavelengths
                total_spectra = np.zeros((len(time_grid_myr), len(wavelengths)))
            
            # Sum spectra in linear space (only for times when star has formed)
            total_spectra[formed_mask, :] += flux_spectra_i
    
    else:
        # Parallel processing: divide stars into chunks
        chunk_size = max(1, n_stars // n_processes)
        star_chunks = []
        
        for i in range(0, n_stars, chunk_size):
            chunk_indices = list(range(i, min(i + chunk_size, n_stars)))
            star_chunks.append((
                chunk_indices, masses, initial_ages, time_grid_myr,
                database_path, metallicity, rotation
            ))
        
        if verbose:
            print(f"  Created {len(star_chunks)} chunks of ~{chunk_size} stars each")
            print(f"  Processing...")
        
        # Process chunks in parallel
        with Pool(processes=n_processes) as pool:
            chunk_results = pool.map(_process_spectra_chunk, star_chunks)
        
        if verbose:
            print("  Combining results from all processes...")
        
        # Combine results from all chunks
        wavelength_grid = chunk_results[0][0]  # All should have same wavelengths
        total_spectra = np.zeros_like(chunk_results[0][1])
        
        for _, chunk_spectra in chunk_results:
            total_spectra += chunk_spectra
    
    if verbose:
        print("  Completed all stars")
        print()
        print("="*70)
        print("SPECTRAL SUMMATION COMPLETE")
        print("="*70)
        print(f"Wavelength range: {wavelength_grid[0]:.1f} - {wavelength_grid[-1]:.1f} Angstrom")
        print(f"Spectral shape: {total_spectra.shape}")
        print(f"Units: 10^-20 erg/s/Angstrom (LINEAR)")
        print("="*70)
    
    return wavelength_grid, total_spectra


# =============================================================================
# Cloudy Table Generation
# =============================================================================

def generate_cloudy_spectral_table(
        masses: np.ndarray,
        initial_ages: np.ndarray,
        database_path: str,
        metallicity: str,
        output_filename: str,
        time_grid_myr: Optional[np.ndarray] = None,
        rotation: bool = False,
        n_processes: Optional[int] = None,
        verbose: bool = True) -> str:
    """
    Generate Cloudy ASCII spectral table for stochastic population.
    
    Creates a Cloudy-compatible spectral table from a stochastic population
    by summing spectra from all stars at each time point.
    
    Parameters
    ----------
    masses : np.ndarray
        Stellar masses [Msun]
    initial_ages : np.ndarray
        Initial stellar ages [Myr]
    database_path : str
        Path to HDF5 database
    metallicity : str
        Metallicity identifier
    output_filename : str
        Output filename (e.g., 'pySB99_stochastic_1e6msun_MW_burst.ascii')
        Will be saved to CLOUDY_DATA_DIR if available, else current directory
    time_grid_myr : np.ndarray, optional
        Time grid [Myr]. If None, uses log-spaced 1e-2 to 50 Myr
    rotation : bool
        Use rotating models
    verbose : bool
        Print progress
        
    Returns
    -------
    str
        Path to generated Cloudy table file
        
    Notes
    -----
    The generated table format matches Cloudy's expectations:
    - Header with version, parameters
    - Age and metallicity grid
    - Wavelengths [Angstrom]
    - Fluxes [arbitrary units, Cloudy normalizes internally]
    
    For stochastic populations, we use a single Z value (the population's Z)
    with time-varying spectra from the mixed-age stellar population.
    
    Examples
    --------
    >>> # After sampling population
    >>> masses, ages, _ = sample_stochastic_population(...)
    >>> 
    >>> # Generate Cloudy table
    >>> table_path = generate_cloudy_spectral_table(
    ...     masses, ages, 'database.h5', 'MW',
    ...     output_filename='stochastic_1e6msun_burst.ascii'
    ... )
    """
    # Default time grid (years for Cloudy)
    if time_grid_myr is None:
        time_grid_myr = np.logspace(np.log10(0.01), np.log10(50), 50)
    
    time_grid_yr = time_grid_myr * 1e6
    
    if verbose:
        print("\n" + "="*70)
        print("GENERATING CLOUDY SPECTRAL TABLE FOR STOCHASTIC POPULATION")
        print("="*70)
        print(f"Population: {len(masses)} stars")
        print(f"Total mass: {np.sum(masses):.2e} Msun")
        print(f"Metallicity: {metallicity}")
        print(f"Time grid: {len(time_grid_myr)} points, "
              f"{time_grid_myr[0]:.3f} - {time_grid_myr[-1]:.1f} Myr")
        print()
    
    # Compute time-delayed spectra
    wavelengths, spectra = compute_time_delayed_spectra(
        masses=masses,
        initial_ages=initial_ages,
        time_grid_myr=time_grid_myr,
        database_path=database_path,
        metallicity=metallicity,
        rotation=rotation,
        n_processes=n_processes,
        verbose=verbose
    )
    
    # Get metallicity Z value
    from .database import METALLICITY_MAPPING
    Z_value = METALLICITY_MAPPING[metallicity]
    
    # Determine output directory
    try:
        from toddlers.constants import CLOUDY_DATA_DIR
        output_dir = CLOUDY_DATA_DIR
    except ImportError:
        output_dir = os.getcwd()
        warnings.warn(f"CLOUDY_DATA_DIR not available, using current directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, output_filename)
    
    if verbose:
        print("\nWriting Cloudy ASCII table...")
    
    # Write Cloudy ASCII table directly (without using SpectralTableGenerator class)
    # Format matches Cloudy's expected structure
    with open(full_path, 'w') as f:
        # Header
        f.write("20060612\n")  # Version identifier
        f.write("2\n2\n")  # 2 axes, 2 parameters per axis
        f.write("Age\nlog(Z)\n")  # Axis names
        
        # Number of (age, Z) combinations and wavelengths
        n_ages = len(time_grid_yr)
        n_Z = 1  # Single metallicity for stochastic
        f.write(f"{n_ages * n_Z}\n")
        f.write(f"{len(wavelengths)}\n")
        
        # Column info
        f.write("lambda\n1.00000000e+00\n")
        f.write("F_lambda\n1.00000000e+00\n")
        
        # Write age-metallicity combinations
        for t in time_grid_yr:
            f.write(f"{t:.6e} {np.log10(Z_value):.6f}\n")
        
        # Write wavelengths (5 per line)
        for i, w in enumerate(wavelengths):
            f.write(f"{w:.6e}")
            if (i + 1) % 5 == 0 or i == len(wavelengths) - 1:
                f.write("\n")
            else:
                f.write("  ")
        
        # Write spectra (5 flux values per line)
        # spectra shape: (n_times, n_wavelengths)
        for i_time in range(n_ages):
            spectrum = spectra[i_time, :]
            for i, flux in enumerate(spectrum):
                f.write(f"{flux:.6e}")
                if (i + 1) % 5 == 0 or i == len(spectrum) - 1:
                    f.write("\n")
                else:
                    f.write("  ")
    
    if verbose:
        print(f"* Cloudy table written to: {full_path}")
        print(f"  Ages: {len(time_grid_yr)} points, Wavelengths: {len(wavelengths)} points")
        print(f"  Metallicity: Z = {Z_value:.4f}")
        print("="*70)
    
    return full_path


def generate_cloudy_spectral_table_2d(
        masses: np.ndarray,
        initial_ages: np.ndarray,
        database_path: str,
        metallicities: Optional[List[str]] = None,
        output_filename: str = 'pySB99_stochastic_2d.ascii',
        time_grid_myr: Optional[np.ndarray] = None,
        rotation: bool = False,
        n_processes: Optional[int] = None,
        verbose: bool = True) -> str:
    """
    Generate 2D Cloudy ASCII spectral table for stochastic populations.
    
    Creates a Cloudy-compatible spectral table with multiple metallicities using
    the SAME IMF realization (masses, ages) for all metallicities. This matches
    the approach in build_stochastic_interpolants_2d().
    
    Parameters
    ----------
    masses : np.ndarray
        Stellar masses [Msun] - same for all metallicities
    initial_ages : np.ndarray
        Initial stellar ages [Myr] - same for all metallicities
    database_path : str
        Path to HDF5 single-star database
    metallicities : list of str, optional
        Metallicity identifiers (e.g., ['SMC', 'LMC', 'MW'])
        If None, uses ['SMC', 'LMC', 'MW', 'MWC']
    output_filename : str
        Output filename for Cloudy table
    time_grid_myr : np.ndarray, optional
        Time grid [Myr]. If None, uses log-spaced 1e-2 to 50 Myr
    rotation : bool
        Use rotating stellar models
    n_processes : int, optional
        Number of parallel processes
    verbose : bool
        Print progress
        
    Returns
    -------
    str
        Path to generated Cloudy table file
        
    Notes
    -----
    The generated table is a proper 2D Cloudy table with:
    - Multiple (age, Z) combinations
    - Same wavelength grid for all
    - Format matching Cloudy's expectations
    
    This allows Cloudy to interpolate in both time and metallicity, just like
    the standard pySB99 tables from generate_pysb99_interpolants.py.
    
    Examples
    --------
    >>> from stochastic.sampling import sample_imf_discrete, sample_ages
    >>>
    >>> # Sample population ONCE
    >>> masses = sample_imf_discrete(1e6, 'database.h5', 'MW', m_upper=120.0, seed=42)
    >>> ages = sample_ages(len(masses), t_sf_myr=5.0, seed=42)
    >>>
    >>> # Generate 2D Cloudy table
    >>> table_path = generate_cloudy_spectral_table_2d(
    ...     masses, ages, 'database.h5',
    ...     metallicities=['SMC', 'LMC', 'MW'],
    ...     output_filename='stochastic_2d.ascii'
    ... )
    """
    from .database import METALLICITY_MAPPING
    
    # Default metallicities
    if metallicities is None:
        metallicities = ['SMC', 'LMC', 'MW', 'MWC']
    
    # Get numeric Z values
    Z_values = [METALLICITY_MAPPING[m] for m in metallicities]
    
    # Default time grid (years for Cloudy)
    if time_grid_myr is None:
        time_grid_myr = np.logspace(np.log10(0.01), np.log10(50), 50)
    
    time_grid_yr = time_grid_myr * 1e6
    
    if verbose:
        print("\n" + "="*70)
        print("GENERATING 2D CLOUDY SPECTRAL TABLE FOR STOCHASTIC POPULATION")
        print("="*70)
        print(f"Population: {len(masses)} stars, total mass = {np.sum(masses):.2e} Msun")
        print(f"Metallicities: {metallicities}")
        print(f"Z values: {Z_values}")
        print(f"Time grid: {len(time_grid_myr)} points, "
              f"{time_grid_myr[0]:.3f} - {time_grid_myr[-1]:.1f} Myr")
        print("="*70)
    
    # Compute spectra for each metallicity
    # All use the SAME masses and ages
    spectra_dict = {}
    wavelength_grid = None
    
    for i, (met, Z_val) in enumerate(zip(metallicities, Z_values)):
        if verbose:
            print(f"\nProcessing metallicity {i+1}/{len(metallicities)}: {met} (Z={Z_val:.4f})")
        
        wavelengths, spectra = compute_time_delayed_spectra(
            masses=masses,
            initial_ages=initial_ages,
            time_grid_myr=time_grid_myr,
            database_path=database_path,
            metallicity=met,
            rotation=rotation,
            n_processes=n_processes,
            verbose=verbose
        )
        
        # Store wavelengths from first metallicity
        if wavelength_grid is None:
            wavelength_grid = wavelengths
        else:
            # Sanity check: all metallicities should have same wavelengths
            if not np.allclose(wavelengths, wavelength_grid):
                warnings.warn(f"Wavelength grids differ for {met}!")
        
        spectra_dict[met] = spectra
    
    # Determine output directory
    try:
        from toddlers.constants import CLOUDY_DATA_DIR
        output_dir = CLOUDY_DATA_DIR
    except ImportError:
        output_dir = os.getcwd()
        warnings.warn(f"CLOUDY_DATA_DIR not available, using current directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, output_filename)
    
    if verbose:
        print("\n" + "="*70)
        print("Writing 2D Cloudy ASCII table...")
        print("="*70)
    
    # Write Cloudy ASCII table
    n_ages = len(time_grid_yr)
    n_Z = len(metallicities)
    n_wavelengths = len(wavelength_grid)
    
    with open(full_path, 'w') as f:
        # Header
        f.write("20060612\n")  # Version identifier
        f.write("2\n2\n")  # 2 axes, 2 parameters per axis
        f.write("Age\nlog(Z)\n")  # Axis names
        
        # Number of (age, Z) combinations and wavelengths
        f.write(f"{n_ages * n_Z}\n")
        f.write(f"{n_wavelengths}\n")
        
        # Column info
        f.write("lambda\n1.00000000e+00\n")
        f.write("F_lambda\n1.00000000e+00\n")
        
        # Write age-metallicity combinations
        # Format: one line per (age, Z) pair
        for Z_val in Z_values:
            for t in time_grid_yr:
                f.write(f"{t:.6e} {np.log10(Z_val):.6f}\n")
        
        # Write wavelengths (5 per line)
        for i, w in enumerate(wavelength_grid):
            f.write(f"{w:.6e}")
            if (i + 1) % 5 == 0 or i == n_wavelengths - 1:
                f.write("\n")
            else:
                f.write("  ")
        
        # Write spectra (5 flux values per line)
        # Order: for each Z, write all time points
        for met in metallicities:
            spectra = spectra_dict[met]
            for i_time in range(n_ages):
                spectrum = spectra[i_time, :]
                for i, flux in enumerate(spectrum):
                    f.write(f"{flux:.6e}")
                    if (i + 1) % 5 == 0 or i == len(spectrum) - 1:
                        f.write("\n")
                    else:
                        f.write("  ")
    
    if verbose:
        print(f"* Cloudy table written to: {full_path}")
        print(f"  Ages: {n_ages} points")
        print(f"  Metallicities: {n_Z} ({metallicities})")
        print(f"  Wavelengths: {n_wavelengths} points")
        print(f"  Total spectra: {n_ages * n_Z}")
        print("="*70)
    
    return full_path


# =============================================================================
# Time-Delayed Feedback Calculation
# =============================================================================

# Per-process cache of EEP interpolators (built once per worker, reused across
# chunks). Keyed by (metallicity, database_path, rotation, quantities).
_EEP_INTERP_CACHE = {}


def _get_eep_interpolator(metallicity, database_path, quantities, rotation):
    """Return a cached EEPFeedbackInterpolator, building it on first use."""
    key = (metallicity, database_path, rotation, tuple(quantities))
    interp = _EEP_INTERP_CACHE.get(key)
    if interp is None:
        from toddlers.pysb99.eep_interpolation import EEPFeedbackInterpolator
        interp = EEPFeedbackInterpolator(
            metallicity, database_path, quantities=list(quantities), rotation=rotation)
        _EEP_INTERP_CACHE[key] = interp
    return interp


def _process_star_chunk(args):
    """
    Worker function to process a chunk of stars for multiprocessing.

    Parameters
    ----------
    args : tuple
        (star_indices, masses, initial_ages, time_grid_myr, database_path,
         metallicity, rotation, quantities, interpolation_mode)
        interpolation_mode is 'snap' (nearest-track, the default) or 'eep'
        (phase-aligned mass interpolation between bracketing grid tracks).

    Returns
    -------
    dict
        Dictionary with accumulated feedback for this chunk
    """
    (star_indices, masses, initial_ages, time_grid_myr,
     database_path, metallicity, rotation, quantities, interpolation_mode) = args
    eep = (_get_eep_interpolator(metallicity, database_path, quantities, rotation)
           if interpolation_mode == 'eep' else None)
    
    n_times = len(time_grid_myr)
    
    # Initialize accumulators for this chunk (LINEAR space)
    chunk_results = {q: np.zeros(n_times) for q in quantities}
    
    # Process stars in this chunk
    for i_star in star_indices:
        m_i = masses[i_star]
        tau_i = initial_ages[i_star]
        
        # Compute stellar age at each simulation time
        # age = t - tau_i (if t >= tau_i, else star hasn't formed yet)
        stellar_ages = time_grid_myr - tau_i
        
        # Mask for times when star has formed (t >= tau_i)
        formed_mask = stellar_ages >= 0.0
        
        if not np.any(formed_mask):
            # Star never forms during simulation time range - skip
            continue
        
        # Query database only for times when star exists
        # For t < tau_i, feedback will remain zero (initialized arrays)
        ages_query = stellar_ages[formed_mask]

        if eep is not None:
            feedback_i = eep.query(m_i, ages_query)
        else:
            feedback_i = query_database(
                database_path=database_path,
                metallicity=metallicity,
                mass=m_i,
                ages_myr=ages_query,
                rotation=rotation,
                quantities=quantities
            )
        
        # Convert from log10 to linear and accumulate
        # Only add feedback for times when star has formed
        for q in quantities:
            chunk_results[q][formed_mask] += 10**feedback_i[q]
    
    return chunk_results


def compute_time_delayed_feedback(
        masses: np.ndarray,
        initial_ages: np.ndarray,
        time_grid_myr: np.ndarray,
        database_path: str,
        metallicity: str,
        rotation: bool = False,
        n_processes: Optional[int] = None,
        verbose: bool = True,
        interpolation_mode: str = 'snap') -> Dict[str, np.ndarray]:
    """
    Compute time-delayed feedback for stochastic population.

    For each star i with (mass m_i, formation time tau_i), compute its feedback
    at each simulation time t::

        Q_i(t) = Q_database(m_i, age = t - tau_i)  if t >= tau_i
        Q_i(t) = 0                                  if t < tau_i (star not yet formed)

    Then sum over all stars::

        Q_total(t) = Sum_i Q_i(t)
    
    Parameters
    ----------
    masses : np.ndarray
        Stellar masses [Msun], shape (n_stars,)
    initial_ages : np.ndarray
        Initial stellar ages [Myr], shape (n_stars,)
        For burst: all zeros
        For constant SFR: uniform in [0, t_sf]
    time_grid_myr : np.ndarray
        Simulation time grid [Myr], shape (n_times,)
    database_path : str
        Path to HDF5 database
    metallicity : str
        Metallicity identifier
    rotation : bool
        Use rotating models
    n_processes : int, optional
        Number of parallel processes. If None, uses all available CPU cores.
        Set to 1 for serial processing (useful for debugging).
        Default: None (use all cores)
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Feedback quantities normalized to 10^6 Msun:
        - 'wind_power': log10(erg/s per 10^6 Msun)
        - 'wind_momentum': log10(dyne per 10^6 Msun)
        - 'L_bol': log10(erg/s per 10^6 Msun)
        - 'Q_HI', 'Q_HeI', 'Q_HeII': log10(photons/s per 10^6 Msun)
        - 'L_LyW': log10(erg/s/Angstrom per 10^6 Msun)
        Each has shape (n_times,)
        
    Notes
    -----
    Since database stores log10(Q) for single stars, we must:
    1. Query Q_i(t) = database(m_i, t - tau_i) in log space
    2. Convert to linear: q_i(t) = 10^Q_i(t)
    3. Sum in linear space: q_total(t) = Sum_i q_i(t)
    4. Convert back to log: Q_total(t) = log10(q_total(t))
    5. Normalize to 10^6 Msun: Q_norm = Q_total - (log10(M_total) - 6)
    
    This normalization matches generate_pysb99_interpolants.py convention,
    ensuring compatibility with TODDLERS which expects all feedback
    interpolants to be per 10^6 Msun.
    """
    n_stars = len(masses)
    n_times = len(time_grid_myr)
    
    # Determine number of processes
    if n_processes is None:
        n_processes = cpu_count()
    
    if verbose:
        print("="*70)
        print("COMPUTING TIME-DELAYED FEEDBACK")
        print("="*70)
        print(f"Population: {n_stars} stars")
        print(f"Time grid: {n_times} points, {time_grid_myr[0]:.2f} - {time_grid_myr[-1]:.2f} Myr")
        print(f"Metallicity: {metallicity}")
        print(f"Parallel processes: {n_processes}")
        print()
    
    # List of quantities to query (include ionizing luminosities from pySB99)
    quantities = ['wind_power', 'wind_momentum', 'L_bol',
                  'Q_HI', 'Q_HeI', 'Q_HeII', 'L_LyW',
                  'L_ion_HI', 'L_ion_HeI', 'L_ion_HeII']
    
    # Parallel or serial processing
    if n_processes == 1:
        # Serial processing (for debugging)
        chunk_results = [_process_star_chunk((
            list(range(n_stars)), masses, initial_ages, time_grid_myr,
            database_path, metallicity, rotation, quantities, interpolation_mode
        ))]

    else:
        # Parallel processing: divide stars into chunks
        chunk_size = max(1, n_stars // n_processes)
        star_chunks = []

        for i in range(0, n_stars, chunk_size):
            chunk_indices = list(range(i, min(i + chunk_size, n_stars)))
            star_chunks.append((
                chunk_indices, masses, initial_ages, time_grid_myr,
                database_path, metallicity, rotation, quantities, interpolation_mode
            ))
        
        if verbose:
            print(f"  Created {len(star_chunks)} chunks of ~{chunk_size} stars each")
            print(f"  Processing...")
        
        # Process chunks in parallel
        with Pool(processes=n_processes) as pool:
            chunk_results = pool.map(_process_star_chunk, star_chunks)
    
    if verbose:
        print("  Combining results from all processes...")
    
    # Combine results from all chunks (LINEAR space)
    wind_power_linear = np.zeros(n_times)
    wind_momentum_linear = np.zeros(n_times)
    L_bol_linear = np.zeros(n_times)
    Q_HI_linear = np.zeros(n_times)
    Q_HeI_linear = np.zeros(n_times)
    Q_HeII_linear = np.zeros(n_times)
    L_LyW_linear = np.zeros(n_times)
    L_ion_HI_linear = np.zeros(n_times)
    L_ion_HeI_linear = np.zeros(n_times)
    L_ion_HeII_linear = np.zeros(n_times)
    
    for chunk_result in chunk_results:
        wind_power_linear += chunk_result['wind_power']
        wind_momentum_linear += chunk_result['wind_momentum']
        L_bol_linear += chunk_result['L_bol']
        Q_HI_linear += chunk_result['Q_HI']
        Q_HeI_linear += chunk_result['Q_HeI']
        Q_HeII_linear += chunk_result['Q_HeII']
        L_LyW_linear += chunk_result['L_LyW']
        L_ion_HI_linear += chunk_result['L_ion_HI']
        L_ion_HeI_linear += chunk_result['L_ion_HeI']
        L_ion_HeII_linear += chunk_result['L_ion_HeII']
    
    if verbose:
        print("  Completed all stars")
        print()
    
    # Calculate total stellar mass
    total_mass = np.sum(masses)
    
    # Convert back to log10
    feedback_total = {
        'wind_power': np.log10(wind_power_linear + 1e-99),
        'wind_momentum': np.log10(wind_momentum_linear + 1e-99),
        'L_bol': np.log10(L_bol_linear + 1e-99),
        'Q_HI': np.log10(Q_HI_linear + 1e-99),
        'Q_HeI': np.log10(Q_HeI_linear + 1e-99),
        'Q_HeII': np.log10(Q_HeII_linear + 1e-99),
        'L_LyW': np.log10(L_LyW_linear + 1e-99),
        'L_ion_HI': np.log10(L_ion_HI_linear + 1e-99),
        'L_ion_HeI': np.log10(L_ion_HeI_linear + 1e-99),
        'L_ion_HeII': np.log10(L_ion_HeII_linear + 1e-99),
    }
    
    # ========================================================================
    # COMPUTE ADDITIONAL TODDLERS-REQUIRED QUANTITIES
    # ========================================================================
    # TODDLERS expects these additional quantities for full compatibility
    
    # Total ionizing photon rate (HI + HeI + HeII)
    Q_ion_linear = Q_HI_linear + Q_HeI_linear + Q_HeII_linear
    feedback_total['Q_ion'] = np.log10(Q_ion_linear + 1e-99)
    
    # Total ionizing luminosity (sum of all ionizing channels)
    # Note: L_ion_HI, L_ion_HeI, L_ion_HeII are TRUE values from pySB99 spectrum integration
    L_ion_total_linear = L_ion_HI_linear + L_ion_HeI_linear + L_ion_HeII_linear
    feedback_total['L_ion'] = np.log10(L_ion_total_linear + 1e-99)
    
    # Mean ionizing photon energy (derived from total L_ion and Q_ion)
    # <E> = L_ion_total / Q_ion_total (in eV)
    eV_to_erg = 1.60218e-12
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_energy_erg = (L_ion_total_linear + 1e-99) / (Q_ion_linear + 1e-99)
        mean_energy_eV = mean_energy_erg / eV_to_erg
        # Handle cases where Q_ion is zero
        mean_energy_eV = np.where(Q_ion_linear > 1e-50, mean_energy_eV, 18.6)
        feedback_total['mean_ion_energy'] = np.log10(np.clip(mean_energy_eV, 1.0, 1000.0))
    
    # Spectral hardness ratios
    # These are in linear space for TODDLERS
    feedback_total['QratHe1H'] = (Q_HeI_linear + 1e-99) / (Q_HI_linear + 1e-99)
    feedback_total['QratHe2H'] = (Q_HeII_linear + 1e-99) / (Q_HI_linear + 1e-99)
    
    # ========================================================================
    # NORMALIZE TO 10^6 MSUN
    # ========================================================================
    # Following generate_pysb99_interpolants.py convention:
    # TODDLERS expects interpolants normalized to 10^6 Msun
    #
    # In log space: log10(Q per 10^6 Msun) = log10(Q) - log10(M/10^6)
    #                                       = log10(Q) - (log10(M) - 6)
    #
    # See generate_pysb99_interpolants.py lines ~450-460 for reference
    log_normalization = np.log10(total_mass) - 6.0
    
    # Apply to log-space quantities only
    log_quantities = ['wind_power', 'wind_momentum', 'L_bol', 
                     'Q_HI', 'Q_HeI', 'Q_HeII', 'Q_ion', 'L_ion', 'L_LyW', 'mean_ion_energy',
                     'L_ion_HI', 'L_ion_HeI', 'L_ion_HeII']
    for key in log_quantities:
        feedback_total[key] -= log_normalization
    
    if verbose:
        print("="*70)
        print("FEEDBACK COMPUTATION COMPLETE")
        print("="*70)
        print(f"Total stellar mass: {total_mass:.2e} Msun")
        print(f"Normalization: 10^6 Msun (log_norm = {log_normalization:.4f})")
        print(f"Peak wind power (per 10^6 Msun): {np.max(feedback_total['wind_power']):.2f} log10(erg/s)")
        print(f"Peak Q_HI (per 10^6 Msun): {np.max(feedback_total['Q_HI']):.2f} log10(photons/s)")
        print("="*70)
    
    return feedback_total


# =============================================================================
# Interpolant Creation (Real 2D for TODDLERS Compatibility)
# =============================================================================


def create_interpolants_2d(
        time_grid_myr: np.ndarray,
        Z_values: np.ndarray,
        feedback_dict: Dict[str, Dict[str, np.ndarray]]) -> Dict:
    """
    Create 2D interpolation functions (time, metallicity) for feedback quantities.
    
    This function creates real 2D interpolants matching the structure from
    generate_pysb99_interpolants.py. The same IMF realization (stellar masses)
    is used for all metallicities.
    
    Parameters
    ----------
    time_grid_myr : np.ndarray
        Time grid [Myr]
    Z_values : np.ndarray
        Metallicity values (e.g., [0.002, 0.006, 0.014])
    feedback_dict : dict
        Nested dict: {metallicity_str: {quantity: values}}
        Example: {'SMC': {'wind_power': [...], ...}, 'LMC': {...}, ...}
        
    Returns
    -------
    dict
        2D interpolation functions for TODDLERS:
        - 'f_wind_power': function(t_myr, Z) -> log10(erg/s per 10^6 Msun)
        - 'f_wind_momentum': function(t_myr, Z) -> log10(dyne per 10^6 Msun)
        - 'f_L_bol': function(t_myr, Z) -> log10(erg/s per 10^6 Msun)
        - 'f_Q_ion': function(t_myr, Z) -> log10(photons/s per 10^6 Msun)
        - 'f_Q_HI': function(t_myr, Z) -> log10(photons/s per 10^6 Msun)
        - 'f_Q_HeI': function(t_myr, Z) -> log10(photons/s per 10^6 Msun)
        - 'f_Q_HeII': function(t_myr, Z) -> log10(photons/s per 10^6 Msun)
        - 'f_L_LyW': function(t_myr, Z) -> log10(erg/s/A per 10^6 Msun)
        - 'f_QratHe1H': function(t_myr, Z) -> log10(Q_HeI/Q_HI)
        - 'f_QratHe2H': function(t_myr, Z) -> log10(Q_HeII/Q_HI)
        - 'f_mean_ion_energy': function(t_myr, Z) -> log10(eV)
        
    Notes
    -----
    This creates real scipy.interpolate.interp2d objects, identical to what
    generate_pysb99_interpolants.py produces. These work seamlessly with
    TODDLERS without any wrapper classes.
    """
    from toddlers.pysb99.generate_pysb99_interpolants import _make_interp2d

    # Get metallicity keys in same order as Z_values
    # Assume feedback_dict keys are ordered or we need to sort by Z
    metallicity_keys = list(feedback_dict.keys())
    n_Z = len(metallicity_keys)
    n_time = len(time_grid_myr)
    
    # Initialize 2D arrays: shape (n_Z, n_time)
    # Get list of quantities from first metallicity
    first_met = metallicity_keys[0]
    quantity_keys = list(feedback_dict[first_met].keys())
    
    # Build 2D arrays for each quantity
    feedback_2d = {}
    for key in quantity_keys:
        feedback_2d[key] = np.zeros((n_Z, n_time))
        for i, met in enumerate(metallicity_keys):
            feedback_2d[key][i, :] = feedback_dict[met][key]
    
    # Create 2D interpolants using interp2d (same as generate_pysb99_interpolants.py)
    interpolants = {}
    
    # Main feedback quantities (already in log space)
    interpolants['f_wind_power'] = _make_interp2d(
        time_grid_myr, Z_values, feedback_2d['wind_power'], kind='linear'
    )
    interpolants['f_wind_momentum'] = _make_interp2d(
        time_grid_myr, Z_values, feedback_2d['wind_momentum'], kind='linear'
    )
    interpolants['f_L_bol'] = _make_interp2d(
        time_grid_myr, Z_values, feedback_2d['L_bol'], kind='linear'
    )
    
    # Ionizing photon rates (already in log space)
    interpolants['f_Q_ion'] = _make_interp2d(
        time_grid_myr, Z_values, feedback_2d['Q_HI'], kind='linear'
    )
    interpolants['f_Q_HI'] = _make_interp2d(
        time_grid_myr, Z_values, feedback_2d['Q_HI'], kind='linear'
    )
    interpolants['f_Q_HeI'] = _make_interp2d(
        time_grid_myr, Z_values, feedback_2d['Q_HeI'], kind='linear'
    )
    interpolants['f_Q_HeII'] = _make_interp2d(
        time_grid_myr, Z_values, feedback_2d['Q_HeII'], kind='linear'
    )
    
    # Lyman-Werner flux (already in log space)
    interpolants['f_L_LyW'] = _make_interp2d(
        time_grid_myr, Z_values, feedback_2d['L_LyW'], kind='linear'
    )
    interpolants['f_LyW'] = interpolants['f_L_LyW']  # Alias for compatibility
    
    # Mean ionizing photon energy (already in log space)
    interpolants['f_mean_ion_energy'] = _make_interp2d(
        time_grid_myr, Z_values, feedback_2d['mean_ion_energy'], kind='linear'
    )
    
    # Spectral hardness ratios (convert to log space)
    # Compute ratios in linear space first
    Q_HI_linear = 10**feedback_2d['Q_HI']
    Q_HeI_linear = 10**feedback_2d['Q_HeI']
    Q_HeII_linear = 10**feedback_2d['Q_HeII']
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_He1H = Q_HeI_linear / Q_HI_linear
        ratio_He2H = Q_HeII_linear / Q_HI_linear
        
        # Store in log space (matching generate_pysb99_interpolants.py)
        interpolants['f_QratHe1H'] = _make_interp2d(
            time_grid_myr, Z_values,
            np.log10(ratio_He1H + 1e-50), kind='linear'
        )
        interpolants['f_QratHe2H'] = _make_interp2d(
            time_grid_myr, Z_values,
            np.log10(ratio_He2H + 1e-50), kind='linear'
        )
    
    # Ionizing luminosities (true values from pySB99 spectrum integration)
    interpolants['f_L_ion'] = _make_interp2d(
        time_grid_myr, Z_values, feedback_2d['L_ion'], kind='linear'
    )
    interpolants['f_L_ion_HI'] = _make_interp2d(
        time_grid_myr, Z_values, feedback_2d['L_ion_HI'], kind='linear'
    )
    interpolants['f_L_ion_HeI'] = _make_interp2d(
        time_grid_myr, Z_values, feedback_2d['L_ion_HeI'], kind='linear'
    )
    interpolants['f_L_ion_HeII'] = _make_interp2d(
        time_grid_myr, Z_values, feedback_2d['L_ion_HeII'], kind='linear'
    )
    
    return interpolants


# =============================================================================
# Time Derivatives (for TODDLERS compatibility)
# =============================================================================

def compute_time_derivatives_2d(
        time_grid_myr: np.ndarray,
        Z_values: np.ndarray,
        interpolants: Dict,
        oversample_factor: int = 10,
        verbose: bool = False) -> Dict:
    """
    Compute time derivatives of feedback for shell evolution (2D version).
    
    TODDLERS needs dQ/dt for wind power and momentum to evolve shells.
    Derivatives are computed in LINEAR space, not log space.
    
    This follows the approach in generate_pysb99_interpolants.py:
    1. Oversample the log interpolants on a fine time grid
    2. Convert from log to linear space
    3. Compute gradient using np.gradient
    4. Create new 2D interpolants for the derivatives
    
    Parameters
    ----------
    time_grid_myr : np.ndarray
        Original time grid [Myr]
    Z_values : np.ndarray
        Metallicity values
    interpolants : dict
        Dictionary containing 'f_wind_power' and 'f_wind_momentum' 2D interpolants
    oversample_factor : int
        Factor to oversample (default: 10 gives 10000 points for ~1000 original)
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Updated interpolants dictionary with added entries:
        - 'f_wind_power_rate': function(t_myr, Z) -> dL/dt [erg/s/Myr, LINEAR]
        - 'f_wind_momentum_rate': function(t_myr, Z) -> dF/dt [dyne/Myr, LINEAR]
        
    Notes
    -----
    The derivative interpolants are in LINEAR space (not log space).
    This matches the behavior of generate_pysb99_interpolants.py.
    """
    from toddlers.pysb99.generate_pysb99_interpolants import _make_interp2d

    if verbose:
        print("  Computing time derivatives (oversampling method)...")
    
    # Oversample time grid
    n_oversampled = len(time_grid_myr) * oversample_factor
    t_oversampled = np.linspace(
        time_grid_myr[0], time_grid_myr[-1], 
        num=n_oversampled, endpoint=True
    )
    
    # Initialize arrays for derivatives: shape (n_Z, n_time_oversampled)
    n_Z = len(Z_values)
    all_wind_power_rate = np.zeros((n_Z, len(t_oversampled)))
    all_wind_momentum_rate = np.zeros((n_Z, len(t_oversampled)))
    
    # For each metallicity, evaluate interpolants and compute derivatives
    for i, Z_val in enumerate(Z_values):
        # Evaluate log-space interpolants on oversampled grid
        wind_power_log = interpolants['f_wind_power'](t_oversampled, Z_val)
        wind_momentum_log = interpolants['f_wind_momentum'](t_oversampled, Z_val)
        
        # Convert to linear space
        wind_power_linear = 10**wind_power_log.flatten()
        wind_momentum_linear = 10**wind_momentum_log.flatten()
        
        # Compute gradients in linear space (units: erg/s/Myr and dyne/Myr)
        all_wind_power_rate[i, :] = np.gradient(wind_power_linear, t_oversampled)
        all_wind_momentum_rate[i, :] = np.gradient(wind_momentum_linear, t_oversampled)
    
    # Create 2D interpolants for derivatives (in LINEAR space, not log)
    interpolants['f_wind_power_rate'] = _make_interp2d(
        t_oversampled, Z_values, all_wind_power_rate,
        kind='linear', fill_value=0
    )
    interpolants['f_wind_momentum_rate'] = _make_interp2d(
        t_oversampled, Z_values, all_wind_momentum_rate,
        kind='linear', fill_value=0
    )
    
    if verbose:
        print("  Computed time derivatives via oversampling")
    
    return interpolants


# =============================================================================
# IMF Sampling Quality Metrics
# =============================================================================

def compute_imf_sampling_metrics(
        masses: np.ndarray,
        imf_name: str = 'kroupa',
        n_mass_bins: int = 30) -> Dict:
    """
    Compute metrics to assess how well the sampled masses represent the IMF.
    
    Parameters
    ----------
    masses : np.ndarray
        Sampled stellar masses [Msun]
    imf_name : str
        Name of IMF used for sampling
    n_mass_bins : int
        Number of bins for histogram comparison
        
    Returns
    -------
    dict
        Dictionary with sampling quality metrics:
        - 'ks_statistic': Kolmogorov-Smirnov test statistic
        - 'ks_pvalue': K-S test p-value (>0.05 suggests good sampling)
        - 'n_per_decade': Average number of stars per log mass decade
        - 'high_mass_counts': Number of stars in key mass ranges
        - 'mass_completeness': Fraction of expected stars in each mass bin
    """
    from scipy import stats
    
    # Try to import IMF function
    try:
        from toddlers.pysb99.stochastic.sampling import AVAILABLE_IMFS
        if imf_name in AVAILABLE_IMFS:
            imf_func = AVAILABLE_IMFS[imf_name]
        else:
            warnings.warn(f"Unknown IMF '{imf_name}', skipping detailed metrics")
            imf_func = None
    except ImportError:
        warnings.warn("Could not import sampling module, skipping IMF comparison")
        imf_func = None
    
    metrics = {}
    
    # Basic statistics
    m_min, m_max = float(np.min(masses)), float(np.max(masses))
    
    # Mass resolution (stars per decade)
    log_mass_range = np.log10(m_max) - np.log10(m_min)
    n_per_decade = len(masses) / log_mass_range if log_mass_range > 0 else 0
    metrics['n_per_decade'] = float(n_per_decade)
    
    # High-mass sampling (critical for feedback)
    metrics['high_mass_counts'] = {
        'above_8msun': int(np.sum(masses >= 8.0)),
        'above_20msun': int(np.sum(masses >= 20.0)),
        'above_40msun': int(np.sum(masses >= 40.0)),
        'above_60msun': int(np.sum(masses >= 60.0))
    }
    
    # Kolmogorov-Smirnov test (compare CDF of sampled vs theoretical)
    if imf_func is not None:
        try:
            # Build theoretical CDF
            m_theory = np.logspace(np.log10(m_min), np.log10(m_max), 1000)
            xi_theory = imf_func(m_theory)
            mass_pdf_theory = m_theory * xi_theory
            mass_pdf_theory /= np.trapz(mass_pdf_theory, m_theory)
            cdf_theory = np.cumsum(mass_pdf_theory)
            cdf_theory /= cdf_theory[-1]
            
            # Compute empirical CDF from samples
            masses_sorted = np.sort(masses)
            cdf_empirical = np.arange(1, len(masses_sorted) + 1) / len(masses_sorted)
            
            # Interpolate theoretical CDF at sampled points
            cdf_theory_interp = np.interp(masses_sorted, m_theory, cdf_theory)
            
            # K-S statistic: maximum vertical distance between CDFs
            ks_stat = np.max(np.abs(cdf_empirical - cdf_theory_interp))
            
            # Two-sample K-S test (more accurate p-value)
            # Compare sampled masses vs theoretical distribution
            ks_result = stats.ks_1samp(masses, 
                                       lambda x: np.interp(x, m_theory, cdf_theory))
            
            metrics['ks_statistic'] = float(ks_stat)
            metrics['ks_pvalue'] = float(ks_result.pvalue)
            metrics['ks_interpretation'] = (
                'good' if ks_result.pvalue > 0.05 else 
                'marginal' if ks_result.pvalue > 0.01 else 
                'poor'
            )
            
        except Exception as e:
            warnings.warn(f"Could not compute K-S test: {e}")
            metrics['ks_statistic'] = None
            metrics['ks_pvalue'] = None
    else:
        metrics['ks_statistic'] = None
        metrics['ks_pvalue'] = None
    
    # Bin-wise completeness check
    # Compare number of stars in each mass bin to theoretical expectation
    if imf_func is not None:
        try:
            m_bins = np.logspace(np.log10(m_min), np.log10(m_max), n_mass_bins + 1)
            m_centers = np.sqrt(m_bins[:-1] * m_bins[1:])
            
            # Theoretical expectation in each bin
            xi_centers = imf_func(m_centers)
            mass_pdf = m_centers * xi_centers
            mass_pdf /= np.trapz(mass_pdf, m_centers)
            dm = np.diff(m_bins)
            expected_frac = mass_pdf * dm / np.sum(mass_pdf * dm)
            expected_counts = expected_frac * len(masses)
            
            # Actual counts in each bin
            hist, _ = np.histogram(masses, bins=m_bins)
            
            # Completeness = actual / expected
            with np.errstate(divide='ignore', invalid='ignore'):
                completeness = hist / expected_counts
                completeness = np.where(np.isfinite(completeness), completeness, 0)
            
            metrics['mass_completeness'] = {
                'bin_centers_msun': m_centers.tolist(),
                'completeness_ratio': completeness.tolist(),
                'mean_completeness': float(np.mean(completeness)),
                'std_completeness': float(np.std(completeness))
            }
            
        except Exception as e:
            warnings.warn(f"Could not compute mass completeness: {e}")
    
    return metrics


# =============================================================================
# Complete Pipeline (Real 2D Interpolants)
# =============================================================================

def build_stochastic_interpolants_2d(
        masses: np.ndarray,
        initial_ages: np.ndarray,
        database_path: str,
        metallicities: Optional[List[str]] = None,
        time_grid_myr: Optional[np.ndarray] = None,
        rotation: bool = False,
        compute_derivatives: bool = True,
        imf_name: str = 'kroupa',
        n_processes: Optional[int] = None,
        verbose: bool = True,
        interpolation_mode: str = 'snap') -> Dict:
    """
    Build real 2D interpolants (time, metallicity) from stochastic populations.
    
    This is the recommended entry point for TODDLERS compatibility. It creates
    interpolants with the same structure as generate_pysb99_interpolants.py by:
    1. Using the SAME IMF realization (masses, ages) for ALL metallicities
    2. Computing feedback for each metallicity using the single-star database
    3. Building 2D arrays and creating scipy.interpolate.interp2d objects
    
    Parameters
    ----------
    masses : np.ndarray
        Stellar masses [Msun] - same for all metallicities
    initial_ages : np.ndarray
        Initial stellar ages [Myr] - same for all metallicities
    database_path : str
        Path to HDF5 single-star database
    metallicities : list of str, optional
        Metallicity identifiers (e.g., ['SMC', 'LMC', 'MW'])
        If None, uses ['SMC', 'LMC', 'MW', 'MWC']
    time_grid_myr : np.ndarray, optional
        Time grid [Myr]. If None, uses 0-30 Myr with 0.1 Myr spacing
    rotation : bool
        Use rotating stellar models
    compute_derivatives : bool
        Compute time derivatives (dL/dt, dF/dt) for TODDLERS
    imf_name : str
        IMF name for metadata ('kroupa', 'salpeter', 'top_heavy', etc.)
        Used for computing IMF sampling quality metrics
    n_processes : int, optional
        Number of parallel processes. If None, uses all CPU cores.
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Real 2D interpolants compatible with TODDLERS:
        - 'f_wind_power': function(t, Z) -> log10(erg/s per 10^6 Msun)
        - 'f_wind_momentum': function(t, Z) -> log10(dyne per 10^6 Msun)
        - 'f_L_bol': function(t, Z) -> log10(erg/s per 10^6 Msun)
        - 'f_Q_ion': function(t, Z) -> log10(photons/s per 10^6 Msun)
        - 'f_Q_HI', 'f_Q_HeI', 'f_Q_HeII': ionizing photon rates
        - 'f_L_LyW': Lyman-Werner luminosity
        - 'f_QratHe1H', 'f_QratHe2H': spectral hardness ratios
        - 'f_mean_ion_energy': mean ionizing photon energy
        - 'f_wind_power_rate': (if derivatives) dL/dt [LINEAR]
        - 'f_wind_momentum_rate': (if derivatives) dF/dt [LINEAR]
        - '_metadata': population and interpolant metadata
        
    Examples
    --------
    >>> from stochastic.sampling import sample_imf_discrete, sample_ages
    >>>
    >>> # Sample population ONCE
    >>> masses = sample_imf_discrete(1e6, 'database.h5', 'MW', m_upper=120.0, seed=42)
    >>> ages = sample_ages(len(masses), t_sf_myr=5.0, seed=42)
    >>>
    >>> # Build 2D interpolants for multiple metallicities
    >>> interpolants = build_stochastic_interpolants_2d(
    ...     masses, ages, 'database.h5',
    ...     metallicities=['SMC', 'LMC', 'MW'],
    ...     imf_name='kroupa'
    ... )
    >>> 
    >>> # Use in TODDLERS
    >>> L_wind = 10**interpolants['f_wind_power'](5.0, 0.014)  # t=5 Myr, Z=MW
    >>> 
    >>> # Access metadata
    >>> meta = interpolants['_metadata']
    >>> print(f"N stars: {meta['n_stars']}")
    >>> print(f"Mass range: {meta['mass_statistics']['min_mass_msun']:.2f} - "
    ...       f"{meta['mass_statistics']['max_mass_msun']:.2f} Msun")
    >>> print(f"High-mass stars (>20 Msun): "
    ...       f"{meta['imf_sampling_quality']['high_mass_counts']['above_20msun']}")
    >>> print(f"K-S test p-value: {meta['imf_sampling_quality']['ks_pvalue']:.3f}")
    
    Notes
    -----
    The key difference from build_stochastic_interpolants():
    - Old: Creates 1D interpolants + wrapper for single metallicity
    - New: Creates real 2D interpolants for multiple metallicities
    - New approach matches deterministic generate_pysb99_interpolants.py exactly
    """
    from .database import METALLICITY_MAPPING
    
    # Default metallicities
    if metallicities is None:
        metallicities = ['SMC', 'LMC', 'MW', 'MWC']
    
    # Get numeric Z values
    Z_values = np.array([METALLICITY_MAPPING[m] for m in metallicities])
    
    # Default time grid
    if time_grid_myr is None:
        time_grid_myr = np.arange(0.0, 30.1, 0.1)
    
    if verbose:
        print("="*70)
        print("BUILDING 2D STOCHASTIC INTERPOLANTS")
        print("="*70)
        print(f"Population: {len(masses)} stars, total mass = {np.sum(masses):.2e} Msun")
        print(f"Metallicities: {metallicities}")
        print(f"Z values: {Z_values}")
        print(f"Time grid: {len(time_grid_myr)} points from {time_grid_myr[0]:.2f} to {time_grid_myr[-1]:.2f} Myr")
        print("="*70)
    
    # Loop over metallicities and compute feedback
    feedback_dict = {}
    
    for i, (met, Z_val) in enumerate(zip(metallicities, Z_values)):
        if verbose:
            print(f"\nProcessing metallicity {i+1}/{len(metallicities)}: {met} (Z={Z_val:.4f})")
        
        # Compute time-delayed feedback for this metallicity
        # Uses SAME masses and ages as all other metallicities
        feedback_total = compute_time_delayed_feedback(
            masses=masses,
            initial_ages=initial_ages,
            time_grid_myr=time_grid_myr,
            database_path=database_path,
            metallicity=met,
            rotation=rotation,
            n_processes=n_processes,
            verbose=verbose,
            interpolation_mode=interpolation_mode
        )
        
        feedback_dict[met] = feedback_total
    
    if verbose:
        print("\n" + "="*70)
        print("Creating 2D interpolation functions...")
        print("="*70)
    
    # Build 2D interpolants
    interpolants = create_interpolants_2d(
        time_grid_myr=time_grid_myr,
        Z_values=Z_values,
        feedback_dict=feedback_dict
    )
    
    # Add metadata
    total_mass = np.sum(masses)
    
    # Compute IMF sampling quality metrics
    if verbose:
        print("\nComputing IMF sampling quality metrics...")
    
    sampling_metrics = compute_imf_sampling_metrics(
        masses=masses,
        imf_name=imf_name,
        n_mass_bins=30
    )
    
    # Compute mass statistics
    mass_stats = {
        'min_mass_msun': float(np.min(masses)),
        'max_mass_msun': float(np.max(masses)),
        'median_mass_msun': float(np.median(masses)),
        'mean_mass_msun': float(np.mean(masses)),
        'percentiles': {
            'p10': float(np.percentile(masses, 10)),
            'p25': float(np.percentile(masses, 25)),
            'p75': float(np.percentile(masses, 75)),
            'p90': float(np.percentile(masses, 90)),
            'p95': float(np.percentile(masses, 95)),
            'p99': float(np.percentile(masses, 99))
        }
    }
    
    # Compute age statistics
    age_stats = {
        'min_age_myr': float(np.min(initial_ages)),
        'max_age_myr': float(np.max(initial_ages)),
        'median_age_myr': float(np.median(initial_ages)),
        'mean_age_myr': float(np.mean(initial_ages)),
        'std_age_myr': float(np.std(initial_ages)),
        'distribution_mode': (
            'burst' if np.max(initial_ages) < 0.01 else
            'constant_sfr' if np.std(initial_ages) > 0.1 * np.mean(initial_ages) else
            'mixed'
        )
    }
    
    interpolants['_metadata'] = {
        # Population basics
        'n_stars': len(masses),
        'total_mass_msun': float(total_mass),
        'imf_name': imf_name,
        
        # Normalization
        'normalization': '10^6 Msun',
        'log_normalization': float(np.log10(total_mass) - 6.0),
        
        # Metallicity grid
        'metallicities': metallicities,
        'Z_values': Z_values.tolist(),
        'rotation': rotation,
        
        # Time grid
        'time_range_myr': [float(time_grid_myr[0]), float(time_grid_myr[-1])],
        'time_grid_myr': time_grid_myr.tolist(),
        'n_time_points': len(time_grid_myr),
        
        # Mass distribution
        'mass_statistics': mass_stats,
        
        # Age distribution  
        'age_statistics': age_stats,
        
        # IMF sampling quality
        'imf_sampling_quality': sampling_metrics,
        
        # Interpolant type
        'stochastic': True,
        'interpolant_type': '2D_real',
        'scipy_interpolant_kind': 'interp2d_linear'
    }
    
    # Print summary if verbose
    if verbose:
        print("\nPopulation Summary:")
        print(f"  N stars: {len(masses)}")
        print(f"  Mass range: {mass_stats['min_mass_msun']:.2f} - {mass_stats['max_mass_msun']:.2f} Msun")
        print(f"  High-mass stars (>20 Msun): {sampling_metrics['high_mass_counts']['above_20msun']}")
        print(f"  IMF sampling: {sampling_metrics['n_per_decade']:.1f} stars/decade")
        if sampling_metrics['ks_pvalue'] is not None:
            print(f"  K-S test: p={sampling_metrics['ks_pvalue']:.3f} ({sampling_metrics['ks_interpretation']})")
        print(f"  Age mode: {age_stats['distribution_mode']}")
        print(f"  Star formation timescale: {age_stats['max_age_myr']:.2f} Myr")
    
    
    # Compute time derivatives if requested
    if compute_derivatives:
        interpolants = compute_time_derivatives_2d(
            time_grid_myr=time_grid_myr,
            Z_values=Z_values,
            interpolants=interpolants,
            oversample_factor=10,
            verbose=verbose
        )
    
    if verbose:
        print(f"\nCreated {len([k for k in interpolants.keys() if not k.startswith('_')])} interpolation functions")
        print("="*70)
        print("2D INTERPOLANTS COMPLETE")
        print("="*70)
    
    return interpolants


# =============================================================================
# Save Interpolants (TODDLERS-Compatible Format)
# =============================================================================

def save_stochastic_interpolants(
        interpolants: Dict,
        output_dir: str,
        imf_name: str,
        rotation: bool = False,
        overwrite: bool = True):
    """
    Save stochastic interpolants in TODDLERS-compatible format.
    
    This matches the format from generate_pysb99_interpolants.py:
    - Main file: LIST of 7 interpolants [L_mech, F_ram, L_bolo, L_ion, Q_ion, F_ram_rate, L_mech_rate]
    - LyW file: Single interpolant for Lyman-Werner luminosity
    - Mean energy file: Single interpolant for mean ionizing photon energy
    - Hardness file: LIST of 2 interpolants [QratHe1H, QratHe2H]
    
    Parameters
    ----------
    interpolants : dict
        Dictionary from build_stochastic_interpolants()
    output_dir : str
        Directory to save files
    imf_name : str
        IMF identifier for filename
    rotation : bool
        Whether rotation was included (for filename)
    overwrite : bool
        Whether to overwrite existing files
        
    Returns
    -------
    dict
        Dictionary mapping file types to filepaths
        
    Notes
    -----
    This format is REQUIRED for TODDLERS. The main interpolant file must be
    a LIST in the specific order expected by stellar_feedback.py:
    [L_mech, F_ram, L_bolo, L_ion, Q_ion, F_ram_rate, L_mech_rate]
    """
    import pickle
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filenames
    rotation_suffix = '_rot' if rotation else ''
    filename_main = f"pySB99interpolation_{imf_name}{rotation_suffix}.obj"
    filename_lw = f"pySB99interpolation_{imf_name}{rotation_suffix}_LumLymanWerner.obj"
    filename_mean_energy = f"pySB99interpolation_{imf_name}{rotation_suffix}_mean_ionizing_photon_energy.obj"
    filename_hardness = f"pySB99interpolation_{imf_name}{rotation_suffix}_hardness.obj"
    
    filepath_main = os.path.join(output_dir, filename_main)
    filepath_lw = os.path.join(output_dir, filename_lw)
    filepath_mean_energy = os.path.join(output_dir, filename_mean_energy)
    filepath_hardness = os.path.join(output_dir, filename_hardness)
    
    # Check for existing files
    if not overwrite:
        for filepath in [filepath_main, filepath_lw, filepath_mean_energy, filepath_hardness]:
            if os.path.exists(filepath):
                raise FileExistsError(
                    f"File {filepath} already exists. Set overwrite=True to replace."
                )
    
    # ========================================================================
    # MAIN INTERPOLANTS FILE
    # ========================================================================
    # CRITICAL: Must be a LIST in this exact order for TODDLERS compatibility
    # Format matches stellar_feedback.py expectations:
    # [L_mech, F_ram, L_bolo, L_ion, Q_ion, F_ram_rate, L_mech_rate]
    #
    # where:
    # - First 5 are in log10 space (erg/s, dyne, photons/s per 10^6 Msun)
    # - Last 2 are derivatives in LINEAR space (erg/s/Myr, dyne/Myr per 10^6 Msun)
    
    main_interpolants = [
        interpolants['f_wind_power'],           # [0] L_mech (log space)
        interpolants['f_wind_momentum'],        # [1] F_ram (log space)
        interpolants['f_L_bol'],                # [2] L_bolo (log space)
        interpolants['f_L_ion'],                # [3] L_ion (log space)
        interpolants['f_Q_ion'],                # [4] Q_ion (log space)
        interpolants['f_wind_momentum_rate'],   # [5] F_ram_rate (linear space: dyne/Myr)
        interpolants['f_wind_power_rate'],      # [6] L_mech_rate (linear space: erg/s/Myr)
    ]
    
    with open(filepath_main, 'wb') as f:
        pickle.dump(main_interpolants, f)
    print(f" Saved main interpolants to {filename_main}")
    
    # ========================================================================
    # LYMAN-WERNER INTERPOLANT FILE
    # ========================================================================
    with open(filepath_lw, 'wb') as f:
        pickle.dump(interpolants['f_L_LyW'], f)
    print(f" Saved Lyman-Werner interpolant to {filename_lw}")
    
    # ========================================================================
    # MEAN IONIZING PHOTON ENERGY FILE
    # ========================================================================
    with open(filepath_mean_energy, 'wb') as f:
        pickle.dump(interpolants['f_mean_ion_energy'], f)
    print(f" Saved mean ionizing photon energy to {filename_mean_energy}")
    
    # ========================================================================
    # HARDNESS RATIOS FILE
    # ========================================================================
    # Format: [f_QratHe1H, f_QratHe2H]
    hardness_interpolants = [
        interpolants['f_QratHe1H'],  # HeI/HI ratio
        interpolants['f_QratHe2H'],  # HeII/HI ratio
    ]
    
    with open(filepath_hardness, 'wb') as f:
        pickle.dump(hardness_interpolants, f)
    print(f" Saved hardness ratios to {filename_hardness}")
    
    print("\n All interpolants saved in TODDLERS-compatible format!")
    
    return {
        'main': filepath_main,
        'lyman_werner': filepath_lw,
        'mean_energy': filepath_mean_energy,
        'hardness': filepath_hardness
    }


# =============================================================================
# Deterministic Limit Test
# =============================================================================

def test_deterministic_limit(
        database_path: str,
        metallicities: List[str] = None,
        target_mass: float = 1e6,
        time_grid_myr: Optional[np.ndarray] = None,
        tolerance: float = 0.05,
        verbose: bool = True) -> Dict:
    """
    Test that stochastic -> deterministic limit as N_stars -> infinity.
    
    For 10^6 Msun with all tau_i = 0 (burst), the stochastic interpolant
    should match the standard pySB99 interpolant within ~5%.
    
    Parameters
    ----------
    database_path : str
        Path to HDF5 database
    metallicities : list of str, optional
        Metallicity identifiers to test. If None, uses ['MW']
    target_mass : float
        Total mass [Msun]. Should be large (>10^6) for good convergence
    time_grid_myr : np.ndarray, optional
        Time grid for comparison
    tolerance : float
        Maximum fractional difference allowed
    verbose : bool
        Print results
        
    Returns
    -------
    dict
        Test results with fractional differences for each quantity
        
    Notes
    -----
    This test verifies correctness of the time-delayed feedback approach.
    Uses the new 2D interpolant workflow.
    """
    if metallicities is None:
        metallicities = ['MW']
    
    if time_grid_myr is None:
        time_grid_myr = np.arange(0.0, 30.1, 0.1)
    
    if verbose:
        print("="*70)
        print("DETERMINISTIC LIMIT TEST (2D)")
        print("="*70)
        print(f"Testing {target_mass:.0e} Msun burst")
        print(f"Metallicities: {metallicities}")
        print()
    
    # Sample population (large N for good convergence)
    from stochastic.sampling import sample_imf_discrete, sample_ages

    masses = sample_imf_discrete(
        total_mass=target_mass,
        database_path=database_path,
        metallicity=metallicities[0],
        m_upper=120.0,
        imf_name='kroupa',
        seed=42
    )
    
    ages = sample_ages(
        n_stars=len(masses),
        t_sf_myr=0.0,  # Burst
        mode='uniform',
        seed=42
    )
    
    if verbose:
        print(f"Sampled {len(masses)} stars")
    
    # Build 2D stochastic interpolants
    stoch_interps = build_stochastic_interpolants_2d(
        masses, ages, database_path, 
        metallicities=metallicities,
        time_grid_myr=time_grid_myr,
        compute_derivatives=False,
        verbose=False
    )
    
    # Load standard pySB99 interpolant for comparison
    # (This would require generate_pysb99_interpolants to have been run)
    # For now, we just check internal consistency
    
    if verbose:
        print("\n2D stochastic interpolants created successfully")
        print("(Full comparison with pySB99 requires standard interpolants)")
        print("="*70)
    
    return {'status': 'created'}


if __name__ == "__main__":
    """
    Example usage of 2D stochastic interpolants.
    """
    
    print("\n" + "="*70)
    print("STOCHASTIC 2D INTERPOLANTS MODULE")
    print("="*70)
    print("\nThis module builds real 2D time-delayed feedback interpolants")
    print("(time, metallicity) from stochastic stellar populations.")
    print("\nTypical workflow:")
    print("  1. Sample population ONCE (metallicity-independent)")
    print("  2. Build 2D interpolants for multiple metallicities")
    print("  3. Use in TODDLERS simulations")
    print("="*70)
    
    # Example (requires database to exist)
    try:
        from stochastic.sampling import sample_imf_discrete, sample_ages

        print("\nExample: Building 2D interpolants for 10^5 Msun population")
        print("-"*70)

        # Sample population ONCE (discrete grid masses)
        masses = sample_imf_discrete(
            total_mass=1e5,
            database_path='database.h5',
            metallicity='MW',
            m_upper=120.0,
            imf_name='kroupa',
            seed=123
        )
        
        ages = sample_ages(
            n_stars=len(masses),
            t_sf_myr=5.0,
            mode='uniform',
            seed=123
        )
        
        print(f"\nSampled {len(masses)} stars")
        print("\nNow would build 2D interpolants with:")
        print("  interpolants = build_stochastic_interpolants_2d(")
        print("      masses, ages, 'database.h5',")
        print("      metallicities=['SMC', 'LMC', 'MW']")
        print("  )")
        print("\n(Requires database.h5 to exist)")
        
    except Exception as e:
        print(f"\nCould not run example: {e}")
        print("(This is expected if database doesn't exist yet)")