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
from .database import query_database, query_spectrum, get_supernova_table

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

def _stochastic_table_key(masses, initial_ages, database_path, metallicities, rotation,
                          time_grid_myr):
    """Content key of a stochastic Cloudy table: everything the spectra depend on."""
    import hashlib
    from toddlers.cloudy_stellar_spectra_generator import _file_md5, STAR_TABLE_CODE_VERSION
    arr_md5 = lambda a: hashlib.md5(np.ascontiguousarray(a, dtype=float).tobytes()).hexdigest()
    return {'kind': 'pySB99_stochastic', 'table_code_version': STAR_TABLE_CODE_VERSION,
            'masses_md5': arr_md5(masses), 'initial_ages_md5': arr_md5(initial_ages),
            'n_stars': int(len(masses)), 'metallicities': list(metallicities),
            'rotation': bool(rotation), 'time_grid_myr': [float(t) for t in time_grid_myr],
            'database_md5': _file_md5(database_path)}


def _stochastic_table_path(output_filename, output_dir=None):
    """Destination of a stochastic table: output_dir if given, else CLOUDY_DATA_DIR if
    available, else the cwd."""
    if output_dir is None:
        try:
            from toddlers.constants import CLOUDY_DATA_DIR
            output_dir = CLOUDY_DATA_DIR
        except ImportError:
            output_dir = os.getcwd()
            warnings.warn(f"CLOUDY_DATA_DIR not available, using current directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, os.path.join(output_dir, output_filename)


def _reuse_stochastic_table(full_path, output_dir, key, time_grid_yr, Z_values, verbose):
    """True if the table at full_path holds this request; an outdated one is moved aside."""
    from toddlers.cloudy_stellar_spectra_generator import (star_table_is_current,
                                                           move_star_table_aside)
    if not os.path.exists(full_path):
        return False
    if star_table_is_current(full_path, key, time_grid_yr, Z_values):
        if verbose:
            print(f"* Cloudy table '{os.path.basename(full_path)}' is current. Reusing it.")
        return True
    move_star_table_aside(full_path, output_dir)
    return False


def _paired_md5(paired_feedback_files):
    if not paired_feedback_files:
        return None
    from toddlers.cloudy_stellar_spectra_generator import files_md5
    return files_md5(paired_feedback_files)


def generate_cloudy_spectral_table(
        masses: np.ndarray,
        initial_ages: np.ndarray,
        database_path: str,
        metallicity: str,
        output_filename: str,
        time_grid_myr: Optional[np.ndarray] = None,
        rotation: bool = False,
        n_processes: Optional[int] = None,
        verbose: bool = True,
        paired_feedback_files: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        paired_feedback_content_md5: Optional[str] = None) -> str:
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
    paired_feedback_files : list of str, optional
        Files of the feedback table this spectral table pairs with; their md5 is recorded
        in the sidecar as ``paired_feedback_md5`` (run-time generations record the table's
        content md5 as ``paired_feedback_content_md5`` instead).
    output_dir : str, optional
        Directory to write the table to; default CLOUDY_DATA_DIR if available, else the
        current directory.
    paired_feedback_content_md5 : str, optional
        Content md5 of the unified feedback table this star table pairs with (a run-time
        generation table, FeedbackTable.content_md5); recorded in the sidecar.
        
    Returns
    -------
    str
        Path to generated Cloudy table file
        
    Notes
    -----
    The table is written by
    :func:`toddlers.cloudy_stellar_spectra_generator.write_cloudy_star_table` (grid checks,
    sidecar ``<table>.meta.json``). An existing table is reused only if its sidecar key
    (population, metallicity, rotation, time grid, database md5) matches; otherwise it is
    moved to ``stale_tables/``.

    The generated table format matches Cloudy's expectations:
    - Header with version, parameters
    - Age and metallicity grid
    - Wavelengths [Angstrom]
    - Fluxes [arbitrary units, Cloudy normalizes internally]
    
    For stochastic populations, we use a single Z value (the population's Z)
    with time-varying spectra from the mixed-age stellar population.
    
    Examples
    --------
    >>> # After sampling the population (m_upper is required)
    >>> from toddlers.pysb99.stochastic.population import IMFSpec, draw_stop_after, generation_rng
    >>> masses = draw_stop_after(IMFSpec('kroupa', 0.1, 100.), 1e6, generation_rng(42, 0))
    >>> masses = masses[masses >= 0.8]   # stars below the lowest track give no light
    >>> ages = sample_ages(len(masses), t_sf_myr=5.0)
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
    
    # Get metallicity Z value
    from .database import METALLICITY_MAPPING
    Z_value = METALLICITY_MAPPING[metallicity]

    output_dir, full_path = _stochastic_table_path(output_filename, output_dir)
    key = _stochastic_table_key(masses, initial_ages, database_path, [metallicity], rotation,
                                time_grid_myr)
    if _reuse_stochastic_table(full_path, output_dir, key, time_grid_yr, [Z_value], verbose):
        return full_path

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

    if verbose:
        print("\nWriting Cloudy ASCII table...")

    # One metallicity: spectra (n_times, n_wavelengths) -> (1, n_times, n_wavelengths)
    from toddlers.cloudy_stellar_spectra_generator import write_cloudy_star_table
    side = {'origin': 'generated', 'key': key,
            'paired_feedback_md5': _paired_md5(paired_feedback_files),
            'units': {'lambda': 'Angstrom', 'F_lambda': '1e20 erg s-1 Angstrom-1',
                      'F_lambda_note': 'sum over the stars of the population (not normalized)'}}
    if paired_feedback_content_md5 is not None:
        side['paired_feedback_content_md5'] = paired_feedback_content_md5
    write_cloudy_star_table(full_path, time_grid_yr, [Z_value], wavelengths, [spectra], sidecar=side)

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
        verbose: bool = True,
        paired_feedback_files: Optional[List[str]] = None) -> str:
    """
    Generate 2D Cloudy ASCII spectral table for stochastic populations.
    
    Creates a Cloudy-compatible spectral table with multiple metallicities using
    the SAME IMF realization (masses, ages) for all metallicities.
    
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
    paired_feedback_files : list of str, optional
        The paired feedback table files; their md5 goes into the sidecar
        (``paired_feedback_md5``).
        
    Returns
    -------
    str
        Path to generated Cloudy table file
        
    Notes
    -----
    Written by :func:`toddlers.cloudy_stellar_spectra_generator.write_cloudy_star_table`,
    with the same sidecar and reuse rule as generate_cloudy_spectral_table.

    The generated table is a proper 2D Cloudy table with:
    - Multiple (age, Z) combinations
    - Same wavelength grid for all
    - Format matching Cloudy's expectations
    
    This allows Cloudy to interpolate in both time and metallicity, just like
    the standard pySB99 tables from generate_pysb99_interpolants.py.
    
    Examples
    --------
    >>> from toddlers.pysb99.stochastic.population import IMFSpec, draw_stop_after, generation_rng
    >>> from toddlers.pysb99.stochastic.sampling import sample_ages
    >>>
    >>> # Sample population ONCE (full IMF range; keep the stars that have tracks)
    >>> masses = draw_stop_after(IMFSpec('kroupa', 0.1, 100.), 1e6, generation_rng(42, 0))
    >>> masses = masses[masses >= 0.8]
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
    
    output_dir, full_path = _stochastic_table_path(output_filename)
    key = _stochastic_table_key(masses, initial_ages, database_path, metallicities, rotation,
                                time_grid_myr)
    if _reuse_stochastic_table(full_path, output_dir, key, time_grid_yr, Z_values, verbose):
        return full_path

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
    
    if verbose:
        print("\n" + "="*70)
        print("Writing 2D Cloudy ASCII table...")
        print("="*70)

    n_ages = len(time_grid_yr)
    n_Z = len(metallicities)
    n_wavelengths = len(wavelength_grid)

    # Order: for each Z, all time points
    from toddlers.cloudy_stellar_spectra_generator import write_cloudy_star_table
    write_cloudy_star_table(full_path, time_grid_yr, Z_values, wavelength_grid,
                            [spectra_dict[met] for met in metallicities],
                            sidecar={'origin': 'generated', 'key': key,
                                     'paired_feedback_md5': _paired_md5(paired_feedback_files)})

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


def supernova_feedback(masses, initial_ages, time_grid_myr, database_path, metallicity) -> Dict:
    """SN power and momentum per time bin for a stochastic population.

    Each star with sn_mass_min <= m <= sn_mass_max explodes once, at its formation
    time plus its death time; t_death and M_final are interpolated between the
    database masses by monotone PCHIP in log-log (exact at database masses), the same
    rule as the deterministic pySB99 SN term, in both snap and eep modes. Bins are centred on time_grid_myr (midpoint edges); events outside the grid
    are dropped. Returns per-bin 'sn_number', 'sn_power' [erg/s] and 'sn_momentum'
    [dyn], plus 'n_exploding' and 'n_sn_in_grid'.
    """
    from ..pysb99_core import StellarPopulationSynthesis, YEAR_S, MSUN_G, in_sn_window, loglog_pchip
    tab = get_supernova_table(database_path, metallicity)
    m = np.asarray(masses, float)
    explodes = in_sn_window(m, tab['sn_mass_min'], tab['sn_mass_max'])
    t_d = loglog_pchip(tab['mass'], tab['t_death_myr'])(m)
    m_ej = np.maximum(loglog_pchip(tab['mass'], tab['M_final_msun'])(m) - tab['sn_remnant_mass'], 0.0)
    p_sn = np.sqrt(2.0 * tab['sn_energy_erg'] * m_ej * MSUN_G)
    t_sn = np.asarray(initial_ages, float) + t_d
    edges = StellarPopulationSynthesis.centred_bin_edges(time_grid_myr)
    dt_s = np.diff(edges) * 1e6 * YEAR_S
    w = explodes.astype(float)
    n_bin = StellarPopulationSynthesis.deposit_events(edges, t_sn[explodes], t_sn[explodes], w[explodes])
    p_bin = StellarPopulationSynthesis.deposit_events(edges, t_sn[explodes], t_sn[explodes], p_sn[explodes])
    return {'sn_number': n_bin, 'sn_power': n_bin * tab['sn_energy_erg'] / dt_s,
            'sn_momentum': p_bin / dt_s, 'n_exploding': int(explodes.sum()),
            'n_sn_in_grid': float(n_bin.sum())}


def compute_time_delayed_feedback(
        masses: np.ndarray,
        initial_ages: np.ndarray,
        time_grid_myr: np.ndarray,
        database_path: str,
        metallicity: str,
        rotation: bool = False,
        n_processes: Optional[int] = None,
        verbose: bool = True,
        interpolation_mode: str = 'snap',
        include_supernovae: bool = True) -> Dict[str, np.ndarray]:
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
    
    # Core-collapse supernovae: one instantaneous event per exploding star, at
    # tau_i + t_death(m_i), deposited into the time bin (centred on time_grid_myr) that
    # holds it, as E_SN / dt and sqrt(2 E_SN M_ej) / dt. t_death and M_final are PCHIP in
    # log-log between the database masses, in both interpolation modes.
    sn = supernova_feedback(masses, initial_ages, time_grid_myr, database_path, metallicity) \
        if include_supernovae else None
    if sn is not None:
        wind_power_linear += sn['sn_power']
        wind_momentum_linear += sn['sn_momentum']
        if verbose:
            print(f"  Supernovae: {sn['n_sn_in_grid']:.0f} in the time grid "
                  f"({sn['n_exploding']} exploding stars)")

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
    
    # Sum of the three (nested) band photon rates: diagnostic only, see below
    Q_ion_linear = Q_HI_linear + Q_HeI_linear + Q_HeII_linear
    # The pySB99 bands are nested (Q_HI already counts every photon below 912 A), so the
    # hydrogen-ionizing rate StellarFeedback reads is Q_HI, as in the deterministic tables
    # (generate_pysb99_interpolants.py f_Q_ion). Q_ion_linear (the band sum) is kept only for
    # the mean-energy diagnostic below, which uses the same sums as the deterministic tables.
    feedback_total['Q_ion'] = np.log10(Q_HI_linear + 1e-99)
    
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

    # Supernova diagnostics (NOT normalized: totals for this population, linear units)
    if sn is not None:
        feedback_total['sn_power_linear'] = sn['sn_power']        # erg/s
        feedback_total['sn_momentum_linear'] = sn['sn_momentum']  # dyn
        feedback_total['sn_number'] = sn['sn_number']
    
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
    This creates real Interp2DLinear objects (toddlers._interp_compat), identical to what
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
# Pre-drawn stochastic tables (removed)
# =============================================================================

_PRE_DRAWN_REMOVED = (
    "{name} was removed: it built a stochastic feedback table from one pre-drawn star list, "
    "which Evolution does not read. Stochastic populations are drawn at run time, one "
    "independent draw per star-forming generation: Evolution(template='pySB99', imf=None, "
    "stochastic=StochasticSpec(IMFSpec(name, m_low, m_up), base_seed=...)) "
    "(toddlers.pysb99.stochastic.population).")


def build_stochastic_interpolants_2d(*args, **kwargs):
    """Removed: see _PRE_DRAWN_REMOVED. Use StochasticSpec in Evolution."""
    raise RuntimeError(_PRE_DRAWN_REMOVED.format(name='build_stochastic_interpolants_2d'))


def save_stochastic_interpolants(*args, **kwargs):
    """Removed: see _PRE_DRAWN_REMOVED. Use StochasticSpec in Evolution."""
    raise RuntimeError(_PRE_DRAWN_REMOVED.format(name='save_stochastic_interpolants'))
