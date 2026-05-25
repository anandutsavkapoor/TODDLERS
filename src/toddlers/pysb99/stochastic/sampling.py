#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stochastic/sampling.py
======================

Sample stochastic stellar populations from an IMF with age distribution.

This module generates populations of individual stars by:
1. Sampling masses from an IMF (e.g., Kroupa, Salpeter, or Custom Top-Heavy)
2. Sampling ages uniformly over star formation duration
3. Capping masses that exceed database limits
4. Visualizing and SAVING the resulting populations to disk

Author: Anand Utsav Kapoor
Created: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple, Optional, Callable
import warnings

# ============================================================================
# IMF Definitions & Generators
# ============================================================================

def create_broken_power_law(alpha1: float, alpha2: float, m_break: float = 0.5) -> Callable:
    """Factory function to create a general 2-part broken power law IMF."""
    def custom_imf(m: np.ndarray) -> np.ndarray:
        xi = np.zeros_like(m)
        mask_low = m < m_break
        xi[mask_low] = m[mask_low]**(-alpha1)
        
        mask_high = m >= m_break
        norm_factor = m_break**(alpha2 - alpha1)
        xi[mask_high] = norm_factor * m[mask_high]**(-alpha2)
        return xi
    return custom_imf

def kroupa_imf(m: np.ndarray) -> np.ndarray:
    """Standard Kroupa (2001) IMF (alpha1=1.3, alpha2=2.3, break=0.5)."""
    return create_broken_power_law(1.3, 2.3, 0.5)(m)

def top_heavy_imf(m: np.ndarray) -> np.ndarray:
    """Example Top-Heavy IMF (alpha1=1.3, alpha2=1.0, break=0.5)."""
    return create_broken_power_law(1.3, 1.0, 0.5)(m)

def salpeter_imf(m: np.ndarray) -> np.ndarray:
    """Salpeter (1955) IMF: single power law."""
    xi = np.zeros_like(m)
    mask = (m >= 0.1) & (m <= 100.0)
    xi[mask] = m[mask]**(-2.35)
    return xi

def chabrier_imf(m: np.ndarray) -> np.ndarray:
    """Chabrier (2003) IMF: log-normal + power law."""
    xi = np.zeros_like(m)
    mask_low = (m >= 0.01) & (m < 1.0)
    m_c, sigma = 0.08, 0.69
    xi[mask_low] = (1.0 / m[mask_low]) * np.exp(
        -(np.log10(m[mask_low]) - np.log10(m_c))**2 / (2 * sigma**2)
    )
    mask_high = (m >= 1.0) & (m <= 100.0)
    norm_factor = (1.0 / 1.0) * np.exp(0) / (1.0**(-2.3))
    xi[mask_high] = norm_factor * m[mask_high]**(-2.3)
    return xi

AVAILABLE_IMFS = {
    'kroupa': kroupa_imf,
    'salpeter': salpeter_imf,
    'chabrier': chabrier_imf,
    'top_heavy': top_heavy_imf
}

# ============================================================================
# IMF Sampling
# ============================================================================

def sample_imf(
        total_mass: float,
        imf_name: str = 'kroupa',
        imf_func: Optional[Callable] = None,
        m_min: float = 0.08,
        m_max: float = 120.0,
        n_samples: int = 10000,
        seed: Optional[int] = None) -> np.ndarray:
    """
    Sample stellar masses until total_mass is reached.

    Uses a stop-after approach: draws stars until cumulative mass exceeds
    the target, then truncates to the star that crosses the threshold.
    The resulting total mass will be close to but not exactly equal to
    total_mass.

    NOTE: This produces continuous mass values. For exact database grid masses,
    use sample_imf_discrete() instead.
    """
    if seed is not None:
        np.random.seed(seed)
    
    if imf_func is None:
        if imf_name not in AVAILABLE_IMFS:
            raise ValueError(f"Unknown IMF: {imf_name}. Available: {list(AVAILABLE_IMFS.keys())}")
        imf_func = AVAILABLE_IMFS[imf_name]
    
    # 1. Prepare PDF/CDF
    m_grid = np.logspace(np.log10(m_min), np.log10(m_max), n_samples)
    xi = imf_func(m_grid)
    mass_dist = m_grid * xi
    mass_dist = mass_dist / np.trapz(mass_dist, m_grid)
    cdf = np.cumsum(mass_dist)
    cdf = cdf / cdf[-1]
    
    mean_mass = np.trapz(m_grid * mass_dist, m_grid)
    
    # 2. Iterative Sampling (Fill up the bucket)
    # We sample in batches until we have enough mass
    current_mass = 0.0
    all_masses = []
    
    while current_mass < total_mass:
        remaining_mass = total_mass - current_mass
        
        # Estimate how many stars we need for the remaining mass
        # Add a 20% buffer + 10 stars to ensure we overshoot eventually
        n_needed = int(remaining_mass / mean_mass * 1.2) + 10
        
        u = np.random.uniform(0, 1, n_needed)
        batch_masses = np.interp(u, cdf, m_grid)
        
        all_masses.append(batch_masses)
        current_mass += np.sum(batch_masses)
    
    # Flatten list of arrays
    masses = np.concatenate(all_masses)
    
    # 3. Truncate and Rescale
    # We now have > total_mass. We trim the excess.
    cumsum_mass = np.cumsum(masses)
    cutoff_idx = np.searchsorted(cumsum_mass, total_mass)
    
    # Select stars up to the cutoff
    masses = masses[:cutoff_idx+1]
    
    return masses


def sample_imf_discrete(
        total_mass: float,
        database_path: str,
        metallicity: str,
        m_upper: float,
        imf_name: str = 'kroupa',
        imf_func: Optional[Callable] = None,
        rotation: bool = False,
        seed: Optional[int] = None,
        max_mass_error_pct: float = 1.0,
        verbose: bool = False) -> np.ndarray:
    """
    Sample stellar masses from discrete database grid.

    This ensures all sampled masses exactly match database entries,
    avoiding mass interpolation in query operations. The mass grid
    is truncated at m_upper before sampling, so the full mass budget
    goes into stars <= m_upper.

    Parameters
    ----------
    total_mass : float
        Target total stellar mass [Msun]
    database_path : str
        Path to HDF5 database
    metallicity : str
        Metallicity identifier (MW, LMC, etc.)
    m_upper : float
        Upper stellar mass limit [Msun]. The database grid is truncated
        at this value before sampling.
    imf_name : str
        IMF name ('kroupa', 'salpeter', etc.)
    imf_func : callable, optional
        Custom IMF function. If provided, overrides imf_name
    rotation : bool
        Use rotating stellar models
    seed : int, optional
        Random seed for reproducibility
    max_mass_error_pct : float
        Maximum allowed error in total mass (default: 1%)
        If exceeded, raises warning
    verbose : bool
        Print diagnostics
        
    Returns
    -------
    np.ndarray
        Stellar masses [Msun], all exact grid values
        
    Notes
    -----
    This function:
    1. Samples from continuous IMF distribution
    2. Rounds each mass to nearest database grid point
    3. Final total mass may differ from target by ~0.1-1% due to rounding
    
    All returned masses are guaranteed to be exact database grid values,
    eliminating interpolation overhead in feedback queries.
    
    Examples
    --------
    >>> from stochastic.sampling import sample_imf_discrete
    >>> masses = sample_imf_discrete(
    ...     total_mass=1e6,
    ...     database_path='database.h5',
    ...     metallicity='MW',
    ...     m_upper=120.0,
    ...     seed=42
    ... )
    >>> # All masses are now exact grid values - no interpolation needed
    """
    # Import here to avoid circular dependency
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from toddlers.pysb99.stochastic.database import get_available_mass_grid
    
    if seed is not None:
        np.random.seed(seed)
    
    # Get discrete mass grid from database, truncated at m_upper
    mass_grid = get_available_mass_grid(metallicity, rotation)
    mass_grid = mass_grid[mass_grid <= m_upper]

    if len(mass_grid) == 0:
        raise ValueError(
            f"No database masses <= {m_upper:.1f} Msun for metallicity {metallicity}"
        )

    if verbose:
        print(f"Database mass grid: {len(mass_grid)} masses")
        print(f"  Range: {mass_grid.min():.2f} - {mass_grid.max():.2f} Msun")

    # Sample from continuous IMF first
    m_min = mass_grid.min()
    m_max = mass_grid.max()
    
    continuous_masses = sample_imf(
        total_mass=total_mass,
        imf_name=imf_name,
        imf_func=imf_func,
        m_min=m_min,
        m_max=m_max,
        seed=seed
    )
    
    if verbose:
        print(f"Sampled {len(continuous_masses)} stars from continuous IMF")
        print(f"  Continuous total mass: {np.sum(continuous_masses):.2e} Msun")
    
    # Round each mass to nearest grid point
    discrete_masses = np.array([
        mass_grid[np.argmin(np.abs(mass_grid - m))]
        for m in continuous_masses
    ])
    
    # Check mass error
    actual_total = np.sum(discrete_masses)
    error_pct = 100 * abs(actual_total - total_mass) / total_mass
    
    if verbose or error_pct > max_mass_error_pct:
        print(f"Discrete sampling results:")
        print(f"  Target mass:  {total_mass:.2e} Msun")
        print(f"  Actual mass:  {actual_total:.2e} Msun")
        print(f"  Error:        {error_pct:.2f}%")
        print(f"  N stars:      {len(discrete_masses)}")
    
    if error_pct > max_mass_error_pct:
        warnings.warn(
            f"Discrete sampling mass error ({error_pct:.2f}%) exceeds threshold "
            f"({max_mass_error_pct:.1f}%). Consider using larger population "
            f"or accepting continuous masses with interpolation."
        )
    
    # Verify all masses are in grid (sanity check)
    for m in discrete_masses:
        if m not in mass_grid:
            raise ValueError(f"Mass {m:.1f} not in database grid - this is a bug!")
    
    return discrete_masses


# ============================================================================
# Age Sampling & Capping
# ============================================================================

def sample_ages(n_stars: int, t_sf_myr: float, mode: str = 'uniform', seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    if mode == 'uniform':
        return np.random.uniform(0, t_sf_myr, n_stars)
    elif mode == 'burst':
        return np.zeros(n_stars)
    else:
        raise ValueError(f"Unknown age mode: {mode}")

def cap_masses(masses: np.ndarray, m_max_database: float, verbose: bool = False) -> Tuple[np.ndarray, Dict]:
    capped_masses = masses.copy()
    mask_exceed = masses > m_max_database
    n_capped = np.sum(mask_exceed)
    mass_before = np.sum(masses)
    mass_exceeded = np.sum(masses[mask_exceed] - m_max_database)
    capped_masses[mask_exceed] = m_max_database
    
    diagnostics = {
        'n_capped': n_capped,
        'mass_capped': mass_exceeded,
        'fraction_mass_lost': mass_exceeded / mass_before if mass_before > 0 else 0,
    }
    if verbose and n_capped > 0:
        print(f"  [Capping] {n_capped} stars capped. Mass lost: {mass_exceeded:.2e} Msun.")
    return capped_masses, diagnostics

# ============================================================================
# Plotting & Visualization (Modified)
# ============================================================================

def plot_population_diagnostics(
        masses: np.ndarray,
        ages: np.ndarray,
        imf_func: Optional[Callable] = None,
        m_min: float = 0.08,
        m_max: float = 120.0,
        title: str = "Stochastic Population",
        filename: str = "population_diagnostics.png",
        target_mass: Optional[float] = None):
    """
    Generate diagnostic plots and save them.
    
    Parameters
    ----------
    target_mass : float, optional
        The requested mass, used to display 'Actual vs Target' stats.
    """
    
    # --- Directory Setup ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, filename)

    # --- Plotting ---
    fig = plt.figure(figsize=(12/1.75, 10/1.75))
    
    # Create Title with Mass Stats (Using raw f-string rf"..." to fix SyntaxWarning)
    actual_mass = np.sum(masses)
    title_text = rf"{title}" + "\n" + rf"Actual Mass: {actual_mass:.1f} $M_{{\odot}}$"
    if target_mass is not None:
        title_text += rf" / Target: {target_mass:.1f} $M_{{\odot}}$"
    
    fig.suptitle(title_text, fontsize=12)
    
    # -------------------------------------------------------
    # Plot 1: Mass Function (Top Left)
    # -------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    bins = np.logspace(np.log10(m_min), np.log10(np.max(masses)), 30)
    ax1.hist(masses, bins=bins, density=True, alpha=0.6, color='royalblue', label='Sampled')
    
    if imf_func is not None:
        m_smooth = np.logspace(np.log10(m_min), np.log10(m_max), 500)
        xi = imf_func(m_smooth)
        norm = np.trapz(xi, m_smooth)
        ax1.plot(m_smooth, xi/norm, 'r--', lw=1.5, label='Theory')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'Mass [$M_{\odot}$]')
    ax1.set_ylabel(r'$dN/dm$')
    ax1.legend(fontsize='small')

    # -------------------------------------------------------
    # Plot 2: Age Distribution (Top Right)
    # -------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(ages, bins=15, color='seagreen', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Age [Myr]')
    ax2.set_ylabel('N Stars')
    ax2.set_title('Age Distribution')

    # -------------------------------------------------------
    # Plot 3: Mass vs Age (Bottom Left)
    # -------------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 3)
    sc = ax3.scatter(ages, masses, c=np.log10(masses), cmap='viridis', alpha=0.5, s=10)
    ax3.set_yscale('log')
    ax3.set_xlabel('Age [Myr]')
    ax3.set_ylabel(r'Mass [$M_{\odot}$]')
    ax3.set_title('Mass vs. Age')

    # -------------------------------------------------------
    # Plot 4: Cumulative Mass Assembly (Bottom Right)
    # -------------------------------------------------------
    ax4 = fig.add_subplot(2, 2, 4)
    sorted_idx = np.argsort(ages)
    sorted_ages = ages[sorted_idx]
    sorted_masses = masses[sorted_idx]
    cum_mass = np.cumsum(sorted_masses)
    
    ax4.plot(sorted_ages, cum_mass, color='darkorange', lw=2, label='Sampled')
    
    # Ideal line
    ax4.plot([0, np.max(ages)], [0, np.sum(masses)], 'k--', alpha=0.6, label='Ideal Constant SFR')
    
    ax4.set_xlabel('Age [Myr]')
    ax4.set_ylabel(r'Cumul. Mass [$M_{\odot}$]')
    ax4.set_title('Mass Assembly Check')
    ax4.legend(fontsize='small')

    plt.tight_layout()
    
    # --- Saving ---
    print(f"Saving plot to: {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":

    print(">>> Discrete Sampling (Exact Grid Masses)")
    print("This requires a database file. Attempting to use default location...")

    try:
        from toddlers._paths import get_database_path
        database_path = get_database_path()   # $TODDLERS_DATA or <package>/database

        print(f"Using database at: {database_path}")

        if os.path.exists(database_path):
            masses_discrete = sample_imf_discrete(
                total_mass=500.0,
                database_path=database_path,
                metallicity='MW',
                m_upper=120.0,
                imf_name='kroupa',
                seed=123,
                verbose=True
            )

            ages_discrete = sample_ages(
                n_stars=len(masses_discrete),
                t_sf_myr=2.0,
                mode='uniform',
                seed=456
            )

            plot_population_diagnostics(
                masses_discrete, ages_discrete,
                imf_func=kroupa_imf,
                title="Discrete Grid Sampling (Kroupa)",
                filename="diag_kroupa_discrete.png",
                target_mass=500.0
            )

            unique_discrete = len(np.unique(masses_discrete))
            print(f"\n  {len(masses_discrete)} stars, {unique_discrete} unique grid masses")
            print(f"  Total mass: {np.sum(masses_discrete):.2f} Msun (target: 500.0)")

        else:
            print(f"  Database not found at: {database_path}")
            print("  Run: python execute_stochastic_examples.py --quick")
            print("  to generate database first")

    except Exception as e:
        print(f"  Could not run discrete example: {e}")
        print("  (This is expected if database doesn't exist yet)")