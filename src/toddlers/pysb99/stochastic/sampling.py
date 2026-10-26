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
    # Continuity at 1 Msun: the power law takes the log-normal's value there.
    norm_factor = np.exp(-(np.log10(m_c))**2 / (2 * sigma**2))
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

    NOTE: This uses numpy's global random state. For run-time stochastic populations
    (one independent stream per generation) use
    toddlers.pysb99.stochastic.population.draw_stop_after.
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


def sample_imf_discrete(*args, **kwargs):
    """Removed in 2.1.0.

    Use toddlers.pysb99.stochastic.population instead: IMFSpec(name, m_low, m_up) names
    the full range, draw_stop_after(...) draws continuous masses, and
    StellarFeedback/Evolution(template='pySB99', stochastic=StochasticSpec(...)) draws one
    independent population per generation at run time."""
    raise RuntimeError(
        "sample_imf_discrete was removed in 2.1.0. "
        "Use toddlers.pysb99.stochastic.population (IMFSpec, draw_stop_after, StochasticSpec; "
        "Evolution(template='pySB99', imf=None, stochastic=StochasticSpec(...))).")


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
    from toddlers.pysb99.stochastic.population import IMFSpec, draw_stop_after, generation_rng
    m = draw_stop_after(IMFSpec('kroupa', 0.1, 100.0), 500.0, generation_rng(123, 0))
    print(f"{len(m)} stars, {m.sum():.1f} Msun drawn (target 500.0), "
          f"{np.sum(m >= 8)} above 8 Msun")
