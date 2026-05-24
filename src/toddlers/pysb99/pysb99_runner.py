#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pySB99 Runner Module
====================

High-level interface for running stellar population synthesis calculations
and generating publication-quality plots. This module provides simple functions
for common use cases and example workflows.

Author: Anand Utsav KAPOOR
Refactored from https://github.com/CalumHawcroft/Starburst/tree/main and 
described in https://arxiv.org/html/2505.24841v1
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Any
import argparse
import json
from datetime import datetime

from .pysb99_core import (
    StellarPopulationConfig, 
    StellarPopulationResults,
    StellarDataLoader,
    run_population_synthesis,
    save_results,
    create_default_config
)
from .pysb99_plotting import (
    PopulationPlotter,
    compare_models,
    plot_quick_summary,
    CustomPopulationPlotter,
    create_custom_population_report
)

def get_available_mass_grid(metallicity: str, rotation: bool = False) -> np.ndarray:
    """
    Get the available mass grid for a given metallicity and rotation setting.
    
    Parameters
    ----------
    metallicity : str
        Metallicity choice: 'MWC', 'MW', 'LMC', 'SMC', 'IZw18', 'Z0'
    rotation : bool
        Whether to include stellar rotation
        
    Returns
    -------
    np.ndarray
        Array of available stellar masses in solar masses
        
    Examples
    --------
    >>> # Get mass grid for Milky Way metallicity
    >>> mass_grid = get_available_mass_grid('MW')
    >>> print(f"Available masses: {mass_grid}")
    
    >>> # Get mass grid for rotating Z0 models
    >>> mass_grid = get_available_mass_grid('Z0', rotation=True)
    >>> print(f"Available masses: {mass_grid}")
    """
    # Create a temporary configuration
    config = create_default_config(
        metallicity=metallicity,
        rotation=rotation
    )
    
    # Create a loader to get the mass grid
    loader = StellarDataLoader(config)
    
    return loader.mass_grid

def create_custom_population(mass_star_counts: Dict[float, int],
                           metallicity: str = 'MW',
                           spectral_library: str = 'FW',
                           rotation: bool = False,
                           output_dir: Optional[str] = None,
                           model_name: str = 'custom_population') -> Tuple[StellarPopulationResults, StellarPopulationConfig]:
    """
    Create and run a stellar population with a custom distribution of star counts.
    
    Parameters
    ----------
    mass_star_counts : Dict[float, int]
        Dictionary mapping stellar mass (in solar masses) to number of stars
    metallicity : str
        Metallicity choice: 'MWC', 'MW', 'LMC', 'SMC', 'IZw18', 'Z0'
    spectral_library : str
        Spectral library: 'FW' (Fastwind) or 'WM' (WMbasic)
    rotation : bool
        Include stellar rotation
    output_dir : str, optional
        Output directory for saving results
    model_name : str
        Name for output files
        
    Returns
    -------
    Tuple[StellarPopulationResults, StellarPopulationConfig]
        Results and configuration objects
        
    Examples
    --------
    >>> # Create a population with 100 stars of 20 M☉ and 50 stars of 40 M☉
    >>> results, config = create_custom_population({
    ...     20.0: 100,
    ...     40.0: 50,
    ... })
    
    >>> # Check the closest available masses that were used
    >>> print("Available masses:", config.available_masses)
    """
    # Get available mass grid and print warnings for mismatches
    available_masses = get_available_mass_grid(metallicity, rotation)
    
    # Check if requested masses are close to available ones
    for requested_mass in mass_star_counts.keys():
        closest_idx = np.abs(available_masses - requested_mass).argmin()
        closest_mass = available_masses[closest_idx]
        
        if abs(closest_mass - requested_mass) / requested_mass > 0.05:  # More than 5% difference
            warnings.warn(
                f"Requested mass {requested_mass:.1f} M☉ not available in grid. " 
                f"Using closest available mass {closest_mass:.1f} M☉."
            )
    
    # Run the model with custom star distribution
    results, config = run_standard_model(
        metallicity=metallicity,
        spectral_library=spectral_library,
        rotation=rotation,
        custom_star_numbers=mass_star_counts,
        output_dir=output_dir,
        model_name=model_name,
        verbose=True
    )
    
    # Store available masses in config for reference
    config.available_masses = available_masses
    
    return results, config

def run_standard_model(total_mass: float = 1e6,
                      metallicity: str = 'MW',
                      spectral_library: str = 'FW',
                      rotation: bool = False,
                      time_end_myr: float = 50.0,
                      custom_star_numbers: Optional[Dict[float, int]] = None,
                      output_dir: Optional[str] = None,
                      model_name: str = 'standard_model',
                      create_plots: bool = True,
                      verbose: bool = True) -> Tuple[StellarPopulationResults, StellarPopulationConfig]:
    """
    Run a standard stellar population synthesis model with common parameters.
    
    Parameters
    ----------
    total_mass : float
        Total stellar mass in solar masses (ignored if custom_star_numbers is provided)
    metallicity : str
        Metallicity choice: 'MWC', 'MW', 'LMC', 'SMC', 'IZw18', 'Z0'
    spectral_library : str
        Spectral library: 'FW' (Fastwind) or 'WM' (WMbasic)
    rotation : bool
        Include stellar rotation
    time_end_myr : float
        End time in Myr
    custom_star_numbers : Dict[float, int], optional
        Custom star distribution mapping stellar mass to count (overrides IMF)
    output_dir : str, optional
        Output directory for saving results
    model_name : str
        Name for output files
    create_plots : bool
        Whether to create summary plots
    verbose : bool
        Print progress information
        
    Returns
    -------
    Tuple[StellarPopulationResults, StellarPopulationConfig]
        Results and configuration objects
        
    Examples
    --------
    >>> # Run a standard Milky Way metallicity model
    >>> results, config = run_standard_model(
    ...     total_mass=1e7,
    ...     metallicity='MW',
    ...     time_end_myr=100.0,
    ...     output_dir='./outputs'
    ... )
    
    >>> # Run a model with custom star distribution
    >>> custom_stars = {
    ...     20.0: 100,   # 100 stars of 20 solar masses
    ...     50.0: 50,    # 50 stars of 50 solar masses
    ...     85.0: 10,    # 10 stars of 85 solar masses
    ... }
    >>> results, config = run_standard_model(
    ...     metallicity='MW',
    ...     custom_star_numbers=custom_stars,
    ...     model_name='custom_stars'
    ... )
    """
    
    if verbose:
        print(f"Running stellar population synthesis model: {model_name}")
        print(f"Parameters:")
        if custom_star_numbers:
            print(f"  Custom star distribution with {len(custom_star_numbers)} mass points")
            total_stars = sum(custom_star_numbers.values())
            estimated_mass = sum(mass * count for mass, count in custom_star_numbers.items())
            print(f"  Total stars: {total_stars}")
            print(f"  Estimated total mass: {estimated_mass:.2e} M☉")
            
            # Print distribution in mass order
            print("  Star distribution:")
            for mass in sorted(custom_star_numbers.keys()):
                count = custom_star_numbers[mass]
                mass_percent = (mass * count / estimated_mass) * 100
                count_percent = (count / total_stars) * 100
                print(f"    {mass:.1f} M☉: {count} stars ({count_percent:.1f}% by number, {mass_percent:.1f}% by mass)")
        else:
            print(f"  Total mass: {total_mass:.0e} M☉")
        print(f"  Metallicity: {metallicity}")
        print(f"  Spectral library: {spectral_library}")
        print(f"  Rotation: {rotation}")
        print(f"  Time range: 0.01 - {time_end_myr} Myr")
    
    # Create configuration
    config = create_default_config(
        total_mass=total_mass,
        metallicity=metallicity,
        spectral_library=spectral_library,
        rotation=rotation,
        time_end=time_end_myr * 1e6,  # Convert to years
        run_speed_mode='FAST',  # Use fast mode for standard runs
        custom_star_numbers=custom_star_numbers
    )
    
    # If using custom star numbers, check against available masses
    if custom_star_numbers and verbose:
        # Get available masses for this metallicity/rotation
        available_masses = get_available_mass_grid(metallicity, rotation)
        
        print("\nChecking requested masses against available grid:")
        for requested_mass in custom_star_numbers.keys():
            closest_idx = np.abs(available_masses - requested_mass).argmin()
            closest_mass = available_masses[closest_idx]
            
            if abs(closest_mass - requested_mass) / requested_mass > 0.05:  # More than 5% difference
                print(f"  Warning: Requested mass {requested_mass:.1f} M☉ not available in grid.")
                print(f"           Using closest available mass {closest_mass:.1f} M☉ instead.")
            else:
                print(f"  Mass {requested_mass:.1f} M☉ is available in the grid (exact or very close).")
        
        # Store available masses in config for reference
        config.available_masses = available_masses.tolist()
    
    # Run synthesis
    if verbose:
        print("\nStarting population synthesis calculation...")
    
    results = run_population_synthesis(config)
    
    if verbose:
        print("Calculation completed successfully!")
    
    # Save results if output directory specified
    if output_dir:
        if verbose:
            print(f"\nSaving results to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        save_results(results, output_dir, model_name)
        
        # Save configuration
        config_dict = _config_to_dict(config)
        config_path = os.path.join(output_dir, f'{model_name}_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        if verbose:
            print("Results saved successfully!")
    
    # Create visualizations for custom star distribution if used
    if custom_star_numbers and output_dir:
        if verbose:
            print("\nCreating custom star distribution visualizations...")
        
        # Get available masses for the current configuration if not already done
        if not hasattr(config, 'available_masses'):
            available_masses = get_available_mass_grid(metallicity, rotation)
            config.available_masses = available_masses.tolist()
        else:
            available_masses = np.array(config.available_masses)
        
        # Create directory for custom population visualizations
        custom_viz_dir = os.path.join(output_dir, 'custom_population')
        os.makedirs(custom_viz_dir, exist_ok=True)
        
        # Create basic distribution plot
        if verbose:
            print("  Creating distribution visualization...")
        
        custom_plotter = CustomPopulationPlotter(results, config, custom_star_numbers, available_masses)
        dist_fig = custom_plotter.plot_distribution(figsize=(10, 10))
        dist_path = os.path.join(custom_viz_dir, f'{model_name}_distribution.pdf')
        dist_fig.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close(dist_fig)
        
        # Create summary plot
        if verbose:
            print("  Creating custom population summary plot...")
        
        summary_fig = custom_plotter.create_summary_plot(figsize=(15, 12))
        summary_path = os.path.join(custom_viz_dir, f'{model_name}_custom_summary.pdf')
        summary_fig.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close(summary_fig)
        
        # Create HR diagram if possible
        if hasattr(results, 'stellar_temperatures') and hasattr(results, 'stellar_luminosities'):
            if verbose:
                print("  Creating HR diagram by mass...")
            
            hr_fig = custom_plotter.plot_hr_diagram_by_mass(time_myr=5.0, figsize=(10, 8))
            hr_path = os.path.join(custom_viz_dir, f'{model_name}_hr_diagram.pdf')
            hr_fig.savefig(hr_path, dpi=300, bbox_inches='tight')
            plt.close(hr_fig)
        
        if verbose:
            print(f"Custom population visualizations saved to {custom_viz_dir}/")
    
    # Create standard plots if requested
    if create_plots:
        if verbose:
            print("\nCreating summary plots...")
        
        plotter = PopulationPlotter(results, config)
        
        # Create comprehensive summary plot
        summary_fig = plotter.create_summary_plot(figsize=(16, 12))
        if output_dir:
            summary_path = os.path.join(output_dir, f'{model_name}_summary.pdf')
            summary_fig.savefig(summary_path, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"Summary plot saved to {summary_path}")
        else:
            plt.show()
        
        # Create quick summary plot
        quick_fig = plot_quick_summary(results, config, figsize=(12, 8))
        if output_dir:
            quick_path = os.path.join(output_dir, f'{model_name}_quick.pdf')
            quick_fig.savefig(quick_path, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"Quick summary plot saved to {quick_path}")
        else:
            plt.show()
    
    return results, config


def run_metallicity_sequence(mass: float = 1e6,
                            metallicities: Optional[List[str]] = None,
                            output_dir: str = './metallicity_sequence',
                            verbose: bool = True) -> Dict[str, Tuple[StellarPopulationResults, StellarPopulationConfig]]:
    """
    Run a sequence of models with different metallicities for comparison.
    
    Parameters
    ----------
    mass : float
        Total stellar mass for all models
    metallicities : List[str], optional
        List of metallicities to run. If None, uses standard sequence.
    output_dir : str
        Output directory for results
    verbose : bool
        Print progress information
        
    Returns
    -------
    Dict[str, Tuple[StellarPopulationResults, StellarPopulationConfig]]
        Dictionary mapping metallicity names to (results, config) tuples
        
    Examples
    --------
    >>> # Run standard metallicity sequence
    >>> results_dict = run_metallicity_sequence(
    ...     mass=1e7,
    ...     output_dir='./z_sequence'
    ... )
    
    >>> # Run custom metallicity sequence
    >>> results_dict = run_metallicity_sequence(
    ...     mass=1e6,
    ...     metallicities=['MW', 'LMC', 'SMC'],
    ...     output_dir='./custom_sequence'
    ... )
    """
    
    if metallicities is None:
        metallicities = ['MW', 'LMC', 'SMC', 'IZw18']  # Standard sequence
    
    if verbose:
        print(f"Running metallicity sequence: {metallicities}")
        print(f"Total mass: {mass:.0e} M☉")
        print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results_dict = {}
    
    for i, metallicity in enumerate(metallicities):
        if verbose:
            print(f"\n--- Running model {i+1}/{len(metallicities)}: {metallicity} ---")
        
        model_name = f"Z_{metallicity}"
        model_output_dir = os.path.join(output_dir, model_name)
        
        try:
            results, config = run_standard_model(
                total_mass=mass,
                metallicity=metallicity,
                output_dir=model_output_dir,
                model_name=model_name,
                create_plots=False,  # Create comparison plot later
                verbose=verbose
            )
            
            results_dict[metallicity] = (results, config)
            
        except Exception as e:
            if verbose:
                print(f"Error running {metallicity} model: {str(e)}")
                print("Continuing with remaining models...")
            continue
    
    # Create comparison plot
    if len(results_dict) > 1:
        if verbose:
            print("\nCreating metallicity comparison plot...")
        
        results_list = [results for results, config in results_dict.values()]
        labels = [f"Z = {z}" for z in results_dict.keys()]
        config_list = [config for results, config in results_dict.values()]
        
        comparison_fig = compare_models(
            results_list, labels, config_list, figsize=(16, 12)
        )
        
        comparison_path = os.path.join(output_dir, 'metallicity_comparison.pdf')
        comparison_fig.savefig(comparison_path, dpi=300, bbox_inches='tight')
        
        if verbose:
            print(f"Comparison plot saved to {comparison_path}")
    
    return results_dict


def run_mass_sequence(masses: Optional[List[float]] = None,
                     metallicity: str = 'MW',
                     output_dir: str = './mass_sequence',
                     verbose: bool = True) -> Dict[float, Tuple[StellarPopulationResults, StellarPopulationConfig]]:
    """
    Run a sequence of models with different total masses.
    
    Parameters
    ----------
    masses : List[float], optional
        List of total masses in solar masses. If None, uses standard sequence.
    metallicity : str
        Metallicity for all models
    output_dir : str
        Output directory for results
    verbose : bool
        Print progress information
        
    Returns
    -------
    Dict[float, Tuple[StellarPopulationResults, StellarPopulationConfig]]
        Dictionary mapping masses to (results, config) tuples
        
    Examples
    --------
    >>> # Run standard mass sequence
    >>> results_dict = run_mass_sequence(
    ...     metallicity='LMC',
    ...     output_dir='./mass_sequence'
    ... )
    
    >>> # Run custom mass sequence
    >>> results_dict = run_mass_sequence(
    ...     masses=[1e5, 1e6, 1e7, 1e8],
    ...     output_dir='./custom_masses'
    ... )
    """
    
    if masses is None:
        masses = [1e5, 1e6, 1e7, 1e8]  # Standard mass sequence
    
    if verbose:
        print(f"Running mass sequence: {[f'{m:.0e}' for m in masses]} M☉")
        print(f"Metallicity: {metallicity}")
        print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results_dict = {}
    
    for i, mass in enumerate(masses):
        if verbose:
            print(f"\n--- Running model {i+1}/{len(masses)}: {mass:.0e} M☉ ---")
        
        model_name = f"M_{mass:.0e}"
        model_output_dir = os.path.join(output_dir, model_name)
        
        try:
            results, config = run_standard_model(
                total_mass=mass,
                metallicity=metallicity,
                output_dir=model_output_dir,
                model_name=model_name,
                create_plots=False,  # Create comparison plot later
                verbose=verbose
            )
            
            results_dict[mass] = (results, config)
            
        except Exception as e:
            if verbose:
                print(f"Error running {mass:.0e} M☉ model: {str(e)}")
                print("Continuing with remaining models...")
            continue
    
    # Create comparison plot
    if len(results_dict) > 1:
        if verbose:
            print("\nCreating mass comparison plot...")
        
        results_list = [results for results, config in results_dict.values()]
        labels = [f"M = {m:.0e} M☉" for m in results_dict.keys()]
        config_list = [config for results, config in results_dict.values()]
        
        comparison_fig = compare_models(
            results_list, labels, config_list, figsize=(16, 12)
        )
        
        comparison_path = os.path.join(output_dir, 'mass_comparison.pdf')
        comparison_fig.savefig(comparison_path, dpi=300, bbox_inches='tight')
        
        if verbose:
            print(f"Comparison plot saved to {comparison_path}")
    
    return results_dict


def create_publication_figures(results: StellarPopulationResults,
                             config: StellarPopulationConfig,
                             output_dir: str,
                             figure_prefix: str = 'figure',
                             formats: List[str] = ['pdf', 'png'],
                             dpi: int = 300) -> Dict[str, str]:
    """
    Create a set of publication-quality figures.
    
    Parameters
    ----------
    results : StellarPopulationResults
        Population synthesis results
    config : StellarPopulationConfig
        Model configuration
    output_dir : str
        Output directory for figures
    figure_prefix : str
        Prefix for figure filenames
    formats : List[str]
        List of output formats ('pdf', 'png', 'eps', etc.)
    dpi : int
        DPI for raster formats
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping figure names to file paths
        
    Examples
    --------
    >>> results, config = run_standard_model(total_mass=1e7)
    >>> figure_paths = create_publication_figures(
    ...     results, config, './figures',
    ...     figure_prefix='paper_fig',
    ...     formats=['pdf', 'png']
    ... )
    """
    
    os.makedirs(output_dir, exist_ok=True)
    plotter = PopulationPlotter(results, config)
    figure_paths = {}
    
    # Define figure specifications
    figures = {
        'ionizing_fluxes': {
            'function': plotter.plot_ionizing_fluxes,
            'figsize': (8, 6),
            'description': 'Ionizing photon rates'
        },
        'wind_properties': {
            'function': plotter.plot_wind_properties,
            'figsize': (8, 10),
            'description': 'Stellar wind properties'
        },
        'uv_slope': {
            'function': plotter.plot_uv_slope,
            'figsize': (8, 6),
            'description': 'UV slope evolution'
        },
        'equivalent_widths': {
            'function': plotter.plot_equivalent_widths,
            'figsize': (8, 6),
            'description': 'Hydrogen line equivalent widths'
        },
        'color_evolution': {
            'function': plotter.plot_color_evolution,
            'figsize': (8, 6),
            'description': 'Broad-band color evolution'
        },
        'sed_evolution': {
            'function': plotter.plot_sed_evolution,
            'figsize': (12, 8),
            'description': 'Spectral energy distribution evolution'
        },
        'spectral_2d': {
            'function': plotter.plot_spectral_evolution_2d,
            'figsize': (12, 8),
            'description': '2D spectral evolution'
        }
    }
    
    print(f"Creating {len(figures)} publication figures...")
    
    for fig_name, fig_spec in figures.items():
        print(f"  Creating {fig_name}: {fig_spec['description']}")
        
        # Create figure
        fig = fig_spec['function'](figsize=fig_spec['figsize'])
        
        # Save in requested formats
        for fmt in formats:
            filename = f"{figure_prefix}_{fig_name}.{fmt}"
            filepath = os.path.join(output_dir, filename)
            
            fig.savefig(filepath, format=fmt, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            
            figure_paths[f"{fig_name}_{fmt}"] = filepath
        
        plt.close(fig)  # Free memory
    
    print(f"Figures saved to {output_dir}")
    return figure_paths


def analyze_population_properties(results: StellarPopulationResults,
                                config: Optional[StellarPopulationConfig] = None,
                                verbose: bool = True) -> Dict[str, Any]:
    """
    Analyze key properties of the stellar population evolution.
    
    Parameters
    ----------
    results : StellarPopulationResults
        Population synthesis results
    config : StellarPopulationConfig, optional
        Model configuration
    verbose : bool
        Print analysis results
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing analysis results
        
    Examples
    --------
    >>> results, config = run_standard_model()
    >>> analysis = analyze_population_properties(results, config)
    >>> print(f"Peak ionizing flux: {analysis['peak_hi_flux']:.2f}")
    """
    
    analysis = {}
    time_myr = results.times / 1e6
    
    # Time evolution characteristics
    analysis['time_range_myr'] = (time_myr.min(), time_myr.max())
    analysis['n_timesteps'] = len(results.times)
    
    # Peak values and their timing
    analysis['peak_hi_flux'] = np.max(results.hi_ionizing_flux)
    analysis['peak_hi_time_myr'] = time_myr[np.argmax(results.hi_ionizing_flux)]
    
    analysis['peak_wind_power'] = np.max(results.wind_power)
    analysis['peak_wind_time_myr'] = time_myr[np.argmax(results.wind_power)]
    
    analysis['peak_bolo_lum'] = np.max(results.bolometric_luminosity)
    analysis['peak_bolo_time_myr'] = time_myr[np.argmax(results.bolometric_luminosity)]
    
    # UV slope statistics
    valid_beta = results.uv_slope_beta[~np.isnan(results.uv_slope_beta)]
    if len(valid_beta) > 0:
        analysis['uv_slope_mean'] = np.mean(valid_beta)
        analysis['uv_slope_std'] = np.std(valid_beta)
        analysis['uv_slope_range'] = (np.min(valid_beta), np.max(valid_beta))
    else:
        analysis['uv_slope_mean'] = np.nan
        analysis['uv_slope_std'] = np.nan
        analysis['uv_slope_range'] = (np.nan, np.nan)
    
    # Color evolution
    ub_color = results.u_magnitude - results.b_magnitude
    bv_color = results.b_magnitude - results.v_magnitude
    
    analysis['ub_color_range'] = (np.min(ub_color), np.max(ub_color))
    analysis['bv_color_range'] = (np.min(bv_color), np.max(bv_color))
    analysis['ub_color_final'] = ub_color[-1]
    analysis['bv_color_final'] = bv_color[-1]
    
    # Equivalent width evolution
    analysis['ha_ew_peak'] = np.max(results.ha_equivalent_width)
    analysis['ha_ew_peak_time_myr'] = time_myr[np.argmax(results.ha_equivalent_width)]
    analysis['ha_ew_final'] = results.ha_equivalent_width[-1]
    
    # Integrated quantities
    dt = np.diff(results.times)
    dt = np.append(dt, dt[-1])  # Handle last point
    
    # Total energy output
    total_energy = np.trapz(10**results.bolometric_luminosity, results.times * 3.15e7)  # erg
    analysis['total_energy_erg'] = total_energy
    
    # Total ionizing photons
    total_hi_photons = np.trapz(10**results.hi_ionizing_flux, results.times)  # s^-1 * yr
    analysis['total_hi_photons'] = total_hi_photons * 3.15e7  # Convert to total photons
    
    # Efficiency metrics
    analysis['ionizing_efficiency'] = results.hi_ionizing_flux - results.bolometric_luminosity
    analysis['wind_efficiency'] = results.wind_momentum - results.bolometric_luminosity
    
    # Time-averaged properties
    analysis['avg_hi_flux'] = np.mean(results.hi_ionizing_flux)
    analysis['avg_wind_power'] = np.mean(results.wind_power)
    analysis['avg_uv_slope'] = np.nanmean(results.uv_slope_beta)
    
    # Configuration-dependent analysis
    if config:
        analysis['total_mass'] = config.total_mass
        analysis['metallicity'] = config.metallicity
        analysis['imf_exponents'] = config.imf_exponents
        analysis['spectral_library'] = config.spectral_library
        analysis['rotation'] = config.rotation
        
        # Mass-normalized quantities
        analysis['specific_hi_flux'] = analysis['peak_hi_flux'] - np.log10(config.total_mass)
        analysis['specific_wind_power'] = analysis['peak_wind_power'] - np.log10(config.total_mass)
    
    if verbose:
        print("Population Analysis Summary")
        print("=" * 40)
        print(f"Time range: {analysis['time_range_myr'][0]:.2f} - {analysis['time_range_myr'][1]:.2f} Myr")
        print(f"Peak HI ionizing flux: {analysis['peak_hi_flux']:.2f} at {analysis['peak_hi_time_myr']:.2f} Myr")
        print(f"Peak wind power: {analysis['peak_wind_power']:.2f} at {analysis['peak_wind_time_myr']:.2f} Myr")
        print(f"UV slope range: {analysis['uv_slope_range'][0]:.2f} to {analysis['uv_slope_range'][1]:.2f}")
        print(f"Final U-B color: {analysis['ub_color_final']:.2f}")
        print(f"Final B-V color: {analysis['bv_color_final']:.2f}")
        print(f"Total energy output: {analysis['total_energy_erg']:.2e} erg")
        
        if config:
            print(f"\nModel Configuration:")
            print(f"Total mass: {config.total_mass:.0e} M☉")
            print(f"Metallicity: {config.metallicity}")
            print(f"Spectral library: {config.spectral_library}")
            print(f"Rotation: {config.rotation}")
    
    return analysis


def _config_to_dict(config: StellarPopulationConfig) -> Dict[str, Any]:
    """Convert configuration object to dictionary for JSON serialization."""
    config_dict = {
        'total_mass': config.total_mass,
        'imf_exponents': config.imf_exponents,
        'imf_mass_limits': list(config.imf_mass_limits),
        'metallicity': config.metallicity,
        'spectral_library': config.spectral_library,
        'rotation': config.rotation,
        'run_speed_mode': config.run_speed_mode,
        'use_powr': config.use_powr,
        'time_start': config.time_start,
        'time_end': config.time_end,
        'time_step': config.time_step,
        'creation_time': datetime.now().isoformat()
    }
    
    # Add custom star distribution if used
    if config.custom_star_numbers is not None:
        # Convert float keys to strings for JSON compatibility
        custom_stars_json = {str(mass): count for mass, count in config.custom_star_numbers.items()}
        config_dict['custom_star_numbers'] = custom_stars_json
        config_dict['using_custom_distribution'] = True
    else:
        config_dict['using_custom_distribution'] = False
    
    return config_dict


def load_results_from_directory(results_dir: str) -> Tuple[StellarPopulationResults, StellarPopulationConfig]:
    """
    Load results and configuration from a directory.
    
    Parameters
    ----------
    results_dir : str
        Directory containing saved results
        
    Returns
    -------
    Tuple[StellarPopulationResults, StellarPopulationConfig]
        Loaded results and configuration
        
    Note
    ----
    This is a placeholder function. Full implementation would require
    modifications to the save/load format in the core module.
    """
    
    # This would need to be implemented based on the actual save format
    # For now, raise a helpful error
    raise NotImplementedError(
        "Loading results from directory is not yet implemented. "
        "This would require modifications to the save format in pysb99_core."
    )


def main():
    """
    Command-line interface for pySB99.
    """
    
    parser = argparse.ArgumentParser(
        description='pySB99: Stellar Population Synthesis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pysb99_runner.py --mass 1e7 --metallicity LMC --output ./results
  python pysb99_runner.py --custom-stars 20:100,50:50,85:10 --output ./custom_run
  python pysb99_runner.py --mass-sequence --output ./mass_sequence
  python pysb99_runner.py --metallicity-sequence --output ./z_sequence
  python pysb99_runner.py --list-available-masses MW --rotation
  python pysb99_runner.py --custom-stars 20:100,50:50 --custom-report
        """
    )
    
    # Model parameters
    parser.add_argument('--mass', type=float, default=1e6,
                       help='Total stellar mass in solar masses (default: 1e6)')
    parser.add_argument('--custom-stars', type=str,
                       help='Custom star distribution in format "mass1:count1,mass2:count2,..."')
    parser.add_argument('--metallicity', type=str, default='MW',
                       choices=['MWC', 'MW', 'LMC', 'SMC', 'IZw18', 'Z0'],
                       help='Metallicity (default: MW)')
    parser.add_argument('--spectral-library', type=str, default='FW',
                       choices=['FW', 'WM'],
                       help='Spectral library: FW (Fastwind) or WM (WMbasic)')
    parser.add_argument('--rotation', action='store_true',
                       help='Include stellar rotation')
    parser.add_argument('--time-end', type=float, default=50.0,
                       help='End time in Myr (default: 50)')
    
    # Sequence runs
    parser.add_argument('--mass-sequence', action='store_true',
                       help='Run mass sequence')
    parser.add_argument('--masses', type=float, nargs='+',
                       help='Masses for sequence (default: 1e5 1e6 1e7 1e8)')
    parser.add_argument('--metallicity-sequence', action='store_true',
                       help='Run metallicity sequence')
    parser.add_argument('--metallicities', type=str, nargs='+',
                       help='Metallicities for sequence (default: MW LMC SMC IZw18)')
    
    # Custom population options
    parser.add_argument('--custom-report', action='store_true',
                       help='Create a comprehensive report for custom star distributions')
    parser.add_argument('--list-available-masses', type=str, choices=['MWC', 'MW', 'LMC', 'SMC', 'IZw18', 'Z0'],
                       help='List available masses for a given metallicity')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, default='./pysb99_output',
                       help='Output directory (default: ./pysb99_output)')
    parser.add_argument('--name', type=str, default='model',
                       help='Model name for output files (default: model)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot creation')
    parser.add_argument('--publication-figures', action='store_true',
                       help='Create publication-quality figures')
    
    # Utility options
    parser.add_argument('--quick-run', action='store_true',
                       help='Quick run with basic plots')
    parser.add_argument('--analyze', action='store_true',
                       help='Perform detailed analysis')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()

    # List available masses if requested
    if args.list_available_masses:
        available_masses = get_available_mass_grid(args.list_available_masses, args.rotation)
        print(f"\nAvailable stellar masses for {args.list_available_masses} metallicity" + 
              f" {'with' if args.rotation else 'without'} rotation:")
        print("-" * 70)
        
        # Format masses in a grid
        masses_per_row = 8
        for i in range(0, len(available_masses), masses_per_row):
            row = available_masses[i:i+masses_per_row]
            print("  ".join(f"{mass:.1f}" for mass in row))
        
        print("\nUse these values in your custom star distribution for best results.")
        sys.exit(0)

    # Parse custom star distribution if provided
    custom_star_numbers = None
    if args.custom_stars:
        custom_star_numbers = {}
        try:
            for pair in args.custom_stars.split(','):
                mass, count = pair.split(':')
                custom_star_numbers[float(mass)] = int(count)
            
            if args.verbose:
                total_stars = sum(custom_star_numbers.values())
                total_mass_estimate = sum(mass * count for mass, count in custom_star_numbers.items())
                
                print("\nCustom star distribution specified:")
                print(f"Total stars: {total_stars}")
                print(f"Estimated total mass: {total_mass_estimate:.2e} M☉")
                print("\nStar distribution:")
                
                # Print distribution in mass order
                for mass in sorted(custom_star_numbers.keys()):
                    count = custom_star_numbers[mass]
                    mass_percent = (mass * count / total_mass_estimate) * 100
                    count_percent = (count / total_stars) * 100
                    print(f"  {mass:.1f} M☉: {count} stars ({count_percent:.1f}% by number, {mass_percent:.1f}% by mass)")
        
        except ValueError:
            parser.error("Custom stars must be in format 'mass1:count1,mass2:count2,...'")
    
    # Create custom report if requested and custom stars are provided
    if args.custom_report and custom_star_numbers:
        if args.verbose:
            print("\nCreating comprehensive custom population report...")
        
        # Run the model
        results, config = run_standard_model(
            total_mass=args.mass,
            metallicity=args.metallicity,
            spectral_library=args.spectral_library,
            rotation=args.rotation,
            time_end_myr=args.time_end,
            custom_star_numbers=custom_star_numbers,
            output_dir=args.output,
            model_name=args.name,
            create_plots=False,  # Skip standard plots, we'll create custom ones
            verbose=args.verbose
        )
        
        # Create report directory
        report_dir = os.path.join(args.output, 'custom_report')
        
        # Get available masses
        available_masses = get_available_mass_grid(args.metallicity, args.rotation)
        
        # Create the report
        report_path = create_custom_population_report(
            results, config, custom_star_numbers,
            output_dir=report_dir,
            report_name=args.name
        )
        
        if args.verbose:
            print(f"Custom population report created at {report_path}")
        
        # Exit if only report was requested
        if args.no_plots:
            sys.exit(0)
    elif args.custom_report and not custom_star_numbers:
        parser.error("--custom-report requires --custom-stars to be specified")

    # Handle sequence runs
    if args.mass_sequence:
        masses = args.masses if args.masses else [1e5, 1e6, 1e7, 1e8]
        results_dict = run_mass_sequence(
            masses=masses,
            metallicity=args.metallicity,
            output_dir=args.output,
            verbose=args.verbose
        )
        return
    
    if args.metallicity_sequence:
        metallicities = args.metallicities if args.metallicities else ['MW', 'LMC', 'SMC', 'IZw18']
        results_dict = run_metallicity_sequence(
            mass=args.mass,
            metallicities=metallicities,
            output_dir=args.output,
            verbose=args.verbose
        )
        return
    
    # Quick run with shorter time and fewer plots
    if args.quick_run:
        args.time_end = min(args.time_end, 10.0)  # Max 10 Myr for quick run
        args.publication_figures = False  # Skip publication figures
    
    # Standard single model run
    results, config = run_standard_model(
        total_mass=args.mass,
        metallicity=args.metallicity,
        spectral_library=args.spectral_library,
        rotation=args.rotation,
        time_end_myr=args.time_end,
        custom_star_numbers=custom_star_numbers,
        output_dir=args.output,
        model_name=args.name,
        create_plots=not args.no_plots,
        verbose=args.verbose
    )
    
    # Create publication figures if requested
    if args.publication_figures:
        if args.verbose:
            print("\nCreating publication figures...")
        
        figure_dir = os.path.join(args.output, 'figures')
        figure_paths = create_publication_figures(
            results, config, figure_dir, 
            figure_prefix=args.name
        )
        
        if args.verbose:
            print(f"Publication figures saved to {figure_dir}")
    
    # Perform analysis if requested
    if args.analyze:
        if args.verbose:
            print("\nPerforming population analysis...")
        
        analysis = analyze_population_properties(results, config, verbose=True)
        
        # Save analysis results
        analysis_path = os.path.join(args.output, f'{args.name}_analysis.json')
        with open(analysis_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            analysis_serializable = {k: float(v) if isinstance(v, np.floating) else v 
                                   for k, v in analysis.items()}
            json.dump(analysis_serializable, f, indent=2)
        
        if args.verbose:
            print(f"Analysis results saved to {analysis_path}")


if __name__ == "__main__":
    main()