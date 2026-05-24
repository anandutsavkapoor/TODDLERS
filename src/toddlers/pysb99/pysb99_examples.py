#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pysb99_complete_demo.py
========================

Universal demonstration and testing script for pySB99 integration with TODDLERS.

This script provides:
  - Quick demonstrations of IMF generation
  - Comprehensive testing suite
  - Verification and validation
  - Visualization and comparison tools
  - TODDLERS integration examples

Usage:
    # Quick demo
    python pysb99_complete_demo.py --mode quick
    
    # Full workflow
    python pysb99_complete_demo.py --mode full
    
    # Testing only
    python pysb99_complete_demo.py --mode test
    
    # Custom IMF comparison
    python pysb99_complete_demo.py --mode compare --imfs kroupa100 topheavy salpeter

Author: Anand Utsav KAPOOR
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
import argparse
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from toddlers.pysb99.generate_pysb99_interpolants import (
    generate_kroupa_like_interpolants,
    generate_custom_population_interpolants,
    METALLICITY_MAPPING,
    PySB99InterpolantGenerator
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default metallicities for quick vs full runs
QUICK_METALLICITIES = ['LMC', 'MW']
FULL_METALLICITIES = ['IZw18', 'SMC', 'LMC', 'MW', 'MWC']

# IMF presets
IMF_PRESETS = {
    'kroupa100': {
        'exponents': [1.3, 2.3],
        'limits': (0.1, 0.5, 100.0),
        'description': 'Standard Kroupa IMF (M_max=100)'
    },
    'kroupa120': {
        'exponents': [1.3, 2.3],
        'limits': (0.1, 0.5, 120.0),
        'description': 'Extended Kroupa IMF (M_max=120)'
    },
    'salpeter': {
        'exponents': [2.35],
        'limits': (0.1, 100.0),
        'description': 'Salpeter single power-law'
    },
    'topheavy': {
        'exponents': [1.0, 1.7],
        'limits': (0.1, 0.5, 100.0),
        'description': 'Top-heavy IMF (flatter slope)'
    },
    'steeper': {
        'exponents': [1.5, 2.7],
        'limits': (0.1, 0.5, 100.0),
        'description': 'Steeper IMF (bottom-heavy)'
    }
}

# Custom population presets
POPULATION_PRESETS = {
    'ob_assoc': {
        120.0: 1, 85.0: 2, 60.0: 5, 40.0: 10, 25.0: 20, 15.0: 40
    },
    'single_60msun': {
        60.0: 1
    },
    'binary_rich': {
        60.0: 10, 40.0: 10, 30.0: 20, 20.0: 20, 15.0: 40
    }
}


# ============================================================================
# CORE: IMF GENERATION FUNCTIONS
# ============================================================================

def generate_imf_set(imf_names, output_dir='./demo_database', 
                     metallicities=QUICK_METALLICITIES, rotation=False, 
                     verbose=True):
    """
    Generate interpolants for multiple IMF configurations.
    
    Parameters
    ----------
    imf_names : list of str
        Names of IMF presets to generate (from IMF_PRESETS)
    output_dir : str
        Output directory for interpolant files
    metallicities : list of str
        Metallicities to include
    rotation : bool
        Whether to include stellar rotation
    verbose : bool
        Print detailed progress
        
    Returns
    -------
    generators : dict
        Dictionary mapping imf_name -> generator object
    """
    os.makedirs(output_dir, exist_ok=True)
    generators = {}
    
    for imf_name in imf_names:
        if imf_name not in IMF_PRESETS:
            print(f"⚠ Unknown IMF preset: {imf_name}, skipping...")
            continue
            
        config = IMF_PRESETS[imf_name]
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Generating: {imf_name} - {config['description']}")
            print(f"{'='*70}")
        
        try:
            generator = generate_kroupa_like_interpolants(
                imf_exponents=config['exponents'],
                imf_mass_limits=config['limits'],
                output_dir=output_dir,
                imf_name=imf_name,
                metallicities=metallicities,
                rotation=rotation,
                verbose=verbose
            )
            generators[imf_name] = generator
            
        except Exception as e:
            print(f"✗ Failed to generate {imf_name}: {e}")
            continue
    
    return generators


def generate_custom_populations(pop_names, output_dir='./demo_database',
                                metallicities=QUICK_METALLICITIES, 
                                rotation=False, verbose=True):
    """
    Generate interpolants for custom stellar populations.
    
    Parameters
    ----------
    pop_names : list of str
        Names of population presets (from POPULATION_PRESETS)
    output_dir : str
        Output directory for interpolant files
    metallicities : list of str
        Metallicities to include
    rotation : bool
        Include stellar rotation
    verbose : bool
        Print detailed progress
        
    Returns
    -------
    generators : dict
        Dictionary mapping pop_name -> generator object
    """
    os.makedirs(output_dir, exist_ok=True)
    generators = {}
    
    for pop_name in pop_names:
        if pop_name not in POPULATION_PRESETS:
            print(f"⚠ Unknown population preset: {pop_name}, skipping...")
            continue
        
        custom_stars = POPULATION_PRESETS[pop_name]
        total_stars = sum(custom_stars.values())
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Generating: {pop_name} ({total_stars} stars)")
            print(f"{'='*70}")
            for mass, count in sorted(custom_stars.items(), reverse=True):
                print(f"  {mass:6.1f} M☉: {count:4d} stars")
        
        try:
            generator = generate_custom_population_interpolants(
                custom_star_numbers=custom_stars,
                output_dir=output_dir,
                imf_name=pop_name,
                metallicities=metallicities,
                rotation=rotation,
                verbose=verbose
            )
            generators[pop_name] = generator
            
        except Exception as e:
            print(f"✗ Failed to generate {pop_name}: {e}")
            continue
    
    return generators


# ============================================================================
# VERIFICATION: FILE AND INTERPOLANT CHECKS
# ============================================================================

def verify_files(imf_name, output_dir='./demo_database'):
    """
    Verify that all required interpolant files were created.
    
    Parameters
    ----------
    imf_name : str
        Name of the IMF to check
    output_dir : str
        Directory containing interpolant files
        
    Returns
    -------
    all_present : bool
        True if all files exist
    """
    expected_files = [
        f'pySB99interpolation_{imf_name}.obj',
        f'pySB99interpolation_{imf_name}_LumLymanWerner.obj',
        f'pySB99interpolation_{imf_name}_mean_ionizing_photon_energy.obj',
        f'pySB99interpolation_{imf_name}_hardness.obj'
    ]
    
    print(f"\nVerifying files for {imf_name}:")
    all_present = True
    
    for filename in expected_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  ✓ {filename:<60s} ({size_kb:6.1f} KB)")
        else:
            print(f"  ✗ {filename:<60s} MISSING")
            all_present = False
    
    return all_present


def test_interpolation(imf_name, output_dir='./demo_database', 
                       test_times=[1.0, 5.0, 10.0], test_Z=0.006):
    """
    Test interpolation at specific time and metallicity points.
    
    Parameters
    ----------
    imf_name : str
        Name of IMF to test
    output_dir : str
        Directory containing interpolant files
    test_times : array-like
        Times (Myr) to test interpolation
    test_Z : float
        Metallicity to test
        
    Returns
    -------
    success : bool
        True if interpolation works correctly
    """
    main_file = os.path.join(output_dir, f'pySB99interpolation_{imf_name}.obj')
    
    if not os.path.exists(main_file):
        print(f"✗ Cannot test {imf_name}: file not found")
        return False
    
    print(f"\nTesting interpolation for {imf_name} at Z={test_Z}:")
    
    try:
        with open(main_file, 'rb') as f:
            interpolants = pickle.load(f)
        
        print(f"  Loaded {len(interpolants)} interpolation functions")
        
        # Test L_mech interpolation
        L_mech_interp = interpolants[0]
        
        for t in test_times:
            log_L_mech = L_mech_interp(t, test_Z)[0]
            L_mech = 10**log_L_mech
            print(f"    t={t:5.1f} Myr: L_mech = {L_mech:.2e} erg/s/M☉")
        
        print("  ✓ Interpolation successful")
        return True
        
    except Exception as e:
        print(f"  ✗ Interpolation failed: {e}")
        return False


def verify_toddlers_compatibility(imf_name, output_dir='./demo_database'):
    """
    Verify that interpolants are compatible with TODDLERS format.
    
    Parameters
    ----------
    imf_name : str
        Name of IMF to check
    output_dir : str
        Directory containing interpolant files
        
    Returns
    -------
    compatible : bool
        True if format is compatible
    """
    print(f"\nChecking TODDLERS compatibility for {imf_name}:")
    
    main_file = os.path.join(output_dir, f'pySB99interpolation_{imf_name}.obj')
    
    if not os.path.exists(main_file):
        print("  ✗ Main interpolant file not found")
        return False
    
    try:
        with open(main_file, 'rb') as f:
            interpolants = pickle.load(f)
        
        # Check structure
        expected_count = 7
        if len(interpolants) == expected_count:
            print(f"  ✓ Correct number of interpolants: {len(interpolants)}")
        else:
            print(f"  ✗ Expected {expected_count}, got {len(interpolants)}")
            return False
        
        # Test interpolation interface
        test_result = interpolants[0](1.0, 0.006)
        if isinstance(test_result, np.ndarray) and test_result.shape == (1, 1):
            print("  ✓ Interpolation interface correct")
            print(f"    Example: log(L_mech) = {test_result[0,0]:.2f}")
        else:
            print(f"  ✗ Unexpected return format: {type(test_result)}, shape {test_result.shape}")
            return False
        
        print("  ✓ Compatible with TODDLERS format")
        return True
        
    except Exception as e:
        print(f"  ✗ Compatibility check failed: {e}")
        return False


# ============================================================================
# VISUALIZATION: COMPARISON AND EVOLUTION PLOTS
# ============================================================================

def plot_imf_comparison(generators, output_dir='./demo_database', 
                        metallicity='MW', figsize=(15, 10)):
    """
    Create comparison plots for different IMFs at a given metallicity.
    
    Parameters
    ----------
    generators : dict
        Dictionary mapping imf_name -> generator object
    output_dir : str
        Output directory for plots
    metallicity : str
        Metallicity to compare (e.g., 'MW', 'LMC')
    figsize : tuple
        Figure size
    """
    # Try to load pretty.mplstyle if available
    style_path = Path('../tests/pretty.mplstyle')
    if style_path.exists():
        plt.style.use(str(style_path))
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'pySB99 IMF Comparison at {metallicity} Metallicity', 
                 fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(generators)))
    
    for (imf_name, gen), color in zip(generators.items(), colors):
        if metallicity not in gen.results_dict:
            print(f"⚠ {imf_name} has no results for {metallicity}")
            continue
            
        results = gen.results_dict[metallicity]
        time_myr = results.times / 1e6
        
        # Wind power
        axes[0, 0].loglog(time_myr, results.wind_power, 
                         label=imf_name, color=color, lw=2)
        axes[0, 0].set_ylabel(r'Wind Power (erg s$^{-1}$)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Ionizing flux
        axes[0, 1].loglog(time_myr, results.hi_ionizing_flux, 
                         label=imf_name, color=color, lw=2)
        axes[0, 1].set_ylabel(r'HI Ionizing Flux (s$^{-1}$)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Bolometric luminosity
        axes[0, 2].loglog(time_myr, results.bolometric_luminosity, 
                         label=imf_name, color=color, lw=2)
        axes[0, 2].set_ylabel(r'$L_{\rm bol}$ (erg s$^{-1}$)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Wind momentum
        axes[1, 0].loglog(time_myr, results.wind_momentum, 
                         label=imf_name, color=color, lw=2)
        axes[1, 0].set_xlabel('Time (Myr)')
        axes[1, 0].set_ylabel(r'Wind Momentum (dyne)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # HeI/HI ratio
        ratio_hei = results.hei_ionizing_flux / np.maximum(results.hi_ionizing_flux, 1e-50)
        axes[1, 1].semilogy(time_myr, ratio_hei, 
                           label=imf_name, color=color, lw=2)
        axes[1, 1].set_xlabel('Time (Myr)')
        axes[1, 1].set_ylabel(r'HeI/HI Ratio')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Cumulative energy
        dt = np.diff(results.times, prepend=results.times[0])
        cumulative_energy = np.cumsum(results.wind_power * dt)
        axes[1, 2].loglog(time_myr, cumulative_energy, 
                         label=imf_name, color=color, lw=2)
        axes[1, 2].set_xlabel('Time (Myr)')
        axes[1, 2].set_ylabel(r'Cumulative Energy (erg)')
        axes[1, 2].grid(True, alpha=0.3)
    
    for ax in axes.flat:
        ax.legend(fontsize=8, loc='best')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'imf_comparison_{metallicity}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved: {output_file}")
    plt.close()


def plot_metallicity_study(imf_name, generator, output_dir='./demo_database',
                           figsize=(10, 6)):
    """
    Plot metallicity dependence of ionizing flux for a single IMF.
    
    Parameters
    ----------
    imf_name : str
        Name of IMF
    generator : PySB99InterpolantGenerator
        Generator object with results
    output_dir : str
        Output directory for plots
    figsize : tuple
        Figure size
    """
    # Try to load pretty.mplstyle if available
    style_path = Path('../tests/pretty.mplstyle')
    if style_path.exists():
        plt.style.use(str(style_path))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    metallicities = sorted(generator.results_dict.keys(), 
                          key=lambda x: METALLICITY_MAPPING[x])
    colors = plt.cm.viridis(np.linspace(0, 1, len(metallicities)))
    
    for met, color in zip(metallicities, colors):
        results = generator.results_dict[met]
        time_myr = results.times / 1e6
        Z_val = METALLICITY_MAPPING[met]
        
        ax.loglog(time_myr, results.hi_ionizing_flux,
                 label=f'{met} (Z={Z_val:.4f})', color=color, lw=2.5)
    
    ax.set_xlabel('Time (Myr)', fontsize=12)
    ax.set_ylabel(r'Ionizing Photon Rate per M$_\odot$ (s$^{-1}$ M$_\odot^{-1}$)', 
                 fontsize=12)
    ax.set_title(f'Metallicity Dependence of Ionizing Flux ({imf_name})', 
                fontsize=13)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'metallicity_study_{imf_name}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Metallicity study saved: {output_file}")
    plt.close()


def plot_evolution_example(imf_name, output_dir='./demo_database',
                           M_stars=1e6, Z=0.014, figsize=(15, 4)):
    """
    Plot example feedback evolution for a stellar population.
    
    Parameters
    ----------
    imf_name : str
        Name of IMF to use
    output_dir : str
        Directory containing interpolants
    M_stars : float
        Total stellar mass (M☉)
    Z : float
        Metallicity
    figsize : tuple
        Figure size
    """
    # Try to load pretty.mplstyle if available
    style_path = Path('../tests/pretty.mplstyle')
    if style_path.exists():
        plt.style.use(str(style_path))
    
    main_file = os.path.join(output_dir, f'pySB99interpolation_{imf_name}.obj')
    
    with open(main_file, 'rb') as f:
        interpolants = pickle.load(f)
    
    L_mech_interp = interpolants[0]
    F_ram_interp = interpolants[1]
    Q_ion_interp = interpolants[4]
    
    time_grid = np.logspace(-2, 1.7, 100)
    
    # Evaluate feedback
    L_mech = np.array([10**L_mech_interp(t, Z)[0] * M_stars for t in time_grid])
    F_ram = np.array([10**F_ram_interp(t, Z)[0] * M_stars for t in time_grid])
    Q_ion = np.array([10**Q_ion_interp(t, Z)[0] * M_stars for t in time_grid])
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f'Stellar Feedback Evolution ({imf_name}, M★ = {M_stars:.0e} M☉, Z = {Z})', 
                fontsize=12)
    
    axes[0].loglog(time_grid, L_mech)
    axes[0].set_xlabel('Time (Myr)')
    axes[0].set_ylabel(r'$L_{\rm mech}$ (erg s$^{-1}$)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Mechanical Luminosity')
    
    axes[1].loglog(time_grid, F_ram)
    axes[1].set_xlabel('Time (Myr)')
    axes[1].set_ylabel(r'$\dot{p}_{\rm wind}$ (dyne)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Wind Momentum')
    
    axes[2].loglog(time_grid, Q_ion)
    axes[2].set_xlabel('Time (Myr)')
    axes[2].set_ylabel(r'$Q_{\rm ion}$ (s$^{-1}$)')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Ionizing Photon Rate')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'feedback_evolution_{imf_name}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Evolution plot saved: {output_file}")
    plt.close()


# ============================================================================
# INTEGRATION: TODDLERS USAGE EXAMPLES
# ============================================================================

def print_toddlers_usage(imf_names, output_dir='./demo_database'):
    """
    Print example code for using generated interpolants with TODDLERS.
    
    Parameters
    ----------
    imf_names : list of str
        Names of generated IMFs
    output_dir : str
        Directory containing interpolant files
    """
    print("\n" + "="*70)
    print("TODDLERS INTEGRATION")
    print("="*70)
    
    print("\nStep 1: Copy interpolant files to TODDLERS database directory")
    print(f"  cp {output_dir}/*.obj /path/to/toddlers_evolution_package/database/")
    
    print("\nStep 2: Update stellar_feedback.py (around line 126)")
    print("  Change:")
    print("    file_prefix = f'SB99interpolation_{self.imf}'")
    print("  To:")
    print("    file_prefix = f'pySB99interpolation_{self.imf}'")
    
    print("\nStep 3: Use in your TODDLERS simulations")
    print("```python")
    print("from toddlers_evolution_package.stellar_feedback import StellarFeedback")
    print()
    
    for imf_name in imf_names[:3]:  # Show first 3 examples
        print(f"# Example: {imf_name}")
        print("feedback = StellarFeedback(")
        print("    template='SB99',")
        print("    Z=0.014,              # MW metallicity")
        print("    M_cl_init=1e7,        # 10^7 Msun cloud")
        print("    eta_sf=0.1,           # 10% efficiency")
        print("    t_list_collapse=[],   # Single burst")
        print("    mode='burst',")
        print(f"    imf='{imf_name}'")
        print(")")
        print()
    
    print("```")
    
    print("\nAvailable IMFs:")
    for imf_name in imf_names:
        files_exist = verify_files(imf_name, output_dir)
        status = "✓" if files_exist else "✗"
        print(f"  {status} {imf_name}")


# ============================================================================
# WORKFLOWS: DIFFERENT EXECUTION MODES
# ============================================================================

def quick_demo(output_dir='./demo_database'):
    """
    Quick demonstration: Generate one IMF and verify.
    
    Parameters
    ----------
    output_dir : str
        Output directory
    """
    print("\n" + "="*70)
    print("QUICK DEMO: Standard Kroupa IMF")
    print("="*70)
    
    # Generate
    generators = generate_imf_set(
        imf_names=['kroupa100'],
        output_dir=output_dir,
        metallicities=QUICK_METALLICITIES,
        verbose=True
    )
    
    # Verify
    verify_files('kroupa100', output_dir)
    test_interpolation('kroupa100', output_dir)
    verify_toddlers_compatibility('kroupa100', output_dir)
    
    # Simple plot
    plot_evolution_example('kroupa100', output_dir)
    
    print("\n" + "="*70)
    print("QUICK DEMO COMPLETE ✓")
    print("="*70)
    print(f"\nGenerated files in {output_dir}/")
    print_toddlers_usage(['kroupa100'], output_dir)


def full_workflow(output_dir='./demo_database'):
    """
    Comprehensive workflow: Multiple IMFs with full analysis.
    
    Parameters
    ----------
    output_dir : str
        Output directory
    """
    print("\n" + "="*70)
    print("FULL WORKFLOW: Comprehensive IMF Suite")
    print("="*70)
    
    # Generate multiple IMFs
    imf_names = ['kroupa100', 'salpeter', 'topheavy']
    generators = generate_imf_set(
        imf_names=imf_names,
        output_dir=output_dir,
        metallicities=FULL_METALLICITIES,
        verbose=True
    )
    
    # Generate custom population
    pop_generators = generate_custom_populations(
        pop_names=['ob_assoc'],
        output_dir=output_dir,
        metallicities=FULL_METALLICITIES,
        verbose=True
    )
    
    # Combine
    all_generators = {**generators, **pop_generators}
    
    # Verification
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    for name in all_generators.keys():
        verify_files(name, output_dir)
        test_interpolation(name, output_dir)
    
    # Visualization
    print("\n" + "="*70)
    print("VISUALIZATION")
    print("="*70)
    
    # IMF comparison at MW metallicity
    plot_imf_comparison(all_generators, output_dir, metallicity='MW')
    
    # Metallicity study for Kroupa
    if 'kroupa100' in generators:
        plot_metallicity_study('kroupa100', generators['kroupa100'], output_dir)
    
    # Evolution examples
    for name in all_generators.keys():
        plot_evolution_example(name, output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("FULL WORKFLOW COMPLETE ✓")
    print("="*70)
    print(f"\nGenerated files in {output_dir}/")
    print_toddlers_usage(list(all_generators.keys()), output_dir)


def test_suite(output_dir='./test_database'):
    """
    Systematic testing: Generate and thoroughly verify interpolants.
    
    Parameters
    ----------
    output_dir : str
        Output directory for test files
    """
    print("\n" + "="*70)
    print("TEST SUITE: pySB99 Interpolants")
    print("="*70)
    
    # Test 1: Generate Kroupa
    print("\nTest 1: Generate Kroupa 100")
    print("-" * 70)
    generators = generate_imf_set(
        imf_names=['kroupa100'],
        output_dir=output_dir,
        metallicities=['SMC', 'LMC', 'MW'],
        verbose=True
    )
    
    # Test 2: Generate custom population
    print("\nTest 2: Generate Custom Population")
    print("-" * 70)
    pop_generators = generate_custom_populations(
        pop_names=['ob_assoc'],
        output_dir=output_dir,
        metallicities=['SMC', 'LMC', 'MW'],
        verbose=True
    )
    
    # Test 3: File verification
    print("\nTest 3: File Verification")
    print("-" * 70)
    test_passed = True
    for name in ['kroupa100', 'ob_assoc']:
        if not verify_files(name, output_dir):
            test_passed = False
    
    # Test 4: Interpolation testing
    print("\nTest 4: Interpolation Testing")
    print("-" * 70)
    for name in ['kroupa100', 'ob_assoc']:
        if not test_interpolation(name, output_dir):
            test_passed = False
    
    # Test 5: TODDLERS compatibility
    print("\nTest 5: TODDLERS Compatibility")
    print("-" * 70)
    for name in ['kroupa100', 'ob_assoc']:
        if not verify_toddlers_compatibility(name, output_dir):
            test_passed = False
    
    # Test 6: Comparison plots
    print("\nTest 6: Comparison Plots")
    print("-" * 70)
    all_generators = {**generators, **pop_generators}
    plot_imf_comparison(all_generators, output_dir, metallicity='MW')
    
    # Results
    print("\n" + "="*70)
    if test_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*70)
    
    # Cleanup option
    response = input(f"\nDelete test files in {output_dir}/? (y/n): ")
    if response.lower() == 'y':
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print("✓ Test files removed")
    else:
        print(f"Test files kept in {output_dir}/")


def compare_imfs(imf_names, output_dir='./demo_database'):
    """
    Generate and compare specific IMFs.
    
    Parameters
    ----------
    imf_names : list of str
        IMF names to compare
    output_dir : str
        Output directory
    """
    print("\n" + "="*70)
    print(f"IMF COMPARISON: {', '.join(imf_names)}")
    print("="*70)
    
    # Generate
    generators = generate_imf_set(
        imf_names=imf_names,
        output_dir=output_dir,
        metallicities=QUICK_METALLICITIES,
        verbose=True
    )
    
    # Compare
    plot_imf_comparison(generators, output_dir, metallicity='MW')
    
    # Individual evolution plots
    for name in imf_names:
        plot_evolution_example(name, output_dir)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE ✓")
    print("="*70)
    print(f"\nPlots saved in {output_dir}/")


# ============================================================================
# REFERENCE INFORMATION
# ============================================================================

def print_reference_info():
    """Print reference tables for metallicities and IMF presets."""
    print("\n" + "="*70)
    print("METALLICITY MAPPING")
    print("="*70)
    print(f"{'Region':<10} {'Z (numerical)':<15} {'Description':<30}")
    print("-"*70)
    
    descriptions = {
        'MWC': 'Galactic Center (super-solar)',
        'MW': 'Milky Way / Solar',
        'LMC': 'Large Magellanic Cloud',
        'SMC': 'Small Magellanic Cloud',
        'IZw18': 'IZw18 (very low Z)',
        'Z0': 'Zero metallicity (Pop III)'
    }
    
    for region, z_val in sorted(METALLICITY_MAPPING.items(), 
                                key=lambda x: x[1], reverse=True):
        desc = descriptions.get(region, '')
        print(f"{region:<10} {z_val:<15.4f} {desc:<30}")
    
    print("\n" + "="*70)
    print("IMF PRESETS")
    print("="*70)
    for name, config in IMF_PRESETS.items():
        print(f"{name:<15} {config['description']}")
        print(f"{'':15} Exponents: {config['exponents']}")
        print(f"{'':15} Limits: {config['limits']}")
    
    print("\n" + "="*70)
    print("POPULATION PRESETS")
    print("="*70)
    for name, stars in POPULATION_PRESETS.items():
        total = sum(stars.values())
        print(f"{name:<15} {total} stars")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main execution with argument parsing."""
    parser = argparse.ArgumentParser(
        description='pySB99 demonstration and testing tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode quick
  %(prog)s --mode full --output-dir ./my_database
  %(prog)s --mode test
  %(prog)s --mode compare --imfs kroupa100 topheavy salpeter
  %(prog)s --info
        """
    )
    
    parser.add_argument('--mode', 
                       choices=['quick', 'full', 'test', 'compare', 'custom'],
                       default='quick',
                       help='Execution mode (default: quick)')
    
    parser.add_argument('--output-dir', 
                       default='./demo_database',
                       help='Output directory (default: ./demo_database)')
    
    parser.add_argument('--imfs', 
                       nargs='+',
                       help='IMF names for comparison mode')
    
    parser.add_argument('--populations',
                       nargs='+',
                       help='Population names for custom mode')
    
    parser.add_argument('--metallicities',
                       nargs='+',
                       help='Metallicities to include')
    
    parser.add_argument('--info',
                       action='store_true',
                       help='Print reference information and exit')
    
    args = parser.parse_args()
    
    if args.info:
        print_reference_info()
        return
    
    # Set metallicities
    if args.metallicities:
        metallicities = args.metallicities
    elif args.mode == 'full':
        metallicities = FULL_METALLICITIES
    else:
        metallicities = QUICK_METALLICITIES
    
    # Execute requested mode
    if args.mode == 'quick':
        quick_demo(args.output_dir)
        
    elif args.mode == 'full':
        full_workflow(args.output_dir)
        
    elif args.mode == 'test':
        test_suite(args.output_dir)
        
    elif args.mode == 'compare':
        if not args.imfs:
            print("Error: --imfs required for compare mode")
            return
        compare_imfs(args.imfs, args.output_dir)
        
    elif args.mode == 'custom':
        if args.imfs:
            generators = generate_imf_set(
                imf_names=args.imfs,
                output_dir=args.output_dir,
                metallicities=metallicities,
                verbose=True
            )
        if args.populations:
            pop_generators = generate_custom_populations(
                pop_names=args.populations,
                output_dir=args.output_dir,
                metallicities=metallicities,
                verbose=True
            )


if __name__ == "__main__":
    main()