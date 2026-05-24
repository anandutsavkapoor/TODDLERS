#!/usr/bin/env python3
"""
plot_database_feedback.py
=========================

Query and visualize single-star feedback tracks from HDF5 database.
This helps verify database integrity and understand stellar feedback evolution.

Usage:
    python plot_database_feedback.py
    python plot_database_feedback.py --metallicity LMC --masses 10,25,60,120
    python plot_database_feedback.py --show-all-metallicities

Author: Anand Utsav Kapoor
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import os
from argparse import ArgumentParser

# Add project to path
sys.path.insert(0, '/Users/akapoor/toddlers_evolution_package/pysb99/stochastic')
from database import query_database, METALLICITY_MAPPING, get_database_info

# Use pretty plotting style
STYLE_FILE = '/Users/akapoor/toddlers_evolution_package/tests/pretty.mplstyle'
if os.path.exists(STYLE_FILE):
    plt.style.use(STYLE_FILE)


def plot_feedback_vs_time(database_path, metallicity='MW', masses=[10, 25, 60, 120],
                          time_max=30.0, output_file=None):
    """
    Plot feedback quantities vs time for different stellar masses.
    
    Parameters
    ----------
    database_path : str
        Path to HDF5 database
    metallicity : str
        Metallicity identifier (MW, LMC, SMC, etc.)
    masses : list
        Stellar masses to plot [Msun]
    time_max : float
        Maximum time to plot [Myr]
    output_file : str, optional
        Save figure to this file
    """
    
    print("="*70)
    print(f"PLOTTING FEEDBACK vs TIME")
    print(f"Metallicity: {metallicity} (Z={METALLICITY_MAPPING[metallicity]})")
    print(f"Masses: {masses} Msun")
    print("="*70)
    
    # Query time grid
    time_grid = np.linspace(0.1, time_max, 50)
    
    # Setup figure
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle(f'Single-Star Feedback Evolution ({metallicity}, Z={METALLICITY_MAPPING[metallicity]})',
                 fontsize=14, y=0.995)
    
    # Colors for different masses
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(masses)))
    
    # Query and plot each mass
    for i_mass, mass in enumerate(masses):
        print(f"\nQuerying mass = {mass:.1f} Msun...")
        
        try:
            feedback = query_database(
                database_path=database_path,
                metallicity=metallicity,
                mass=mass,
                ages_myr=time_grid,
                quantities=['wind_power', 'wind_momentum', 'L_bol',
                           'Q_HI', 'Q_HeI', 'Q_HeII', 'L_LyW']
            )
            
            # Calculate Q_ion
            Q_HI_linear = 10**feedback['Q_HI']
            Q_HeI_linear = 10**feedback['Q_HeI']
            Q_HeII_linear = 10**feedback['Q_HeII']
            Q_ion_linear = Q_HI_linear + Q_HeI_linear + Q_HeII_linear
            Q_ion_log = np.log10(Q_ion_linear + 1e-99)
            
            # Calculate hardness ratios
            ratio_HeI_HI = Q_HeI_linear / (Q_HI_linear + 1e-50)
            ratio_HeII_HI = Q_HeII_linear / (Q_HI_linear + 1e-50)
            
            # Plot each quantity
            label = rf'{mass:.0f} M$_\odot$'
            color = colors[i_mass]
            
            # Row 1: Wind properties
            ax = axes[0, 0]
            ax.plot(time_grid, 10**feedback['wind_power'], color=color,
                   label=label, linewidth=2, alpha=0.8)
            
            ax = axes[0, 1]
            ax.plot(time_grid, 10**feedback['wind_momentum'], color=color,
                   label=label, linewidth=2, alpha=0.8)
            
            ax = axes[0, 2]
            ax.plot(time_grid, 10**feedback['L_bol'], color=color,
                   label=label, linewidth=2, alpha=0.8)
            
            # Row 2: Ionizing photons
            ax = axes[1, 0]
            ax.plot(time_grid, 10**feedback['Q_HI'], color=color,
                   label=label, linewidth=2, alpha=0.8)
            
            ax = axes[1, 1]
            ax.plot(time_grid, 10**Q_ion_log, color=color,
                   label=label, linewidth=2, alpha=0.8)
            
            ax = axes[1, 2]
            ax.plot(time_grid, 10**feedback['L_LyW'], color=color,
                   label=label, linewidth=2, alpha=0.8)
            
            # Row 3: Component comparison and hardness
            ax = axes[2, 0]
            ax.plot(time_grid, 10**feedback['Q_HI'], color=color,
                   linewidth=2, alpha=0.8, linestyle='-', label=f'{mass:.0f} HI')
            ax.plot(time_grid, 10**feedback['Q_HeI'], color=color,
                   linewidth=1.5, alpha=0.6, linestyle='--')
            ax.plot(time_grid, 10**feedback['Q_HeII'], color=color,
                   linewidth=1.5, alpha=0.6, linestyle=':')
            
            ax = axes[2, 1]
            ax.plot(time_grid, ratio_HeI_HI, color=color,
                   label=label, linewidth=2, alpha=0.8)
            
            ax = axes[2, 2]
            ax.plot(time_grid, ratio_HeII_HI, color=color,
                   label=label, linewidth=2, alpha=0.8)
            
            print(f"  Peak Q_ion: {np.max(Q_ion_log):.2f} log10(photons/s)")
            print(f"  Peak L_bol: {np.max(feedback['L_bol']):.2f} log10(erg/s)")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Format axes
    # Row 1
    axes[0, 0].set_ylabel(r'Wind Power (erg s$^{-1}$)', fontsize=11)
    axes[0, 0].set_title('Mechanical Luminosity', fontsize=12)
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=9, loc='best')
    
    axes[0, 1].set_ylabel(r'Wind Momentum (dyne)', fontsize=11)
    axes[0, 1].set_title('Ram Pressure', fontsize=12)
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_ylabel(r'$L_{\rm bol}$ (erg s$^{-1}$)', fontsize=11)
    axes[0, 2].set_title('Bolometric Luminosity', fontsize=12)
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2
    axes[1, 0].set_ylabel(r'$Q_{\rm HI}$ (photons s$^{-1}$)', fontsize=11)
    axes[1, 0].set_title('HI Ionizing Photon Rate', fontsize=12)
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_ylabel(r'$Q_{\rm ion}$ (photons s$^{-1}$)', fontsize=11)
    axes[1, 1].set_title('Total Ionizing Photon Rate', fontsize=12)
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].set_ylabel(r'$L_{\rm LyW}$ (erg s$^{-1}$ $\AA^{-1}$)', fontsize=11)
    axes[1, 2].set_title('Lyman-Werner Flux', fontsize=12)
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Row 3
    axes[2, 0].set_xlabel(r'Time (Myr)', fontsize=11)
    axes[2, 0].set_ylabel(r'Ionizing Photons (s$^{-1}$)', fontsize=11)
    axes[2, 0].set_title('Q Components (solid=HI, dash=HeI, dot=HeII)', fontsize=12)
    axes[2, 0].set_yscale('log')
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].set_xlabel(r'Time (Myr)', fontsize=11)
    axes[2, 1].set_ylabel(r'$Q_{\rm HeI} / Q_{\rm HI}$', fontsize=11)
    axes[2, 1].set_title('HeI/HI Hardness Ratio', fontsize=12)
    axes[2, 1].set_yscale('log')
    axes[2, 1].grid(True, alpha=0.3)
    
    axes[2, 2].set_xlabel(r'Time (Myr)', fontsize=11)
    axes[2, 2].set_ylabel(r'$Q_{\rm HeII} / Q_{\rm HI}$', fontsize=11)
    axes[2, 2].set_title('HeII/HI Hardness Ratio', fontsize=12)
    axes[2, 2].set_yscale('log')
    axes[2, 2].grid(True, alpha=0.3)
    
    # Set xlim for all
    for ax in axes.flat:
        ax.set_xlim(0, time_max)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n[SAVED] {output_file}")
    else:
        plt.savefig('/home/claude/database_feedback_vs_time.png', dpi=300, bbox_inches='tight')
        print(f"\n[SAVED] /home/claude/database_feedback_vs_time.png")
    
    plt.close()


def plot_feedback_vs_mass(database_path, metallicity='MW', times=[1.0, 5.0, 10.0],
                          mass_min=1.0, mass_max=120.0, output_file=None):
    """
    Plot feedback quantities vs stellar mass at fixed times.
    
    Parameters
    ----------
    database_path : str
        Path to HDF5 database
    metallicity : str
        Metallicity identifier
    times : list
        Times to plot [Myr]
    mass_min, mass_max : float
        Mass range [Msun]
    output_file : str, optional
        Save figure to this file
    """
    
    print("\n" + "="*70)
    print(f"PLOTTING FEEDBACK vs MASS")
    print(f"Metallicity: {metallicity}")
    print(f"Times: {times} Myr")
    print("="*70)
    
    # Mass grid
    mass_grid = np.array([300., 250., 180., 120., 85., 60., 40., 32., 25., 20., 15., 12., 9., 7., 5., 4., 3., 2.5, 2., 1.7, 1.5, 1.35, 1.25, 1.1, 1., 0.9, 0.8])
    
    # Setup figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f'Feedback vs Stellar Mass ({metallicity}, Z={METALLICITY_MAPPING[metallicity]})',
                 fontsize=14, y=0.995)
    
    # Colors for different times
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(times)))
    
    # Query each time
    for i_time, time_myr in enumerate(times):
        print(f"\nQuerying time = {time_myr:.1f} Myr...")
        
        # Arrays to store results
        Q_HI_arr = []
        Q_ion_arr = []
        L_bol_arr = []
        L_mech_arr = []
        F_ram_arr = []
        L_LyW_arr = []
        
        for mass in mass_grid:
            try:
                feedback = query_database(
                    database_path=database_path,
                    metallicity=metallicity,
                    mass=mass,
                    ages_myr=np.array([time_myr]),
                    quantities=['wind_power', 'wind_momentum', 'L_bol',
                               'Q_HI', 'Q_HeI', 'Q_HeII', 'L_LyW'],
                    suppress_warnings=True
                )
                
                # Calculate Q_ion
                Q_HI_linear = 10**feedback['Q_HI'][0]
                Q_HeI_linear = 10**feedback['Q_HeI'][0]
                Q_HeII_linear = 10**feedback['Q_HeII'][0]
                Q_ion_linear = Q_HI_linear + Q_HeI_linear + Q_HeII_linear
                
                Q_HI_arr.append(Q_HI_linear)
                Q_ion_arr.append(Q_ion_linear)
                L_bol_arr.append(10**feedback['L_bol'][0])
                L_mech_arr.append(10**feedback['wind_power'][0])
                F_ram_arr.append(10**feedback['wind_momentum'][0])
                L_LyW_arr.append(10**feedback['L_LyW'][0])
                
            except Exception as e:
                Q_HI_arr.append(np.nan)
                Q_ion_arr.append(np.nan)
                L_bol_arr.append(np.nan)
                L_mech_arr.append(np.nan)
                F_ram_arr.append(np.nan)
                L_LyW_arr.append(np.nan)
        
        # Convert to arrays
        Q_HI_arr = np.array(Q_HI_arr)
        Q_ion_arr = np.array(Q_ion_arr)
        L_bol_arr = np.array(L_bol_arr)
        L_mech_arr = np.array(L_mech_arr)
        F_ram_arr = np.array(F_ram_arr)
        L_LyW_arr = np.array(L_LyW_arr)
        
        # Plot
        label = rf'{time_myr:.1f} Myr'
        color = colors[i_time]
        
        axes[0, 0].loglog(mass_grid, L_mech_arr, color=color, label=label,
                         linewidth=2, alpha=0.8)
        axes[0, 1].loglog(mass_grid, F_ram_arr, color=color, label=label,
                         linewidth=2, alpha=0.8)
        axes[0, 2].loglog(mass_grid, L_bol_arr, color=color, label=label,
                         linewidth=2, alpha=0.8)
        axes[1, 0].loglog(mass_grid, Q_HI_arr, color=color, label=label,
                         linewidth=2, alpha=0.8)
        axes[1, 1].loglog(mass_grid, Q_ion_arr, color=color, label=label,
                         linewidth=2, alpha=0.8)
        axes[1, 2].loglog(mass_grid, L_LyW_arr, color=color, label=label,
                         linewidth=2, alpha=0.8)
    
    # Format axes
    axes[0, 0].set_ylabel(r'Wind Power (erg s$^{-1}$)', fontsize=11)
    axes[0, 0].set_title('Mechanical Luminosity', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=9, loc='best')
    
    axes[0, 1].set_ylabel(r'Wind Momentum (dyne)', fontsize=11)
    axes[0, 1].set_title('Ram Pressure', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_ylabel(r'$L_{\rm bol}$ (erg s$^{-1}$)', fontsize=11)
    axes[0, 2].set_title('Bolometric Luminosity', fontsize=12)
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel(r'Stellar Mass (M$_\odot$)', fontsize=11)
    axes[1, 0].set_ylabel(r'$Q_{\rm HI}$ (photons s$^{-1}$)', fontsize=11)
    axes[1, 0].set_title('HI Ionizing Photon Rate', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel(r'Stellar Mass (M$_\odot$)', fontsize=11)
    axes[1, 1].set_ylabel(r'$Q_{\rm ion}$ (photons s$^{-1}$)', fontsize=11)
    axes[1, 1].set_title('Total Ionizing Photon Rate', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].set_xlabel(r'Stellar Mass (M$_\odot$)', fontsize=11)
    axes[1, 2].set_ylabel(r'$L_{\rm LyW}$ (erg s$^{-1}$ $\AA^{-1}$)', fontsize=11)
    axes[1, 2].set_title('Lyman-Werner Flux', fontsize=12)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n[SAVED] {output_file}")
    else:
        plt.savefig('/home/claude/database_feedback_vs_mass.png', dpi=300, bbox_inches='tight')
        print(f"\n[SAVED] /home/claude/database_feedback_vs_mass.png")
    
    plt.close()


def plot_all_metallicities(database_path, mass=25.0, time_max=30.0, output_file=None):
    """
    Plot feedback for one mass across all metallicities.
    
    Parameters
    ----------
    database_path : str
        Path to HDF5 database
    mass : float
        Stellar mass [Msun]
    time_max : float
        Maximum time [Myr]
    output_file : str, optional
        Save figure to this file
    """
    
    print("\n" + "="*70)
    print(f"PLOTTING ALL METALLICITIES")
    print(f"Mass: {mass:.1f} Msun")
    print("="*70)
    
    # Get available metallicities from database
    with h5py.File(database_path, 'r') as f:
        metallicities = [k for k in f.keys() if k.startswith('Z_')]
    
    print(f"Available metallicities: {metallicities}")
    
    # Time grid
    time_grid = np.linspace(0.01, time_max, 200)
    
    # Setup figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f'Metallicity Dependence (M = {mass:.0f} M$_\\odot$)',
                 fontsize=14, y=0.995)
    
    # Colors
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(metallicities)))
    
    # Query each metallicity
    for i_met, met_key in enumerate(sorted(metallicities)):
        # Extract Z value from key (e.g., 'Z_0.014' -> 0.014)
        Z_val = float(met_key.split('_')[1])
        
        # Find corresponding metallicity name
        met_name = None
        for name, Z in METALLICITY_MAPPING.items():
            if abs(Z - Z_val) < 1e-5:
                met_name = name
                break
        
        if met_name is None:
            met_name = f'Z={Z_val:.4f}'
        
        print(f"\nQuerying {met_name} (Z={Z_val})...")
        
        try:
            feedback = query_database(
                database_path=database_path,
                metallicity=met_name if met_name in METALLICITY_MAPPING else list(METALLICITY_MAPPING.keys())[0],
                mass=mass,
                ages_myr=time_grid,
                quantities=['wind_power', 'wind_momentum', 'L_bol',
                           'Q_HI', 'Q_HeI', 'Q_HeII', 'L_LyW']
            )
            
            # Calculate Q_ion
            Q_HI_linear = 10**feedback['Q_HI']
            Q_HeI_linear = 10**feedback['Q_HeI']
            Q_HeII_linear = 10**feedback['Q_HeII']
            Q_ion_linear = Q_HI_linear + Q_HeI_linear + Q_HeII_linear
            Q_ion_log = np.log10(Q_ion_linear + 1e-99)
            
            # Plot
            label = f'{met_name} (Z={Z_val:.4f})'
            color = colors[i_met]
            
            axes[0, 0].plot(time_grid, 10**feedback['wind_power'], color=color,
                           label=label, linewidth=2, alpha=0.8)
            axes[0, 1].plot(time_grid, 10**feedback['wind_momentum'], color=color,
                           label=label, linewidth=2, alpha=0.8)
            axes[0, 2].plot(time_grid, 10**feedback['L_bol'], color=color,
                           label=label, linewidth=2, alpha=0.8)
            axes[1, 0].plot(time_grid, 10**feedback['Q_HI'], color=color,
                           label=label, linewidth=2, alpha=0.8)
            axes[1, 1].plot(time_grid, 10**Q_ion_log, color=color,
                           label=label, linewidth=2, alpha=0.8)
            axes[1, 2].plot(time_grid, 10**feedback['L_LyW'], color=color,
                           label=label, linewidth=2, alpha=0.8)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Format axes
    axes[0, 0].set_ylabel(r'Wind Power (erg s$^{-1}$)', fontsize=11)
    axes[0, 0].set_title('Mechanical Luminosity', fontsize=12)
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc='best')
    
    axes[0, 1].set_ylabel(r'Wind Momentum (dyne)', fontsize=11)
    axes[0, 1].set_title('Ram Pressure', fontsize=12)
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_ylabel(r'$L_{\rm bol}$ (erg s$^{-1}$)', fontsize=11)
    axes[0, 2].set_title('Bolometric Luminosity', fontsize=12)
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel(r'Time (Myr)', fontsize=11)
    axes[1, 0].set_ylabel(r'$Q_{\rm HI}$ (photons s$^{-1}$)', fontsize=11)
    axes[1, 0].set_title('HI Ionizing Photon Rate', fontsize=12)
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel(r'Time (Myr)', fontsize=11)
    axes[1, 1].set_ylabel(r'$Q_{\rm ion}$ (photons s$^{-1}$)', fontsize=11)
    axes[1, 1].set_title('Total Ionizing Photon Rate', fontsize=12)
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].set_xlabel(r'Time (Myr)', fontsize=11)
    axes[1, 2].set_ylabel(r'$L_{\rm LyW}$ (erg s$^{-1}$ $\AA^{-1}$)', fontsize=11)
    axes[1, 2].set_title('Lyman-Werner Flux', fontsize=12)
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    for ax in axes.flat:
        ax.set_xlim(0, time_max)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n[SAVED] {output_file}")
    else:
        plt.savefig('/home/claude/database_metallicity_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n[SAVED] /home/claude/database_metallicity_comparison.png")
    
    plt.close()


def main():
    """Main execution."""
    
    parser = ArgumentParser(description='Query and plot HDF5 database feedback tracks')
    
    parser.add_argument('--database', type=str,
                       default='/Users/akapoor/toddlers_evolution_package/src/database/single_star_tracks.h5',
                       help='Path to HDF5 database')
    parser.add_argument('--metallicity', type=str, default='MW',
                       help='Metallicity (MW, LMC, SMC, etc.)')
    parser.add_argument('--masses', type=str, default='10,25,60,120',
                       help='Comma-separated masses for time evolution plot')
    parser.add_argument('--times', type=str, default='1.0,5.0,10.0',
                       help='Comma-separated times for mass scaling plot')
    parser.add_argument('--time-max', type=float, default=30.0,
                       help='Maximum time for plots [Myr]')
    parser.add_argument('--show-all-metallicities', action='store_true',
                       help='Plot all metallicities for a single mass')
    parser.add_argument('--info', action='store_true',
                       help='Show database info and exit')
    parser.add_argument('--output-dir', type=str, default='./database_plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Check database exists
    if not os.path.exists(args.database):
        print(f"ERROR: Database not found: {args.database}")
        print("\nGenerate database with:")
        print("  python execute_stochastic_examples.py --quick")
        return
    
    # Show info if requested
    if args.info:
        get_database_info(args.database, verbose=True)
        return
    
    # Parse masses and times
    masses = [float(m) for m in args.masses.split(',')]
    times = [float(t) for t in args.times.split(',')]
    
    print("\n" + "#"*70)
    print("# DATABASE FEEDBACK PLOTTING")
    print("#"*70)
    print(f"Database: {args.database}")
    print(f"Output directory: {args.output_dir}")
    
    # Make output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot 1: Feedback vs time
    output_file = os.path.join(args.output_dir, 
                              f'database_feedback_vs_time_{args.metallicity}.png')
    plot_feedback_vs_time(
        database_path=args.database,
        metallicity=args.metallicity,
        masses=masses,
        time_max=args.time_max,
        output_file=output_file
    )
    
    # Plot 2: Feedback vs mass
    output_file = os.path.join(args.output_dir,
                              f'database_feedback_vs_mass_{args.metallicity}.png')
    plot_feedback_vs_mass(
        database_path=args.database,
        metallicity=args.metallicity,
        times=times,
        output_file=output_file
    )
    
    # Plot 3: All metallicities (if requested)
    if args.show_all_metallicities:
        output_file = os.path.join(args.output_dir,
                                  'database_metallicity_comparison.png')
        plot_all_metallicities(
            database_path=args.database,
            mass=25.0,
            time_max=args.time_max,
            output_file=output_file
        )
    
    print("\n" + "="*70)
    print("ALL PLOTS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()