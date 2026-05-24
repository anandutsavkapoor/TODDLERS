#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_all_paper_figures.py
=============================
Master script to generate ALL publication figures for the TODDLERS 2.0 paper.

Usage:
    python main/generate_all_paper_figures.py                # full pipeline
    python main/generate_all_paper_figures.py --fig 3        # only Fig 3
    python main/generate_all_paper_figures.py --fig 5 7      # only Figs 5 and 7
    python main/generate_all_paper_figures.py --plot-only     # skip simulations, replot from cached output
    python main/generate_all_paper_figures.py --fig meth     # methodology figures only

Author: Anand Utsav KAPOOR
Created: 2026-04-01
"""

import os
import sys
import re
import time
import json
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# `toddlers` is installed (pip install -e .); no sys.path manipulation needed.

# Import TODDLERS modules
from toddlers._paths import get_data_dir
from toddlers.evolution import Evolution
from toddlers.stellar_feedback import StellarFeedback
from toddlers.constants import *  # M_SUN, MYR_TO_SEC, PC_TO_CM, KM_TO_CM, M_P, MU_N, etc.
from toddlers.cloudy_stellar_spectra_generator import SpectralTableGenerator
from toddlers.cloudy_parallel_manager import ParallelCloudyManager
from toddlers.cloudy_simulation_manager import CloudySimulationManager
from toddlers.cloudy_output_handler import CloudyTableConsolidator, ReadCloudyMainOutput
from toddlers.track_simulation import load_output_file

# Import pySB99 modules
from toddlers.pysb99.generate_pysb99_interpolants import (
    generate_kroupa_like_interpolants as pysb99_generate_kroupa,
    generate_custom_population_interpolants as pysb99_generate_custom,
)
from toddlers.pysb99.stochastic.sampling import sample_imf_discrete, sample_ages
from toddlers.pysb99.stochastic.interpolants import (
    build_stochastic_interpolants_2d,
    save_stochastic_interpolants,
)
from toddlers.pysb99.stochastic.database import generate_single_star_database

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
DATABASE_DIR = get_data_dir()                        # $TODDLERS_DATA or <pkg>/database
FIGURES_DIR = os.path.join(THIS_DIR, 'figures')      # output figures (local)
RESULTS_DIR = os.path.join(THIS_DIR, 'evolution_output')
GRAIN_DIR = os.path.join(THIS_DIR, 'grain_comparison')
DATA_DIR = os.path.join(THIS_DIR, 'data')            # shipped reference data (BPT CSVs, ...)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAIN_DIR, exist_ok=True)  # grain figure runs Cloudy with cwd=GRAIN_DIR

DATABASE_FILE = os.path.join(DATABASE_DIR, 'single_star_tracks.h5')

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
STYLE_FILE = os.path.join(THIS_DIR, 'paper_style.mplstyle')
plt.style.use(STYLE_FILE)

# A&A column widths
SINGLE_COL = 8.8 / 2.54   # inches
_DOUBLE_COL_PAGE = 18.6 / 2.54  # inches (full page width)
# Scale figsize to 0.75x so that style-sheet font sizes (7-9 pt) remain
# readable when LaTeX includes the figure at 0.85\textwidth.
DOUBLE_COL = _DOUBLE_COL_PAGE * 0.75

# ---------------------------------------------------------------------------
# Standardised cloud parameters (Table 2)
# ---------------------------------------------------------------------------
# Shared parameters
ETA_SF = 0.05
N_CL = 160.0            # cm^-3, all sections
Z_DEFAULT = 0.014       # solar, Sects 4.1, 4.3-4.5
Z_DYNDENSITY = 0.0001  # Sect 4.2
LOG_MCL = 6.0
M_CL_INIT = 10**LOG_MCL * M_SUN  # grams

# Sect 4.1 now also uses Z=0.014 (MW) for consistency
Z_PROFILE = Z_DEFAULT

# Stochastic section uses low-mass cluster
STOCHASTIC_MASS = 1e3         # Msun stellar mass
STOCHASTIC_MMAX = 100         # Msun upper mass limit
STOCHASTIC_MCL = 2e4 * M_SUN  # cloud mass (grams) such that eta_sf * M_cl = 1e3 Msun

METALLICITY_MAPPING = {'MWC': 0.020, 'MW': 0.014, 'LMC': 0.006}


# ===================================================================
# Utility functions
# ===================================================================

def set_square_panels(fig):
    """Force all axes in the figure to have square plotting area."""
    for ax in fig.get_axes():
        ax.set_box_aspect(1)


def add_panel_label(ax, label, x=0.05, y=0.92):
    """Add panel label like (a), (b) etc."""
    ax.text(x, y, label, transform=ax.transAxes, fontweight='bold',
            fontsize=9, va='top')


def parse_summary_file(output_path):
    """Parse .summary file for phase transition times (Myr)."""
    summary_path = os.path.splitext(output_path)[0] + '.summary'
    transitions = {}
    if not os.path.exists(summary_path):
        print(f"  [WARN] Summary file not found: {summary_path}")
        return transitions
    with open(summary_path, 'r') as f:
        for line in f:
            match = re.match(r'\s+(\w+)\s+at\s+([\d.]+)\s+Myr', line)
            if match:
                transitions[match.group(1)] = float(match.group(2))
    return transitions


def get_fragmentation_time(generation):
    """Extract fragmentation time from phase_transitions list."""
    for transition in generation.get('phase_transitions', []):
        if transition[0] == 'phase1_to_fragmentation':
            return transition[1] / MYR_TO_SEC
    return None


def save_figure(fig, name):
    """Save figure as PDF and PNG to the figures directory."""
    pdf_path = os.path.join(FIGURES_DIR, f'{name}.pdf')
    png_path = os.path.join(FIGURES_DIR, f'{name}.png')
    fig.savefig(pdf_path, format='pdf')
    fig.savefig(png_path, format='png', dpi=150)
    plt.close(fig)
    print(f"  [OK] Saved: {pdf_path}")


def check_interpolant_exists(imf_name):
    """Check if pySB99 interpolant files exist."""
    main_file = os.path.join(DATABASE_DIR, f'pySB99interpolation_{imf_name}.obj')
    lw_file = os.path.join(DATABASE_DIR, f'pySB99interpolation_{imf_name}_LumLymanWerner.obj')
    mean_file = os.path.join(DATABASE_DIR, f'pySB99interpolation_{imf_name}_mean_ionizing_photon_energy.obj')
    return os.path.exists(main_file) and os.path.exists(lw_file) and os.path.exists(mean_file)


def generate_cloudy_tables(imf_name):
    """Generate Cloudy spectral tables for a given IMF."""
    time_points = np.logspace(np.log10(1e4), np.log10(50e6), 50)
    Z_values = [0.006, 0.014, 0.020]  # LMC, MW, MWC
    gen = SpectralTableGenerator(
        template='pySB99', imf=imf_name, star_type='sin',
        max_age=time_points[-1], wavelength_resolution_factor=1,
    )
    gen.generate_spectral_table(
        time_points=time_points, Z_values=Z_values,
        formation_timescale=None,
    )


def save_pysb99_metadata(imf_name, imf_config):
    """Save metadata JSON for a pySB99 interpolant."""
    metallicities = ['LMC', 'MW', 'MWC']
    Z_values = [METALLICITY_MAPPING[m] for m in metallicities]
    metadata = {
        'population_name': imf_name,
        'metallicities': metallicities,
        'Z_values': Z_values,
        'spectral_library': 'FW',
        'rotation': False,
        'time_start_myr': 0.01,
        'time_end_myr': 31.0,
        'time_step_myr': 0.1,
        'generation_date': datetime.now().isoformat(),
    }
    if 'exponents' in imf_config:
        metadata['imf_type'] = 'power_law'
        metadata['exponents'] = imf_config['exponents']
        metadata['mass_limits'] = imf_config['mass_limits']
    elif 'custom_star_numbers' in imf_config:
        metadata['imf_type'] = 'custom'
        metadata['custom_star_numbers'] = {str(k): v for k, v in imf_config['custom_star_numbers'].items()}
    path = os.path.join(DATABASE_DIR, f'pySB99interpolation_{imf_name}_metadata.json')
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)


# ===================================================================
# Fig 3: Profile / SF mode comparison (4 panels)
# ===================================================================

def generate_fig3(plot_only=False):
    """
    Fig 3: Impact of cloud structure and star formation mode.

    Four configurations compared: uniform+burst, MBE+burst, uniform+CSF, MBE+CSF.
    Panels: (a) shell radius, (b) velocity, (c) shell density, (d) shell mass.
    """
    print("\n" + "="*70)
    print("FIGURE 3: Profile / SF mode comparison")
    print("="*70)

    configs = [
        {"name": "Uniform_Burst", "profile_type": "uniform",
         "cluster_formation_mode": "burst", "formation_timescale": None},
        {"name": "ModBE_Burst", "profile_type": "modified_bonnor_ebert",
         "cluster_formation_mode": "burst", "formation_timescale": None, "alpha": 2},
        {"name": "Uniform_CSF", "profile_type": "uniform",
         "cluster_formation_mode": "constant_sfr",
         "formation_timescale": 2.0 * MYR_TO_SEC},
        {"name": "ModBE_CSF", "profile_type": "modified_bonnor_ebert",
         "cluster_formation_mode": "constant_sfr",
         "formation_timescale": 2.0 * MYR_TO_SEC, "alpha": 2},
    ]

    display_names = {
        "Uniform_Burst": r"Uni., burst",
        "ModBE_Burst": r"MBE, burst",
        "Uniform_CSF": r"Uni., CSF",
        "ModBE_CSF": r"MBE, CSF",
    }

    # --- Run / load simulations ---
    results = {}
    for cfg in configs:
        profile_params = {}
        if cfg['profile_type'] == 'modified_bonnor_ebert':
            profile_params['alpha'] = cfg.get('alpha', 2)

        evolution = Evolution(
            Z=Z_PROFILE, eta_sf=ETA_SF, n_cl=N_CL,
            M_cl_init=M_CL_INIT,
            template='BPASS',
            cluster_formation_mode=cfg['cluster_formation_mode'],
            formation_timescale=cfg['formation_timescale'],
            imf='chab100', star_type='bin',
            profile_type=cfg['profile_type'],
            profile_params=profile_params,
            dynamic_cloud_density=False,
            add_cover_frac=False,
            init_with_fragmentation=False,
            use_Tion_model=False,
        )
        output_path, _ = evolution.get_output_paths()

        if os.path.exists(output_path) and plot_only:
            _, data = load_output_file(output_path)
        elif os.path.exists(output_path):
            _, data = load_output_file(output_path)
        else:
            if plot_only:
                print(f"  [SKIP] No cached output for {cfg['name']}, use --run to generate")
                return
            data = evolution.run_simulation()

        results[cfg['name']] = data[0]  # first generation
        print(f"  [OK] {cfg['name']}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE_COL, DOUBLE_COL / 4 + 0.7),
                              sharex=True, constrained_layout=True)

    plot_specs = [
        {'key': 'radius', 'scale': 1/PC_TO_CM, 'ylabel': r'$R_\mathrm{sh}$ (pc)',
         'yscale': 'log'},
        {'key': 'velocity', 'scale': 1/KM_TO_CM,
         'ylabel': r'$V_\mathrm{sh}$ (km s$^{-1}$)', 'yscale': 'linear'},
        {'key': ('shell_properties', 'n_shell_in'), 'scale': 1.0,
         'ylabel': r'$n_\mathrm{in}$ (cm$^{-3}$)', 'yscale': 'log'},
        {'key': 'mass', 'scale': 1/M_SUN,
         'ylabel': r'$M_\mathrm{sh}$ ($M_\odot$)', 'yscale': 'log'},
    ]

    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color',
                [f'C{i}' for i in range(4)])

    for i, (spec, ax) in enumerate(zip(plot_specs, axes)):
        for j, cfg in enumerate(configs):
            gen = results[cfg['name']]
            t = gen['time'] / MYR_TO_SEC

            if isinstance(spec['key'], tuple):
                y = gen[spec['key'][0]][spec['key'][1]] * spec['scale']
            else:
                y = gen[spec['key']] * spec['scale']

            label = display_names[cfg['name']] if i == 3 else None
            ax.plot(t, y, color=colors[j], label=label)

            # Fragmentation time as dashed vertical line (panel a only)
            if i == 0:
                t_frag = get_fragmentation_time(gen)
                if t_frag is not None:
                    ax.axvline(t_frag, ls='--', color=colors[j], lw=0.7,
                               alpha=0.7)

        ax.set_yscale(spec['yscale'])
        ax.set_xlabel(r'Time (Myr)')
        ax.set_ylabel(spec['ylabel'])
        ax.set_xticks([0, 5, 10, 15, 20])
        add_panel_label(ax, panel_labels[i])

    axes[3].legend(loc='center right', fontsize=5, handlelength=1.2)

    set_square_panels(fig)
    save_figure(fig, 'fig3_profile_sf_comparison')


# ===================================================================
# Fig 4: Dynamic cloud density (6 panels, 2x3)
# ===================================================================

def generate_fig4(plot_only=False):
    """
    Fig 4: Dynamic vs static cloud density evolution.

    2x3 grid: (a) radius, (b) velocity, (c) shell density,
              (d) eta_rad + Lya multiplier, (e) cloud density, (f) mass.
    """
    print("\n" + "="*70)
    print("FIGURE 4: Dynamic cloud density")
    print("="*70)

    M_CL_INIT_DYNDENSITY = 10**6.5 * M_SUN  # log M=6.5 for this test case

    def _run_or_load(dynamic):
        evolution = Evolution(
            Z=Z_DYNDENSITY, eta_sf=ETA_SF, n_cl=N_CL,
            M_cl_init=M_CL_INIT_DYNDENSITY,
            template='BPASS',
            cluster_formation_mode='burst',
            formation_timescale=None,
            imf='chab100', star_type='bin',
            profile_type='uniform',
            add_cover_frac=False,
            dynamic_cloud_density=dynamic,
        )
        output_path, _ = evolution.get_output_paths()
        tag = "dynamic" if dynamic else "static"

        if os.path.exists(output_path):
            _, data = load_output_file(output_path)
            print(f"  [OK] Loaded {tag}: {output_path}")
        else:
            if plot_only:
                print(f"  [SKIP] No cached output for {tag}")
                return None, None
            data = evolution.run_simulation()
            print(f"  [OK] Ran {tag}")
        return data[0], output_path

    static_gen, static_path = _run_or_load(dynamic=False)
    dynamic_gen, dynamic_path = _run_or_load(dynamic=True)

    if static_gen is None or dynamic_gen is None:
        return

    # --- Plot ---
    fig, axs = plt.subplots(2, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.58),
                             sharex=True, constrained_layout=True,
                             gridspec_kw={'hspace': 0.02})

    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color',
                [f'C{i}' for i in range(4)])
    c_static, c_dynamic = colors[0], colors[1]

    datasets = [
        ("Static", static_gen, c_static),
        ("Dynamic", dynamic_gen, c_dynamic),
    ]

    # Twin axis for panel (d)
    ax_eta = axs[1, 0]
    ax_lya = ax_eta.twinx()

    for name, gen, color in datasets:
        t = gen['time'] / MYR_TO_SEC
        style = dict(color=color, lw=1.0)

        # (a) Radius
        axs[0, 0].semilogy(t, gen['radius'] / PC_TO_CM, label=name, **style)

        # (b) Velocity
        axs[0, 1].plot(t, gen['velocity'] / KM_TO_CM, label=name, **style)

        # (c) Shell density
        axs[0, 2].semilogy(t, gen['shell_properties']['n_shell_in'],
                            label=name, **style)

        # (d) eta_rad + Lya multiplier
        ax_eta.plot(t, gen['stellar_feedback']['eta_rad'], **style)
        ax_lya.plot(t, gen['stellar_feedback']['Lya_multiplier'],
                    ls='--', **style)

        # (e) Cloud density
        axs[1, 1].semilogy(t, gen['n_cloud_avg'], label=name, **style)

        # (f) Shell mass + cloud mass
        axs[1, 2].semilogy(t, gen['mass'] / M_SUN, **style)
        cloud_mass = gen['cloud_mass'] / M_SUN
        axs[1, 2].axhline(y=cloud_mass, ls=':', **style)

        # Fragmentation time as dashed vertical line (panel a only)
        for transition in gen.get('phase_transitions', []):
            if transition[0] == 'phase1_to_fragmentation':
                t_frag = transition[1] / MYR_TO_SEC
                axs[0, 0].axvline(t_frag, ls='--', color=color, lw=0.7,
                                  alpha=0.7)

    # Labels
    axs[0, 0].set_ylabel(r'$R_\mathrm{sh}$ (pc)')
    axs[0, 1].set_ylabel(r'$V_\mathrm{sh}$ (km s$^{-1}$)')
    axs[0, 2].set_ylabel(r'$n_\mathrm{in}$ (cm$^{-3}$)')
    ax_eta.set_ylabel(r'$\eta_\mathrm{rad}$')
    ax_lya.set_ylabel(r'Ly$\alpha$ mult.')
    axs[1, 1].set_ylabel(r'$n_\mathrm{cloud}$ (cm$^{-3}$)')
    axs[1, 2].set_ylabel(r'$M$ ($M_\odot$)')

    for ax in axs[1, :]:
        ax.set_xlabel(r'Time (Myr)')

    # Panel labels
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for i, ax in enumerate(axs.flatten()):
        add_panel_label(ax, panel_labels[i])

    # Legends
    axs[0, 0].legend(loc='lower right', fontsize=6)

    # Twin axis legend
    eta_line = Line2D([0], [0], color='k', lw=1.0)
    lya_line = Line2D([0], [0], color='k', lw=1.0, ls='--')
    ax_eta.legend([eta_line, lya_line], [r'$\eta_\mathrm{rad}$', r'Ly$\alpha$ mult.'],
                  loc='upper right', fontsize=6)

    # Mass panel legend
    shell_line = Line2D([0], [0], color='k', lw=1.0, ls='-')
    cloud_line = Line2D([0], [0], color='k', lw=1.0, ls=':')
    axs[1, 2].legend([shell_line, cloud_line], ['Shell', 'Cloud'],
                      loc='lower right', fontsize=6)

    for ax in axs[1, :]:
        ax.set_xticks([0, 5, 10, 15, 20, 25])

    set_square_panels(fig)
    save_figure(fig, 'fig4_dynamic_density')


# ===================================================================
# Fig 5: IMF comparison (3 panels)
# ===================================================================

def generate_fig5(plot_only=False):
    """
    Fig 5: Kroupa vs top-heavy IMF comparison.

    Panels: (a) Q(H), (b) ram force, (c) shell radius.
    """
    print("\n" + "="*70)
    print("FIGURE 5: IMF comparison")
    print("="*70)

    # --- Generate interpolants if needed ---
    kroupa_name = 'kroupa_100'
    topheavy_name = 'topheavy_100'

    kroupa_config = {
        'exponents': [1.3, 2.3],
        'mass_limits': [0.1, 0.5, 100.0],
    }
    topheavy_config = {
        'exponents': [1.3, 2.0],
        'mass_limits': [0.1, 0.5, 100.0],
    }

    if not check_interpolant_exists(kroupa_name):
        print("  Generating Kroupa interpolants...")
        pysb99_generate_kroupa(
            imf_exponents=kroupa_config['exponents'],
            imf_mass_limits=tuple(kroupa_config['mass_limits']),
            output_dir=DATABASE_DIR, imf_name=kroupa_name,
            metallicities=['LMC', 'MW', 'MWC'], rotation=False,
            time_start_myr=0.01, time_end_myr=31.0, time_step_myr=0.1,
            auto_extend_imf=False, verbose=True,
        )
        save_pysb99_metadata(kroupa_name, kroupa_config)
        generate_cloudy_tables(kroupa_name)

    if not check_interpolant_exists(topheavy_name):
        print("  Generating top-heavy interpolants...")
        pysb99_generate_kroupa(
            imf_exponents=topheavy_config['exponents'],
            imf_mass_limits=tuple(topheavy_config['mass_limits']),
            output_dir=DATABASE_DIR, imf_name=topheavy_name,
            metallicities=['LMC', 'MW', 'MWC'], rotation=False,
            time_start_myr=0.01, time_end_myr=31.0, time_step_myr=0.1,
            auto_extend_imf=False, verbose=True,
        )
        save_pysb99_metadata(topheavy_name, topheavy_config)
        generate_cloudy_tables(topheavy_name)

    # --- Run simulations ---
    evo_data = {}
    for imf_name, label in [(kroupa_name, 'Kroupa'), (topheavy_name, 'Top-heavy')]:
        evolution = Evolution(
            Z=Z_DEFAULT, eta_sf=ETA_SF, n_cl=N_CL,
            M_cl_init=M_CL_INIT,
            template='pySB99', imf=imf_name, star_type='sin',
            cluster_formation_mode='burst', formation_timescale=None,
            profile_type='uniform', add_cover_frac=False,
        )
        output_path, _ = evolution.get_output_paths()

        if os.path.exists(output_path):
            _, data = load_output_file(output_path)
        else:
            if plot_only:
                print(f"  [SKIP] No cached output for {label}")
                return
            data = evolution.run_simulation()

        evo_data[label] = {'gen': data[0], 'path': output_path}
        print(f"  [OK] {label}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL / 3 + 0.7),
                              sharex=True, constrained_layout=True)

    for label, style in [('Kroupa', {}), ('Top-heavy', {})]:
        gen = evo_data[label]['gen']
        t = gen['time'] / MYR_TO_SEC

        axes[0].semilogy(t, gen['stellar_feedback']['Q_i'], label=label)
        axes[1].semilogy(t, gen['stellar_feedback']['F_ram'])
        axes[2].semilogy(t, gen['radius'] / PC_TO_CM)

        # Fragmentation line in panel (c)
        t_frag = get_fragmentation_time(gen)
        if t_frag is not None:
            for ax_idx in [2]:
                axes[ax_idx].axvline(t_frag, ls='--', lw=0.7, alpha=0.7,
                                     color=axes[0].get_lines()[-1].get_color())

    axes[0].set_ylabel(r'$Q(\mathrm{H})$ (s$^{-1}$)')
    axes[1].set_ylabel(r'$\dot{p}$ (dyne)')
    axes[2].set_ylabel(r'$R_\mathrm{sh}$ (pc)')
    for ax in axes:
        ax.set_xlabel(r'Time (Myr)')

    add_panel_label(axes[0], '(a)')
    add_panel_label(axes[1], '(b)')
    add_panel_label(axes[2], '(c)')
    axes[0].legend(loc='lower left')

    set_square_panels(fig)
    save_figure(fig, 'pysb99_imf_comparison')


# ===================================================================
# Fig 6: Custom stellar populations (2 panels)
# ===================================================================

def generate_fig6(plot_only=False):
    """
    Fig 6: Custom population feedback comparison.

    Panels: (a) Q(H), (b) ram force.
    Three populations: massive-dominated, intermediate, low-mass.
    """
    print("\n" + "="*70)
    print("FIGURE 6: Custom feedback")
    print("="*70)

    populations = [
        ('example_massive', {100.0: 1, 30.0: 1, 15.0: 2}, 160.0,
         r"Massive-dom."),
        ('example_intermediate', {30.0: 1, 15.0: 2}, 60.0,
         r"Intermediate"),
        ('example_lowmass', {8.0: 10, 5.0: 20}, 180.0,
         r"Low-mass"),
    ]

    # Generate interpolants
    for name, pop, _, _ in populations:
        if not check_interpolant_exists(name):
            print(f"  Generating interpolant for {name}...")
            config = {'custom_star_numbers': pop}
            pysb99_generate_custom(
                output_dir=DATABASE_DIR, imf_name=name,
                custom_star_numbers=pop,
                metallicities=['LMC', 'MW', 'MWC'], rotation=False,
                time_start_myr=0.01, time_end_myr=31.0, time_step_myr=0.1,
                verbose=True,
            )
            save_pysb99_metadata(name, config)
            generate_cloudy_tables(name)

    # Calculate feedback
    time_myr = np.linspace(0.01, 10.0, 200)
    time_sec = time_myr * MYR_TO_SEC

    results = {}
    for name, pop, total_mass, label in populations:
        feedback = StellarFeedback(
            template='pySB99', Z=Z_DEFAULT,
            M_cl_init=1e6 * M_SUN, eta_sf=1.0,
            t_list_collapse=[], mode='burst',
            imf=name, star_type='sin',
        )
        Q = np.array([feedback.get_ionizing_photon_rate(t) for t in time_sec])
        F = np.array([feedback.get_ram_force(t) for t in time_sec])
        scale = total_mass / 1e6
        results[name] = {'Q': Q * scale, 'F': F * scale, 'label': label}
        print(f"  [OK] {label}: {total_mass} Msun")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL * 0.75, 3.0 * 0.75),
                              constrained_layout=True)

    linestyles = ['-', '--', '-']
    for (name, _, _, _), ls in zip(populations, linestyles):
        d = results[name]
        axes[0].semilogy(time_myr, d['Q'], ls=ls, label=d['label'])
        axes[1].semilogy(time_myr, d['F'], ls=ls)

    axes[0].set_xlabel(r'Time (Myr)')
    axes[0].set_ylabel(r'$Q(\mathrm{H})$ (s$^{-1}$)')
    axes[1].set_xlabel(r'Time (Myr)')
    axes[1].set_ylabel(r'$\dot{p}$ (dyne)')

    add_panel_label(axes[0], '(a)')
    add_panel_label(axes[1], '(b)')
    axes[0].legend(loc='upper right', fontsize=6)

    set_square_panels(fig)
    save_figure(fig, 'custom_feedback_examples')


# ===================================================================
# Fig 7: Stochastic sampling (3 panels)
# ===================================================================


def characterize_sampling_mass_error(
        total_mass, m_upper, n_realizations=500, metallicity='MW',
        imf_name='kroupa', rotation=False):
    """
    Characterize the mass error of discrete IMF sampling over many realizations.

    Decomposes the total error into contributions from (i) stop-after truncation
    (continuous sum vs target) and (ii) grid snapping (discrete sum vs continuous
    sum). Prints summary statistics and returns the arrays for further use.

    Parameters
    ----------
    total_mass : float
        Target stellar mass [Msun]
    m_upper : float
        Upper mass limit [Msun]
    n_realizations : int
        Number of independent draws
    metallicity : str
        Metallicity identifier for the grid
    imf_name : str
        IMF name
    rotation : bool
        Use rotating models

    Returns
    -------
    dict
        'cont_err_pct': array of continuous mass errors [%]
        'disc_err_pct': array of discrete (total) mass errors [%]
        'snap_err_pct': array of grid-snapping-only errors [%]
        'n_stars': array of star counts per realization
    """
    from toddlers.pysb99.stochastic.sampling import sample_imf
    from toddlers.pysb99.stochastic.database import get_available_mass_grid

    grid = get_available_mass_grid(metallicity, rotation)
    grid = grid[grid <= m_upper]

    cont_err = np.zeros(n_realizations)
    disc_err = np.zeros(n_realizations)
    snap_err = np.zeros(n_realizations)
    n_stars_arr = np.zeros(n_realizations, dtype=int)

    for i in range(n_realizations):
        continuous = sample_imf(
            total_mass, imf_name=imf_name,
            m_min=grid.min(), m_max=grid.max(), seed=i,
        )
        cont_sum = np.sum(continuous)

        discrete = np.array([
            grid[np.argmin(np.abs(grid - m))] for m in continuous
        ])
        disc_sum = np.sum(discrete)

        cont_err[i] = 100.0 * (cont_sum - total_mass) / total_mass
        disc_err[i] = 100.0 * (disc_sum - total_mass) / total_mass
        snap_err[i] = 100.0 * (disc_sum - cont_sum) / total_mass
        n_stars_arr[i] = len(continuous)

    print(f"\n  Sampling mass error characterization "
          f"({n_realizations} realizations, M*={total_mass:.0e} Msun, "
          f"m_upper={m_upper:.0f} Msun):")
    print(f"  {'':30s} {'mean':>8s} {'median':>8s} {'std':>8s} "
          f"{'min':>8s} {'max':>8s}")
    for label, arr in [('Stop-after truncation [%]', cont_err),
                       ('Grid snapping only [%]', snap_err),
                       ('Total (discrete) [%]', disc_err)]:
        print(f"  {label:30s} {np.mean(arr):+8.2f} {np.median(arr):+8.2f} "
              f"{np.std(arr):8.2f} {np.min(arr):+8.2f} {np.max(arr):+8.2f}")
    print(f"  {'|Total| [%]':30s} {np.mean(np.abs(disc_err)):8.2f} "
          f"{np.median(np.abs(disc_err)):8.2f} {np.std(np.abs(disc_err)):8.2f} "
          f"{np.min(np.abs(disc_err)):8.2f} {np.max(np.abs(disc_err)):8.2f}")
    print(f"  N_stars: mean={np.mean(n_stars_arr):.0f}, "
          f"std={np.std(n_stars_arr):.0f}, "
          f"range=[{np.min(n_stars_arr)}, {np.max(n_stars_arr)}]")

    return {
        'cont_err_pct': cont_err,
        'disc_err_pct': disc_err,
        'snap_err_pct': snap_err,
        'n_stars': n_stars_arr,
    }


def generate_fig7(plot_only=False):
    """
    Fig 7: Stochastic vs deterministic comparison.

    Panels: (a) Q(H), (b) ram force, (c) shell radius.
    Survey 50 populations, select 2 contrasting, compare with deterministic.
    """
    print("\n" + "="*70)
    print("FIGURE 7: Stochastic comparison")
    print("="*70)

    # --- Ensure single-star database exists ---
    if not os.path.exists(DATABASE_FILE):
        print("  Generating single-star database...")
        generate_single_star_database(DATABASE_FILE, metallicity='MW',
                                       rotation=False, verbose=True)

    # --- Ensure deterministic Kroupa interpolant exists ---
    det_imf = 'kroupa_100'
    if not check_interpolant_exists(det_imf):
        print("  Generating Kroupa interpolant...")
        pysb99_generate_kroupa(
            output_dir=DATABASE_DIR, imf_name=det_imf,
            metallicities=['LMC', 'MW', 'MWC'], rotation=False,
            time_start_myr=0.01, time_end_myr=31.0, time_step_myr=0.1,
            verbose=True,
        )
        generate_cloudy_tables(det_imf)

    # --- Characterize sampling mass error ---
    characterize_sampling_mass_error(
        total_mass=STOCHASTIC_MASS, m_upper=STOCHASTIC_MMAX,
        n_realizations=500, metallicity='MW',
    )

    # --- Survey stochastic populations ---
    survey_file = os.path.join(FIGURES_DIR, 'stochastic_population_survey.csv')
    n_survey = 50

    if os.path.exists(survey_file) and plot_only:
        import csv
        survey_data = []
        with open(survey_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                survey_data.append(row)
        print(f"  Loaded survey from {survey_file}")
    else:
        print(f"  Surveying {n_survey} stochastic populations...")

        survey_data = []
        for seed in range(n_survey):
            masses = sample_imf_discrete(
                total_mass=STOCHASTIC_MASS,
                database_path=DATABASE_FILE,
                metallicity='MW',
                m_upper=STOCHASTIC_MMAX,
                seed=seed,
            )
            m_max = float(np.max(masses)) if len(masses) > 0 else 0
            n_stars = len(masses)
            n_above_20 = int(np.sum(masses > 20))
            survey_data.append({
                'seed': seed, 'm_max': m_max,
                'n_stars': n_stars, 'n_above_20': n_above_20,
                'total_mass': float(np.sum(masses)),
            })

        # Save survey
        import csv
        with open(survey_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=survey_data[0].keys())
            writer.writeheader()
            writer.writerows(survey_data)
        print(f"  Saved survey to {survey_file}")

    # --- Select two contrasting cases ---
    sorted_by_mmax = sorted(survey_data, key=lambda x: float(x['m_max']),
                             reverse=True)
    high_case = sorted_by_mmax[0]
    low_case = sorted_by_mmax[-1]
    # Pick a low case that still has some stars above 20 Msun or just the lowest
    for entry in reversed(sorted_by_mmax):
        if float(entry['m_max']) > 10:
            low_case = entry
            break

    selected = [high_case, low_case]
    print(f"  Selected seeds: {high_case['seed']} (m_max={float(high_case['m_max']):.1f}), "
          f"{low_case['seed']} (m_max={float(low_case['m_max']):.1f})")

    # --- Generate stochastic interpolants ---
    stoch_imf_names = []
    pop_infos = []

    for case in selected:
        seed = int(case['seed'])
        masses = sample_imf_discrete(
            total_mass=STOCHASTIC_MASS,
            database_path=DATABASE_FILE,
            metallicity='MW',
            m_upper=STOCHASTIC_MMAX,
            seed=seed,
        )

        imf_name = f'stochastic_seed{seed}_M{int(STOCHASTIC_MASS)}'

        if not check_interpolant_exists(imf_name):
            print(f"  Building interpolants for seed {seed}...")
            # Instantaneous burst: all ages = 0
            ages = sample_ages(
                n_stars=len(masses), t_sf_myr=0.0,
                mode='uniform', seed=seed * 2,
            )
            time_grid = np.logspace(np.log10(0.01), np.log10(50), 50)
            interps = build_stochastic_interpolants_2d(
                masses=masses, initial_ages=ages,
                database_path=DATABASE_FILE,
                metallicities=['LMC', 'MW', 'MWC'],
                time_grid_myr=time_grid,
                rotation=False, imf_name='kroupa', verbose=True,
            )
            save_stochastic_interpolants(
                interpolants=interps, output_dir=DATABASE_DIR,
                imf_name=imf_name, rotation=False, overwrite=True,
            )

        stoch_imf_names.append(imf_name)
        pop_infos.append({
            'seed': seed,
            'm_max': float(case['m_max']),
            'n_stars': int(case['n_stars']),
            'n_above_20': int(case['n_above_20']),
        })

    # --- Run evolutions ---
    all_runs = [
        ('Fully-sampled', det_imf),
        (f"$m_\\mathrm{{max}}\\approx{float(high_case['m_max']):.0f}\\,M_\\odot$",
         stoch_imf_names[0]),
        (f"$m_\\mathrm{{max}}\\approx{float(low_case['m_max']):.0f}\\,M_\\odot$",
         stoch_imf_names[1]),
    ]

    evo_data = {}
    for label, imf_name in all_runs:
        # Stochastic uses lower cloud mass
        if 'stochastic' in imf_name:
            mcl = STOCHASTIC_MCL
        else:
            mcl = STOCHASTIC_MCL  # same cloud for deterministic comparison

        evolution = Evolution(
            Z=Z_DEFAULT, eta_sf=ETA_SF, n_cl=N_CL,
            M_cl_init=mcl,
            template='pySB99', imf=imf_name, star_type='sin',
            cluster_formation_mode='burst', formation_timescale=None,
            profile_type='uniform', add_cover_frac=False,
        )
        output_path, _ = evolution.get_output_paths()

        if os.path.exists(output_path):
            _, data = load_output_file(output_path)
        else:
            if plot_only:
                print(f"  [SKIP] No cached output for {label}")
                return
            data = evolution.run_simulation()

        evo_data[label] = {'gen': data[0], 'path': output_path}
        print(f"  [OK] {label}")

    # --- Save diagnostics ---
    diagnostics = {'params': {
        'Z': Z_DEFAULT, 'eta_sf': ETA_SF, 'n_cl': N_CL,
        'M_cl_init_Msun': STOCHASTIC_MCL / M_SUN,
    }}
    for label in evo_data:
        gen = evo_data[label]['gen']
        t_frag = get_fragmentation_time(gen)
        if t_frag is None:
            transitions = parse_summary_file(evo_data[label]['path'])
            t_frag = transitions.get('phase1_to_fragmentation')
        diagnostics[label] = {
            'peak_Q_ion': float(np.max(gen['stellar_feedback']['Q_i'])),
            'peak_F_ram': float(np.max(gen['stellar_feedback']['F_ram'])),
            't_frag_myr': t_frag,
            'max_radius_pc': float(np.max(gen['radius'] / PC_TO_CM)),
        }
    diag_file = os.path.join(FIGURES_DIR, 'pysb99_stochastic_diagnostics.json')
    with open(diag_file, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    print(f"  [OK] Diagnostics: {diag_file}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL / 3 + 0.7),
                              sharex=True, constrained_layout=True)

    for label in evo_data:
        gen = evo_data[label]['gen']
        t = gen['time'] / MYR_TO_SEC

        lbl = label if label == 'Fully-sampled' else label
        axes[0].semilogy(t, gen['stellar_feedback']['Q_i'],
                          label=lbl if True else None)
        axes[1].semilogy(t, gen['stellar_feedback']['F_ram'])
        axes[2].semilogy(t, gen['radius'] / PC_TO_CM)

        # Fragmentation line
        t_frag = get_fragmentation_time(gen)
        if t_frag is None:
            transitions = parse_summary_file(evo_data[label]['path'])
            t_frag = transitions.get('phase1_to_fragmentation')
        if t_frag is not None:
            c = axes[0].get_lines()[-1].get_color()
            axes[2].axvline(t_frag, ls='--', lw=0.7, alpha=0.7, color=c)

    axes[0].set_ylabel(r'$Q(\mathrm{H})$ (s$^{-1}$)')
    axes[1].set_ylabel(r'$\dot{p}$ (dyne)')
    axes[2].set_ylabel(r'$R_\mathrm{sh}$ (pc)')
    for ax in axes:
        ax.set_xlabel(r'Time (Myr)')

    add_panel_label(axes[0], '(a)')
    add_panel_label(axes[1], '(b)')
    add_panel_label(axes[2], '(c)')
    axes[0].legend(loc='lower left', fontsize=7)
    for ax in axes:
        ax.tick_params(labelsize=8)

    set_square_panels(fig)
    save_figure(fig, 'pysb99_stochastic_comparison')


# ===================================================================
# Methodology figures
# ===================================================================

def generate_fig_density_profiles():
    """Methodology: density profile shapes (single panel)."""
    print("\n" + "="*70)
    print("METHODOLOGY: Density profiles")
    print("="*70)

    from toddlers.cloud_density_profiles import create_density_profile

    # Match Sect. 3.1 parameters: log M = 6.0, n_cl = 160 cm^-3
    M_cl = 10**6.0 * M_SUN
    n_avg = N_CL  # 160 cm^-3

    profiles = {
        'Uniform': create_density_profile('uniform', M_cl, n_avg),
        'Bonnor-Ebert': create_density_profile('bonnor_ebert', M_cl, n_avg, xi_max=6.5),
        r'Mod. BE ($\alpha=2$)': create_density_profile('modified_bonnor_ebert', M_cl, n_avg, alpha=2),
        'Gaussian': create_density_profile('gaussian', M_cl, n_avg, sigma_fraction=0.2),
    }

    fig, ax1 = plt.subplots(figsize=(SINGLE_COL * 0.8, SINGLE_COL * 0.8),
                          constrained_layout=True)
    ax2 = ax1.twinx()

    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color',
                [f'C{i}' for i in range(4)])

    for i, (name, profile) in enumerate(profiles.items()):
        color = colors[i % len(colors)]
        r_range = np.logspace(np.log10(profile.R_cl / 10000),
                               np.log10(profile.R_cl), 5000, endpoint=False)
        rho = np.array([profile.density(r) for r in r_range])
        n_H = rho / MU_N  # convert mass density to number density
        m_enc = np.array([profile.mass_enclosed(r) for r in r_range])
        r_norm = r_range / profile.R_cl

        ax1.semilogy(r_norm, n_H, color=color, label=name)
        ax2.semilogy(r_norm, m_enc / M_cl, color=color, ls='--', alpha=0.6)

    ax1.set_xlabel(r'$r / R_\mathrm{cl}$')
    ax1.set_ylabel(r'$n$ (cm$^{-3}$)')
    ax2.set_ylabel(r'$M(<r) / M_\mathrm{cl}$')
    ax2.set_ylim(1e-8, 2)

    # Combined legend: profile names + line style explanation
    legend_elements = []
    for i, name in enumerate(profiles):
        legend_elements.append(Line2D([0], [0], color=colors[i], lw=1.0, label=name))
    legend_elements.append(Line2D([0], [0], color='gray', lw=1.0, label='Density'))
    legend_elements.append(Line2D([0], [0], color='gray', lw=1.0, ls='--', label='Enclosed mass'))
    ax1.legend(handles=legend_elements, fontsize=5, loc='lower left')

    set_square_panels(fig)
    save_figure(fig, 'density_profiles')


def generate_fig_ram_force():
    """Methodology: ram force comparison (single panel) with cumulative on twin axis."""
    print("\n" + "="*70)
    print("METHODOLOGY: Ram force")
    print("="*70)

    Z = 0.014
    eta_sf = 0.05
    n_cl = 100.0
    M_cl_init = 1e5 * M_SUN

    time_myr = np.linspace(0.01, 15.0, 500)
    time_sec = time_myr * MYR_TO_SEC
    dt = np.diff(time_sec, prepend=0)

    # Burst mode
    fb_burst = StellarFeedback(
        template='SB99', Z=Z, M_cl_init=M_cl_init, eta_sf=eta_sf,
        t_list_collapse=[], mode='burst',
    )
    F_burst = np.array([fb_burst.get_ram_force(t) for t in time_sec])
    cum_burst = np.cumsum(F_burst * dt)

    # Constant SFR
    fb_csf = StellarFeedback(
        template='SB99', Z=Z, M_cl_init=M_cl_init, eta_sf=eta_sf,
        t_list_collapse=[], mode='constant_sfr',
        formation_timescale=2.0 * MYR_TO_SEC,
    )
    F_csf = np.array([fb_csf.get_ram_force(t) for t in time_sec])
    cum_csf = np.cumsum(F_csf * dt)

    fig, ax1 = plt.subplots(figsize=(SINGLE_COL * 0.8, SINGLE_COL * 0.8),
                          constrained_layout=True)
    ax2 = ax1.twinx()

    l1, = ax1.semilogy(time_myr, F_burst, label='Burst')
    l2, = ax1.semilogy(time_myr, F_csf,
                        label=r'CSF ($\tau_\mathrm{SF}=2$ Myr)')

    l3, = ax2.semilogy(time_myr, cum_burst, ls='--',
                        color=l1.get_color(), label='Burst (cumul.)')
    l4, = ax2.semilogy(time_myr, cum_csf, ls='--',
                        color=l2.get_color(), label='CSF (cumul.)')

    ax1.set_xlabel(r'Time (Myr)')
    ax1.set_ylabel(r'$\dot{p}$ (dyne)')
    ax2.set_ylabel(r'$\int \dot{p}\,\mathrm{d}t$ (dyne s)')
    ax1.set_xlim(0, 15)
    ax1.set_ylim(bottom=5e27)

    ax1.legend([l1, l2, l3, l4],
               ['Burst', r'CSF ($\tau_\mathrm{SF}=2$ Myr)',
                'Burst (cumul.)', 'CSF (cumul.)'],
               fontsize=5, loc='lower right')

    set_square_panels(fig)
    save_figure(fig, 'ram_force_dyne_short_term_single_gen_SB99')


# ===================================================================
# Fig 8: BPT diagram with / without DIG
# ===================================================================

# BPT line keys in consolidated table files (species_wavelength format).
BPT_LINES = {
    'oiii_5007': 'O3_5006.84A',
    'hbeta':     'H1_4861.32A',
    'nii_6584':  'N2_6583.45A',
    'halpha':    'H1_6562.80A',
}


def _parse_intensity_file(filepath):
    """Parse a consolidated intensity text file into {key: linear_value}."""
    intensities = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            key = f"{parts[0]}_{parts[1]}"
            intensities[key] = 10**float(parts[2])
    return intensities


def _gather_bpt_ratios(directory, intensity_type='emergent'):
    """Read consolidated table files and return BPT log10 ratios vs time."""
    timestamps, oiii_hb, nii_ha = [], [], []
    if not os.path.isdir(directory):
        print(f"  [WARN] Directory not found: {directory}")
        return nii_ha, oiii_hb, timestamps

    for fname in sorted(os.listdir(directory)):
        if not (fname.startswith(f"database_lines_{intensity_type} line intensities_")
                and fname.endswith('.txt')):
            continue
        t = float(re.search(r'_([0-9.]+)\.txt$', fname).group(1))
        intensities = _parse_intensity_file(os.path.join(directory, fname))

        # H-like iso-sequence file has Balmer lines
        h_fname = f"H-like_iso-sequence_{intensity_type} line intensities_{t:.2f}.txt"
        h_path = os.path.join(directory, h_fname)
        if os.path.exists(h_path):
            intensities.update(_parse_intensity_file(h_path))

        oiii = intensities.get(BPT_LINES['oiii_5007'])
        hb   = intensities.get(BPT_LINES['hbeta'])
        nii  = intensities.get(BPT_LINES['nii_6584'])
        ha   = intensities.get(BPT_LINES['halpha'])

        if all(v and v > 0 for v in [oiii, hb, nii, ha]):
            oiii_hb.append(np.log10(oiii / hb))
            nii_ha.append(np.log10(nii / ha))
            timestamps.append(t)

    return nii_ha, oiii_hb, timestamps


def _consolidate_cloudy_tables(sim_file, with_dig):
    """Run CloudyTableConsolidator on Cloudy output (mirrors main/execute.py)."""
    manager = CloudySimulationManager(sim_file, complete_init=False)
    cloudy_dir = manager.cloudy_run_dir
    tag = 'with_dig' if with_dig else 'no_dig'
    output_dir = os.path.join(cloudy_dir, f'consolidated_tables_{tag}')
    os.makedirs(output_dir, exist_ok=True)

    consolidator = CloudyTableConsolidator(ReadCloudyMainOutput, output_dir=output_dir)

    model_types = ['shell', 'unified']
    if with_dig:
        model_types.append('dig')

    timesteps = set()
    for mtype in model_types:
        for f in os.listdir(cloudy_dir):
            if f.startswith(f'{mtype}_') and f.endswith('.out'):
                try:
                    timesteps.add(float(f.split('_')[1].replace('.out', '')))
                except (ValueError, IndexError):
                    pass

    ok, fail = 0, 0
    for ts in sorted(timesteps):
        shell_f   = os.path.join(cloudy_dir, f'shell_{ts:.2f}.out')
        unified_f = os.path.join(cloudy_dir, f'unified_{ts:.2f}.out')
        dig_f     = os.path.join(cloudy_dir, f'dig_{ts:.2f}.out') if with_dig else None
        is_within = os.path.exists(unified_f)

        if (os.path.exists(shell_f) or os.path.exists(unified_f) or
                (with_dig and dig_f and os.path.exists(dig_f))):
            try:
                consolidator.consolidate_tables(
                    timestep=ts, shell_file=shell_f, unified_file=unified_f,
                    dig_file=dig_f, is_within_cloud=is_within, add_dig=with_dig)
                ok += 1
            except Exception as e:
                print(f"  [WARN] Consolidation failed at t={ts:.2f}: {e}")
                fail += 1

    print(f"  [OK] Consolidated {ok} timesteps ({tag}), {fail} failed")
    return output_dir


def generate_fig8(plot_only=False):
    """
    Fig 8: BPT diagram with and without DIG contribution.

    Pipeline: evolution -> Cloudy with DIG -> consolidate tables -> plot BPT.
    """
    print("\n" + "="*70)
    print("FIGURE 8: BPT diagram (DIG effect)")
    print("="*70)

    # --- Step 1: evolution ---
    evolution = Evolution(
        Z=Z_DEFAULT, eta_sf=ETA_SF, n_cl=N_CL, M_cl_init=M_CL_INIT,
        template='BPASS', cluster_formation_mode='burst',
        formation_timescale=None, imf='chab100', star_type='bin',
        profile_type='uniform', add_cover_frac=False,
    )
    output_path, _ = evolution.get_output_paths()

    if os.path.exists(output_path):
        print(f"  [OK] Evolution output exists: {output_path}")
    else:
        if plot_only:
            print("  [SKIP] No evolution output, run without --plot-only")
            return
        evolution.run_simulation()
        print(f"  [OK] Evolution done: {output_path}")

    # --- Step 2: Cloudy with DIG ---
    manager = CloudySimulationManager(output_path, complete_init=False)
    cloudy_dir = manager.cloudy_run_dir
    dig_files = ([f for f in os.listdir(cloudy_dir)
                  if f.startswith('dig_') and f.endswith('.out')]
                 if os.path.isdir(cloudy_dir) else [])

    if len(dig_files) > 0:
        print(f"  [OK] Found {len(dig_files)} DIG output files")
    elif not plot_only:
        print("  Running Cloudy with DIG (this is slow, use a cluster)...")
        pm = ParallelCloudyManager(
            max_workers=56, time_sampling='adaptive',
            continue_after_dissolution=True, add_dig=True,
        )
        pm.run_parallel_simulations([output_path])
        print("  [OK] Cloudy with DIG done")
    else:
        print("  [SKIP] No Cloudy DIG output, run without --plot-only")
        return

    # --- Step 3: consolidate tables ---
    cons_dig   = os.path.join(cloudy_dir, 'consolidated_tables_with_dig')
    cons_nodig = os.path.join(cloudy_dir, 'consolidated_tables_no_dig')

    if not os.path.isdir(cons_dig) or len(os.listdir(cons_dig)) == 0:
        print("  Consolidating tables (with DIG)...")
        cons_dig = _consolidate_cloudy_tables(output_path, with_dig=True)
    else:
        print(f"  [OK] Consolidated tables (with DIG) exist")

    if not os.path.isdir(cons_nodig) or len(os.listdir(cons_nodig)) == 0:
        print("  Consolidating tables (no DIG)...")
        cons_nodig = _consolidate_cloudy_tables(output_path, with_dig=False)
    else:
        print(f"  [OK] Consolidated tables (no DIG) exist")

    # --- Step 4: plot BPT ---
    nii_dig, oiii_dig, t_dig       = _gather_bpt_ratios(cons_dig, 'emergent')
    nii_nodig, oiii_nodig, t_nodig = _gather_bpt_ratios(cons_nodig, 'emergent')

    if not t_dig and not t_nodig:
        print("  [WARN] No BPT data found, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 0.8, SINGLE_COL * 0.8),
                           constrained_layout=True)

    t_min, t_max = 0, 12
    cmap = plt.cm.viridis

    if t_dig:
        sc = ax.scatter(nii_dig, oiii_dig, c=t_dig, cmap=cmap,
                        vmin=t_min, vmax=t_max, s=15, marker='o',
                        alpha=0.6, label='With DIG', zorder=5)
    if t_nodig:
        ax.scatter(nii_nodig, oiii_nodig, c=t_nodig, cmap=cmap,
                   vmin=t_min, vmax=t_max, s=15, marker='x',
                   alpha=0.6, label='No DIG', zorder=4)

    # Kewley & Kauffmann demarcation curves
    x_kew = np.linspace(-4, 0.4, 200)
    x_kau = np.linspace(-4, 0.0, 200)
    ax.plot(x_kew, 0.61 / (x_kew - 0.47) + 1.19, 'r--', lw=0.8,
            label='Kewley+01')
    ax.plot(x_kau, 0.61 / (x_kau - 0.05) + 1.30, 'b-.', lw=0.8,
            label='Kauffmann+03')

    ax.set_xlabel(r'$\log\,([\mathrm{N\,II}]/\mathrm{H}\alpha)$')
    ax.set_ylabel(r'$\log\,([\mathrm{O\,III}]/\mathrm{H}\beta)$')
    ax.set_xlim(-1.75, 0.25)
    ax.set_ylim(-0.25, 1.1)
    ax.legend(loc='lower left')

    if t_dig:
        cbar = fig.colorbar(sc, ax=ax, orientation='horizontal',
                            pad=0.02, aspect=30)
        cbar.set_label('Time (Myr)')

    save_figure(fig, 'bpt_dig_comparison')


# ===================================================================
# Appendix: Grain SED comparison (Orion vs ISM grains)
# ===================================================================

def _read_cloudy_cont(filepath):
    """Read a Cloudy .cont file.  Returns arrays: nu_ryd, incident, diffout, total."""
    nu, incident, diffout, total = [], [], [], []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            cols = line.split()
            nu.append(float(cols[0]))
            incident.append(float(cols[1]))
            diffout.append(float(cols[3]))
            total.append(float(cols[6]))
    return {
        'nu_ryd': np.array(nu),
        'incident': np.array(incident),
        'diffout': np.array(diffout),
        'total': np.array(total),
    }


def _run_cloudy_in_dir(input_file, work_dir):
    """Run Cloudy on an input file inside work_dir."""
    import subprocess
    inp_base = os.path.basename(input_file).replace('.in', '')
    result = subprocess.run(
        [CLOUDY_EXE, '-r', inp_base],
        cwd=work_dir, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  [ERROR] Cloudy failed for {inp_base}: {result.stderr[:200]}")
    else:
        print(f"  [OK] Cloudy finished: {inp_base}")


def generate_fig_grain_comparison(plot_only=False):
    """
    Appendix figure: SED comparison between Orion-like (ratio=0.1) and
    ISM-like (ratio=0.4) grain size distributions for the same HII region.

    Runs Cloudy from the .in files in paper/grain_comparison/ if output
    does not exist, then plots the resulting continua.
    """
    print("\n" + "="*70)
    print("APPENDIX: Grain SED comparison")
    print("="*70)

    orion_cont = os.path.join(GRAIN_DIR, 'hii_orion.cont')
    ism_cont   = os.path.join(GRAIN_DIR, 'hii_ism.cont')

    # Run Cloudy if output files are missing
    for label, cont_file, in_file in [
        ('Orion', orion_cont, os.path.join(GRAIN_DIR, 'hii_orion.in')),
        ('ISM',   ism_cont,   os.path.join(GRAIN_DIR, 'hii_ism.in')),
    ]:
        if os.path.exists(cont_file):
            print(f"  [OK] {label} continuum exists: {cont_file}")
        elif plot_only:
            print(f"  [SKIP] {label} continuum not found, run without --plot-only")
            return
        else:
            print(f"  Running Cloudy for {label} grains...")
            _run_cloudy_in_dir(in_file, GRAIN_DIR)

    if not os.path.exists(orion_cont) or not os.path.exists(ism_cont):
        print("  [ERROR] Cloudy output still missing after run, skipping")
        return

    orion = _read_cloudy_cont(orion_cont)
    ism   = _read_cloudy_cont(ism_cont)

    # Frequency (Ryd) -> wavelength (micron)
    RYD_TO_CM = 6.6261e-27 * 2.998e10 / (13.6057 * 1.6022e-12)
    orion_lam = RYD_TO_CM / orion['nu_ryd'] * 1e4   # micron
    ism_lam   = RYD_TO_CM / ism['nu_ryd'] * 1e4

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 0.7, SINGLE_COL * 0.7 * 0.85),
                           constrained_layout=True)

    # Cloudy .cont columns are already nu*L_nu (= 4pi r^2 nu F_nu).
    # Plot directly without extra nu multiplication.
    ax.loglog(orion_lam, orion['total'],
              label=r'Orion grains ($r_\mathrm{s/l}=0.1$)')
    ax.loglog(ism_lam, ism['total'],
              label=r'ISM grains ($r_\mathrm{s/l}=0.4$)')

    # Incident spectrum
    ax.loglog(orion_lam, orion['incident'],
              ls=':', color='gray', lw=0.7, label='Incident')

    ax.set_xlabel(r'$\lambda$ ($\mu$m)')
    ax.set_ylabel(r'$\nu L_\nu$ (erg s$^{-1}$)')
    ax.set_xlim(0.05, 1000)
    ax.set_ylim(1e35, 1e42)
    ax.legend(loc='lower left')

    save_figure(fig, 'grain_sed_comparison')


def generate_fig_grain_distribution(plot_only=False):
    """
    Appendix figure: grain mass distribution dm/da for different
    small-to-large grain mass ratios.
    """
    print("\n" + "="*70)
    print("APPENDIX: Grain size distribution")
    print("="*70)

    from toddlers.cloudy_grains_generator import CloudyGrainsGenerator

    generator = CloudyGrainsGenerator("cloudy.exe", "")
    n_cutoff = 3
    a_values = np.logspace(-3, 0, 1000)  # 0.001 to 1 micron

    ratios = [0.01, 0.10, 0.20, 0.40]
    labels = [
        r'0.01',
        r'0.10 (def.)',
        r'0.20',
        r'0.40 (ISM)',
    ]

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 0.7, SINGLE_COL * 0.7 * 0.85),
                           constrained_layout=True)

    linestyles = ['-', '--', '-.', ':']
    for ratio, label, ls in zip(ratios, labels, linestyles):
        dist = generator.create_modified_orion_distribution(ratio)
        dm_da = np.array([
            generator.power_law(a) * generator.truncation(a, dist['sigma_l'], n_cutoff, dist['a_l'])
            for a in a_values
        ])
        ax.loglog(a_values, dm_da, ls=ls, label=label)

    ax.axvline(0.03, ls=':', color='gray', lw=0.7, alpha=0.7)
    ax.text(0.035, 1e-2, r'$a_L$', fontsize=7, color='gray')

    ax.set_xlabel(r'Grain size $a$ ($\mu$m)')
    ax.set_ylabel(r'$dm/da$ (arb. units)')
    ax.set_xlim(1e-3, 1)
    ax.set_ylim(1e-8, 5e2)
    ax.legend(loc='lower right')

    save_figure(fig, 'grain_size_distribution')
    print(f"  [OK] Saved grain_size_distribution.pdf")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate all publication figures for TODDLERS 2.0')
    parser.add_argument('--fig', nargs='+', type=str, default=None,
                        help='Generate specific figures (e.g. 3 5 7 meth)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip simulations, replot from cached output')
    args = parser.parse_args()

    total_start = time.time()

    # Figure dispatch
    figure_map = {
        '3': ('Fig 3: Profile/SF comparison', generate_fig3),
        '4': ('Fig 4: Dynamic density', generate_fig4),
        '5': ('Fig 5: IMF comparison', generate_fig5),
        '6': ('Fig 6: Custom feedback', generate_fig6),
        '7': ('Fig 7: Stochastic', generate_fig7),
        '8': ('Fig 8: BPT/DIG', generate_fig8),
        'meth': ('Methodology figures', None),
        'grain': ('Appendix: Grain SED comparison', generate_fig_grain_comparison),
        'graindist': ('Appendix: Grain size distribution', generate_fig_grain_distribution),
    }

    if args.fig is None:
        targets = list(figure_map.keys())
    else:
        targets = args.fig

    for target in targets:
        if target == 'meth':
            generate_fig_density_profiles()
            generate_fig_ram_force()
        elif target in figure_map:
            _, func = figure_map[target]
            func(plot_only=args.plot_only)
        else:
            print(f"  [WARN] Unknown figure: {target}")

    elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"All done in {elapsed:.1f} s")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
