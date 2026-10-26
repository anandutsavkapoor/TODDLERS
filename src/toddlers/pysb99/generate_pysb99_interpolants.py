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
from typing import Dict, List, Optional, Tuple
import warnings

# Add pysb99 to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), 'pysb99'))

from .pysb99_core import (
    StellarPopulationConfig,
    run_population_synthesis,
    StellarPopulationResults,
    SN_ENERGY_ERG, SN_MASS_MIN, SN_MASS_MAX, SN_REMNANT_MASS,
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
                 run_speed_mode: str = 'DEFAULT',
                 include_supernovae: bool = True):
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
        include_supernovae : bool
            If True (default), L_mech and F_ram include core-collapse supernovae
            (see pysb99_core._attach_supernovae); False gives the stellar winds alone.
        """
        self.include_supernovae = include_supernovae
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
            auto_extend_imf=auto_extend_imf,
            include_supernovae=self.include_supernovae
        )

        # Set IMF parameters if provided
        if imf_exponents is not None:
            config.imf_exponents = imf_exponents
        if imf_mass_limits is not None:
            config.imf_mass_limits = imf_mass_limits
        
        # Run synthesis
        results = run_population_synthesis(config)
        
        # Store results (and the configuration as run: auto_extend_imf may have raised the
        # IMF upper limit, which the table's IMF definition must record)
        self.results_dict[metallicity] = results
        self.configs = getattr(self, 'configs', {})
        self.configs[metallicity] = config
        
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
    
    def table_quantities(self):
        """Linear per-1e6-Msun arrays (n_Z, n_t) of the unified table, the population mass of
        each metallicity, and the L_LyW zero-node count.

        Normalization: per 1e6 Msun of the requested IMF mass (IMF populations) or of the
        population's own mass (custom populations). L_ion is the < 912 A integral L_HI: the
        pySB99 bands are nested (pysb99_core: each band integrates wave <= its edge), so
        adding the He bands would count He-ionizing photons twice or three times.
        Mechanical feedback is stored as wind and SN channels.
        """
        if not self.results_dict:
            raise ValueError("No results available. Run simulations first.")
        keys = ['L_bol', 'L_ion', 'Q_HI', 'Q_HeI', 'Q_HeII', 'L_LyW', 'L_mech_wind', 'L_mech_SN',
                'F_ram_wind', 'F_ram_SN']
        q = {k: [] for k in keys}
        masses, lyw_zero = [], 0
        for metallicity in self.metallicities:
            r = self.results_dict[metallicity]
            initial_masses = np.array(r.stellar_masses[0])
            actual_mass = float(np.sum(r.number_of_stars * initial_masses))
            if getattr(self, '_using_custom_distribution', False):
                total_mass = actual_mass
            else:
                total_mass = float(self._requested_total_mass)
            masses.append(total_mass)
            lognorm = np.log10(total_mass) - 6.0
            per = 1e6 / total_mass
            q['L_bol'].append(10 ** (r.bolometric_luminosity - lognorm))
            q['L_ion'].append(10 ** (r.hi_ionizing_luminosity - lognorm))
            q['Q_HI'].append(10 ** (r.hi_ionizing_flux - lognorm))
            q['Q_HeI'].append(10 ** (r.hei_ionizing_flux - lognorm))
            q['Q_HeII'].append(10 ** (r.heii_ionizing_flux - lognorm))
            q['L_mech_wind'].append(10 ** (r.wind_power - lognorm))
            q['F_ram_wind'].append(10 ** (r.wind_momentum - lognorm))
            n_t = len(r.wind_power)
            sn_p = np.zeros(n_t) if r.sn_power is None else np.asarray(r.sn_power, float)
            sn_m = np.zeros(n_t) if r.sn_momentum is None else np.asarray(r.sn_momentum, float)
            q['L_mech_SN'].append(sn_p * per)
            q['F_ram_SN'].append(sn_m * per)
            band = np.logical_and(r.wavelength_grid >= 995., r.wavelength_grid <= 1005.)
            if not np.any(band):
                raise ValueError(f"Lyman-Werner band 995-1005 A not in the wavelength grid ({metallicity})")
            # flux_spectra is linear, stored as L_lambda / 1e20 (x 1e20 gives erg/s/A)
            lyw = np.mean(r.flux_spectra[:, band], axis=1) * per * 1e20
            lyw_zero += int(np.sum(lyw == 0))
            q['L_LyW'].append(lyw)
        return {k: np.array(v) for k, v in q.items()}, masses, lyw_zero

    def imf_record(self, population_config):
        """(imf_definition, normalization) of the population as run (the checked IMF record)."""
        if population_config.get('population_type') == 'custom':
            # the requested masses map to the nearest track of each metallicity's grid, so the
            # stars (and the population mass) are recorded per metallicity
            stars, Ms = {}, {}
            for metallicity in self.metallicities:
                r = self.results_dict[metallicity]
                m0 = np.array(r.stellar_masses[0], dtype=float)
                n = np.asarray(r.number_of_stars, dtype=float)
                stars[metallicity] = {repr(float(m)): float(k) for m, k in zip(m0, n) if k > 0}
                Ms[metallicity] = float(np.sum(n * m0))
            d = {'kind': 'custom_population', 'star_numbers_msun_per_Z': stars,
                 'requested': population_config.get('custom_star_numbers')}
            n = {'rule': 'per 1e6 Msun of the custom population (grid stars as run, per metallicity)',
                 'per_mass_msun': 1e6, 'population_mass_msun': None, 'population_mass_msun_per_Z': Ms,
                 'm_low_msun': None, 'm_up_msun': None}
            return d, n
        from ..feedback_tables import broken_power_law
        cfg = self.configs[self.metallicities[0]]
        exps, lims = list(cfg.imf_exponents), list(cfg.imf_mass_limits)
        for m in self.metallicities[1:]:
            c = self.configs[m]
            if list(c.imf_exponents) != exps or list(c.imf_mass_limits) != lims:
                raise ValueError("IMF as run differs between metallicities (auto_extend_imf?)")
        M = float(self._requested_total_mass)
        d = broken_power_law(exps, lims, M)
        d['requested_mass_limits_msun'] = list(population_config.get('imf_mass_limits') or lims)
        d['auto_extend_imf'] = population_config.get('auto_extend_imf')
        n = {'rule': 'per 1e6 Msun of the full IMF', 'per_mass_msun': 1e6, 'population_mass_msun': M,
             'm_low_msun': lims[0], 'm_up_msun': lims[-1]}
        return d, n

    def build_table(self, imf_name='custom', population_config=None):
        """The unified feedback table (toddlers.feedback_tables.FeedbackTable) of the runs."""
        from ..feedback_tables import build_feedback_table
        cfg = population_config or _population_config(
            self, imf_name, population_type='custom' if getattr(self, '_using_custom_distribution', False) else 'imf',
            imf_exponents=getattr(self.configs[self.metallicities[0]], 'imf_exponents', None),
            imf_mass_limits=getattr(self.configs[self.metallicities[0]], 'imf_mass_limits', None),
            time_start_myr=self.time_start_myr, time_end_myr=self.time_end_myr,
            time_step_myr=self.time_step_myr)
        q, masses, lyw_zero = self.table_quantities()
        imf_def, norm = self.imf_record(cfg)
        meta = {
            'template': 'pySB99', 'imf': imf_name, 'star_type': None,
            'normalization': norm, 'imf_definition': imf_def,
            'source': {'code': 'pySB99 (toddlers.pysb99)', 'generator_config': cfg,
                       'population_mass_msun_per_Z': dict(zip(self.metallicities, masses))},
            'conventions': {
                'L_ion': 'hi_ionizing_luminosity: the < 912 A integral (nested bands)',
                'Q_HI_Q_HeI_Q_HeII': 'nested: photons below 912, 504, 228 A',
                'L_LyW': 'mean of flux_spectra over its nodes in 995-1005 A (pySB99 grid: 2 nodes)',
                'L_LyW_zero_nodes': lyw_zero,
                'mechanical': 'wind channel from the wind power / momentum, SN channel from the '
                              'per-bin SN power / momentum (pysb99_core._attach_supernovae)',
                **supernova_metadata(self.include_supernovae),
                'sn_time_deposition_mode': self.sn_time_deposition_mode(),
            },
        }
        return build_feedback_table(self.time_grid, self.Z_values, q, meta)

    def sn_time_deposition_mode(self):
        """'point', 'spread' or 'none': how the runs deposited their supernovae in time
        (pysb99_core._attach_supernovae); StellarFeedback reads a 'point' table's SN channel
        linearly in t."""
        if not self.include_supernovae:
            return 'none'
        by_z = {z: r.supernova_events['time_deposition'] for z, r in self.results_dict.items()
                 if getattr(r, 'supernova_events', None)}
        modes = set(by_z.values())
        if not modes:
            missing = sorted(z for z in self.results_dict if z not in by_z)
            raise ValueError(f"include_supernovae is True but no run carries its supernova "
                             f"record (supernova_events, with 'time_deposition'); missing at "
                             f"Z = {missing}. Rerun the populations with supernovae.")
        if len(modes) != 1:
            raise ValueError(f"supernova time deposition differs between metallicities: {by_z}")
        return modes.pop()

    def save_interpolants(self,
                         output_dir: str = './database',
                         imf_name: str = 'custom',
                         overwrite: bool = False,
                         population_config: Optional[Dict] = None):
        """
        Write the unified feedback table ``pySB99_<imf_name>[_rot]_feedback.h5``
        (toddlers.feedback_tables); the generator configuration the Cloudy stage needs to
        rerun the population is stored in its ``source`` attribute (generator_config).

        Returns
        -------
        FeedbackTable
            the table written (``.path``, ``.md5``).
        """
        from ..feedback_tables import write_table_file, table_filename
        name = imf_name + ('_rot' if self.rotation else '')
        path = os.path.join(output_dir, table_filename('pySB99', name))
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"File {path} already exists. Set overwrite=True to replace.")
        table = self.build_table(name, population_config)
        write_table_file(table, path)
        print(f"Saved feedback table to {path}")
        return table


# Version of the pySB99 SN treatment recorded in product metadata (see supernova_metadata).
PYSB99_SN_VERSION = 2


def supernova_metadata(include_supernovae: bool) -> Dict:
    """Metadata keys describing the SN treatment of a pySB99 product.

    ``include_supernovae`` records whether L_mech and F_ram include the SN channel.
    ``pysb99_sn_version`` (PYSB99_SN_VERSION) names the SN treatment: the SN death time and
    final mass are interpolated between the Geneva tracks by monotone PCHIP in log-log.
    """
    md = {'include_supernovae': bool(include_supernovae), 'pysb99_sn_version': PYSB99_SN_VERSION}
    if include_supernovae:
        md.update({
            'sn_death_time_interpolation': 'pchip_loglog',
            'sn_energy_erg': SN_ENERGY_ERG,
            'sn_mass_min_msun': SN_MASS_MIN,
            'sn_mass_max_msun': SN_MASS_MAX,
            'sn_above_max': 'direct_collapse',
            'sn_remnant_mass_msun': SN_REMNANT_MASS,
            'sn_ejecta_convention': 'M_final_minus_M_rem',
            'sn_momentum_convention': 'sqrt(2 E_SN M_ej) per SN, summed',
            'sn_time_deposition': 'spread_over_row_death_interval (IMF) | point (custom)',
            'sn_time_binning': 'centred',
        })
    return md


def product_includes_supernovae(metadata: Dict) -> bool:
    """Whether a pySB99 product's L_mech / F_ram include SNe (its ``include_supernovae`` key)."""
    return bool(metadata.get('include_supernovae', False))


def _population_config(generator, imf_name, *, population_type, custom_star_numbers=None,
                       imf_exponents=None, imf_mass_limits=None, time_start_myr=0.01,
                       time_end_myr=31.0, time_step_myr=0.1, auto_extend_imf=None):
    """The generator configuration of a pySB99 table (stored in the table's ``source``
    attribute as ``generator_config``).

    SpectralTableGenerator reads it to rerun the population for the Cloudy star table, so
    the deterministic pySB99 -> Cloudy path works directly after table generation.
    ``auto_extend_imf`` and ``run_speed_mode`` are recorded so the Cloudy table is built from
    the same population as the feedback table. It matters for custom populations too: the
    IMF upper limit selects the track set that is loaded. No timestamp is recorded (a rebuild
    on the same machine gives the same file md5).
    """
    if population_type == 'custom' and auto_extend_imf is None:
        auto_extend_imf = True  # how custom-population interpolants are generated
    metadata = {
        'population_name': imf_name,
        'metallicities': list(generator.metallicities),
        'Z_values': list(generator.Z_values),
        'spectral_library': generator.spectral_library,
        'rotation': generator.rotation,
        'time_start_myr': time_start_myr,
        'time_end_myr': time_end_myr,
        'time_step_myr': time_step_myr,
        'run_speed_mode': generator.run_speed_mode,
        'auto_extend_imf': auto_extend_imf,
        'population_type': population_type,
        **supernova_metadata(getattr(generator, 'include_supernovae', False)),
    }
    if population_type == 'custom':
        metadata['custom_star_numbers'] = {str(m): n for m, n in custom_star_numbers.items()}
    else:
        metadata['imf_exponents'] = [float(x) for x in imf_exponents]
        metadata['imf_mass_limits'] = [float(x) for x in imf_mass_limits]
    return metadata


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
        verbose: bool = True,
        include_supernovae: bool = True):
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
    include_supernovae : bool
        If True (default), L_mech and F_ram include core-collapse supernovae.

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
        time_step_myr=time_step_myr,
        include_supernovae=include_supernovae
    )

    generator.run_all_metallicities(
        total_mass=1e6,
        imf_exponents=imf_exponents,
        imf_mass_limits=imf_mass_limits,
        auto_extend_imf=auto_extend_imf,
        verbose=verbose
    )
    
    config = _population_config(generator, imf_name, population_type='imf',
                                imf_exponents=imf_exponents, imf_mass_limits=imf_mass_limits,
                                time_start_myr=time_start_myr, time_end_myr=time_end_myr,
                                time_step_myr=time_step_myr, auto_extend_imf=auto_extend_imf)
    generator.save_interpolants(output_dir=output_dir, imf_name=imf_name, overwrite=True,
                                population_config=config)

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
        verbose: bool = True,
        include_supernovae: bool = True):
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
    include_supernovae : bool
        If True (default), L_mech and F_ram include core-collapse supernovae.
        
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
        time_step_myr=time_step_myr,
        include_supernovae=include_supernovae
    )
    
    # auto_extend_imf=True (the default custom populations have always used): the IMF
    # upper limit still selects which track set (with or without VMS) is loaded.
    generator.run_all_metallicities(
        custom_star_numbers=custom_star_numbers,
        auto_extend_imf=True,
        verbose=verbose
    )

    config = _population_config(generator, imf_name, population_type='custom',
                                custom_star_numbers=custom_star_numbers,
                                time_start_myr=time_start_myr, time_end_myr=time_end_myr,
                                time_step_myr=time_step_myr, auto_extend_imf=True)
    generator.save_interpolants(output_dir=output_dir, imf_name=imf_name, overwrite=True,
                                population_config=config)

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
    parser.add_argument('--imf-limits', type=float, nargs='+', default=[0.1, 0.5, 100.0],
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
    name = f"{args.imf_name}_rot" if args.rotation else args.imf_name
    print(f"  1. Put pySB99_{name}_feedback.h5 in the TODDLERS data directory "
          "($TODDLERS_DATA, default src/toddlers/database/)")
    print(f"  2. Evolution(..., template='pySB99', imf='{name}', star_type='sin')")