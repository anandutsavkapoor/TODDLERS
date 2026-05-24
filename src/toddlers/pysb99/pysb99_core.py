#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pySB99 Core Module
==================

A Python stellar population synthesis tool for modeling stellar evolution and spectra.
This module closely follows the original pySB99.py implementation while providing
improved code organization.

Author: Anand Utsav KAPOOR
Refactored from https://github.com/CalumHawcroft/Starburst/tree/main and 
described in https://arxiv.org/html/2505.24841v1
"""

import numpy as np
import pandas as pd
import time
import os
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
import warnings

def safe_log10(array, min_val=1e-50):
    """Safe logarithm that avoids log(0) warnings."""
    safe_array = np.maximum(np.asarray(array), min_val)
    return np.log10(safe_array)

@dataclass
class StellarPopulationConfig:
    """
    Configuration for stellar population synthesis calculations.
    
    Notes
    -----
    When using custom_star_numbers, be aware of the available mass grid for your
    chosen metallicity. The provided masses will be mapped to the closest available
    grid points. Available mass ranges depend on the metallicity:
    
    - MWC: Masses from 0.8 to 300 M☉
    - MW: Masses from 0.8 to 500 M☉ (non-rotating) or 0.8 to 300 M☉ (rotating)
    - LMC: Masses from 0.8 to 300 M☉
    - SMC: Masses from 0.8 to 120 M☉
    - IZw18: Masses from 1.7 to 120 M☉
    - Z0: Masses from 1.7 to 300 M☉ (non-rotating) or 1.7 to 120 M☉ (rotating)
    
    You can obtain the exact grid by inspecting the StellarDataLoader.mass_grid property.
    """
    # Basic population parameters
    total_mass: float = 1.0e6  # Total stellar mass in solar masses
    imf_exponents: List[float] = None  # IMF power law exponents
    imf_mass_limits: Tuple[float, ...] = (0.1, 0.5, 120.0)  # Default mass limits for IMF segments
    
    # extend to VMS regime, else default to imf_mass_limits
    auto_extend_imf: bool = True

    # Custom star number configuration
    custom_star_numbers: Optional[Dict[float, int]] = None  # Maps stellar mass to number of stars

    # Metallicity and spectral library options
    metallicity: str = 'MW'  # 'MWC', 'MW', 'LMC', 'SMC', 'IZw18', 'Z0'
    spectral_library: str = 'FW'  # 'FW' (Fastwind) or 'WM' (WMbasic)
    rotation: bool = False  # Include stellar rotation
    
    # Computation parameters
    run_speed_mode: str = 'FAST'  # 'FAST', 'DEFAULT', 'HIGH_RES'
    use_powr: bool = False  # Use PoWR OB grid (not yet implemented)
    
    # Time grid
    time_start: float = 0.01e6  # Start time in years
    time_end: float = 50e6    # End time in years
    time_step: float = 0.1e6  # Time step in years
    
    def __post_init__(self):
        if self.imf_exponents is None:
            self.imf_exponents = [1.3, 2.3]


@dataclass
class StellarPopulationResults:
    """Results from stellar population synthesis calculations."""
    
    # Time grid
    times: np.ndarray
    
    # Spectral energy distributions
    wavelength_grid: np.ndarray
    flux_spectra: np.ndarray  # Shape: (n_times, n_wavelengths)
    flux_spectra_with_nebular: np.ndarray
    
    # Ionizing fluxes
    bolometric_luminosity: np.ndarray
    hi_ionizing_flux: np.ndarray  # HI ionizing photon rate (photons/s)
    hei_ionizing_flux: np.ndarray  # HeI ionizing photon rate (photons/s)
    heii_ionizing_flux: np.ndarray  # HeII ionizing photon rate (photons/s)
    hi_ionizing_luminosity: np.ndarray  # HI ionizing energy flux (erg/s)
    hei_ionizing_luminosity: np.ndarray  # HeI ionizing energy flux (erg/s)
    heii_ionizing_luminosity: np.ndarray  # HeII ionizing energy flux (erg/s)
    
    # Wind properties
    wind_power: np.ndarray
    wind_momentum: np.ndarray
    
    # UV slopes and equivalent widths
    uv_slope_beta: np.ndarray
    ha_equivalent_width: np.ndarray
    hb_equivalent_width: np.ndarray
    pb_equivalent_width: np.ndarray
    bg_equivalent_width: np.ndarray
    
    # Photometric properties
    v_magnitude: np.ndarray
    u_magnitude: np.ndarray
    i_magnitude: np.ndarray
    b_magnitude: np.ndarray
    abs_v_magnitude: np.ndarray
    
    # Stellar population properties
    stellar_masses: List[np.ndarray]
    stellar_temperatures: List[np.ndarray]
    stellar_luminosities: List[np.ndarray]
    number_of_stars: np.ndarray


def f(x, A, B):
    """Linear function for curve fitting: y = Ax + B"""
    return A*x + B


class StellarDataLoader:
    """Handles loading of stellar evolution tracks and spectral grids."""
    
    def __init__(self, config: StellarPopulationConfig):
        self.config = config
        self.file_path = self._get_file_path()
        self.mass_grid = self._get_mass_grid()
        self.minimum_wr_mass = self._get_minimum_wr_mass()
        
    def _get_file_path(self) -> str:
        """Get the file path based on metallicity."""
        # Get directory where THIS file is located
        module_dir = os.path.dirname(os.path.abspath(__file__))

        metallicity_paths = {
            'MWC': 'pySB99_files/Z020_pySB99_files/',
            'MW': 'pySB99_files/Z014_pySB99_files/',
            'LMC': 'pySB99_files/Z006_pySB99_files/',
            'SMC': 'pySB99_files/Z002_pySB99_files/',
            'IZw18': 'pySB99_files/Z0004_pySB99_files/',
            'Z0': 'pySB99_files/Z00_pySB99_files/'
        }

        # Return absolute path
        relative_path = metallicity_paths[self.config.metallicity]
        return os.path.join(module_dir, relative_path)
    
    def _get_mass_grid(self) -> np.ndarray:
        """Get the mass grid based on metallicity and rotation."""
        if self.config.metallicity == 'MWC':
            mass_grid = [300, 200., 150., 120., 85., 60., 40., 32., 25., 20., 15., 12., 9., 7., 5., 4., 3., 2.5, 2., 1.7, 1.5, 1.35, 1.25, 1.1, 1., 0.9, 0.8]
            if self.config.imf_mass_limits[-1] > 300.:
                raise ValueError('Tracks do not exist at Z=0.20 above 300Msol')
        
        elif self.config.metallicity == 'MW':
            if self.config.rotation:
                mass_grid = [300., 250., 180., 120., 85., 60., 40., 32., 25., 20., 15., 12., 9., 7., 5., 4., 3., 2.5, 2., 1.7, 1.5, 1.35, 1.25, 1.1, 1., 0.9, 0.8]
                if self.config.imf_mass_limits[-1] > 300.:
                    raise ValueError('Tracks with rotation do not exist at Z=0.014 above 300Msol')
            else:
                mass_grid = [500., 300, 200., 150., 120., 85., 60., 50., 40., 32., 25., 23., 22., 20., 17., 15., 14., 12., 11.75, 11.5, 10., 9., 8., 7., 5., 4., 3., 2.5, 2., 1.7, 1.5, 1.35, 1.25, 1.1, 1., 0.9, 0.8]
                if self.config.imf_mass_limits[-1] > 500.:
                    raise ValueError('Tracks do not exist at Z=0.014 above 500Msol')
        
        elif self.config.metallicity == 'LMC':
            mass_grid = [300, 250., 180., 120., 85., 60., 40., 32., 25., 20., 15., 12., 9., 7., 5., 4., 3., 2.5, 2., 1.7, 1.5, 1.35, 1.25, 1.1, 1., 0.9, 0.8]
            if self.config.imf_mass_limits[-1] > 300.:
                raise ValueError('Tracks do not exist at Z=0.006 above 300Msol')
        
        elif self.config.metallicity == 'SMC':
            mass_grid = [120., 85., 60., 40., 32., 25., 20., 15., 12., 9., 7., 5., 4., 3., 2.5, 2., 1.7, 1.5, 1.35, 1.25, 1.1, 1., 0.9, 0.8]
            if self.config.imf_mass_limits[-1] > 120.:
                raise ValueError('Tracks do not exist at Z=0.002 above 120Msol')
        
        elif self.config.metallicity == 'IZw18':
            mass_grid = [120., 85., 60., 40., 32., 25., 20., 15., 12., 9., 7., 5., 4., 3., 2.5, 2., 1.7]
            if self.config.imf_mass_limits[-1] > 120.:
                raise ValueError('Tracks do not exist at 0.0004 above 120Msol')
        
        elif self.config.metallicity == 'Z0':
            if self.config.rotation:
                mass_grid = [120., 85., 60., 40., 30., 20., 15., 12., 9., 7., 5., 4., 3., 2.5, 2., 1.7]
                if self.config.imf_mass_limits[-1] > 120.:
                    raise ValueError('Tracks with rotation do not exist at Z=0.0 above 120Msol')
            else:
                mass_grid = [300., 250., 120., 85., 60., 40., 30., 20., 15., 12., 9., 7., 5., 4., 3., 2.5, 2., 1.7]
                if self.config.imf_mass_limits[-1] > 300.:
                    raise ValueError('Tracks do not exist at Z=0.0 above 300Msol')
        
        return np.array(mass_grid)
    
    def _get_minimum_wr_mass(self) -> float:
        """Get minimum mass for Wolf-Rayet stars based on metallicity and rotation."""
        if self.config.metallicity == 'MWC':
            return 20.0
        elif self.config.metallicity == 'MW':
            return 20.0 if self.config.rotation else 25.0
        elif self.config.metallicity == 'LMC':
            return 25.0
        elif self.config.metallicity == 'SMC' or self.config.metallicity == 'IZw18' or self.config.metallicity == 'Z0':
            return 84.0
        return 20.0  # Default fallback
    
    def load_evolution_tracks(self) -> np.ndarray:
        """Load stellar evolution tracks."""
        rotation_suffix = 'v40' if self.config.rotation else 'v00'
        z_string = self._get_z_string()
        
        if self.config.metallicity in ['MWC', 'MW', 'LMC']:
            if self.config.imf_mass_limits[-1] > 120.0:
                filename = f'Z{z_string}{rotation_suffix}_VMS_tracks.npy'
                print(f"Loading VMS tracks from {filename}")
            else:
                filename = f'Z{z_string}{rotation_suffix}_VMS_tracks.npy'
                print(f"Loading VMS tracks from {filename}")
        else:
            filename = f'Z{z_string}{rotation_suffix}_tracks.npy'
        
        return np.load(self.file_path + filename)
    
    def _get_z_string(self) -> str:
        """Get metallicity string for filenames."""
        z_strings = {
            'MWC': '020',
            'MW': '014',
            'LMC': '006',
            'SMC': '002',
            'IZw18': '0004',
            'Z0': '00'
        }
        return z_strings[self.config.metallicity]
    
    def load_spectral_grid(self) -> Tuple[List[str], List[str]]:
        """Load spectral grid file."""
        if self.config.spectral_library == 'WM':
            if self.config.metallicity == 'MWC':
                filename = 'galaxy/lejeune/WMbasic_OB_Z020_test.dat'
            elif self.config.metallicity == 'MW':
                filename = 'WMbasic_OB_Z020_test.dat'
            elif self.config.metallicity == 'LMC':
                filename = 'galaxy/lejeune/WMbasic_OB_Z008_tst.dat'
            elif self.config.metallicity == 'SMC':
                filename = 'galaxy/lejeune/WMbasic_OB_Z004_tst.dat'
            elif self.config.metallicity == 'IZw18' or self.config.metallicity == 'Z0':
                filename = 'galaxy/lejeune/WMbasic_OB_Z001_tst.dat'
        else:  # FW
            if self.config.metallicity == 'MWC':
                filename = 'FW_SB_grid_Z020_VMS.txt'
            elif self.config.metallicity == 'MW':
                filename = 'FW_SB_grid_Z014.txt'
                if self.config.imf_mass_limits[-1] > 120.0:
                    filename = 'FW_SB_grid_Z014_VMS.txt'
            elif self.config.metallicity == 'LMC':
                filename = 'FW_SB_grid_Z006.txt'
                if self.config.imf_mass_limits[-1] > 120.0:
                    filename = 'FW_SB_grid_Z006_VMS.txt'
            elif self.config.metallicity == 'SMC':
                filename = 'FW_SB_grid_Z002.txt'
            elif self.config.metallicity == 'IZw18':
                filename = 'FW_SB_grid_Z0004.txt'
            elif self.config.metallicity == 'Z0':
                filename = 'FW_SB_grid_Z0.txt'
                if self.config.imf_mass_limits[-1] > 120.0:
                    filename = 'FW_SB_grid_Z0_VMS.txt'
        
        return self._read_spectra_grid(self.file_path + filename)
    
    def _read_spectra_grid(self, spec_file: str) -> Tuple[List[str], List[str]]:
        """Read spectral grid from file."""
        df = pd.read_csv(spec_file)
        ind_spec = [i for i in range(len(df)) if df['start'][i] == ' CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC']
        
        spec_params = []
        spectra = []
        
        for i in range(len(ind_spec)-1):
            spec_param = df['start'][ind_spec[i]+1]
            spec_params.append(spec_param)
            
            spec_data = df['start'][ind_spec[i]+2 : ind_spec[i+1]]
            spectra.append(spec_data)
        
        return spec_params, spectra
    
    def load_auxiliary_data(self) -> Dict[str, np.ndarray]:
        """Load auxiliary spectral data (low-mass, high-res, WR spectra)."""
        # Determine metallicity suffix for filenames
        if self.config.metallicity in ['MWC', 'MW']:
            metallicity_suffix = 'p00'
            wr_metallicity_suffix = 'Z020'
        elif self.config.metallicity == 'LMC':
            metallicity_suffix = 'm04'
            wr_metallicity_suffix = 'Z008'
        elif self.config.metallicity == 'SMC':
            metallicity_suffix = 'm07'
            wr_metallicity_suffix = 'Z004'
        elif self.config.metallicity in ['IZw18', 'Z0']:
            metallicity_suffix = 'm13'
            wr_metallicity_suffix = 'Z001'
        
        data = {}
        
        # Low-mass stellar spectra
        data['lowmass_params'] = np.load(self.file_path + f'spec_params_lowmass{metallicity_suffix}.npy')
        data['lowmass_flux'] = np.load(self.file_path + f'lcb97_{metallicity_suffix}_reform.npy')
        
        # High-resolution spectra
        data['hires_params'] = np.load(self.file_path + f'spec_params_ifa_line_{metallicity_suffix}.npy')
        data['hires_flux'] = np.load(self.file_path + f'ifa_line_{metallicity_suffix}_reform.npy')
        data['hires_cont_flux'] = np.load(self.file_path + f'ifa_cont_{metallicity_suffix}_reform.npy')
        
        # Wolf-Rayet spectra
        data['wn_spec_params'] = np.load(self.file_path + f'WN_spec_params_cmfgen_{wr_metallicity_suffix}.npy')
        data['wn_spectra'] = np.load(self.file_path + f'WN_spectra_cmfgen_{wr_metallicity_suffix}.npy', allow_pickle=True)
        data['wc_spec_params'] = np.load(self.file_path + f'WC_spec_params_cmfgen_{wr_metallicity_suffix}.npy')
        data['wc_spectra'] = np.load(self.file_path + f'WC_spectra_cmfgen_{wr_metallicity_suffix}.npy', allow_pickle=True)
        
        # PoWR WR spectra
        data['wn_spec_params_powr'] = np.load(self.file_path + f'WN_spec_params_powr_{wr_metallicity_suffix}.npy')
        data['wn_spectra_powr'] = np.load(self.file_path + f'WN_spectra_powr_{wr_metallicity_suffix}.npy', allow_pickle=True)
        data['wc_spec_params_powr'] = np.load(self.file_path + f'WC_spec_params_powr_{wr_metallicity_suffix}.npy')
        data['wc_spectra_powr'] = np.load(self.file_path + f'WC_spectra_powr_{wr_metallicity_suffix}.npy', allow_pickle=True)
        
        # Wavelength grid
        data['hires_wave_grid'] = np.load(self.file_path + 'hires_wave_grid.npy')
        
        return data


class StellarPopulationSynthesis:
    """Main class for stellar population synthesis calculations."""
    
    def __init__(self, config: StellarPopulationConfig):
        self.config = config
        self.M_total = config.total_mass  # Keep reference to total mass
        self.loader = StellarDataLoader(config)

        # --- Dynamic IMF handling ---
        if self.config.auto_extend_imf:
            grid_max_mass = float(np.max(self.loader.mass_grid))
            imf_limits = list(self.config.imf_mass_limits)

            if imf_limits[-1] < grid_max_mass:
                old_limit = imf_limits[-1]
                imf_limits[-1] = grid_max_mass
                self.config.imf_mass_limits = tuple(imf_limits)

                warnings.warn(
                    f"IMF upper mass limit extended from {old_limit:.1f} to "
                    f"{grid_max_mass:.1f} Msun (auto_extend_imf=True).",
                    UserWarning
                )
        # else: auto_extend_imf = False → leave IMF untouched
        
        # Load all required data
        print("Loading stellar evolution tracks...")
        self.evolution_tracks = self.loader.load_evolution_tracks()
        
        print("Loading spectral grids...")
        self.spec_params, self.spectra = self.loader.load_spectral_grid()
        
        print("Loading auxiliary data...")
        self.aux_data = self.loader.load_auxiliary_data()
        
        # Initialize data structures
        self.track_data = {}
        self.mass_ordered_tracks = {}
        self.grid_data = {}
        
        # Create time grid
        self.times = np.arange(self.config.time_start, self.config.time_end, self.config.time_step)
        self.times_fine = np.arange(1e-16, self.config.time_end, self.config.time_step)
        self.times_steps_SB99 = np.arange(0.01e6, 50e6, 0.1e6)
        
        # Process evolution tracks
        self._process_evolution_tracks()
        
        # Process spectral grids
        self._process_spectral_grids()
        
        print("Initialization complete.")
    
    def _process_evolution_tracks(self):
        """Process stellar evolution tracks."""
        # Parse tracks
        self._parse_evolution_tracks()
        
        # Interpolate between tracks
        self._interpolate_tracks()
    
    def _parse_evolution_tracks(self):
        """Parse evolution tracks into separate arrays."""
        evo_tracks = self.evolution_tracks
        mass_grid = self.loader.mass_grid
        split_factor = len(mass_grid)
        
        # Extract track components
        ids = evo_tracks[:, 0]
        ages = evo_tracks[:, 1]
        masses = evo_tracks[:, 2]
        luminosities = evo_tracks[:, 3]
        temperatures = evo_tracks[:, 4]
        h_abundances = evo_tracks[:, 5]
        he_abundances = evo_tracks[:, 6]
        c12_abundances = evo_tracks[:, 7]
        n14_abundances = evo_tracks[:, 8]
        o16_abundances = evo_tracks[:, 9]
        core_temperatures = evo_tracks[:, 10]
        mass_loss_rates = evo_tracks[:, 11]
        
        # Split tracks into individual sub-arrays
        self.track_data = {
            'ids': np.array_split(ids, split_factor),
            'ages': np.array_split(ages, split_factor),
            'masses': np.array_split(masses, split_factor),
            'luminosities': np.array_split(luminosities, split_factor),
            'temperatures': np.array_split(temperatures, split_factor),
            'h_abundances': np.array_split(h_abundances, split_factor),
            'he_abundances': np.array_split(he_abundances, split_factor),
            'c12_abundances': np.array_split(c12_abundances, split_factor),
            'n14_abundances': np.array_split(n14_abundances, split_factor),
            'o16_abundances': np.array_split(o16_abundances, split_factor),
            'core_temperatures': np.array_split(core_temperatures, split_factor),
            'mass_loss_rates': np.array_split(mass_loss_rates, split_factor)
        }
        
        # Set initial age for all tracks to 0.0
        for track in self.track_data['ages']:
            track[0] = 1.0e-3
        
        # Extend tracks with post-MS evolution
        for i in range(len(self.track_data['ages'])):
            self.track_data['ages'][i] = np.append(self.track_data['ages'][i], 
                                                np.array([self.track_data['ages'][i][-1] + 100., 
                                                        (self.track_data['ages'][i][-1] + 100.) * 10**6]))
            self.track_data['luminosities'][i] = np.append(self.track_data['luminosities'][i], np.array([-20., -20.]))
            self.track_data['temperatures'][i] = np.append(self.track_data['temperatures'][i], 
                                                        np.array([self.track_data['temperatures'][i][-1], 
                                                                self.track_data['temperatures'][i][-1]]))
            self.track_data['masses'][i] = np.append(self.track_data['masses'][i], 
                                                np.array([self.track_data['masses'][i][-1], 
                                                        self.track_data['masses'][i][-1]]))
            self.track_data['h_abundances'][i] = np.append(self.track_data['h_abundances'][i], 
                                                        np.array([self.track_data['h_abundances'][i][-1], 
                                                                self.track_data['h_abundances'][i][-1]]))
            self.track_data['he_abundances'][i] = np.append(self.track_data['he_abundances'][i], 
                                                        np.array([self.track_data['he_abundances'][i][-1], 
                                                                self.track_data['he_abundances'][i][-1]]))
            self.track_data['c12_abundances'][i] = np.append(self.track_data['c12_abundances'][i], 
                                                        np.array([self.track_data['c12_abundances'][i][-1], 
                                                                self.track_data['c12_abundances'][i][-1]]))
            self.track_data['n14_abundances'][i] = np.append(self.track_data['n14_abundances'][i], 
                                                        np.array([self.track_data['n14_abundances'][i][-1], 
                                                                self.track_data['n14_abundances'][i][-1]]))
            self.track_data['o16_abundances'][i] = np.append(self.track_data['o16_abundances'][i], 
                                                        np.array([self.track_data['o16_abundances'][i][-1], 
                                                                self.track_data['o16_abundances'][i][-1]]))
            self.track_data['core_temperatures'][i] = np.append(self.track_data['core_temperatures'][i], 
                                                            np.array([self.track_data['core_temperatures'][i][-1], 
                                                                    self.track_data['core_temperatures'][i][-1]]))
            self.track_data['mass_loss_rates'][i] = np.append(self.track_data['mass_loss_rates'][i], 
                                                        np.array([self.track_data['mass_loss_rates'][i][-1], 
                                                                self.track_data['mass_loss_rates'][i][-1]]))
        
        # Clean up mass loss rates (set unrealistic values to very low)
        for i in range(len(self.track_data['mass_loss_rates'])):
            for j in range(len(self.track_data['mass_loss_rates'][i])):
                if self.track_data['mass_loss_rates'][i][j] > -1:
                    self.track_data['mass_loss_rates'][i][j] = -1000
        
        # Transpose tracks to mass-increasing order
        for param in self.track_data:
            self.mass_ordered_tracks[param] = np.transpose(self.track_data[param])
    
    def _interpolate_tracks(self):
        """Interpolate between tracks to create a denser grid."""
        # Interpolate each parameter
        for param in ['ages', 'luminosities', 'temperatures', 'h_abundances', 'he_abundances',
                    'c12_abundances', 'n14_abundances', 'o16_abundances', 'core_temperatures', 'mass_loss_rates']:
            
            grid_masses, grid_params = self.interpolate_param(self.track_data[param], self.track_data['masses'])
            
            # Rearrange to get one array per mass
            rearranged_grid = self.rearrange_grid_array(grid_params)
            self.grid_data[param] = rearranged_grid
        
        # Use masses from the ages interpolation
        grid_masses, _ = self.interpolate_param(self.track_data['ages'], self.track_data['masses'])
        self.grid_data['masses'] = self.rearrange_grid_array(grid_masses)
    
    def interpolate_param(self, tracks_parameter, track_masses, run_speed_mode=None):
        """
        Interpolate parameters between tracks.
        This function follows the original implementation in pySB99.py
        """
        if run_speed_mode is None:
            run_speed_mode = self.config.run_speed_mode
            
        grid_masses_adjinterp_total = []
        grid_params_adjinterp_total = []
        
        for i in range(len(track_masses)-1):
            # Get upper and lower tracks
            track_mass_upper = np.flip(track_masses[i])
            track_param_upper = np.flip(tracks_parameter[i])
            track_mass_lower = np.flip(track_masses[i+1])
            track_param_lower = np.flip(tracks_parameter[i+1])

            initial_mass_upper = track_mass_upper[-1]
            initial_mass_lower = track_mass_lower[-1]
            
            # Determine interpolation resolution based on mass range and run speed
            if run_speed_mode == 'DEFAULT':
                if round(initial_mass_upper) > 7 and round(initial_mass_upper) < 35:
                    inter_track_sampling = (initial_mass_upper - initial_mass_lower + 1) * 15
                else:
                    inter_track_sampling = (initial_mass_upper - initial_mass_lower + 1)
            elif run_speed_mode == 'FAST':
                if round(initial_mass_upper) > 7 and round(initial_mass_upper) < 35:
                    inter_track_sampling = (initial_mass_upper - initial_mass_lower + 1) * 10
                else:
                    inter_track_sampling = (initial_mass_upper - initial_mass_lower + 1) * 5
            elif run_speed_mode == 'HIGH_RES':
                if round(initial_mass_upper) > 7 and round(initial_mass_upper) < 35:
                    inter_track_sampling = (initial_mass_upper - initial_mass_lower + 1) * 100
                else:
                    inter_track_sampling = (initial_mass_upper - initial_mass_lower + 1) * 50
            
            # Stack masses and parameters for adjacent tracks
            track_masses_adjinterp = np.column_stack((track_mass_lower, track_mass_upper))
            track_param_adjinterp = np.column_stack((track_param_lower, track_param_upper))
            
            grid_masses_adjinterp = []
            
            # Create mass grid points between tracks
            for j in range(len(track_mass_upper)):
                grid_mass_adjinterp = np.linspace(
                    track_mass_lower[j], track_mass_upper[j], round(inter_track_sampling)
                )
                grid_masses_adjinterp.append(grid_mass_adjinterp)
            
            # Interpolate parameters at each mass grid point
            grid_params_adjinterp = []
            for track_param, track_mass, grid_mass in zip(
                track_param_adjinterp, track_masses_adjinterp, grid_masses_adjinterp
            ):
                grid_params_adjinterp_fn = interp1d(
                    track_mass, track_param, kind='linear', fill_value='extrapolate'
                )

                grid_params_adjinterp_loop = grid_params_adjinterp_fn(grid_mass)
                grid_params_adjinterp.append(grid_params_adjinterp_loop)
            
            grid_masses_adjinterp_total.append(grid_masses_adjinterp)
            grid_params_adjinterp_total.append(grid_params_adjinterp)
        
        return grid_masses_adjinterp_total, grid_params_adjinterp_total
    
    def rearrange_grid_array(self, grid_array):
        """
        Rearrange the grid interpolation array to give one array per mass.
        This function follows the original implementation in pySB99.py
        """
        lengths = []
        for i in range(len(grid_array)):
            grid_sizes = len(grid_array[i][0])
            lengths.append(grid_sizes)
        
        total_grid_size = sum(lengths)
        
        rearranged = []
        for k in range(len(grid_array)):
            for j in range(len(grid_array[k][0])-1, -1, -1):
                for i in range(len(grid_array[k])):
                    rearranged.append(grid_array[k][i][j])
        
        rearranged = np.array_split(rearranged, total_grid_size)
        return rearranged
    
    def _process_spectral_grids(self):
        """Process spectral grids."""
        # Process O/B spectral grid
        self.ob_spectral_grid = self._reform_spec_grid(self.spectra, self.spec_params)
        
        # Get wavelength grid from first spectrum
        self.wavelength_grid = self.ob_spectral_grid['spectra'][0][:, 0]
        
        # Process low-mass stellar grid
        self.lowmass_spectral_grid = self._process_lowmass_grid()
        
        # Process Wolf-Rayet spectral grids
        self.wr_spectral_grids = self._process_wr_grids()
        
        # Process high-resolution spectra
        self.hires_spectral_grid = self._process_hires_grid()
    
    def _reform_spec_grid(self, spectra, spec_params):
        """
        Reform spectral grid.
        Equivalent to reform_spec_grid in pySB99.py
        """
        reformed_spec_grid = []
        for i in range(len(spectra)):
            reformatted_spec = self._reformat_spec(spectra[i])
            reformed_spec_grid.append(reformatted_spec)
        
        spec_params_id = []
        spec_params_teff = []
        spec_params_logl = []
        spec_params_logg = []
        
        for i in spec_params:
            spec_id, spec_teff, spec_logl, spec_logg = i.split()
            spec_id = float(spec_id)
            spec_params_id.append(spec_id)
            spec_teff = float(spec_teff)
            spec_params_teff.append(spec_teff)
            spec_logl = float(spec_logl)
            spec_params_logl.append(spec_logl)
            spec_logg = float(spec_logg)
            spec_params_logg.append(spec_logg)
        
        spec_params_teff = 10**np.array(spec_params_teff)
        spec_params_reform = np.column_stack((spec_params_id, spec_params_teff, spec_params_logl, spec_params_logg))
        
        integrated_spectra = self._integrate_spec_grid(reformed_spec_grid)
        
        return {
            'spectra': reformed_spec_grid,
            'params': spec_params_reform,
            'teffs': spec_params_teff,
            'logls': np.array(spec_params_logl),
            'loggs': np.array(spec_params_logg),
            'integrated': integrated_spectra
        }
    
    def _reformat_spec(self, spectrum, scale_factor=4.0*3.14142):
        """
        Reformat spectrum.
        Equivalent to reformat_spec in pySB99.py
        """
        wave_reform = []
        flux_reform = []
        
        for i in spectrum:
            try:
                spec_wavep, spec_fluxp = i.split()
                wavep_app = float(spec_wavep)
                wave_reform.append(wavep_app)
                fluxp_app = float(spec_fluxp) * scale_factor
                flux_reform.append(fluxp_app)
            except:
                continue
        
        flux_reform = np.array(flux_reform)
        spec_reform = np.column_stack((wave_reform, flux_reform))
        return spec_reform
    
    def _reformat_spec_WR(self, spectrum):
        """
        Reformat Wolf-Rayet spectrum.
        Equivalent to reformat_spec_WR in pySB99.py
        """
        wave_reform = []
        flux_reform = []
        
        for i in spectrum:
            try:
                spec_wavep, spec_fluxp = i.split()
                wavep_app = float(spec_wavep)
                wave_reform.append(wavep_app)
                fluxp_app = float(spec_fluxp)
                flux_reform.append(fluxp_app)
            except:
                continue
        
        flux_reform = np.array(flux_reform)
        spec_reform = np.column_stack((wave_reform, flux_reform))
        return spec_reform
    
    def _process_wr_grids(self):
        """Process Wolf-Rayet spectral grids."""
        wr_grids = {}
        
        # Process WN grid
        wn_reformed_spec_grid, wn_spec_params_reform, wn_spec_params_teff = self._reform_spec_grid_WR(
            self.aux_data['wn_spectra'], self.aux_data['wn_spec_params']
        )
        wn_integrated_spectra = self._integrate_spec_grid(wn_reformed_spec_grid)
        
        wr_grids['wn'] = {
            'spectra': wn_reformed_spec_grid,
            'params': wn_spec_params_reform,
            'teffs': wn_spec_params_teff,
            'integrated': wn_integrated_spectra
        }
        
        # Process WC grid
        wc_reformed_spec_grid, wc_spec_params_reform, wc_spec_params_teff = self._reform_spec_grid_WR(
            self.aux_data['wc_spectra'], self.aux_data['wc_spec_params']
        )
        wc_integrated_spectra = self._integrate_spec_grid(wc_reformed_spec_grid)
        
        wr_grids['wc'] = {
            'spectra': wc_reformed_spec_grid,
            'params': wc_spec_params_reform,
            'teffs': wc_spec_params_teff,
            'integrated': wc_integrated_spectra
        }
        
        # Process PoWR WN grid
        wn_reformed_spec_grid_powr, wn_spec_params_reform_powr, wn_spec_params_teff_powr, wn_spec_params_radius_powr, _ = (
            self._reform_spec_grid_powr(self.aux_data['wn_spectra_powr'], self.aux_data['wn_spec_params_powr'])
        )
        wn_integrated_spectra_powr = self._integrate_spec_grid(wn_reformed_spec_grid_powr)
        
        wr_grids['wn_powr'] = {
            'spectra': wn_reformed_spec_grid_powr,
            'params': wn_spec_params_reform_powr,
            'teffs': wn_spec_params_teff_powr,
            'radii': wn_spec_params_radius_powr,
            'integrated': wn_integrated_spectra_powr
        }
        
        # Process PoWR WC grid
        wc_reformed_spec_grid_powr, wc_spec_params_reform_powr, wc_spec_params_teff_powr, wc_spec_params_radius_powr, _ = (
            self._reform_spec_grid_powr(self.aux_data['wc_spectra_powr'], self.aux_data['wc_spec_params_powr'])
        )
        wc_integrated_spectra_powr = self._integrate_spec_grid(wc_reformed_spec_grid_powr)
        
        wr_grids['wc_powr'] = {
            'spectra': wc_reformed_spec_grid_powr,
            'params': wc_spec_params_reform_powr,
            'teffs': wc_spec_params_teff_powr,
            'radii': wc_spec_params_radius_powr,
            'integrated': wc_integrated_spectra_powr
        }
        
        return wr_grids
    
    def _reform_spec_grid_WR(self, spectra, spec_params):
        """
        Reform Wolf-Rayet spectral grid.
        Equivalent to reform_spec_grid_WR in pySB99.py
        """
        reformed_spec_grid = []
        for i in range(len(spectra)):
            reformatted_spec = self._reformat_spec_WR(spectra[i])
            reformed_spec_grid.append(reformatted_spec)
        
        spec_params_id = []
        spec_params_teff = []
        
        for i in spec_params:
            spec_id, spec_teff = i.split()
            spec_id = float(spec_id)
            spec_params_id.append(spec_id)
            spec_teff = float(spec_teff)
            spec_params_teff.append(spec_teff)
        
        spec_params_teff = np.array(spec_params_teff)
        spec_params_reform = np.column_stack((spec_params_id, spec_params_teff))
        
        return reformed_spec_grid, spec_params_reform, spec_params_teff
    
    def _reform_spec_grid_powr(self, spectra, spec_params):
        """
        Reform PoWR spectral grid.
        Equivalent to reform_spec_grid_powr in pySB99.py
        """
        reformed_spec_grid = []
        for i in range(len(spectra)):
            reformatted_spec = self._reformat_spec_WR(spectra[i])
            reformed_spec_grid.append(reformatted_spec)
        
        spec_params_id = []
        spec_params_teff = []
        spec_params_radius = []
        spec_params_length = []
        
        for i in spec_params:
            spec_id, spec_teff, spec_radius, spec_length = i.split()
            spec_id = float(spec_id)
            spec_params_id.append(spec_id)
            spec_teff = float(spec_teff)
            spec_params_teff.append(spec_teff)
            spec_radius = float(spec_radius)
            spec_params_radius.append(spec_radius)
            spec_length = float(spec_length)
            spec_params_length.append(spec_length)
        
        spec_params_teff = np.array(spec_params_teff)
        spec_params_reform = np.column_stack((spec_params_id, spec_params_teff, spec_params_radius, spec_params_length))
        
        return (reformed_spec_grid, spec_params_reform, spec_params_teff, 
               np.array(spec_params_radius), np.array(spec_params_length))
    
    def _process_lowmass_grid(self):
        """Process low-mass stellar grid."""
        lowmass_teffs = self.aux_data['lowmass_params'][:, 0]
        lowmass_loggs = self.aux_data['lowmass_params'][:, 1]
        lowmass_temps = np.log10(lowmass_teffs)
        
        lowmass_spec = []
        for i in range(len(self.aux_data['lowmass_flux'])):
            lm_spectrum = np.column_stack((self.wavelength_grid, self.aux_data['lowmass_flux'][i]))
            lowmass_spec.append(lm_spectrum)
        
        lowmass_int_spec = self._integrate_spec_grid(lowmass_spec)
        
        return {
            'spectra': lowmass_spec,
            'teffs': lowmass_teffs,
            'loggs': lowmass_loggs,
            'integrated': lowmass_int_spec
        }
    
    def _process_hires_grid(self):
        """Process high-resolution spectral grid."""
        if not self.config.use_powr:  # Using regular hires grid
            hires_teffs = self.aux_data['hires_params'][:, 0]
            hires_loggs = self.aux_data['hires_params'][:, 1]
            
            hires_spec = []
            hires_cont = []
            
            for i in range(len(self.aux_data['hires_flux'])):
                hires_spectrum = np.column_stack((self.aux_data['hires_wave_grid'], self.aux_data['hires_flux'][i]))
                hires_continuum = np.column_stack((self.aux_data['hires_wave_grid'], self.aux_data['hires_cont_flux'][i]))
                hires_spec.append(hires_spectrum)
                hires_cont.append(hires_continuum)
            
            hires_int_spec = self._integrate_spec_grid(hires_spec)
            
            return {
                'spectra': hires_spec,
                'cont': hires_cont,
                'teffs': hires_teffs,
                'loggs': hires_loggs,
                'integrated': hires_int_spec
            }
        else:
            # PoWR grid not fully implemented in original code
            hires_wave_grid = np.linspace(920, 3100, 10000)
            empty_hires_flux = np.full_like(hires_wave_grid, 0.0)
            
            return {
                'wave_grid': hires_wave_grid,
                'empty_flux': empty_hires_flux
            }
    
    def _integrate_spec_grid(self, reformed_spec_grid):
        """
        Integrate spectral grid.
        Equivalent to integrate_spec_grid in pySB99.py
        """
        integrated_spectra = []
        for i in range(len(reformed_spec_grid)):
            integrated_spectrum = np.trapz(reformed_spec_grid[i][:,1], reformed_spec_grid[i][:,0])
            integrated_spectra.append(integrated_spectrum)
        return integrated_spectra
    
    def calculate_population(self) -> StellarPopulationResults:
        """
        Calculate stellar population synthesis for all time steps.
        
        Returns
        -------
        StellarPopulationResults
            Complete results of the population synthesis
        """
        print("Starting population synthesis calculations...")
        start_time = time.time()
        
        # Get timestep mass indices
        timestep_mass_ind = self._get_timestep_0_ind(self.times_steps_SB99[0])
        
        # Initialize result arrays
        results = {
            'times': self.times_fine,
            'population_flux_iterations': [],
            'population_flux_iterations_send': [],
            'population_ion_flux_iterations': [],
            'population_ion_L_flux_iterations': [],
            'population_ion_HI_flux_iterations': [],
            'population_ion_HEI_flux_iterations': [],
            'population_ion_HEII_flux_iterations': [],
            'population_ion_HI_lum_iterations': [],  # Energy fluxes
            'population_ion_HEI_lum_iterations': [],  # Energy fluxes
            'population_ion_HEII_lum_iterations': [],  # Energy fluxes
            'population_continuum_iterations': [],
            'population_flux_total_iterations': [],
            'population_ages': [],
            'population_masses': [],
            'population_temps': [],
            'population_teffs': [],
            'population_radii': [],
            'population_lums': [],
            'population_H_abundances': [],
            'population_12C_abundances': [],
            'population_14N_abundances': [],
            'population_16O_abundances': [],
            'population_assigned_spec_teffs': [],
            'population_assigned_spec_logls': [],
            'population_assigned_spec_loggs': [],
            'population_loggs': [],
            'population_mass_loss_rates': [],
            'population_vinfs': [],
            'population_vescs': [],
            'population_windmoms': [],
            'population_windmoms_leuven': [],
            'population_windmoms_xshootu': [],
            'population_windmoms_vink': [],
            'population_vinfs_vink': [],
            'population_mdots_vink': [],
            'population_windmoms_calc': [],
            'population_windpowers': [],
            'population_windpowers_xshootu': [],
            'population_uv_slopes_x': [],
            'population_uv_slopes_y': [],
            'population_uv_slopes_beta': [],
            'population_Ha_ew': [],
            'population_Hb_ew': [],
            'population_Pb_ew': [],
            'population_Bg_ew': [],
            'population_Vmag': [],
            'population_Umag': [],
            'population_Imag': [],
            'population_Bmag': [],
            'population_absVmag': [],
            'population_Ha_luminosity': [],
            'population_Ha_continuum_flux': [],
            'population_choice_iterations': []
        }
        
        # Main calculation loop
        for i, timestep in enumerate(self.times_fine):
            if i % 50 == 0:
                print(f"Processing timestep {i}/{len(self.times_fine)}: {timestep/1e6:.2f} Myr")
            
            # Get stellar parameters at this timestep
            timestep_ages, timestep_temps, timestep_lums, timestep_masses, timestep_H, \
            timestep_He, timestep_loggs, timestep_mdot, timestep_12C, timestep_14N, timestep_16O, \
            timestep_mass_test, timestep_cnr, timestep_coher = self._get_timestep_params(
                timestep, timestep_mass_ind
            )
            
            # For first timestep, calculate IMF
            if i == 0:
                initial_masses = timestep_masses
                # Use custom star numbers if provided, otherwise calculate from IMF
                if self.config.custom_star_numbers is not None:
                    No_stars = self._create_custom_star_numbers(
                        timestep_masses, self.config.custom_star_numbers
                    )                    
                    total_stars = np.sum(No_stars)
                    total_mass = np.sum(No_stars * timestep_masses)
                    print('Custom star distribution:')
                    print(f'Total No stars = {total_stars:.4E}, Total mass = {total_mass:.4E} Msun')                    
                else:
                    No_stars, c_masses, dens, xmhigh, xmlow = self.calc_Nostars(
                        timestep_masses, self.config.imf_exponents, self.config.imf_mass_limits
                    )
                # Store No_stars as instance variable to emulate global access
                self.No_stars = No_stars
            
            # Calculate linear Teffs and radii
            timestep_teffs = 10**np.array(timestep_temps)
            timestep_radii = self._compute_radii(timestep_temps, timestep_lums)
            
            # Get spectroscopic parameters
            specsyn_teffs, specsyn_loggs, specsyn_radii, specsyn_cotests, specsyn_bbfluxes = self._get_specsyn_params(
                timestep_temps, timestep_masses, timestep_lums
            )
            
            # Assign spectra to stars
            assigned_integrated_spectra, assigned_spec_teff, assigned_spectra, population_choice, \
            assigned_spec_logl, assigned_spec_logg = self._assign_spectra_to_grid_WR(
                timestep_temps, self.ob_spectral_grid['params'], self.ob_spectral_grid['teffs'], 
                self.ob_spectral_grid['logls'], timestep_lums, timestep_H, 
                self.ob_spectral_grid['loggs'], timestep_masses, initial_masses, 
                specsyn_cotests, specsyn_loggs, timestep_cnr, timestep_coher
            )
            
            # Calculate population spectrum
            population_flux, assigned_flux_scaled = self._specsyn(
                assigned_integrated_spectra, specsyn_bbfluxes, assigned_spectra, 
                specsyn_radii, No_stars
            )
            
            # Calculate wind properties
            timestep_vinfs, timestep_windpowers_calc, timestep_windpowers, \
            timestep_windmoms_calc, timestep_windmoms, timestep_windmoms_vink, \
            timestep_vinfs_vink, timestep_mdots_vink, timestep_vescs, \
            timestep_windmoms_leuven, timestep_windmoms_xshootu, timestep_windpowers_xshootu = self._calc_wind(
                timestep_temps, timestep_lums, timestep_masses, timestep_mdot, 
                timestep_H, timestep_He, timestep_12C, timestep_14N, timestep_16O, 
                initial_masses, No_stars, timestep_radii, timestep_cnr
            )
            
            # Calculate ionizing fluxes
            population_ion_flux = self._ionise(self.wavelength_grid, population_flux, 2)
            
            # Calculate nebular continuum
            population_continuum = self._continuum(population_ion_flux[1][0])
            
            # Add nebular continuum to spectrum
            continuum_resampled = np.interp(
                self.ob_spectral_grid['spectra'][0][:, 0], 
                population_continuum[:, 0], 
                population_continuum[:, 1]
            )
            population_flux_total = population_flux + continuum_resampled
            
            # Calculate UV slope
            timestep_uv_slope_x, timestep_uv_slope_y, timestep_uv_slope_beta = self._get_uv_slope(
                np.log10(self.ob_spectral_grid['spectra'][0][:, 0]), 
                safe_log10(population_flux_total) + 20.
            )
            
            # Calculate equivalent widths
            timestep_Ha_ew, timestep_Hb_ew, timestep_Pb_ew, timestep_Bg_ew, \
            timestep_Ha_luminosity, timestep_Ha_continuum_flux = self._get_ew(
                self.ob_spectral_grid['spectra'][0][:, 0], 
                population_flux_total, 
                population_ion_flux[1][0]
            )
            
            # Calculate photometric colors
            timestep_Vmag, timestep_Umag, timestep_Imag, timestep_Bmag, timestep_absVmag = self._colours(
                population_flux_total
            )
            
            # Store results for this timestep
            results['population_flux_iterations'].append(population_flux)
            results['population_flux_iterations_send'].append(safe_log10(population_flux) + 20.)
            results['population_ion_flux_iterations'].append(population_ion_flux)
            results['population_ion_L_flux_iterations'].append(population_ion_flux[0])
            results['population_ion_HI_flux_iterations'].append(population_ion_flux[1][0])  # Photon rate
            results['population_ion_HEI_flux_iterations'].append(population_ion_flux[2][0])  # Photon rate
            results['population_ion_HEII_flux_iterations'].append(population_ion_flux[3][0])  # Photon rate
            results['population_ion_HI_lum_iterations'].append(population_ion_flux[1][1])  # Energy flux
            results['population_ion_HEI_lum_iterations'].append(population_ion_flux[2][1])  # Energy flux
            results['population_ion_HEII_lum_iterations'].append(population_ion_flux[3][1])  # Energy flux
            results['population_continuum_iterations'].append(population_continuum)
            results['population_flux_total_iterations'].append(population_flux_total)
            results['population_choice_iterations'].append(population_choice)
            results['population_uv_slopes_x'].append(timestep_uv_slope_x)
            results['population_uv_slopes_y'].append(timestep_uv_slope_y)
            results['population_uv_slopes_beta'].append(timestep_uv_slope_beta)
            results['population_Ha_ew'].append(timestep_Ha_ew)
            results['population_Hb_ew'].append(timestep_Hb_ew)
            results['population_Pb_ew'].append(timestep_Pb_ew)
            results['population_Bg_ew'].append(timestep_Bg_ew)
            results['population_Vmag'].append(timestep_Vmag)
            results['population_Umag'].append(timestep_Umag)
            results['population_Imag'].append(timestep_Imag)
            results['population_Bmag'].append(timestep_Bmag)
            results['population_absVmag'].append(timestep_absVmag)
            results['population_Ha_luminosity'].append(timestep_Ha_luminosity)
            results['population_Ha_continuum_flux'].append(timestep_Ha_continuum_flux)
            
            results['population_ages'].append(timestep_ages)
            results['population_masses'].append(timestep_masses)
            results['population_temps'].append(timestep_temps)
            results['population_teffs'].append(timestep_teffs)
            results['population_lums'].append(timestep_lums)
            results['population_radii'].append(timestep_radii)
            results['population_H_abundances'].append(timestep_H)
            results['population_12C_abundances'].append(timestep_12C)
            results['population_14N_abundances'].append(timestep_14N)
            results['population_16O_abundances'].append(timestep_16O)
            results['population_assigned_spec_teffs'].append(assigned_spec_teff)
            results['population_assigned_spec_logls'].append(assigned_spec_logl)
            results['population_assigned_spec_loggs'].append(assigned_spec_logg)
            results['population_loggs'].append(timestep_loggs)
            results['population_mass_loss_rates'].append(timestep_mdot)
            results['population_vinfs'].append(timestep_vinfs)
            results['population_vescs'].append(timestep_vescs)
            results['population_windmoms'].append(timestep_windmoms)
            results['population_windmoms_vink'].append(timestep_windmoms_vink)
            results['population_windmoms_leuven'].append(timestep_windmoms_leuven)
            results['population_windmoms_xshootu'].append(timestep_windmoms_xshootu)
            results['population_windpowers_xshootu'].append(timestep_windpowers_xshootu)
            results['population_vinfs_vink'].append(timestep_vinfs_vink)
            results['population_mdots_vink'].append(timestep_mdots_vink)
            results['population_windmoms_calc'].append(timestep_windmoms_calc)
            results['population_windpowers'].append(timestep_windpowers)
        
        elapsed_time = time.time() - start_time
        print(f"Calculation completed in {elapsed_time:.2f} seconds")
        
        # Create final results object
        return self._create_results_object(results, No_stars)
    
    def _get_timestep_0_ind(self, timestep):
        """
        Get indices for unique initial masses.
        Equivalent to get_timestep_0_ind in pySB99.py
        """
        timestep_masses = []
        for i in range(len(self.grid_data['ages'])):
            ind_nearest_age = np.argmin((self.grid_data['ages'][i] - self.times_steps_SB99[0])**2)
            nearest_mass = self.grid_data['masses'][i][ind_nearest_age]
            timestep_masses.append(nearest_mass)
        
        timestep_masses_unique = np.unique(timestep_masses, return_index=True)
        
        timestep_masses_IMF_ind = [
            i for i in range(len(timestep_masses_unique[0])) 
            if timestep_masses_unique[0][i] < self.config.imf_mass_limits[-1]
        ]
        
        timestep_masses_red_ind = timestep_masses_unique[1][timestep_masses_IMF_ind]
        
        return timestep_masses_red_ind
    
    def _get_timestep_params(self, timestep, timestep_mass_ind):
        """
        Get stellar parameters for a given timestep.
        Equivalent to get_timestep_params in pySB99.py
        """
        timestep_ages = []
        timestep_masses = []
        timestep_temps = []
        timestep_lums = []
        timestep_H_abundances = []
        timestep_He_abundances = []
        timestep_12C_abundances = []
        timestep_14N_abundances = []
        timestep_16O_abundances = []
        timestep_core_temps = []
        timestep_mass_loss_rates = []
        
        for i in range(len(self.grid_data['ages'])):
            nearest_agecalc = abs(self.grid_data['ages'][i] - timestep)
            ind_nearest_age = np.argmin(nearest_agecalc)
            nearest_age = self.grid_data['ages'][i][ind_nearest_age]
            
            nearest_mass = self.grid_data['masses'][i][ind_nearest_age]
            nearest_temp = self.grid_data['temperatures'][i][ind_nearest_age]
            nearest_lum = self.grid_data['luminosities'][i][ind_nearest_age]
            nearest_H_abundance = self.grid_data['h_abundances'][i][ind_nearest_age]
            nearest_He_abundance = self.grid_data['he_abundances'][i][ind_nearest_age]
            nearest_12C_abundance = self.grid_data['c12_abundances'][i][ind_nearest_age]
            nearest_14N_abundance = self.grid_data['n14_abundances'][i][ind_nearest_age]
            nearest_16O_abundance = self.grid_data['o16_abundances'][i][ind_nearest_age]
            nearest_core_temp = self.grid_data['core_temperatures'][i][ind_nearest_age]
            nearest_mass_loss_rate = self.grid_data['mass_loss_rates'][i][ind_nearest_age]
            
            timestep_ages.append(nearest_age)
            timestep_masses.append(nearest_mass)
            timestep_temps.append(nearest_temp)
            timestep_lums.append(nearest_lum)
            timestep_H_abundances.append(nearest_H_abundance)
            timestep_He_abundances.append(nearest_He_abundance)
            timestep_12C_abundances.append(nearest_12C_abundance)
            timestep_14N_abundances.append(nearest_14N_abundance)
            timestep_16O_abundances.append(nearest_16O_abundance)
            timestep_core_temps.append(nearest_core_temp)
            timestep_mass_loss_rates.append(nearest_mass_loss_rate)
        
        # Filter parameters by mass indices
        timestep_ages_final = []
        timestep_temps_final = []
        timestep_lums_final = []
        timestep_masses_final = []
        timestep_H_abundances_final = []
        timestep_He_abundances_final = []
        timestep_12C_abundances_final = []
        timestep_14N_abundances_final = []
        timestep_16O_abundances_final = []
        timestep_core_temps_final = []
        timestep_mass_loss_rates_final = []
        
        for i in timestep_mass_ind:
            timestep_ages_final.append(timestep_ages[i])
            timestep_temps_final.append(timestep_temps[i])
            timestep_lums_final.append(timestep_lums[i])
            timestep_masses_final.append(timestep_masses[i])
            timestep_H_abundances_final.append(timestep_H_abundances[i])
            timestep_He_abundances_final.append(timestep_He_abundances[i])
            timestep_12C_abundances_final.append(timestep_12C_abundances[i])
            timestep_14N_abundances_final.append(timestep_14N_abundances[i])
            timestep_16O_abundances_final.append(timestep_16O_abundances[i])
            timestep_core_temps_final.append(timestep_core_temps[i])
            timestep_mass_loss_rates_final.append(timestep_mass_loss_rates[i])
        
        # Calculate derived quantities
        timestep_teffs_calc = 10**np.array(timestep_temps_final)
        timestep_lum_calc = (10**np.array(timestep_lums_final) * 3.839e33)
        radii_denom = (12.566 * 5.670e-5 * (timestep_teffs_calc**4))
        timestep_radii_calc = ((timestep_lum_calc / radii_denom)**0.5) * 100
        
        logg_num = (6.67e-12) * np.array(timestep_masses_final) * (1.9891e33) * 1e8
        timestep_gravities_final = (logg_num) / (timestep_radii_calc**2)
        timestep_loggs_final = np.log10(timestep_gravities_final)

        # CRITICAL: Store timestep_loggs_final as instance attribute
        # This matches the original code where timestep_loggs_final was a global variable
        self.current_timestep_loggs = timestep_loggs_final
        
        timestep_H_abundances_final = np.array(timestep_H_abundances_final)
        timestep_He_abundances_final = np.array(timestep_He_abundances_final)
        timestep_12C_abundances_final = np.array(timestep_12C_abundances_final)
        timestep_14N_abundances_final = np.array(timestep_14N_abundances_final)
        timestep_16O_abundances_final = np.array(timestep_16O_abundances_final)
        
        # Calculate carbon-to-nitrogen ratio
        timestep_cnr = timestep_12C_abundances_final / timestep_14N_abundances_final
        
        # Calculate carbon+oxygen to helium ratio
        timestep_coher = ((timestep_12C_abundances_final/12.) + (timestep_16O_abundances_final/16.)) / (timestep_He_abundances_final/4.)
        
        # Apply Wolf-Rayet temperature correction
        WR_correction_factor = 0.6
        for i in range(len(timestep_masses_final)):
            if timestep_H_abundances_final[i] < 0.1:
                timestep_core_teff_final = 10**timestep_core_temps_final[i]
                corrected_teff = timestep_core_teff_final + (WR_correction_factor - 1.0) * (timestep_core_teff_final - timestep_teffs_calc[i])
                timestep_temps_final[i] = np.log10(corrected_teff)
        
        # Mark supernovae
        ind_SN_stars = [i for i in range(len(timestep_lums_final)) if timestep_lums_final[i] < -19.]
        timestep_temps_final = np.array(timestep_temps_final)
        
        return (timestep_ages_final, timestep_temps_final, timestep_lums_final, 
                timestep_masses_final, timestep_H_abundances_final, timestep_He_abundances_final,
                timestep_loggs_final, timestep_mass_loss_rates_final, timestep_12C_abundances_final, 
                timestep_14N_abundances_final, timestep_16O_abundances_final, timestep_masses, timestep_cnr, 
                timestep_coher)
    
    def _get_specsyn_params(self, timestep_temps_final, timestep_masses_final, timestep_lums_final):
        """
        Get spectroscopic parameters.
        Equivalent to get_specsyn_params in pySB99.py
        """
        specsyn_teffs = []
        specsyn_loggs = []
        specsyn_radii = []
        specsyn_cotests = []
        specsyn_bbfluxes = []
        
        for i in range(len(timestep_temps_final)):
            teff_specsyn = 10**(timestep_temps_final[i])
            
            logg_specsyn = np.log10(timestep_masses_final[i]) + (4.* timestep_temps_final[i]) - timestep_lums_final[i] - 10.6
            radius_specsyn = 10.**(10.8426 + 0.5*timestep_lums_final[i] - (2.*timestep_temps_final[i]) + 7.52)
            cotest = 5.71*np.log10(teff_specsyn) - 21.95
            
            if teff_specsyn == 0.0:
                specsyn_bbflux = 0.0
            else:
                specsyn_bbflux = 5.6696196e-05 * teff_specsyn**4.
            
            specsyn_teffs.append(teff_specsyn)
            specsyn_loggs.append(logg_specsyn)
            specsyn_radii.append(radius_specsyn)
            specsyn_cotests.append(cotest)
            specsyn_bbfluxes.append(specsyn_bbflux)
        
        # Set temperature to 0 for supernova stars
        ind_SN_stars = [i for i in range(len(specsyn_teffs)) if specsyn_teffs[i] == 1.0]
        specsyn_teffs = np.array(specsyn_teffs)
        specsyn_teffs[ind_SN_stars] = 0.0
        
        return (specsyn_teffs, specsyn_loggs, specsyn_radii, specsyn_cotests, specsyn_bbfluxes)
    
    def _assign_spectra_to_grid_WR(self, timestep_temps_final, spec_params_reform, 
                                 spec_params_teff, spec_params_logl, timestep_lums_final, 
                                 timestep_H_abundances_final, spec_params_logg, 
                                 timestep_masses_final, initial_masses, specsyn_cotests, 
                                 specsyn_loggs, timestep_cnr, timestep_coher):
        """
        Assign appropriate spectra to stars based on their properties.
        Equivalent to assign_spectra_to_grid_WR in pySB99.py
        """
        assigned_integrated_spectra = []
        assigned_spec_teff = []
        assigned_spec_logl = []
        assigned_spec_logg = []
        assigned_spectra = []
        timestep_teffs_final = 10**timestep_temps_final
        population_choice = []
        
        # WR spectra data
        WN_spec_params_logl = np.full_like(self.wr_spectral_grids['wn']['teffs'], 1.)
        WC_spec_params_logl = np.full_like(self.wr_spectral_grids['wc']['teffs'], 1.)
        lm_spec_params_logl = np.full_like(self.lowmass_spectral_grid['teffs'], 1.)
        
        # Empty flux for dead stars
        empty_flux = np.full_like(self.wavelength_grid, 0.0)
        
        for j in range(len(timestep_temps_final)):
            distance_to_spec = []
            
            # Dead star (supernova)
            if timestep_lums_final[j] == -20.0:
                population_choice.append('nope')
                assigned_spec_teff.append(0.0)
                assigned_spectra.append(np.column_stack((self.wavelength_grid, empty_flux)))
                assigned_integrated_spectra.append(1e-30)
            
            # Wolf-Rayet star
            elif (timestep_temps_final[j] > 4.4 and 
                  timestep_H_abundances_final[j] < 0.4 and 
                  initial_masses[j] > self.loader.minimum_wr_mass):
                
                population_choice.append('WR')
                
                # WN (late) - hydrogen-rich
                if timestep_H_abundances_final[j] > 0.1:
                    for i in range(len(self.wr_spectral_grids['wn']['params'])):
                        distance_to_spec.append((self.wr_spectral_grids['wn']['teffs'][i] - 10**timestep_temps_final[j])**2)
                    
                    nearest_spec_ind = np.argmin(distance_to_spec)
                    near_spec_temp = self.wr_spectral_grids['wn']['teffs'][nearest_spec_ind]
                    assigned_spec_teff.append(near_spec_temp)
                    near_spec_logl = WN_spec_params_logl[nearest_spec_ind]
                    assigned_spec_logl.append(near_spec_logl)
                    assigned_spec_logg.append(-1.)
                    assigned_spectra.append(self.wr_spectral_grids['wn']['spectra'][nearest_spec_ind])
                    assigned_integrated_spectra.append(self.wr_spectral_grids['wn']['integrated'][nearest_spec_ind])
                
                # WN (early) - nitrogen-rich
                elif timestep_cnr[j] < 10.:
                    for i in range(len(self.wr_spectral_grids['wn']['params'])):
                        distance_to_spec.append((self.wr_spectral_grids['wn']['teffs'][i] - 10**timestep_temps_final[j])**2)
                    
                    nearest_spec_ind = np.argmin(distance_to_spec)
                    near_spec_temp = self.wr_spectral_grids['wn']['teffs'][nearest_spec_ind]
                    assigned_spec_teff.append(near_spec_temp)
                    near_spec_logl = WN_spec_params_logl[nearest_spec_ind]
                    assigned_spec_logl.append(near_spec_logl)
                    assigned_spec_logg.append(-1.)
                    assigned_spectra.append(self.wr_spectral_grids['wn']['spectra'][nearest_spec_ind])
                    assigned_integrated_spectra.append(self.wr_spectral_grids['wn']['integrated'][nearest_spec_ind])
                
                # WC (late) - carbon-rich
                elif timestep_coher[j] < 0.5:
                    for i in range(len(self.wr_spectral_grids['wc']['params'])):
                        distance_to_spec.append((self.wr_spectral_grids['wc']['teffs'][i] - 10**timestep_temps_final[j])**2)
                    
                    nearest_spec_ind = np.argmin(distance_to_spec)
                    near_spec_temp = self.wr_spectral_grids['wc']['teffs'][nearest_spec_ind]
                    assigned_spec_teff.append(near_spec_temp)
                    near_spec_logl = WC_spec_params_logl[nearest_spec_ind]
                    assigned_spec_logl.append(near_spec_logl)
                    assigned_spec_logg.append(-1.)
                    assigned_spectra.append(self.wr_spectral_grids['wc']['spectra'][nearest_spec_ind])
                    assigned_integrated_spectra.append(self.wr_spectral_grids['wc']['integrated'][nearest_spec_ind])
                
                # WC (early)
                elif timestep_coher[j] < 1.:
                    for i in range(len(self.wr_spectral_grids['wc']['params'])):
                        distance_to_spec.append((self.wr_spectral_grids['wc']['teffs'][i] - 10**timestep_temps_final[j])**2)
                    
                    nearest_spec_ind = np.argmin(distance_to_spec)
                    near_spec_temp = self.wr_spectral_grids['wc']['teffs'][nearest_spec_ind]
                    assigned_spec_teff.append(near_spec_temp)
                    near_spec_logl = WC_spec_params_logl[nearest_spec_ind]
                    assigned_spec_logl.append(near_spec_logl)
                    assigned_spec_logg.append(-1.)
                    assigned_spectra.append(self.wr_spectral_grids['wc']['spectra'][nearest_spec_ind])
                    assigned_integrated_spectra.append(self.wr_spectral_grids['wc']['integrated'][nearest_spec_ind])
                
                # WO - oxygen-rich
                elif timestep_coher[j] >= 1.:
                    for i in range(len(self.wr_spectral_grids['wc']['params'])):
                        distance_to_spec.append((self.wr_spectral_grids['wc']['teffs'][i] - 10**timestep_temps_final[j])**2)
                    
                    nearest_spec_ind = np.argmin(distance_to_spec)
                    near_spec_temp = self.wr_spectral_grids['wc']['teffs'][nearest_spec_ind]
                    assigned_spec_teff.append(near_spec_temp)
                    near_spec_logl = WC_spec_params_logl[nearest_spec_ind]
                    assigned_spec_logl.append(near_spec_logl)
                    assigned_spec_logg.append(-1.)
                    assigned_spectra.append(self.wr_spectral_grids['wc']['spectra'][nearest_spec_ind])
                    assigned_integrated_spectra.append(self.wr_spectral_grids['wc']['integrated'][nearest_spec_ind])
            
            # Large, cool star
            elif timestep_temps_final[j] < 3.65 and initial_masses[j] > self.loader.minimum_wr_mass:
                if hasattr(self, 'No_stars'):
                    population_choice.append(self.No_stars[j])
                else:
                    raise AttributeError("self.No_stars not found - this should be set during IMF calculation")
            
            # OB star
            elif (timestep_teffs_final[j] > 20000. and 
                specsyn_loggs[j] > 2.2 and 
                specsyn_loggs[j] < specsyn_cotests[j]):
                
                population_choice.append('OB')
                
                for i in range(len(spec_params_reform)):
                    # Check if current_timestep_loggs exists, otherwise raise error
                    if hasattr(self, 'current_timestep_loggs'):
                        logg_value = self.current_timestep_loggs[j]
                    else:
                        raise AttributeError("current_timestep_loggs not found. Make sure _get_timestep_params() was called first.")
                    distance_to_spec.append(abs(spec_params_teff[i] - 10**timestep_temps_final[j]) + 
                                        abs(spec_params_logg[i] - logg_value))

                nearest_spec_ind = np.argmin(distance_to_spec)
                near_spec_temp = spec_params_teff[nearest_spec_ind]
                assigned_spec_teff.append(near_spec_temp)
                near_spec_logl = spec_params_logl[nearest_spec_ind]
                assigned_spec_logl.append(near_spec_logl)
                near_spec_logg = spec_params_logg[nearest_spec_ind]
                assigned_spec_logg.append(near_spec_logg)
                assigned_spectra.append(self.ob_spectral_grid['spectra'][nearest_spec_ind])
                assigned_integrated_spectra.append(self.ob_spectral_grid['integrated'][nearest_spec_ind])
            
            # Low-mass star
            else:
                population_choice.append('lowmass')
                
                for i in range(len(self.lowmass_spectral_grid['teffs'])):
                    # Check if current_timestep_loggs exists, otherwise raise error
                    if hasattr(self, 'current_timestep_loggs'):
                        logg_value = self.current_timestep_loggs[j]
                    else:
                        raise AttributeError("current_timestep_loggs not found. Make sure _get_timestep_params() was called first.") 
                    distance_to_spec.append((self.lowmass_spectral_grid['teffs'][i] - 10**timestep_temps_final[j])**2 + 
                                        (self.lowmass_spectral_grid['loggs'][i] - logg_value)**2)
                
                nearest_spec_ind = np.argmin(distance_to_spec)
                near_spec_temp = self.lowmass_spectral_grid['teffs'][nearest_spec_ind]
                assigned_spec_teff.append(near_spec_temp)
                near_spec_logl = lm_spec_params_logl[nearest_spec_ind]
                assigned_spec_logl.append(timestep_lums_final[j])
                near_spec_logg = self.lowmass_spectral_grid['loggs'][nearest_spec_ind]
                assigned_spec_logg.append(near_spec_logg)
                assigned_spectra.append(self.lowmass_spectral_grid['spectra'][nearest_spec_ind])
                assigned_integrated_spectra.append(self.lowmass_spectral_grid['integrated'][nearest_spec_ind])
        
        return (assigned_integrated_spectra, assigned_spec_teff, 
                assigned_spectra, population_choice, 
                assigned_spec_logl, assigned_spec_logg)
    
    def _specsyn(self, assigned_integrated_spectra, specsyn_bbfluxes, assigned_spectra, radii, No_stars):
        """
        Calculate population spectrum.
        Equivalent to specsyn in pySB99.py
        """
        assigned_spec_renormed = []
        assigned_flux_renormed = []
        template_wave = assigned_spectra[0][:,0]
        
        # Normalize spectra by blackbody flux
        for i in range(len(assigned_integrated_spectra)):
            xinte = assigned_integrated_spectra[i] / specsyn_bbfluxes[i]
            renormed_flux = assigned_spectra[i][:,1] / xinte
            renormed_spec = np.column_stack((template_wave, renormed_flux))
            assigned_spec_renormed.append(renormed_spec)
            assigned_flux_renormed.append(renormed_flux)
        
        test_assigned_flux = assigned_flux_renormed
        test_radii = radii
        
        assigned_flux_scaled = []
        assigned_flux_scaled_test = []
        
        # Scale flux by radius and number of stars
        for i in range(len(test_assigned_flux)):
            assigned_flux_scaled_i = 12.566 * test_radii[i] * test_radii[i] / 1e20 * test_assigned_flux[i] * No_stars[i]
            assigned_flux_scaled.append(assigned_flux_scaled_i)
            assigned_flux_scaled_test.append(assigned_flux_scaled_i)
        
        # Sum over all stars
        assigned_flux_scaled = np.array(assigned_flux_scaled)
        population_flux = assigned_flux_scaled.sum(axis=0)
        
        return population_flux, assigned_flux_scaled_test
    
    def _ionise(self, wave, pop_flux, code):
        """
        Calculate ionizing fluxes.
        Equivalent to ionise in pySB99.py
        """
        spectrum_flux = np.array(pop_flux)
        
        spectrum_freq = 2.997925e18 / wave
        spectrum_flux_conv1 = 3.33564e-19 * spectrum_flux * wave * wave
        spectrum_flux_conv = spectrum_flux_conv1 / spectrum_freq / 6.6262e-27
        
        full_lum = np.trapz(np.flip(spectrum_flux), np.flip(wave))
        
        if code == 1:
            bolo_lum = np.log10(full_lum + 1.0e-30) + 20.
        if code == 2:
            bolo_lum = np.log10(-1.0 * full_lum + 1.0e-30) + 20.
        
        flux_lim_HI = 912.
        flux_lim_HeI = 504.
        flux_lim_HeII = 228.
        
        ionising_flux_HI = self._compute_ion_flux(wave, spectrum_freq, spectrum_flux, spectrum_flux_conv, flux_lim_HI, code)
        ionising_flux_HeI = self._compute_ion_flux(wave, spectrum_freq, spectrum_flux, spectrum_flux_conv, flux_lim_HeI, code)
        ionising_flux_HeII = self._compute_ion_flux(wave, spectrum_freq, spectrum_flux, spectrum_flux_conv, flux_lim_HeII, code)
        
        return (bolo_lum, ionising_flux_HI, ionising_flux_HeI, ionising_flux_HeII)
    
    def _compute_ion_flux(self, wave, freq, flux_wave, flux_freq, limit, code):
        """
        Compute ionizing flux for a specific ion.
        Equivalent to compute_ion_flux in pySB99.py
        """
        ind_ION = [i for i in range(len(wave)) if wave[i] <= limit]
        
        if len(ind_ION) == 0:
            return (np.log10(1.0e-30) + 20.0, np.log10(1.0e-30) + 20.0, np.log10(1.0e-30) + 20.0)
        
        wave_ION = wave[ind_ION]
        freq_ION = freq[ind_ION]
        flux_wave_ION = flux_wave[ind_ION]
        flux_freq_ION = flux_freq[ind_ION]
        
        integral_photons_ION = np.trapz(flux_freq_ION, freq_ION)
        integral_flux_ION = np.trapz(flux_wave_ION, wave_ION)
        
        int_ION = []
        for i in range(len(wave_ION)):
            ION_comp = flux_wave_ION[i] / (6.626e-27 * wave_ION[i])
            int_ION.append(ION_comp)
        
        qi = np.sum(int_ION)
        Qi = qi * 4 * np.pi * 18**2
        Qilog = safe_log10(Qi)
        
        if code == 1:
            No_photons_ION = np.log10(integral_photons_ION + 1.0e-30) + 20.
            ion_flux_ION = np.log10(-1.0 * integral_flux_ION + 1.0e-30) + 20.
        if code == 2:
            No_photons_ION = np.log10(-1.0 * integral_photons_ION + 1.0e-30) + 20.
            ion_flux_ION = np.log10(integral_flux_ION + 1.0e-30) + 20.
        
        return (No_photons_ION, ion_flux_ION, Qilog)
    
    def _continuum(self, pop_ionising_flux):
        """
        Calculate nebular continuum emission.
        Equivalent to continuum in pySB99.py
        """
        xrange = np.array([10., 912., 913., 1300., 1500., 1800., 2200.,
                         2855., 3331., 3421., 3422., 3642., 3648., 5700., 7000., 8207.,
                         8209., 14583., 14585., 22787., 22789., 32813., 32815.,
                         44680., 44682., 2000000.])
        
        gamma = np.array([0., 0., 2.11e-4, 5.647, 9.35, 9.847, 10.582, 16.101, 24.681,
                        26.736, 24.883, 29.979, 6.519, 8.773, 11.545, 13.585, 6.333,
                        10.444, 7.023, 9.361, 7.59, 9.35, 8.32, 9.53, 8.87, 0.])
        
        continuum = 2.998e18 / xrange / xrange / 2.60e-13 * 1.0e-30 * gamma * 10**(pop_ionising_flux - 30.)
        
        return np.column_stack((xrange, continuum))
    
    def _calc_wind(self, timestep_temps, timestep_lums, timestep_masses, timestep_mdot, 
                 timestep_H, timestep_He, timestep_12C, timestep_14N, timestep_16O, initial_masses, 
                 No_stars, timestep_radii, timestep_cnr):
        """
        Calculate stellar wind properties.
        Equivalent to calc_wind in pySB99.py
        """
        timestep_mdot = np.array(timestep_mdot)
        timestep_masses = np.array(timestep_masses)
        timestep_radii = np.array(timestep_radii)
        timestep_lums = np.array(timestep_lums)
        timestep_H = np.array(timestep_H)
        timestep_12C = np.array(timestep_12C)
        timestep_14N = np.array(timestep_14N)
        timestep_16O = np.array(timestep_16O)
        timestep_coher = ((timestep_12C/12.) + (timestep_16O/16.)) / (timestep_He/4.)
        
        vinfs = []
        OB_limit_mass = 8.
        timestep_teffs = 10**timestep_temps
        timestep_vesc = ((2 * 6.67 * 10**-11 * (timestep_masses * 2 * 10**30)) / 
                       (timestep_radii * 7 * 10**8))**0.5 / (1. * 10**3)
        
        vinf_vink21 = []
        mdot_vink00 = []
        mdot_leuven = []
        vinf_xshootu = []
        
        for i in range(len(timestep_temps)):
            # Dead star
            if timestep_lums[i] == -20.0:
                vinf = 0.
                vinf_vink21i = 0.
                mdot_vink00i = timestep_mdot[i]
                mdot_leuveni = timestep_mdot[i]
                vinf_xshootui = 0.
            
            # LBV
            elif timestep_temps[i] < 4.4 and timestep_temps[i] > 3.75 and timestep_mdot[i] > -3.5:
                vinf = 200.
                vinf_vink21i = 200.
                mdot_vink00i = timestep_mdot[i]
                mdot_leuveni = timestep_mdot[i]
                vinf_xshootui = 200.
            
            # Low mass stars
            elif timestep_temps[i] < 3.9:
                vinf = 30.
                vinf_vink21i = 30.
                mdot_vink00i = timestep_mdot[i]
                mdot_leuveni = timestep_mdot[i]
                vinf_xshootui = 30.
            
            # Wolf-Rayet stars
            elif (timestep_temps[i] > 4.4 and timestep_H[i] < 0.4 and 
                  initial_masses[i] > self.loader.minimum_wr_mass):
                
                # WN (late)
                if timestep_H[i] > 0.1:
                    vinf = 1650.
                    vinf_vink21i = 1650.
                    mdot_vink00i = timestep_mdot[i]
                    mdot_leuveni = timestep_mdot[i]
                    vinf_xshootui = 1650.
                
                # WN (early)
                elif timestep_cnr[i] < 10.:
                    vinf = 1900.
                    vinf_vink21i = 1900.
                    mdot_vink00i = timestep_mdot[i]
                    mdot_leuveni = timestep_mdot[i]
                    vinf_xshootui = 1900.
                
                # WC (late)
                elif timestep_coher[i] < 0.5:
                    vinf = 1800.
                    vinf_vink21i = 1800.
                    mdot_vink00i = timestep_mdot[i]
                    mdot_leuveni = timestep_mdot[i]
                    vinf_xshootui = 1800.
                
                # WC (early)
                elif timestep_coher[i] < 1.:
                    vinf = 2800.
                    vinf_vink21i = 2800.
                    mdot_vink00i = timestep_mdot[i]
                    mdot_leuveni = timestep_mdot[i]
                    vinf_xshootui = 2800.
                
                # WO
                elif timestep_coher[i] >= 1.:
                    vinf = 3500.
                    vinf_vink21i = 3500.
                    mdot_vink00i = timestep_mdot[i]
                    mdot_leuveni = timestep_mdot[i]
                    vinf_xshootui = 3500.
            
            # OB stars
            elif timestep_temps[i] > 3.9:
                # Radiation pressure parameter
                gamma0 = 1. - (2.7e-5 * (10.**(timestep_lums[i])) / timestep_masses[i])
                if gamma0 <= 0.:
                    gamma0 = 1e-10
                
                # Wind velocity (Kudritzki & Puls 2000)
                vinf = 618. * np.sqrt(timestep_masses[i] / 
                                    (10.**(0.5*timestep_lums[i] - 2.*timestep_temps[i] + 7.52)) * 
                                    gamma0) * (0.58 + 2.04*(0.5*timestep_lums[i] - 2.*timestep_temps[i] + 7.52))
                
                # Scale mass loss rate
                timestep_mdot[i] = safe_log10((10**timestep_mdot[i]))
                
                # Leuven mass loss rate
                mdot_leuveni = -5.52 + 2.39*np.log10((10**timestep_lums[i])/(10**6)) - \
                             1.48*np.log10(timestep_masses[i]/45.) + \
                             2.12*np.log10(timestep_teffs[i]/45000.) + \
                             (0.75 - 1.87*np.log10(timestep_teffs[i]/40000.)) * np.log10(1./1.)
                
                # X-Shooter wind velocity
                vinf_xshootui = 0.092*timestep_teffs[i] - 1040.0*1.0**0.22
                
                # Vink mass loss rate and wind velocity
                if timestep_teffs[i] < 25883.:
                    log_vinf_vink21i = -7.79 - 0.07*timestep_lums[i] + 2.57*timestep_temps[i]
                    vinf_vink21i = 10**log_vinf_vink21i
                    mdot_vink00i = -6.688 + 2.210*np.log10(((10**timestep_lums[i])/(1.*10**5))) - \
                                 1.339*np.log10(timestep_masses[i]/30.) - \
                                 1.601*np.log10((vinf_vink21i/timestep_vesc[i])/2.) + \
                                 1.07*np.log10(timestep_teffs[i]/20000.)
                else:
                    log_vinf_vink21i = 0.39 - 0.04*timestep_lums[i] + 0.74*timestep_temps[i]
                    vinf_vink21i = 10**log_vinf_vink21i
                    mdot_vink00i = -6.697 + 2.194*np.log10(((10**timestep_lums[i])/(1.*10**5))) - \
                                 1.313*np.log10(timestep_masses[i]/30.) - \
                                 1.226*np.log10((vinf_vink21i/timestep_vesc[i])/2.) + \
                                 0.933*np.log10(timestep_teffs[i]/40000.) - \
                                 10.92*(np.log10(timestep_teffs[i]/40000.))**2
            
            # Default
            else:
                vinf = 0.
                vinf_vink21i = 0.
                mdot_vink00i = timestep_mdot[i]
                mdot_leuveni = timestep_mdot[i]
                vinf_xshootui = 0.
            
            vinf_vink21.append(vinf_vink21i)
            mdot_vink00.append(mdot_vink00i)
            mdot_leuven.append(mdot_leuveni)
            vinfs.append(vinf)
            vinf_xshootu.append(vinf_xshootui)
        
        # Apply metallicity scaling
        if self.config.metallicity == 'MWC':
            Z_value = 1.4285
        elif self.config.metallicity == 'MW':
            Z_value = 1.0
        elif self.config.metallicity == 'LMC':
            Z_value = 0.4285
        elif self.config.metallicity == 'SMC':
            Z_value = 0.1428
        elif self.config.metallicity == 'IZw18':
            Z_value = 0.0285
        elif self.config.metallicity == 'Z0':
            Z_value = 1e-5
        
        # Scale wind velocities by metallicity
        vinfs_Z = np.array(vinfs) * (Z_value**0.13)
        vinf_xshootu = np.array(vinf_xshootu) * (Z_value**0.22)
        
        # Apply weak wind effect
        timestep_mdot_weakwind = []
        for i in range(len(timestep_mdot)):
            if initial_masses[i] > 14.5 and timestep_lums[i] < 5.2:
                timestep_mdot_weakwind.append(timestep_mdot[i])
            else:
                timestep_mdot_weakwind.append(timestep_mdot[i])
        
        vinf_vink21 = np.array(vinf_vink21)
        mdot_vink00 = np.array(mdot_vink00)
        mdot_leuven = np.array(mdot_leuven)
        
        # Calculate wind power and momentum
        wind_power = (10.**timestep_mdot) * vinfs_Z**2 * 3.155
        wind_mom = 2 * (10.**timestep_mdot) * vinfs_Z * 3.155e-5
        wind_mdot = 10.**timestep_mdot
        wind_vinf = vinfs_Z
        wind_mom_vink = 2 * (10.**mdot_vink00) * vinf_vink21 * 3.155e-5
        wind_mom_leuven = 2 * (10.**mdot_leuven) * vinfs_Z * 3.155e-5
        wind_mom_xshootu = 2 * (10.**np.array(timestep_mdot_weakwind)) * vinf_xshootu * 3.155e-5
        wind_power_xshootu = (10.**np.array(timestep_mdot)) * vinf_xshootu**2 * 3.155
        
        # Scale by number of stars
        wind_power_calc = wind_power * No_stars
        wind_mom_calc = wind_mom * No_stars
        wind_mom_vink_calc = wind_mom_vink * No_stars
        wind_mom_leuven_calc = wind_mom_leuven * No_stars
        wind_mom_xshootu_calc = wind_mom_xshootu * No_stars
        wind_power_xshootu_calc = wind_power_xshootu * No_stars
        
        # Sum over all stars
        wind_power_sum = np.sum(wind_power_calc)
        wind_mom_sum = np.sum(wind_mom_calc)
        wind_mom_vink_sum = np.sum(wind_mom_vink_calc)
        wind_mom_leuven_sum = np.sum(wind_mom_leuven_calc)
        wind_mom_xshootu_sum = np.sum(wind_mom_xshootu_calc)
        wind_power_xshootu_sum = np.sum(wind_power_xshootu_calc)
        
        # Convert to log scale
        wpower = np.log10(wind_power_sum + 1e-30) + 35.
        wmom = np.log10(wind_mom_sum + 1e-30) + 35.
        wmom_vink = np.log10(wind_mom_vink_sum + 1e-30) + 35.
        wmom_leuven = np.log10(wind_mom_leuven_sum + 1e-30) + 35.
        wmom_xshootu = (wind_mom_xshootu_sum + 1e-30) + 35.
        wpower_xshootu = np.log10(wind_power_xshootu_sum + 1e-30) + 35.
        
        return (vinfs_Z, wind_power_calc, wpower, wind_mom_calc, wmom, wmom_vink, 
               vinf_vink21, mdot_vink00, timestep_vesc, wmom_leuven, 
               wmom_xshootu, wpower_xshootu)
    
    def _get_uv_slope(self, uv_wave, uv_flux):
        """
        Calculate UV slope.
        Equivalent to get_uv_slope in pySB99.py
        """
        # Define wavelength windows
        uvslope_bounds_1 = [3.103119254, 3.108565024]
        uvslope_bounds_2 = [3.116939647, 3.119255889]
        uvslope_bounds_3 = [3.127752516, 3.137037455]
        uvslope_bounds_4 = [3.156851901, 3.174931594]
        uvslope_bounds_5 = [3.19368103, 3.199480915]
        uvslope_bounds_6 = [3.224533063, 3.240549248]
        
        # Find indices within each window
        ind_uvslope_1 = [i for i in range(len(uv_wave)) 
                        if uv_wave[i] >= uvslope_bounds_1[0] and uv_wave[i] <= uvslope_bounds_1[1]]
        ind_uvslope_2 = [i for i in range(len(uv_wave)) 
                        if uv_wave[i] >= uvslope_bounds_2[0] and uv_wave[i] <= uvslope_bounds_2[1]]
        ind_uvslope_3 = [i for i in range(len(uv_wave)) 
                        if uv_wave[i] >= uvslope_bounds_3[0] and uv_wave[i] <= uvslope_bounds_3[1]]
        ind_uvslope_4 = [i for i in range(len(uv_wave)) 
                        if uv_wave[i] >= uvslope_bounds_4[0] and uv_wave[i] <= uvslope_bounds_4[1]]
        ind_uvslope_5 = [i for i in range(len(uv_wave)) 
                        if uv_wave[i] >= uvslope_bounds_5[0] and uv_wave[i] <= uvslope_bounds_5[1]]
        ind_uvslope_6 = [i for i in range(len(uv_wave)) 
                        if uv_wave[i] >= uvslope_bounds_6[0] and uv_wave[i] <= uvslope_bounds_6[1]]
        
        # Combine all indices
        ind_uvslope_full = np.concatenate((ind_uvslope_1, ind_uvslope_2, ind_uvslope_3, 
                                         ind_uvslope_4, ind_uvslope_5, ind_uvslope_6))
        
        # Extract wavelength and flux within windows
        wave_uv_slope = uv_wave[ind_uvslope_full]
        flux_uv_slope = uv_flux[ind_uvslope_full]
        
        # Fit linear function
        try:
            popt_uvslope, pcov_uvslope = curve_fit(f, wave_uv_slope, flux_uv_slope)
            x_uvslope = np.linspace(3.05, 3.3, 1000)
            y_uvslope = popt_uvslope[0] * x_uvslope + popt_uvslope[1]
            beta_uvslope = popt_uvslope[0]
        except:
            x_uvslope = np.linspace(3.05, 3.3, 1000)
            y_uvslope = x_uvslope * 0
            beta_uvslope = np.nan
        
        return x_uvslope, y_uvslope, beta_uvslope
    
    def _get_ew(self, uv_wave, uv_flux, HI_ionflux):
        """
        Calculate equivalent widths.
        Equivalent to get_ew in pySB99.py
        """
        # Interpolate continuum flux at line wavelengths
        Ha_continuum_flux = np.interp(6563., uv_wave, uv_flux)
        Hb_continuum_flux = np.interp(4861., uv_wave, uv_flux)
        Pb_continuum_flux = np.interp(12818., uv_wave, uv_flux)
        Bg_continuum_flux = np.interp(21655., uv_wave, uv_flux)
        
        # Calculate line luminosities
        Ha_luminosity = 1.36e-12 * 10**(HI_ionflux-30.)
        Hb_luminosity = 4.76e-13 * 10**(HI_ionflux-30.)
        Pb_luminosity = 7.73e-14 * 10**(HI_ionflux-30.)
        Bg_luminosity = 1.31e-14 * 10**(HI_ionflux-30.)
        
        # Calculate equivalent widths
        Ha_ew = Ha_luminosity / Ha_continuum_flux * 1.0e10
        Hb_ew = Hb_luminosity / Hb_continuum_flux * 1.0e10
        Pb_ew = Pb_luminosity / Pb_continuum_flux * 1.0e10
        Bg_ew = Bg_luminosity / Bg_continuum_flux * 1.0e10
        
        # Convert to log scale
        Ha_ew_unit = np.log10(Ha_ew + 1.0e-35)
        Hb_ew_unit = np.log10(Hb_ew + 1.0e-35)
        Pb_ew_unit = np.log10(Pb_ew + 1.0e-35)
        Bg_ew_unit = np.log10(Bg_ew + 1.0e-35)
        
        return Ha_ew_unit, Hb_ew_unit, Pb_ew_unit, Bg_ew_unit, Ha_luminosity, Ha_continuum_flux
    
    def _colours(self, population_flux):
        """
        Calculate photometric colors.
        Equivalent to colours in pySB99.py
        """
        # Define filter profiles
        V_wave = [50.,4750.,4800.,4850.,4900.,4950.,5000.,5050.,5100.,5150.,5200.,5250.,5300.,5350.,5400.,5450.,5500.,5550.,5600.,5650.,5700.,
                 5750.,5800.,5850.,5900.,5950.,6000.,6050.,6100.,6150.,6200.,6250.,6300.,6350.,6400.,6450.,6500.,6550.,6600.,6650.,6700.,
                 6750.,6800.,6850.,6900.,6950.,7000.,7050.,7100.,7150.,7200.,7250.,7300.,7350.,7400.,2000000.]
        V_profile = [0.000,0.000,0.030,0.084,0.163,0.301,0.458,0.630,0.780,0.895,0.967,0.997,1.000,0.988,0.958,0.919,0.877,0.819,0.765,0.711,0.657,
                    0.602,0.545,0.488,0.434,0.386,0.331,0.289,0.250,0.214,0.181,0.151,0.120,0.093,0.069,0.051,0.036,0.027,0.021,0.018,0.016,
                    0.014,0.012,0.011,0.010,0.009,0.008,0.007,0.006,0.005,0.004,0.003,0.002,0.001,0.000,0.000]
        
        U_wave = [50.,3050.,3100.,3150.,3200.,3250.,3300.,3350.,3400.,3450.,3500.,3550.,3600.,3650.,3700.,3750.,3800.,3850.,3900.,3950.,4000.,
                 4050.,4100.,4150.,4200.,2000000.]
        U_profile = [0.000,0.000,0.020,0.077,0.135,0.204,0.282,0.385,0.493,0.600,0.705,0.820,0.900,0.959,0.993,1.000,0.975,0.850,0.645,0.400,0.223,
                    0.125,0.057,0.005,0.000,0.000]
        
        I_wave = [50.,7000.,7100.,7200.,7300.,7400.,7500.,7600.,7700.,7800.,7900.,8000.,8100.,8200.,8300.,8400.,8500.,8600.,8700.,8800.,8900.,
                 9000.,9100.,9200.,2000000.]
        I_profile = [0.00000,0.00000,0.02400,0.23200,0.55500,0.78500,0.91000,0.96500,0.98500,0.99000,0.99500,1.00000,1.00000,0.99000,0.98000,
                    0.95000,0.91000,0.86000,0.75000,0.56000,0.33000,0.15000,0.03000,0.00000,0.00000]
        
        B_wave = [50.,3600.,3650.,3700.,3750.,3800.,3850.,3900.,3950.,4000.,4050.,4100.,4150.,4200.,4250.,4300.,4350.,4400.,4450.,4500.,4550.,
                 4600.,4650.,4700.,4750.,4800.,4850.,4900.,4950.,5000.,5050.,5100.,5150.,5200.,5250.,5300.,5350.,5400.,5450.,5500.,5550.,2000000.]
        B_profile = [0.000,0.000,0.006,0.030,0.060,0.134,0.302,0.567,0.841,0.959,0.983,0.996,1.000,0.996,0.987,0.974,0.957,0.931,0.897,0.849,0.800,
                    0.748,0.698,0.648,0.597,0.545,0.497,0.447,0.397,0.345,0.297,0.252,0.207,0.166,0.129,0.095,0.069,0.043,0.024,0.009,0.000,0.000]
        
        # Interpolate filter profiles onto spectrum wavelength grid
        V_interp = np.interp(self.wavelength_grid, V_wave, V_profile)
        U_interp = np.interp(self.wavelength_grid, U_wave, U_profile)
        I_interp = np.interp(self.wavelength_grid, I_wave, I_profile)
        B_interp = np.interp(self.wavelength_grid, B_wave, B_profile)
        
        # Multiply spectrum by filter profiles
        V_data = population_flux * V_interp
        U_data = population_flux * U_interp
        I_data = population_flux * I_interp
        B_data = population_flux * B_interp
        
        # Integrate flux through each filter
        V_value = np.trapz(self.wavelength_grid, V_data)
        U_value = np.trapz(self.wavelength_grid, U_data)
        I_value = np.trapz(self.wavelength_grid, I_data)
        B_value = np.trapz(self.wavelength_grid, B_data)
        
        # Convert to magnitudes
        V_mag = (-2.5 * np.log10(-1.0*V_value))
        U_mag = (-2.5 * np.log10(-1.0*U_value))
        I_mag = (-2.5 * np.log10(-1.0*I_value))
        B_mag = (-2.5 * np.log10(-1.0*B_value))
        
        absV_mag = (-2.5 * np.log10(-1.0*V_value)) + 36.552
        
        return V_mag, U_mag, I_mag, B_mag, absV_mag
    
    def _compute_radii(self, temps, lums):
        """
        Compute stellar radii from temperature and luminosity.
        Equivalent to compute_radii in pySB99.py
        """
        sigma = 5.670 * 10**(-8)  # Stefan-Boltzmann constant
        radii = []
        lums_sol = 10**np.array(lums) * 3.828*10**26
        teffs = 10**temps
        
        for i in range(len(teffs)):
            if temps[i] == 0.0:
                R = 0.1
            else:
                R = (lums_sol[i] / (4 * np.pi * sigma * teffs[i]**4))**0.5
                R = R / (7*10**8)  # Convert to solar radii
            radii.append(R)
        
        return radii

    def _create_custom_star_numbers(self, available_masses, custom_star_numbers):
        """
        Create a star number array from user-provided custom star counts.
        
        Parameters
        ----------
        available_masses : np.ndarray
            Mass grid from stellar evolution tracks
        custom_star_numbers : Dict[float, int]
            Mapping of stellar mass to number of stars
        
        Returns
        -------
        np.ndarray
            Array of star numbers matching the available_masses grid
        """
        # convert to np array
        available_masses = np.asarray(available_masses)

        # Initialize array with zeros
        star_numbers = np.zeros_like(available_masses)
        
        # Map the custom star numbers to the closest available masses
        for mass, count in custom_star_numbers.items():
            # Find closest mass in the available grid
            idx = np.abs(available_masses - mass).argmin()
            closest_mass = available_masses[idx]
            
            # Check if the mass difference is significant
            if abs(closest_mass - mass) / mass > 0.1:  # More than 10% difference
                warnings.warn(
                    f"Requested mass {mass:.1f} Msun not available in grid. " 
                    f"Using closest available mass {closest_mass:.1f} Msun."
                )
            
            print(f"  Mapping {mass:.1f} Msun -> {closest_mass:.1f} Msun (N={count})")
            
            # Set the star count
            star_numbers[idx] = count
        
        return star_numbers

    def calc_Nostars(self, IMF_masses, IMF_exponents, IMF_mass_limits):
        """
        Calculate number of stars from IMF.
        Equivalent to calc_Nostars in pySB99.py
        """
        # Determine number of IMF intervals
        if len(IMF_exponents) == 1:
            N_IMF_intervals = 1
        elif len(IMF_exponents) > 1:
            N_IMF_intervals = 2  # IMPORTANT: Always 2 in original code
        
        # Compute normalization constants
        if N_IMF_intervals > 1:
            A_ic = np.zeros_like(IMF_exponents)
            A_ic[0] = 1.
            
            for exponent_index in range(len(IMF_exponents)):
                exponent_index = exponent_index + 1  # skip first IMF_exponent
                if exponent_index == len(IMF_exponents):
                    break  # stop at the end
                
                A_i = A_ic[exponent_index-1] * (IMF_mass_limits[exponent_index]**(IMF_exponents[exponent_index] - IMF_exponents[exponent_index-1]))
                A_ic[exponent_index] = A_i
            
            k_ic = []
            for exponent_index in range(len(IMF_exponents)):
                if IMF_exponents[exponent_index] == 2.0:
                    k_i = np.log(IMF_mass_limits[exponent_index+1]) - np.log(IMF_mass_limits[exponent_index])
                else:
                    k_i = (IMF_mass_limits[exponent_index+1]**(2.0 - IMF_exponents[exponent_index]) - 
                          IMF_mass_limits[exponent_index]**(2.0 - IMF_exponents[exponent_index])) / (2 - IMF_exponents[exponent_index])
                k_ic.append(k_i)
            
            Ak_ic = A_ic * k_ic
            Ak = sum(Ak_ic)
        
        else:  # N_IMF_intervals = 1
            A_ic = 1
            k_ic = (IMF_mass_limits[1]**(2.0 - IMF_exponents[0]) - 
                  IMF_mass_limits[0]**(2.0 - IMF_exponents[0])) / (2 - IMF_exponents[0])
            Ak = A_ic * k_ic
        
        S = self.M_total / Ak
        A = A_ic * S  # Normalization constants
        
        # Calculate mass bin boundaries
        xmhigh = np.full_like(IMF_masses, 0)
        xmlow = np.full_like(IMF_masses, 0)
        
        for mass_index in range(len(IMF_masses)):
            if IMF_masses[mass_index] == min(IMF_masses):
                xmhigh[mass_index] = 0.5 * (IMF_masses[mass_index] + IMF_masses[mass_index+1])
                xmlow[mass_index] = IMF_masses[mass_index]
            elif IMF_masses[mass_index] == max(IMF_masses):
                xmhigh[mass_index] = IMF_masses[mass_index]
                xmlow[mass_index] = 0.5 * (IMF_masses[mass_index-1] + IMF_masses[mass_index])
            else:
                xmhigh[mass_index] = 0.5 * (IMF_masses[mass_index] + IMF_masses[mass_index+1])
                xmlow[mass_index] = 0.5 * (IMF_masses[mass_index] + IMF_masses[mass_index-1])
        
        dens = []
        
        # Calculate number density in each mass bin
        if N_IMF_intervals == 2:
            for mass_index in range(len(IMF_masses)):
                dens_i = A[1] * (xmhigh[mass_index]**(1-IMF_exponents[1]) - 
                               xmlow[mass_index]**(1-IMF_exponents[1])) / (1 - IMF_exponents[1])
                dens.append(dens_i)
        else:
            for mass_index in range(len(IMF_masses)):
                dens_i = A * (xmhigh[mass_index]**(1-IMF_exponents[0]) - 
                            xmlow[mass_index]**(1-IMF_exponents[0])) / (1 - IMF_exponents[0])
                dens.append(dens_i)
        
        N_stars = np.array(dens)
        total_mass = N_stars * IMF_masses
        
        print('Total No stars = ', '{:0.4E}'.format(sum(N_stars).item()), 
              'Total mass = ', '{:0.4E}'.format(np.sum(total_mass)))
        
        return N_stars, IMF_masses, dens, xmhigh, xmlow
    
    def _create_results_object(self, results: Dict, n_stars: np.ndarray) -> StellarPopulationResults:
        """Convert calculation results to StellarPopulationResults object."""
        return StellarPopulationResults(
            times=results['times'],
            wavelength_grid=self.wavelength_grid,
            flux_spectra=np.array(results['population_flux_iterations']),
            flux_spectra_with_nebular=np.array(results['population_flux_total_iterations']),
            bolometric_luminosity=np.array(results['population_ion_L_flux_iterations']),
            hi_ionizing_flux=np.array(results['population_ion_HI_flux_iterations']),
            hei_ionizing_flux=np.array(results['population_ion_HEI_flux_iterations']),
            heii_ionizing_flux=np.array(results['population_ion_HEII_flux_iterations']),
            hi_ionizing_luminosity=np.array(results['population_ion_HI_lum_iterations']),
            hei_ionizing_luminosity=np.array(results['population_ion_HEI_lum_iterations']),
            heii_ionizing_luminosity=np.array(results['population_ion_HEII_lum_iterations']),
            wind_power=np.array(results['population_windpowers']),
            wind_momentum=np.array(results['population_windmoms']),
            uv_slope_beta=np.array(results['population_uv_slopes_beta']),
            ha_equivalent_width=np.array(results['population_Ha_ew']),
            hb_equivalent_width=np.array(results['population_Hb_ew']),
            pb_equivalent_width=np.array(results['population_Pb_ew']),
            bg_equivalent_width=np.array(results['population_Bg_ew']),
            v_magnitude=np.array(results['population_Vmag']),
            u_magnitude=np.array(results['population_Umag']),
            i_magnitude=np.array(results['population_Imag']),
            b_magnitude=np.array(results['population_Bmag']),
            abs_v_magnitude=np.array(results['population_absVmag']),
            stellar_masses=results['population_masses'],
            stellar_temperatures=results['population_temps'],
            stellar_luminosities=results['population_lums'],
            number_of_stars=n_stars
        )


def create_default_config(**kwargs) -> StellarPopulationConfig:
    """
    Create a default configuration with optional parameter overrides.
    
    Parameters
    ----------
    **kwargs
        Configuration parameters to override defaults
        
    Returns
    -------
    StellarPopulationConfig
        Configuration object
    """
    config = StellarPopulationConfig()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            warnings.warn(f"Unknown configuration parameter: {key}")
    
    return config


def run_population_synthesis(config: Optional[StellarPopulationConfig] = None, **kwargs) -> StellarPopulationResults:
    """
    Run stellar population synthesis with given configuration.
    
    Parameters
    ----------
    config : StellarPopulationConfig, optional
        Configuration object. If

    None, uses default with kwargs overrides.
    **kwargs
        Configuration parameters to override (if config is None)
        
    Returns
    -------
    StellarPopulationResults
        Complete results of the population synthesis
    """
    if config is None:
        config = create_default_config(**kwargs)
    
    # Initialize and run synthesis
    synthesizer = StellarPopulationSynthesis(config)
    results = synthesizer.calculate_population()
    
    return results


def save_results(results: StellarPopulationResults, output_dir: str, model_name: str = 'pySB_model') -> None:
    """
    Save results to files.
    
    Parameters
    ----------
    results : StellarPopulationResults
        Results to save
    output_dir : str
        Output directory path
    model_name : str
        Model name for file naming
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save key results as text files
    np.savetxt(os.path.join(output_dir, f'{model_name}_times.txt'), results.times)
    np.savetxt(os.path.join(output_dir, f'{model_name}_bololum.txt'), results.bolometric_luminosity)
    np.savetxt(os.path.join(output_dir, f'{model_name}_hi_ionflux.txt'), results.hi_ionizing_flux)
    np.savetxt(os.path.join(output_dir, f'{model_name}_hei_ionflux.txt'), results.hei_ionizing_flux)
    np.savetxt(os.path.join(output_dir, f'{model_name}_heii_ionflux.txt'), results.heii_ionizing_flux)
    np.savetxt(os.path.join(output_dir, f'{model_name}_hi_ionlum.txt'), results.hi_ionizing_luminosity)
    np.savetxt(os.path.join(output_dir, f'{model_name}_hei_ionlum.txt'), results.hei_ionizing_luminosity)
    np.savetxt(os.path.join(output_dir, f'{model_name}_heii_ionlum.txt'), results.heii_ionizing_luminosity)
    np.savetxt(os.path.join(output_dir, f'{model_name}_windpower.txt'), results.wind_power)
    np.savetxt(os.path.join(output_dir, f'{model_name}_windmom.txt'), results.wind_momentum)
    np.savetxt(os.path.join(output_dir, f'{model_name}_uv_slope.txt'), results.uv_slope_beta)
    np.savetxt(os.path.join(output_dir, f'{model_name}_ha_ew.txt'), results.ha_equivalent_width)
    np.savetxt(os.path.join(output_dir, f'{model_name}_hb_ew.txt'), results.hb_equivalent_width)
    np.savetxt(os.path.join(output_dir, f'{model_name}_pb_ew.txt'), results.pb_equivalent_width)
    np.savetxt(os.path.join(output_dir, f'{model_name}_bg_ew.txt'), results.bg_equivalent_width)
    
    # Save spectra as NumPy arrays
    np.save(os.path.join(output_dir, f'{model_name}_wavelength.npy'), results.wavelength_grid)
    np.save(os.path.join(output_dir, f'{model_name}_spectra.npy'), results.flux_spectra)
    np.save(os.path.join(output_dir, f'{model_name}_spectra_nebular.npy'), results.flux_spectra_with_nebular)
    
    # Save combined SEDs (as in original code)
    stacked_seds = np.column_stack((
        safe_log10(results.flux_spectra) + 20.0,
        safe_log10(results.flux_spectra_with_nebular) + 20.0
    ))
    np.save(os.path.join(output_dir, f'{model_name}_SEDs.npy'), stacked_seds)
    
    # Save input parameters
    with open(os.path.join(output_dir, 'input.txt'), 'w') as inputs_file:
        inputs_file.write(f'M_total = {results.number_of_stars.sum()}Msol\n')
        # More parameters could be saved here
    
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    print("pySB99 Core Module")
    print("=================")
    print("This module provides the core functionality for stellar population synthesis.")
    print("Use run_population_synthesis() to perform calculations.")
    print("\nExample:")
    print(">>> from pysb99_core import run_population_synthesis")
    print(">>> results = run_population_synthesis(total_mass=1e6, metallicity='MW')")