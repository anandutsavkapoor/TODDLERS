#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pySB99 Plotting Module
======================

Visualization tools for stellar population synthesis results.
This module provides comprehensive plotting capabilities for analyzing
stellar population properties including spectra, ionizing fluxes,
wind properties, and photometric evolution.

Author: Anand Utsav KAPOOR
Refactored from https://github.com/CalumHawcroft/Starburst/tree/main and 
described in https://arxiv.org/html/2505.24841v1
"""

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from .pysb99_core import StellarPopulationResults, StellarPopulationConfig, safe_log10

# Set publication-quality plotting defaults
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times', 'Computer Modern Roman'],
    'text.usetex': False,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'lines.linewidth': 2,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5
})

class CustomPopulationPlotter:
    """
    Specialized plotter for custom stellar populations.
    
    This class provides visualization tools for populations created with
    custom star distributions rather than standard IMFs.
    """
    
    def __init__(self, results: StellarPopulationResults, 
                config: StellarPopulationConfig, 
                custom_star_numbers: Dict[float, int],
                available_masses: Optional[List[float]] = None):
        """
        Initialize plotter with results and custom star distribution.
        
        Parameters
        ----------
        results : StellarPopulationResults
            Results from population synthesis
        config : StellarPopulationConfig
            Configuration used for the run
        custom_star_numbers : Dict[float, int]
            Dictionary mapping stellar mass to number of stars
        available_masses : List[float], optional
            List of available masses in the model grid
        """
        self.results = results
        self.config = config
        self.custom_star_numbers = custom_star_numbers
        self.available_masses = available_masses
        
        # Set up color schemes - using same approach as PopulationPlotter
        self.colors = self._setup_color_schemes()
        
        # Calculate total stars and mass
        self.total_stars = sum(custom_star_numbers.values())
        self.total_mass = sum(mass * count for mass, count in custom_star_numbers.items())
        
    def _setup_color_schemes(self) -> Dict[str, Any]:
        """Set up publication-quality color schemes."""
        # Use colorbrewer qualitative palette
        qual_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Sequential colormap for mass evolution
        seq_cmap = plt.cm.viridis
        
        return {
            'qualitative': qual_colors,
            'sequential': seq_cmap,
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'accent': '#d62728',
            'masses': plt.cm.plasma  # Colormap for different masses
        }
    
    def plot_distribution(self, figsize: Tuple[float, float] = (10, 10), 
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive visualization of the custom star distribution.
        
        Parameters
        ----------
        figsize : Tuple[float, float]
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig = plt.figure(figsize=figsize)
        
        # Create 2x2 grid layout
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Sort masses for consistent plotting
        masses = sorted(self.custom_star_numbers.keys())
        counts = [self.custom_star_numbers[m] for m in masses]
        mass_fracs = [(m * self.custom_star_numbers[m] / self.total_mass) * 100 for m in masses]
        count_fracs = [(self.custom_star_numbers[m] / self.total_stars) * 100 for m in masses]
        
        # 1. Star counts bar chart
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.bar(masses, counts, width=masses[0]*0.2 if len(masses) > 0 else 1.0, 
                      color=self.colors['qualitative'], edgecolor='black', alpha=0.7)
        
        ax1.set_xlabel('Stellar Mass (M$_\\odot$)')
        ax1.set_ylabel('Number of Stars')
        ax1.set_title('Star Count Distribution')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1*max(counts),
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
        
        # 2. Mass contribution (pie chart)
        ax2 = fig.add_subplot(gs[0, 1])
        mass_contributions = [m * self.custom_star_numbers[m] for m in masses]
        
        # Create labels with percentages
        labels = [f'{m:.1f} M$_\\odot$ ({(m*self.custom_star_numbers[m]/self.total_mass)*100:.1f}%)' 
                 for m in masses]
        
        ax2.pie(mass_contributions, labels=labels, autopct='%1.1f%%',
               colors=self.colors['qualitative'], startangle=90, shadow=True,
               wedgeprops={'edgecolor': 'black', 'linewidth': 0.5, 'antialiased': True})
        
        ax2.set_title('Contribution to Total Mass')
        
        # 3. Combined percentage comparison
        ax3 = fig.add_subplot(gs[1, 0])
        x = np.arange(len(masses))
        width = 0.35
        
        bars2 = ax3.bar(x - width/2, mass_fracs, width, color='tomato', edgecolor='darkred', 
                       alpha=0.7, label='% of Total Mass')
        bars3 = ax3.bar(x + width/2, count_fracs, width, color='mediumseagreen', edgecolor='darkgreen', 
                       alpha=0.7, label='% of Total Stars')
        
        ax3.set_xlabel('Stellar Mass (M$_\\odot$)')
        ax3.set_ylabel('Percentage (%)')
        ax3.set_title('Mass vs. Number Contribution')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'{m:.1f}' for m in masses])
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend()
        
        # Add value labels on top of bars
        for bar in bars2:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=8)
        
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=8)
        
        # 4. Information panel with key statistics
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')  # No axes for text panel
        
        # Create information text
        info_text = [
            "CUSTOM POPULATION SUMMARY",
            "========================",
            f"Total stars: {self.total_stars}",
            f"Total mass: {self.total_mass:.2e} M$_\\odot$",
            f"Average mass: {self.total_mass/self.total_stars:.2f} M$_\\odot$",
            "",
            "Mass Distribution:",
        ]
        
        # Add per-mass info
        for mass in masses:
            count = self.custom_star_numbers[mass]
            info_text.append(f"  {mass:.1f} M$_\\odot$: {count} stars")
        
        # If available_masses are provided, add info about mass mapping
        if self.available_masses is not None:
            available_masses = np.asarray(self.available_masses)
            
            info_text.append("")
            info_text.append("Mass Mapping to Available Grid:")
            
            for mass in masses:
                idx = np.abs(available_masses - mass).argmin()
                closest = available_masses[idx]
                diff_pct = abs(closest - mass) / mass * 100
                
                if diff_pct < 1:
                    status = "exact match"
                elif diff_pct < 5:
                    status = "close match"
                else:
                    status = f"significant difference ({diff_pct:.1f}%)"
                
                info_text.append(f"  {mass:.1f} M$_\\odot$ → {closest:.1f} M$_\\odot$ ({status})")
        
        # Add text to the plot
        ax4.text(0.05, 0.95, "\n".join(info_text), transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        # Add title
        fig.suptitle(f"Custom Stellar Population Analysis\n{self.config.metallicity} Metallicity, " + 
                    f"{'With' if self.config.rotation else 'Without'} Rotation", 
                    fontsize=16, y=0.98)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Custom population visualization saved to {save_path}")
        
        return fig
    
    def plot_contribution_by_mass(self, quantities: List[str] = None,
                                figsize: Tuple[float, float] = (12, 8),
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the contribution of each mass bin to various physical quantities.
        
        Parameters
        ----------
        quantities : List[str], optional
            List of quantities to plot. If None, uses a default set.
        figsize : Tuple[float, float]
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        if quantities is None:
            quantities = ['wind_power', 'hi_ionizing_flux', 'bolometric_luminosity']
        
        fig, axes = plt.subplots(len(quantities), 1, figsize=figsize, sharex=True)
        if len(quantities) == 1:
            axes = [axes]  # Make it iterable
            
        # Sort masses for consistent plotting
        masses = sorted(self.custom_star_numbers.keys())
        
        # Create a color map for different masses
        colors = self.colors['masses'](np.linspace(0, 1, len(masses)))
        
        # Time in Myr for x-axis
        time_myr = self.results.times / 1e6
        
        # Plot each quantity
        for i, quantity in enumerate(quantities):
            ax = axes[i]
            
            # Get the appropriate quantity data
            if quantity == 'wind_power':
                data = self.results.wind_power
                title = 'Wind Power'
                ylabel = r'$\log(\dot{E}_{\rm wind}/{\rm erg\,s}^{-1})$'
            elif quantity == 'hi_ionizing_flux':
                data = self.results.hi_ionizing_flux
                title = 'HI Ionizing Flux'
                ylabel = r'$\log(Q_{\rm HI}/{\rm s}^{-1})$'
            elif quantity == 'bolometric_luminosity':
                data = self.results.bolometric_luminosity
                title = 'Bolometric Luminosity'
                ylabel = r'$\log(L_{\rm bol}/{\rm erg\,s}^{-1})$'
            elif quantity == 'wind_momentum':
                data = self.results.wind_momentum
                title = 'Wind Momentum'
                ylabel = r'$\log(\dot{p}_{\rm wind}/{\rm dyne})$'
            elif quantity == 'uv_slope_beta':
                data = self.results.uv_slope_beta
                title = 'UV Slope'
                ylabel = r'$\beta_{\rm UV}$'
            else:
                # Skip unknown quantities
                ax.text(0.5, 0.5, f"Unknown quantity: {quantity}", 
                       transform=ax.transAxes, ha='center', va='center')
                continue
                
            # Plot the total quantity
            ax.plot(time_myr, data, color='black', linewidth=2.5, 
                   label='Total', zorder=10)
            
            # Add individual mass contributions if available
            # This is a placeholder - actual contribution data would need to be
            # extracted from the results if available
            
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Add legend to last plot
            if i == len(quantities) - 1:
                ax.legend()
                ax.set_xlabel('Time (Myr)')
        
        fig.suptitle("Physical Quantities Evolution by Stellar Mass", fontsize=16)
        fig.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Mass contribution plot saved to {save_path}")
        
        return fig
    
    def plot_hr_diagram_by_mass(self, time_myr: float = 5.0,
                              figsize: Tuple[float, float] = (10, 8),
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Hertzsprung-Russell diagram at a specific time, colored by initial mass.
        
        Parameters
        ----------
        time_myr : float
            Time in Myr for the HR diagram
        figsize : Tuple[float, float]
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Find closest time index
        time_target_yr = time_myr * 1e6
        time_idx = np.argmin(np.abs(self.results.times - time_target_yr))
        actual_time = self.results.times[time_idx] / 1e6
        
        # Get stellar properties at this time
        if hasattr(self.results, 'stellar_temperatures') and len(self.results.stellar_temperatures) > time_idx:
            temperatures = self.results.stellar_temperatures[time_idx]
            luminosities = self.results.stellar_luminosities[time_idx]
            masses = self.results.stellar_masses[time_idx]
            
            # Use a different marker size for different counts
            marker_sizes = []
            colors = []
            
            # Sort masses for consistent coloring
            unique_masses = sorted(self.custom_star_numbers.keys())
            mass_to_color = {m: self.colors['masses'](i/len(unique_masses)) 
                           for i, m in enumerate(unique_masses)}
            
            # Filter out dead stars
            alive_mask = np.array(luminosities) > -19
            temperatures_alive = np.array(temperatures)[alive_mask]
            luminosities_alive = np.array(luminosities)[alive_mask]
            masses_alive = np.array(masses)[alive_mask]
            
            if len(temperatures_alive) > 0:
                # Create scatter plot with custom colors and sizes
                for i, (temp, lum, mass) in enumerate(zip(temperatures_alive, luminosities_alive, masses_alive)):
                    # Find closest mass in custom distribution
                    closest_mass = min(unique_masses, key=lambda m: abs(m - mass))
                    
                    # Get color and size based on this mass
                    color = mass_to_color[closest_mass]
                    size = 50 + 100 * (self.custom_star_numbers[closest_mass] / self.total_stars)
                    
                    ax.scatter(temp, lum, s=size, color=color, edgecolors='black', 
                              linewidth=0.5, alpha=0.7, zorder=i)
                
                # Add legend for masses
                legend_elements = []
                for mass in unique_masses:
                    count = self.custom_star_numbers[mass]
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                    markerfacecolor=mass_to_color[mass],
                                                    markeredgecolor='black', 
                                                    markersize=10, 
                                                    label=f'{mass:.1f} M$_\\odot$ ({count} stars)'))
                
                ax.legend(handles=legend_elements, title='Initial Mass', 
                         loc='upper left', frameon=True, fancybox=True, shadow=True)
                
                ax.invert_xaxis()  # Hot stars on the left
                ax.set_xlabel(r'$\log(T_{\rm eff}/{\rm K})$')
                ax.set_ylabel(r'$\log(L/L_\odot)$')
                ax.set_title(f'Hertzsprung-Russell Diagram at {actual_time:.1f} Myr')
                ax.grid(True, alpha=0.3)
                
                # Add main sequence line
                self._add_evolution_tracks(ax)
            else:
                ax.text(0.5, 0.5, 'No living stars at this time', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=16)
        else:
            ax.text(0.5, 0.5, 'Stellar data not available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=16)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"HR diagram saved to {save_path}")
        
        return fig
    
    def _add_evolution_tracks(self, ax: plt.Axes) -> None:
        """Add theoretical evolution tracks to HR diagram."""
        # Approximate main sequence
        ms_teff = np.linspace(3.5, 4.7, 100)
        ms_lum = 4.0 * (ms_teff - 3.76) + 0.5  # Approximate MS relation
        ax.plot(ms_teff, ms_lum, 'k--', alpha=0.5, linewidth=1, label='Main Sequence')
        
        # Approximate giant branch
        gb_teff = np.linspace(3.5, 3.8, 50)
        gb_lum = -10.0 * (gb_teff - 3.8) + 3.0
        ax.plot(gb_teff, gb_lum, 'r--', alpha=0.5, linewidth=1, label='Giant Branch')
        
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    def create_summary_plot(self, figsize: Tuple[float, float] = (15, 12), 
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive summary plot for the custom stellar population.
        
        Parameters
        ----------
        figsize : Tuple[float, float]
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig = plt.figure(figsize=figsize)
        
        # Create layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Star distribution (upper left)
        ax1 = fig.add_subplot(gs[0, 0])
        masses = sorted(self.custom_star_numbers.keys())
        counts = [self.custom_star_numbers[m] for m in masses]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(masses))
        ax1.barh(y_pos, counts, color=self.colors['qualitative'], 
                height=0.7, edgecolor='black', alpha=0.8)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f'{m:.1f} M$_\\odot$' for m in masses])
        ax1.invert_yaxis()  # To have largest mass at the top
        ax1.set_xlabel('Number of Stars')
        ax1.set_title('Custom Star Distribution')
        
        # Add value labels
        for i, count in enumerate(counts):
            ax1.text(count + 0.5, i, str(count), va='center')
        
        # 2. Ionizing flux evolution (upper middle)
        ax2 = fig.add_subplot(gs[0, 1])
        time_myr = self.results.times / 1e6
        
        ax2.plot(time_myr, self.results.hi_ionizing_flux, 
                color=self.colors['primary'], linewidth=2.5, label=r'H\,{\sc i}')
        ax2.plot(time_myr, self.results.hei_ionizing_flux, 
                color=self.colors['secondary'], linewidth=2.5, label=r'He\,{\sc i}')
        ax2.plot(time_myr, self.results.heii_ionizing_flux, 
                color=self.colors['accent'], linewidth=2.5, label=r'He\,{\sc ii}')
        
        ax2.set_xlabel('Time (Myr)')
        ax2.set_ylabel(r'$\log(Q/{\rm s}^{-1})$')
        ax2.set_title('Ionizing Photon Rates')
        ax2.legend(frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        
        # 3. Wind properties (upper right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3_twin = ax3.twinx()
        
        line1 = ax3.plot(time_myr, self.results.wind_power, 
                        color=self.colors['primary'], linewidth=2.5, label='Power')
        line2 = ax3_twin.plot(time_myr, self.results.wind_momentum, 
                             color=self.colors['secondary'], linewidth=2.5, label='Momentum')
        
        ax3.set_xlabel('Time (Myr)')
        ax3.set_ylabel(r'$\log(\dot{E}_{\rm wind}/{\rm erg\,s}^{-1})$', color=self.colors['primary'])
        ax3_twin.set_ylabel(r'$\log(\dot{p}_{\rm wind}/{\rm dyne})$', color=self.colors['secondary'])
        ax3.set_title('Stellar Wind Properties')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='best')
        
        ax3.tick_params(axis='y', labelcolor=self.colors['primary'])
        ax3_twin.tick_params(axis='y', labelcolor=self.colors['secondary'])
        ax3.grid(True, alpha=0.3)
        
        # 4. HR diagram (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        time_hr = 5.0  # Fixed time for HR diagram (5 Myr)
        
        # Find closest time index
        time_idx = np.argmin(np.abs(self.results.times - time_hr * 1e6))
        
        # Get stellar properties
        if hasattr(self.results, 'stellar_temperatures') and len(self.results.stellar_temperatures) > time_idx:
            temperatures = self.results.stellar_temperatures[time_idx]
            luminosities = self.results.stellar_luminosities[time_idx]
            
            # Filter out dead stars
            alive_mask = np.array(luminosities) > -19
            temperatures_alive = np.array(temperatures)[alive_mask]
            luminosities_alive = np.array(luminosities)[alive_mask]
            
            if len(temperatures_alive) > 0:
                ax4.scatter(temperatures_alive, luminosities_alive, 
                           c='skyblue', edgecolors='navy', s=50, alpha=0.7)
                
                ax4.invert_xaxis()  # Hot stars on the left
                ax4.set_xlabel(r'$\log(T_{\rm eff}/{\rm K})$')
                ax4.set_ylabel(r'$\log(L/L_\odot)$')
                ax4.set_title(f'HR Diagram at {time_hr:.1f} Myr')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No living stars at this time', 
                       transform=ax4.transAxes, ha='center', va='center',
                       fontsize=12)
        else:
            ax4.text(0.5, 0.5, 'Stellar data not available', 
                   transform=ax4.transAxes, ha='center', va='center',
                   fontsize=12)
        
        # 5. UV slope evolution (middle center)
        ax5 = fig.add_subplot(gs[1, 1])
        
        # Filter out NaN values
        valid_mask = ~np.isnan(self.results.uv_slope_beta)
        
        if np.any(valid_mask):
            ax5.plot(time_myr[valid_mask], self.results.uv_slope_beta[valid_mask], 
                    color=self.colors['primary'], linewidth=2.5, marker='o', markersize=4)
        
        ax5.set_xlabel('Time (Myr)')
        ax5.set_ylabel(r'$\beta_{\rm UV}$')
        ax5.set_title(r'UV Slope Evolution ($f_\lambda \propto \lambda^\beta$)')
        ax5.grid(True, alpha=0.3)
        
        # 6. Equivalent widths (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        
        ax6.plot(time_myr, self.results.ha_equivalent_width, 
                color=self.colors['primary'], linewidth=2.5, label=r'H$\alpha$')
        
        if hasattr(self.results, 'hb_equivalent_width'):
            ax6.plot(time_myr, self.results.hb_equivalent_width, 
                    color=self.colors['secondary'], linewidth=2.5, label=r'H$\beta$')
        
        ax6.set_xlabel('Time (Myr)')
        ax6.set_ylabel(r'$\log({\rm EW}/\AA)$')
        ax6.set_title('Hydrogen Line Equivalent Widths')
        ax6.legend(frameon=True, fancybox=True, shadow=True)
        ax6.grid(True, alpha=0.3)
        
        # 7. SED evolution (bottom)
        ax7 = fig.add_subplot(gs[2, :])
        
        # Select representative times
        times = [1, 5, 10, 30]
        colors = plt.cm.plasma(np.linspace(0, 1, len(times)))
        
        for i, t in enumerate(times):
            time_idx = np.argmin(np.abs(time_myr - t))
            flux = self.results.flux_spectra_with_nebular[time_idx]
            
            # Convert to log scale
            flux_log = safe_log10(flux)
            
            ax7.semilogx(self.results.wavelength_grid, flux_log, 
                       color=colors[i], linewidth=2, 
                       label=f'{time_myr[time_idx]:.1f} Myr')
        
        ax7.set_xlabel(r'Wavelength (\AA)')
        ax7.set_ylabel(r'$\log(f_\lambda)$ [arbitrary units]')
        ax7.set_title('Spectral Energy Distribution Evolution')
        ax7.legend(frameon=True, fancybox=True, shadow=True)
        ax7.grid(True, alpha=0.3)
        ax7.set_xlim(800, 7000)
        
        # Add title with population info
        title = (f"Custom Stellar Population: {self.total_stars} stars, " + 
                f"{self.total_mass:.2e} M$_\\odot$, {self.config.metallicity} Metallicity")
        fig.suptitle(title, fontsize=16, y=0.98)
        
        # Add summary stats
        info_text = [
            f"Population: {self.total_stars} stars, {self.total_mass:.2e} M$_\\odot$",
            f"Metallicity: {self.config.metallicity}, Rotation: {'Yes' if self.config.rotation else 'No'}",
            f"Masses: " + ", ".join([f"{m:.1f} M$_\\odot$ ({self.custom_star_numbers[m]})" for m in sorted(self.custom_star_numbers.keys())])
        ]
        
        fig.text(0.02, 0.01, "\n".join(info_text), fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Custom population summary plot saved to {save_path}")
        
        return fig

def plot_custom_population(results: StellarPopulationResults,
                       config: StellarPopulationConfig,
                       custom_star_numbers: Optional[Dict[float, int]] = None,
                       available_masses: Optional[List[float]] = None,
                       plot_type: str = 'summary',
                       output_dir: Optional[str] = None,
                       base_filename: str = 'custom_population',
                       figsize: Tuple[float, float] = (12, 10)) -> Dict[str, plt.Figure]:
    """
    Create comprehensive visualizations for custom stellar populations.
    
    Parameters
    ----------
    results : StellarPopulationResults
        Results from population synthesis
    config : StellarPopulationConfig
        Configuration used for the run
    custom_star_numbers : Dict[float, int], optional
        Dictionary mapping stellar mass to number of stars. If None, tries to extract from config.
    available_masses : List[float], optional
        List of available masses in the model grid. If None, tries to extract from config.
    plot_type : str
        Type of plot: 'summary', 'distribution', 'hr', 'contribution', or 'all'
    output_dir : str, optional
        Directory to save plots. If None, plots are not saved.
    base_filename : str
        Base filename for saved plots
    figsize : Tuple[float, float]
        Figure size
        
    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary mapping plot types to Matplotlib figures
        
    Examples
    --------
    >>> custom_stars = {20.0: 100, 40.0: 50, 85.0: 10}
    >>> results, config = run_standard_model(custom_star_numbers=custom_stars)
    >>> figures = plot_custom_population(results, config, plot_type='all', 
    ...                               output_dir='./custom_plots')
    """
    # Extract custom_star_numbers from config if not provided
    if custom_star_numbers is None:
        if hasattr(config, 'custom_star_numbers') and config.custom_star_numbers is not None:
            custom_star_numbers = config.custom_star_numbers
        else:
            raise ValueError("No custom star distribution found. Please provide custom_star_numbers.")
    
    # Extract available_masses from config if not provided
    if available_masses is None:
        if hasattr(config, 'available_masses'):
            available_masses = config.available_masses
    
    # Create the custom population plotter
    plotter = CustomPopulationPlotter(results, config, custom_star_numbers, available_masses)
    
    # Dictionary to store figures
    figures = {}
    
    # Create requested plots
    if plot_type in ['summary', 'all']:
        save_path = os.path.join(output_dir, f"{base_filename}_summary.pdf") if output_dir else None
        figures['summary'] = plotter.create_summary_plot(figsize=figsize, save_path=save_path)
    
    if plot_type in ['distribution', 'all']:
        save_path = os.path.join(output_dir, f"{base_filename}_distribution.pdf") if output_dir else None
        figures['distribution'] = plotter.plot_distribution(figsize=figsize, save_path=save_path)
    
    if plot_type in ['hr', 'all']:
        save_path = os.path.join(output_dir, f"{base_filename}_hr_diagram.pdf") if output_dir else None
        figures['hr'] = plotter.plot_hr_diagram_by_mass(time_myr=5.0, figsize=figsize, save_path=save_path)
    
    if plot_type in ['contribution', 'all']:
        save_path = os.path.join(output_dir, f"{base_filename}_contribution.pdf") if output_dir else None
        figures['contribution'] = plotter.plot_contribution_by_mass(figsize=figsize, save_path=save_path)
    
    return figures


def create_custom_population_report(results: StellarPopulationResults,
                                   config: StellarPopulationConfig,
                                   custom_star_numbers: Optional[Dict[float, int]] = None,
                                   output_dir: str = './custom_population_report',
                                   report_name: str = 'custom_population_report') -> str:
    """
    Create a comprehensive report with visualizations for a custom stellar population.
    
    Parameters
    ----------
    results : StellarPopulationResults
        Results from population synthesis
    config : StellarPopulationConfig
        Configuration used for the run
    custom_star_numbers : Dict[float, int], optional
        Dictionary mapping stellar mass to number of stars. If None, tries to extract from config.
    output_dir : str
        Directory to save the report
    report_name : str
        Base name for the report files
        
    Returns
    -------
    str
        Path to the report directory
        
    Examples
    --------
    >>> custom_stars = {20.0: 100, 40.0: 50, 85.0: 10}
    >>> results, config = run_standard_model(custom_star_numbers=custom_stars)
    >>> report_path = create_custom_population_report(results, config, 
    ...                                           output_dir='./custom_report')
    """
    # Extract custom_star_numbers from config if not provided
    if custom_star_numbers is None:
        if hasattr(config, 'custom_star_numbers') and config.custom_star_numbers is not None:
            custom_star_numbers = config.custom_star_numbers
        else:
            raise ValueError("No custom star distribution found. Please provide custom_star_numbers.")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all plots
    figures = plot_custom_population(results, config, custom_star_numbers,
                                   plot_type='all', output_dir=output_dir,
                                   base_filename=report_name)
    
    # Create a summary text file
    summary_path = os.path.join(output_dir, f"{report_name}_summary.txt")
    
    with open(summary_path, 'w') as f:
        # Header
        f.write("="*80 + "\n")
        f.write(f"CUSTOM STELLAR POPULATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Configuration
        f.write("MODEL CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Metallicity: {config.metallicity}\n")
        f.write(f"Spectral library: {config.spectral_library}\n")
        f.write(f"Rotation: {'Yes' if config.rotation else 'No'}\n")
        f.write(f"Time range: {config.time_start/1e6:.2f} - {config.time_end/1e6:.2f} Myr\n\n")
        
        # Star distribution
        total_stars = sum(custom_star_numbers.values())
        total_mass = sum(mass * count for mass, count in custom_star_numbers.items())
        
        f.write("CUSTOM STAR DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        f.write(f"Total stars: {total_stars}\n")
        f.write(f"Total mass: {total_mass:.2e} M☉\n")
        f.write(f"Average mass: {total_mass/total_stars:.2f} M☉\n\n")
        
        f.write("Mass distribution:\n")
        for mass in sorted(custom_star_numbers.keys()):
            count = custom_star_numbers[mass]
            mass_percent = (mass * count / total_mass) * 100
            count_percent = (count / total_stars) * 100
            f.write(f"  {mass:.1f} M☉: {count} stars ({count_percent:.1f}% by number, {mass_percent:.1f}% by mass)\n")
        
        f.write("\n")
        
        # Results summary
        f.write("RESULTS SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Peak HI ionizing flux: {np.max(results.hi_ionizing_flux):.2f} at {results.times[np.argmax(results.hi_ionizing_flux)]/1e6:.2f} Myr\n")
        f.write(f"Peak wind power: {np.max(results.wind_power):.2f} at {results.times[np.argmax(results.wind_power)]/1e6:.2f} Myr\n")
        
        # UV slope statistics
        valid_beta = results.uv_slope_beta[~np.isnan(results.uv_slope_beta)]
        if len(valid_beta) > 0:
            f.write(f"UV slope range: {np.min(valid_beta):.2f} to {np.max(valid_beta):.2f}\n")
            f.write(f"Mean UV slope: {np.mean(valid_beta):.2f}\n")
        
        # Equivalent width
        f.write(f"Hα equivalent width range: {np.min(results.ha_equivalent_width):.2f} to {np.max(results.ha_equivalent_width):.2f}\n\n")
        
        # Generated plots
        f.write("GENERATED PLOTS\n")
        f.write("-"*80 + "\n")
        for plot_type, fig in figures.items():
            plot_path = os.path.join(output_dir, f"{report_name}_{plot_type}.pdf")
            f.write(f"- {plot_type.capitalize()}: {os.path.basename(plot_path)}\n")
    
    # Print completion message
    print(f"\nCustom population report generated at: {output_dir}")
    print(f"Summary file: {os.path.basename(summary_path)}")
    print(f"Generated {len(figures)} plots")
    
    return output_dir


def compare_custom_populations(results_list: List[StellarPopulationResults],
                              config_list: List[StellarPopulationConfig],
                              custom_star_numbers_list: List[Dict[float, int]],
                              labels: List[str],
                              figsize: Tuple[float, float] = (15, 12),
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare multiple custom stellar populations.
    
    Parameters
    ----------
    results_list : List[StellarPopulationResults]
        List of results for different populations
    config_list : List[StellarPopulationConfig]
        List of configurations for different populations
    custom_star_numbers_list : List[Dict[float, int]]
        List of custom star distributions
    labels : List[str]
        Labels for each population
    figsize : Tuple[float, float]
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The comparison figure
        
    Examples
    --------
    >>> # Compare three different custom populations
    >>> pop1 = {20.0: 100, 40.0: 50}
    >>> pop2 = {15.0: 150, 30.0: 75}
    >>> pop3 = {25.0: 80, 50.0: 40}
    >>> 
    >>> results1, config1 = run_standard_model(custom_star_numbers=pop1)
    >>> results2, config2 = run_standard_model(custom_star_numbers=pop2)
    >>> results3, config3 = run_standard_model(custom_star_numbers=pop3)
    >>> 
    >>> fig = compare_custom_populations(
    ...     [results1, results2, results3],
    ...     [config1, config2, config3],
    ...     [pop1, pop2, pop3],
    ...     ['Population 1', 'Population 2', 'Population 3']
    ... )
    """
    # Ensure all lists have the same length
    if not (len(results_list) == len(config_list) == len(custom_star_numbers_list) == len(labels)):
        raise ValueError("All input lists must have the same length")
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Color map for different populations
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
    
    # 1. Star distribution comparison (upper left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Find all unique masses across populations
    all_masses = set()
    for custom_stars in custom_star_numbers_list:
        all_masses.update(custom_stars.keys())
    
    all_masses = sorted(all_masses)
    
    # Create bar positions
    bar_width = 0.8 / len(custom_star_numbers_list)
    r = np.arange(len(all_masses))
    
    # Plot bars for each population
    for i, (custom_stars, label, color) in enumerate(zip(custom_star_numbers_list, labels, colors)):
        # Get counts for each mass (0 if not present)
        counts = [custom_stars.get(mass, 0) for mass in all_masses]
        
        # Plot bars
        position = r - 0.4 + (i + 0.5) * bar_width
        ax1.bar(position, counts, width=bar_width, color=color, edgecolor='black',
               alpha=0.7, label=label)
    
    # Set x-axis ticks
    ax1.set_xticks(r)
    ax1.set_xticklabels([f'{m:.1f}' for m in all_masses])
    ax1.set_xlabel('Stellar Mass (M$_\\odot$)')
    ax1.set_ylabel('Number of Stars')
    ax1.set_title('Star Distribution Comparison')
    ax1.legend()
    
    # 2. Ionizing flux comparison (upper right)
    ax2 = fig.add_subplot(gs[0, 1])
    
    for i, (results, label, color) in enumerate(zip(results_list, labels, colors)):
        time_myr = results.times / 1e6
        ax2.plot(time_myr, results.hi_ionizing_flux, color=color, linewidth=2, label=label)
    
    ax2.set_xlabel('Time (Myr)')
    ax2.set_ylabel(r'$\log(Q_{\rm HI}/{\rm s}^{-1})$')
    ax2.set_title('H I Ionizing Flux Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Wind power comparison (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    for i, (results, label, color) in enumerate(zip(results_list, labels, colors)):
        time_myr = results.times / 1e6
        ax3.plot(time_myr, results.wind_power, color=color, linewidth=2, label=label)
    
    ax3.set_xlabel('Time (Myr)')
    ax3.set_ylabel(r'$\log(\dot{E}_{\rm wind}/{\rm erg\,s}^{-1})$')
    ax3.set_title('Wind Power Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. UV slope comparison (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    for i, (results, label, color) in enumerate(zip(results_list, labels, colors)):
        time_myr = results.times / 1e6
        # Filter out NaN values
        valid_mask = ~np.isnan(results.uv_slope_beta)
        if np.any(valid_mask):
            ax4.plot(time_myr[valid_mask], results.uv_slope_beta[valid_mask], 
                    color=color, linewidth=2, label=label)
    
    ax4.set_xlabel('Time (Myr)')
    ax4.set_ylabel(r'$\beta_{\rm UV}$')
    ax4.set_title('UV Slope Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Equivalent width comparison (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    
    for i, (results, label, color) in enumerate(zip(results_list, labels, colors)):
        time_myr = results.times / 1e6
        ax5.plot(time_myr, results.ha_equivalent_width, color=color, linewidth=2, label=label)
    
    ax5.set_xlabel('Time (Myr)')
    ax5.set_ylabel(r'$\log({\rm EW}_{{\rm H}\alpha}/\AA)$')
    ax5.set_title(r'H$\alpha$ Equivalent Width Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Population statistics comparison (bottom right)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')  # No axes for text panel
    
    # Create comparison text
    rows = []
    rows.append(["Population", "Total Stars", "Total Mass (M☉)", "Avg. Mass (M☉)"])
    rows.append(["-"*12, "-"*11, "-"*16, "-"*14])
    
    for i, (custom_stars, label) in enumerate(zip(custom_star_numbers_list, labels)):
        total_stars = sum(custom_stars.values())
        total_mass = sum(mass * count for mass, count in custom_stars.items())
        avg_mass = total_mass / total_stars if total_stars > 0 else 0
        
        rows.append([label, f"{total_stars}", f"{total_mass:.2e}", f"{avg_mass:.2f}"])
    
    # Format as table
    col_widths = [max(len(row[j]) for row in rows) for j in range(len(rows[0]))]
    table_str = []
    
    for i, row in enumerate(rows):
        row_str = "  ".join(val.ljust(col_widths[j]) for j, val in enumerate(row))
        table_str.append(row_str)
        
        # Add separator after header
        if i == 1:
            table_str.append("")
    
    # Add some additional statistics
    table_str.append("")
    table_str.append("Peak HI Ionizing Flux:")
    
    for i, (results, label) in enumerate(zip(results_list, labels)):
        peak = np.max(results.hi_ionizing_flux)
        peak_time = results.times[np.argmax(results.hi_ionizing_flux)] / 1e6
        table_str.append(f"  {label}: {peak:.2f} at {peak_time:.2f} Myr")
    
    table_str.append("")
    table_str.append("Peak Wind Power:")
    
    for i, (results, label) in enumerate(zip(results_list, labels)):
        peak = np.max(results.wind_power)
        peak_time = results.times[np.argmax(results.wind_power)] / 1e6
        table_str.append(f"  {label}: {peak:.2f} at {peak_time:.2f} Myr")
    
    # Add the table to the plot
    ax6.text(0.01, 0.99, "\n".join(table_str), 
            transform=ax6.transAxes, fontsize=9, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add title
    fig.suptitle('Custom Stellar Population Comparison', fontsize=16)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    return fig

class PopulationPlotter:
    """Main plotting class for stellar population synthesis results."""
    
    def __init__(self, results: StellarPopulationResults, config: Optional[StellarPopulationConfig] = None):
        """
        Initialize plotter with results and configuration.
        
        Parameters
        ----------
        results : StellarPopulationResults
            Results from population synthesis
        config : StellarPopulationConfig, optional
            Configuration used for the run
        """
        self.results = results
        self.config = config
        self.time_myr = results.times / 1e6  # Convert to Myr
        
        # Set up color schemes
        self.colors = self._setup_color_schemes()
        
    def _setup_color_schemes(self) -> Dict[str, Any]:
        """Set up publication-quality color schemes."""
        # Use colorbrewer qualitative palette
        qual_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Sequential colormap for time evolution
        seq_cmap = plt.cm.viridis
        
        return {
            'qualitative': qual_colors,
            'sequential': seq_cmap,
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'accent': '#d62728'
        }
    
    def create_summary_plot(self, figsize: Tuple[float, float] = (15, 10), 
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive summary plot showing key population properties.
        
        Parameters
        ----------
        figsize : Tuple[float, float]
            Figure size in inches
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig = plt.figure(figsize=figsize)
        
        # Create subplot layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Ionizing fluxes
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_ionizing_fluxes(ax1)
        
        # 2. Wind properties
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_wind_properties(ax2)
        
        # 3. UV slope evolution
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_uv_slope(ax3)
        
        # 4. Equivalent widths
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_equivalent_widths(ax4)
        
        # 5. Color evolution
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_color_evolution(ax5)
        
        # 6. Spectral evolution (2D)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_spectral_evolution_2d(ax6)
        
        # 7. SED at key times
        ax7 = fig.add_subplot(gs[2, :])
        self._plot_sed_evolution(ax7)
        
        # Add title and metadata
        title = self._create_title()
        fig.suptitle(title, fontsize=16, y=0.98)
        
        # Add configuration info
        if self.config:
            config_text = self._create_config_text()
            fig.text(0.02, 0.02, config_text, fontsize=9, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Summary plot saved to {save_path}")
        
        return fig
    
    def plot_ionizing_fluxes(self, ax: Optional[plt.Axes] = None, 
                           figsize: Tuple[float, float] = (10, 6),
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ionizing photon fluxes as a function of time.
        
        Parameters
        ----------
        ax : plt.Axes, optional
            Axes to plot on. If None, creates new figure.
        figsize : Tuple[float, float]
            Figure size if creating new figure
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The created or modified figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        self._plot_ionizing_fluxes(ax)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Ionizing flux plot saved to {save_path}")
        
        return fig
    
    def _plot_ionizing_fluxes(self, ax: plt.Axes) -> None:
        """Internal method to plot ionizing fluxes."""
        # Plot HI ionizing flux
        ax.plot(np.log10(self.results.times), self.results.hi_ionizing_flux, 
               label=r'H\,{\sc i}', color=self.colors['primary'], linewidth=2.5)
        
        # Plot HeI ionizing flux
        ax.plot(np.log10(self.results.times), self.results.hei_ionizing_flux, 
               label=r'He\,{\sc i}', color=self.colors['secondary'], linewidth=2.5)
        
        # Plot HeII ionizing flux
        ax.plot(np.log10(self.results.times), self.results.heii_ionizing_flux, 
               label=r'He\,{\sc ii}', color=self.colors['accent'], linewidth=2.5)
        
        ax.set_xlim(6.0, 8.0)
        ax.set_ylim(44.0, 54.0)
        ax.set_xlabel(r'$\log(t/{\rm yr})$')
        ax.set_ylabel(r'$\log(Q/{\rm s}^{-1})$')
        ax.set_title('Ionizing Photon Rates')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
    
    def plot_wind_properties(self, ax: Optional[plt.Axes] = None,
                           figsize: Tuple[float, float] = (10, 6),
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot stellar wind properties as a function of time.
        
        Parameters
        ----------
        ax : plt.Axes, optional
            Axes to plot on. If None, creates new figure.
        figsize : Tuple[float, float]
            Figure size if creating new figure
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The created or modified figure
        """
        if ax is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
            fig.subplots_adjust(hspace=0.1)
        else:
            fig = ax.figure
            ax1, ax2 = ax, None
        
        self._plot_wind_properties(ax1, ax2)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Wind properties plot saved to {save_path}")
        
        return fig
    
    def _plot_wind_properties(self, ax1: plt.Axes, ax2: Optional[plt.Axes] = None) -> None:
        """Internal method to plot wind properties."""
        time_log = np.log10(self.results.times)
        
        if ax2 is not None:
            # Two-panel plot
            # Wind power
            ax1.plot(time_log, self.results.wind_power, 
                    color=self.colors['primary'], linewidth=2.5)
            ax1.set_xlim(6.0, 7.5)
            ax1.set_ylim(34.0, 42.0)
            ax1.set_ylabel(r'$\log(\dot{E}_{\rm wind}/{\rm erg\,s}^{-1})$')
            ax1.set_title('Stellar Wind Power')
            ax1.grid(True, alpha=0.3)
            
            # Wind momentum
            ax2.plot(time_log, self.results.wind_momentum, 
                    color=self.colors['secondary'], linewidth=2.5)
            ax2.set_xlim(6.0, 7.5)
            ax2.set_ylim(27.5, 33.0)
            ax2.set_xlabel(r'$\log(t/{\rm yr})$')
            ax2.set_ylabel(r'$\log(\dot{p}_{\rm wind}/{\rm dyne})$')
            ax2.set_title('Stellar Wind Momentum')
            ax2.grid(True, alpha=0.3)
        else:
            # Single panel - plot both with different y-axes
            ax1.plot(time_log, self.results.wind_power, 
                    color=self.colors['primary'], linewidth=2.5, label='Power')
            
            ax1_twin = ax1.twinx()
            ax1_twin.plot(time_log, self.results.wind_momentum, 
                         color=self.colors['secondary'], linewidth=2.5, label='Momentum')
            
            ax1.set_xlim(6.0, 7.5)
            ax1.set_xlabel(r'$\log(t/{\rm yr})$')
            ax1.set_ylabel(r'$\log(\dot{E}_{\rm wind}/{\rm erg\,s}^{-1})$', color=self.colors['primary'])
            ax1_twin.set_ylabel(r'$\log(\dot{p}_{\rm wind}/{\rm dyne})$', color=self.colors['secondary'])
            ax1.set_title('Stellar Wind Properties')
            
            # Color the y-axis labels
            ax1.tick_params(axis='y', labelcolor=self.colors['primary'])
            ax1_twin.tick_params(axis='y', labelcolor=self.colors['secondary'])
            
            ax1.grid(True, alpha=0.3)
    
    def plot_uv_slope(self, ax: Optional[plt.Axes] = None,
                     figsize: Tuple[float, float] = (10, 6),
                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot UV slope (β parameter) evolution.
        
        Parameters
        ----------
        ax : plt.Axes, optional
            Axes to plot on. If None, creates new figure.
        figsize : Tuple[float, float]
            Figure size if creating new figure
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The created or modified figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        self._plot_uv_slope(ax)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"UV slope plot saved to {save_path}")
        
        return fig
    
    def _plot_uv_slope(self, ax: plt.Axes) -> None:
        """Internal method to plot UV slope."""
        time_log = np.log10(self.results.times)
        
        # Filter out NaN values
        valid_mask = ~np.isnan(self.results.uv_slope_beta)
        
        if np.any(valid_mask):
            ax.plot(time_log[valid_mask], self.results.uv_slope_beta[valid_mask], 
                   color=self.colors['primary'], linewidth=2.5, marker='o', markersize=4)
        
        ax.set_xlim(6.0, 8.0)
        ax.set_xlabel(r'$\log(t/{\rm yr})$')
        ax.set_ylabel(r'$\beta_{\rm UV}$')
        ax.set_title(r'UV Slope Evolution ($f_\lambda \propto \lambda^\beta$)')
        ax.invert_yaxis()  # More negative beta values at top
        ax.grid(True, alpha=0.3)
        
        # Add reference lines for different galaxy types
        ax.axhline(-2.3, color='gray', linestyle='--', alpha=0.7, label='Starburst')
        ax.axhline(-1.4, color='gray', linestyle=':', alpha=0.7, label='Irregular')
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    def plot_equivalent_widths(self, ax: Optional[plt.Axes] = None,
                             figsize: Tuple[float, float] = (10, 6),
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot equivalent width evolution for hydrogen lines.
        
        Parameters
        ----------
        ax : plt.Axes, optional
            Axes to plot on. If None, creates new figure.
        figsize : Tuple[float, float]
            Figure size if creating new figure
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The created or modified figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        self._plot_equivalent_widths(ax)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Equivalent width plot saved to {save_path}")
        
        return fig
    
    def _plot_equivalent_widths(self, ax: plt.Axes) -> None:
        """Internal method to plot equivalent widths."""
        time_log = np.log10(self.results.times)
        
        # Plot H-alpha
        ax.plot(time_log, self.results.ha_equivalent_width, 
               label=r'H$\alpha$', color=self.colors['primary'], linewidth=2.5)
        
        # Plot H-beta if available
        if hasattr(self.results, 'hb_equivalent_width'):
            ax.plot(time_log, self.results.hb_equivalent_width, 
                   label=r'H$\beta$', color=self.colors['secondary'], linewidth=2.5)
        
        ax.set_xlim(6.0, 8.0)
        ax.set_xlabel(r'$\log(t/{\rm yr})$')
        ax.set_ylabel(r'$\log({\rm EW}/\AA)$')
        ax.set_title('Hydrogen Line Equivalent Widths')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Add reference lines for different stellar populations
        ax.axhline(2.0, color='gray', linestyle='--', alpha=0.7, label='Strong')
        ax.axhline(1.0, color='gray', linestyle=':', alpha=0.7, label='Moderate')
    
    def plot_color_evolution(self, ax: Optional[plt.Axes] = None,
                           figsize: Tuple[float, float] = (10, 6),
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot broad-band color evolution.
        
        Parameters
        ----------
        ax : plt.Axes, optional
            Axes to plot on. If None, creates new figure.
        figsize : Tuple[float, float]
            Figure size if creating new figure
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The created or modified figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        self._plot_color_evolution(ax)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Color evolution plot saved to {save_path}")
        
        return fig
    
    def _plot_color_evolution(self, ax: plt.Axes) -> None:
        """Internal method to plot color evolution."""
        time_log = np.log10(self.results.times)
        
        # Calculate colors
        ub_color = self.results.u_magnitude - self.results.b_magnitude
        bv_color = self.results.b_magnitude - self.results.v_magnitude
        
        ax.plot(time_log, ub_color, label=r'$U-B$', 
               color=self.colors['primary'], linewidth=2.5)
        ax.plot(time_log, bv_color, label=r'$B-V$', 
               color=self.colors['secondary'], linewidth=2.5)
        
        ax.set_xlim(6.0, 8.0)
        ax.set_xlabel(r'$\log(t/{\rm yr})$')
        ax.set_ylabel('Color (mag)')
        ax.set_title('Broad-band Color Evolution')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
    
    def plot_spectral_evolution_2d(self, ax: Optional[plt.Axes] = None,
                                  figsize: Tuple[float, float] = (12, 8),
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a 2D plot showing spectral evolution over time.
        
        Parameters
        ----------
        ax : plt.Axes, optional
            Axes to plot on. If None, creates new figure.
        figsize : Tuple[float, float]
            Figure size if creating new figure
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The created or modified figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        self._plot_spectral_evolution_2d(ax)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"2D spectral evolution plot saved to {save_path}")
        
        return fig
    
    def _plot_spectral_evolution_2d(self, ax: plt.Axes) -> None:
        """Internal method to create 2D spectral evolution plot."""
        # Subsample for visualization
        time_indices = np.arange(0, len(self.results.times), max(1, len(self.results.times)//50))
        wave_indices = np.arange(0, len(self.results.wavelength_grid), 
                               max(1, len(self.results.wavelength_grid)//500))
        
        # Create meshgrid
        wave_sub = self.results.wavelength_grid[wave_indices]
        time_sub = self.results.times[time_indices] / 1e6  # Convert to Myr
        flux_sub = self.results.flux_spectra[np.ix_(time_indices, wave_indices)]
        
        # Take log of flux for better visualization
        flux_log = np.log10(flux_sub + 1e-30)
        
        # Create 2D plot
        im = ax.imshow(flux_log, aspect='auto', origin='lower', 
                      extent=[wave_sub.min(), wave_sub.max(), time_sub.min(), time_sub.max()],
                      cmap=self.colors['sequential'], interpolation='bilinear')
        
        ax.set_xlabel(r'Wavelength (\AA)')
        ax.set_ylabel('Time (Myr)')
        ax.set_title('Spectral Evolution')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r'$\log(f_\lambda)$ [arbitrary units]')
        
        # Mark important spectral features
        important_lines = {
            'Ly$\\alpha$': 1216,
            'C\,{\\sc iv}': 1549,
            'H$\\beta$': 4861,
            'H$\\alpha$': 6563
        }
        
        for line_name, line_wave in important_lines.items():
            if wave_sub.min() <= line_wave <= wave_sub.max():
                ax.axvline(line_wave, color='white', linestyle='--', alpha=0.7)
                ax.text(line_wave, ax.get_ylim()[1]*0.95, line_name, 
                       rotation=90, ha='right', va='top', color='white',
                       fontsize=10, weight='bold')
    
    def plot_sed_evolution(self, ax: Optional[plt.Axes] = None,
                          times_myr: Optional[List[float]] = None,
                          figsize: Tuple[float, float] = (12, 8),
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot spectral energy distributions at selected times.
        
        Parameters
        ----------
        ax : plt.Axes, optional
            Axes to plot on. If None, creates new figure.
        times_myr : List[float], optional
            Times in Myr to plot. If None, uses default selection.
        figsize : Tuple[float, float]
            Figure size if creating new figure
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The created or modified figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        self._plot_sed_evolution(ax, times_myr)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"SED evolution plot saved to {save_path}")
        
        return fig
    
    def _plot_sed_evolution(self, ax: plt.Axes, times_myr: Optional[List[float]] = None) -> None:
        """Internal method to plot SED evolution."""
        if times_myr is None:
            # Default time selection
            times_myr = [1, 3, 5, 10, 20, 50]
        
        # Find closest time indices
        time_indices = []
        for t_target in times_myr:
            t_target_yr = t_target * 1e6
            closest_idx = np.argmin(np.abs(self.results.times - t_target_yr))
            time_indices.append(closest_idx)
        
        # Color map for different times
        colors = plt.cm.plasma(np.linspace(0, 1, len(time_indices)))
        
        for i, (idx, color) in enumerate(zip(time_indices, colors)):
            actual_time = self.results.times[idx] / 1e6
            flux = self.results.flux_spectra[idx] 
            
            # Convert to log scale and add offset for visibility
            flux_log = safe_log10(flux) # Offset each spectrum
            
            ax.semilogx(self.results.wavelength_grid, flux_log, 
                   color=color, linewidth=2, 
                   label=f'{actual_time:.1f} Myr')
        
        ax.set_xlabel(r'Wavelength (\AA)')
        ax.set_ylabel(r'$\log(f_\lambda) + {\rm offset}$ [arbitrary units]')
        ax.set_ylim(15)
        ax.set_xlim(500, 5e4)
        ax.set_title('Spectral Energy Distribution Evolution')
        ax.legend(frameon=True, fancybox=True, shadow=True, 
                 bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Mark important spectral features
        important_lines = {
            'Ly$\\alpha$': 1216,
            'C\,{\\sc iv}': 1549,
            'H$\\beta$': 4861,
            'H$\\alpha$': 6563
        }
        
        for line_name, line_wave in important_lines.items():
            if 1000 <= line_wave <= 10000:
                ax.axvline(line_wave, color='gray', linestyle=':', alpha=0.5)
                ax.text(line_wave, ax.get_ylim()[1]*0.98, line_name, 
                       rotation=90, ha='right', va='top', 
                       fontsize=9, alpha=0.7)
    
    def plot_hertzsprung_russell_diagram(self, time_myr: float = 5.0,
                                       figsize: Tuple[float, float] = (10, 8),
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Hertzsprung-Russell diagram at a specific time.
        
        Parameters
        ----------
        time_myr : float
            Time in Myr for the HR diagram
        figsize : Tuple[float, float]
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Find closest time index
        time_target_yr = time_myr * 1e6
        time_idx = np.argmin(np.abs(self.results.times - time_target_yr))
        actual_time = self.results.times[time_idx] / 1e6
        
        # Get stellar properties at this time
        if hasattr(self.results, 'stellar_temperatures') and len(self.results.stellar_temperatures) > time_idx:
            temperatures = self.results.stellar_temperatures[time_idx]
            luminosities = self.results.stellar_luminosities[time_idx]
            masses = self.results.stellar_masses[time_idx]
            
            # Convert to effective temperatures
            log_teff = temperatures
            log_lum = luminosities
            
            # Filter out dead stars
            alive_mask = log_lum > -19
            
            if np.any(alive_mask):
                log_teff_alive = log_teff[alive_mask]
                log_lum_alive = log_lum[alive_mask]
                masses_alive = masses[alive_mask]
                
                # Create scatter plot colored by mass
                scatter = ax.scatter(log_teff_alive, log_lum_alive, 
                                   c=masses_alive, cmap='viridis', 
                                   s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(r'Initial Mass ($M_\odot$)')
                
                ax.invert_xaxis()  # Hot stars on the left
                ax.set_xlabel(r'$\log(T_{\rm eff}/{\rm K})$')
                ax.set_ylabel(r'$\log(L/L_\odot)$')
                ax.set_title(f'Hertzsprung-Russell Diagram at {actual_time:.1f} Myr')
                ax.grid(True, alpha=0.3)
                
                # Add stellar evolution tracks if available
                self._add_evolution_tracks(ax)
            else:
                ax.text(0.5, 0.5, 'No living stars at this time', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=16)
        else:
            ax.text(0.5, 0.5, 'Stellar data not available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"HR diagram saved to {save_path}")
        
        return fig
    
    def _add_evolution_tracks(self, ax: plt.Axes) -> None:
        """Add theoretical evolution tracks to HR diagram."""
        # This would require access to the original evolution tracks
        # For now, add approximate main sequence and giant branch lines
        
        # Approximate main sequence
        ms_teff = np.linspace(3.5, 4.7, 100)
        ms_lum = 4.0 * (ms_teff - 3.76) + 0.5  # Approximate MS relation
        ax.plot(ms_teff, ms_lum, 'k--', alpha=0.5, linewidth=1, label='Main Sequence')
        
        # Approximate giant branch
        gb_teff = np.linspace(3.5, 3.8, 50)
        gb_lum = -10.0 * (gb_teff - 3.8) + 3.0
        ax.plot(gb_teff, gb_lum, 'r--', alpha=0.5, linewidth=1, label='Giant Branch')
        
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    def plot_mass_function_evolution(self, times_myr: Optional[List[float]] = None,
                                   figsize: Tuple[float, float] = (12, 8),
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the evolution of the stellar mass function.
        
        Parameters
        ----------
        times_myr : List[float], optional
            Times in Myr to plot. If None, uses default selection.
        figsize : Tuple[float, float]
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if times_myr is None:
            times_myr = [0.1, 1, 5, 10, 20]
        
        # Color map for different times
        colors = plt.cm.viridis(np.linspace(0, 1, len(times_myr)))
        
        for i, (t_myr, color) in enumerate(zip(times_myr, colors)):
            time_idx = np.argmin(np.abs(self.results.times - t_myr * 1e6))
            
            if hasattr(self.results, 'stellar_masses') and len(self.results.stellar_masses) > time_idx:
                masses = self.results.stellar_masses[time_idx]
                luminosities = self.results.stellar_luminosities[time_idx]
                
                # Filter out dead stars
                alive_mask = np.array(luminosities) > -19
                masses_alive = np.array(masses)[alive_mask]
                
                if len(masses_alive) > 0:
                    # Create mass bins
                    mass_bins = np.logspace(np.log10(0.1), np.log10(np.max(masses_alive)), 20)
                    hist, bin_edges = np.histogram(masses_alive, bins=mass_bins)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    bin_widths = bin_edges[1:] - bin_edges[:-1]
                    
                    # Normalize by bin width to get density
                    density = hist / bin_widths
                    
                    # Plot
                    ax.loglog(bin_centers, density, 'o-', color=color, 
                             linewidth=2, markersize=6, 
                             label=f'{t_myr:.1f} Myr ({len(masses_alive)} stars)')
        
        ax.set_xlabel(r'Mass ($M_\odot$)')
        ax.set_ylabel(r'$dN/dM$ (stars per $M_\odot$)')
        ax.set_title('Stellar Mass Function Evolution')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Add initial IMF for comparison
        if self.config:
            mass_range = np.logspace(-1, 2, 100)
            imf_density = self._calculate_imf_density(mass_range)
            ax.loglog(mass_range, imf_density, 'k--', linewidth=2, 
                     alpha=0.7, label='Initial IMF')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Mass function evolution plot saved to {save_path}")
        
        return fig
    
    def _calculate_imf_density(self, masses: np.ndarray) -> np.ndarray:
        """Calculate IMF density for comparison."""
        if not self.config:
            return np.ones_like(masses)
        
        # Simple power-law IMF approximation
        # This is a simplified version - the actual IMF might be more complex
        alpha = self.config.imf_exponents[-1] if self.config.imf_exponents else 2.35
        normalization = 1.0  # Arbitrary normalization
        
        return normalization * masses**(-alpha)
    
    def create_diagnostic_plots(self, figsize: Tuple[float, float] = (15, 12),
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create diagnostic plots for model validation.
        
        Parameters
        ----------
        figsize : Tuple[float, float]
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Bolometric luminosity evolution
        ax1 = fig.add_subplot(gs[0, 0])
        time_log = np.log10(self.results.times)
        ax1.plot(time_log, self.results.bolometric_luminosity, 
                color=self.colors['primary'], linewidth=2)
        ax1.set_xlabel(r'$\log(t/{\rm yr})$')
        ax1.set_ylabel(r'$\log(L_{\rm bol}/{\rm erg\,s}^{-1})$')
        ax1.set_title('Bolometric Luminosity')
        ax1.grid(True, alpha=0.3)
        
        # 2. Ionizing efficiency
        ax2 = fig.add_subplot(gs[0, 1])
        efficiency = self.results.hi_ionizing_flux - self.results.bolometric_luminosity
        ax2.plot(time_log, efficiency, color=self.colors['secondary'], linewidth=2)
        ax2.set_xlabel(r'$\log(t/{\rm yr})$')
        ax2.set_ylabel(r'$\log(\xi_{\rm ion})$')
        ax2.set_title('Ionizing Efficiency')
        ax2.grid(True, alpha=0.3)
        
        # 3. Wind efficiency
        ax3 = fig.add_subplot(gs[0, 2])
        wind_efficiency = self.results.wind_momentum - self.results.bolometric_luminosity
        ax3.plot(time_log, wind_efficiency, color=self.colors['accent'], linewidth=2)
        ax3.set_xlabel(r'$\log(t/{\rm yr})$')
        ax3.set_ylabel(r'$\log(\eta_{\rm wind})$')
        ax3.set_title('Wind Efficiency')
        ax3.grid(True, alpha=0.3)
        
        # 4. Color-color diagram
        ax4 = fig.add_subplot(gs[1, 0])
        ub_color = self.results.u_magnitude - self.results.b_magnitude
        bv_color = self.results.b_magnitude - self.results.v_magnitude
        
        # Color points by time
        scatter = ax4.scatter(bv_color, ub_color, c=self.time_myr, 
                             cmap='viridis', s=30, alpha=0.7)
        ax4.set_xlabel(r'$B-V$')
        ax4.set_ylabel(r'$U-B$')
        ax4.set_title('Color-Color Diagram')
        ax4.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Time (Myr)')
        
        # 5. Spectral index distribution
        ax5 = fig.add_subplot(gs[1, 1])
        # Calculate spectral indices at different wavelengths
        wave_1500 = np.argmin(np.abs(self.results.wavelength_grid - 1500))
        wave_2800 = np.argmin(np.abs(self.results.wavelength_grid - 2800))
        
        flux_1500 = self.results.flux_spectra_with_nebular[:, wave_1500]
        flux_2800 = self.results.flux_spectra_with_nebular[:, wave_2800]
        
        spectral_index = np.log10(flux_2800 / (flux_1500 + 1e-30))
        ax5.plot(time_log, spectral_index, color=self.colors['primary'], linewidth=2)
        ax5.set_xlabel(r'$\log(t/{\rm yr})$')
        ax5.set_ylabel(r'$\log(f_{2800}/f_{1500})$')
        ax5.set_title('UV Spectral Index')
        ax5.grid(True, alpha=0.3)
        
        # 6. Cumulative energy output
        ax6 = fig.add_subplot(gs[1, 2])
        dt = np.diff(self.results.times)
        dt = np.append(dt, dt[-1])  # Handle last point
        
        # Cumulative bolometric energy
        energy_cumulative = np.cumsum(10**self.results.bolometric_luminosity * dt * 3.15e7)  # Convert to erg
        ax6.loglog(self.time_myr, energy_cumulative, color=self.colors['primary'], 
                  linewidth=2, label='Bolometric')
        
        # Cumulative wind energy
        wind_energy_cumulative = np.cumsum(10**self.results.wind_power * dt * 3.15e7)
        ax6.loglog(self.time_myr, wind_energy_cumulative, color=self.colors['secondary'], 
                  linewidth=2, label='Wind')
        
        ax6.set_xlabel('Time (Myr)')
        ax6.set_ylabel('Cumulative Energy (erg)')
        ax6.set_title('Energy Budget')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Line ratio diagnostics (bottom row)
        ax7 = fig.add_subplot(gs[2, :2])
        
        # Create BPT-like diagram using available line ratios
        # This is simplified since we only have equivalent widths
        ha_flux = 10**self.results.ha_equivalent_width
        hb_flux = 10**self.results.hb_equivalent_width if hasattr(self.results, 'hb_equivalent_width') else ha_flux / 2.86
        
        ha_hb_ratio = np.log10(ha_flux / hb_flux)
        
        # Plot evolution in line ratio space
        scatter = ax7.scatter(self.time_myr, ha_hb_ratio, c=self.time_myr, 
                             cmap='plasma', s=50, alpha=0.7)
        ax7.set_xlabel('Time (Myr)')
        ax7.set_ylabel(r'$\log({\rm H}\alpha/{\rm H}\beta)$')
        ax7.set_title('Line Ratio Evolution')
        ax7.grid(True, alpha=0.3)
        
        # Add theoretical Balmer decrement
        ax7.axhline(np.log10(2.86), color='red', linestyle='--', 
                   label='Case B (T=10$^4$ K)', alpha=0.7)
        ax7.legend()
        
        # 8. Model summary statistics
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        
        # Calculate summary statistics
        stats_text = self._create_summary_statistics()
        ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, 
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        fig.suptitle('Model Diagnostics', fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Diagnostic plots saved to {save_path}")
        
        return fig
    
    def _create_summary_statistics(self) -> str:
        """Create summary statistics text."""
        stats = []
        stats.append("SUMMARY STATISTICS")
        stats.append("=" * 20)
        
        # Time range
        stats.append(f"Time range: {self.time_myr.min():.1f} - {self.time_myr.max():.1f} Myr")
        
        # Peak values
        max_hi_flux = np.max(self.results.hi_ionizing_flux)
        max_wind_power = np.max(self.results.wind_power)
        max_bolo_lum = np.max(self.results.bolometric_luminosity)
        
        stats.append(f"Peak HI flux: {max_hi_flux:.1f}")
        stats.append(f"Peak wind power: {max_wind_power:.1f}")
        stats.append(f"Peak bolo lum: {max_bolo_lum:.1f}")
        
        # UV slope range
        valid_beta = self.results.uv_slope_beta[~np.isnan(self.results.uv_slope_beta)]
        if len(valid_beta) > 0:
            stats.append(f"UV slope range: {valid_beta.min():.2f} to {valid_beta.max():.2f}")
        
        # Color evolution
        ub_color = self.results.u_magnitude - self.results.b_magnitude
        bv_color = self.results.b_magnitude - self.results.v_magnitude
        stats.append(f"U-B range: {ub_color.min():.2f} to {ub_color.max():.2f}")
        stats.append(f"B-V range: {bv_color.min():.2f} to {bv_color.max():.2f}")
        
        # Number of stars
        if hasattr(self.results, 'number_of_stars'):
            total_stars = np.sum(self.results.number_of_stars)
            stats.append(f"Total stars: {total_stars:.0f}")
        
        return "\n".join(stats)
    
    def _create_title(self) -> str:
        """Create appropriate title for plots."""
        if self.config:
            mass_str = f"{self.config.total_mass:.0e}"
            return (f"Stellar Population Synthesis: M = {mass_str} M$_\\odot$, "
                   f"Z = {self.config.metallicity}, {self.config.spectral_library}")
        else:
            return "Stellar Population Synthesis Results"
    
    def _create_config_text(self) -> str:
        """Create configuration summary text."""
        lines = []
        lines.append(f"M_total: {self.config.total_mass:.0e} M☉")
        lines.append(f"IMF: {self.config.imf_exponents}")
        lines.append(f"Metallicity: {self.config.metallicity}")
        lines.append(f"Spectra: {self.config.spectral_library}")
        lines.append(f"Rotation: {self.config.rotation}")
        lines.append(f"Speed: {self.config.run_speed_mode}")
        return "\n".join(lines)


def compare_models(results_list: List[StellarPopulationResults], 
                   labels: List[str],
                   config_list: Optional[List[StellarPopulationConfig]] = None,
                   figsize: Tuple[float, float] = (15, 10),
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare multiple stellar population models.
    
    Parameters
    ----------
    results_list : List[StellarPopulationResults]
        List of results to compare
    labels : List[str]
        Labels for each model
    config_list : List[StellarPopulationConfig], optional
        Configuration objects for each model
    figsize : Tuple[float, float]
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The comparison figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Color scheme for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
    
    # 1. Ionizing flux comparison
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (results, label, color) in enumerate(zip(results_list, labels, colors)):
        time_log = np.log10(results.times)
        ax1.plot(time_log, results.hi_ionizing_flux, 
                color=color, linewidth=2, label=label)
    
    ax1.set_xlim(6.0, 8.0)
    ax1.set_xlabel(r'$\log(t/{\rm yr})$')
    ax1.set_ylabel(r'$\log(Q_{\rm HI}/{\rm s}^{-1})$')
    ax1.set_title('H I Ionizing Flux')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Wind momentum comparison
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (results, label, color) in enumerate(zip(results_list, labels, colors)):
        time_log = np.log10(results.times)
        ax2.plot(time_log, results.wind_momentum, 
                color=color, linewidth=2, label=label)
    
    ax2.set_xlim(6.0, 7.5)
    ax2.set_xlabel(r'$\log(t/{\rm yr})$')
    ax2.set_ylabel(r'$\log(\dot{p}_{\rm wind}/{\rm dyne})$')
    ax2.set_title('Wind Momentum')
    ax2.grid(True, alpha=0.3)
    
    # 3. UV slope comparison
    ax3 = fig.add_subplot(gs[0, 2])
    for i, (results, label, color) in enumerate(zip(results_list, labels, colors)):
        time_log = np.log10(results.times)
        valid_mask = ~np.isnan(results.uv_slope_beta)
        if np.any(valid_mask):
            ax3.plot(time_log[valid_mask], results.uv_slope_beta[valid_mask], 
                    color=color, linewidth=2, label=label, marker='o', markersize=3)
    
    ax3.set_xlim(6.0, 8.0)
    ax3.set_xlabel(r'$\log(t/{\rm yr})$')
    ax3.set_ylabel(r'$\beta_{\rm UV}$')
    ax3.set_title('UV Slope')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3)
    
    # 4. Color-color comparison
    ax4 = fig.add_subplot(gs[1, 0])
    for i, (results, label, color) in enumerate(zip(results_list, labels, colors)):
        ub_color = results.u_magnitude - results.b_magnitude
        bv_color = results.b_magnitude - results.v_magnitude
        ax4.plot(bv_color, ub_color, color=color, linewidth=2, 
                label=label, alpha=0.7)
    
    ax4.set_xlabel(r'$B-V$')
    ax4.set_ylabel(r'$U-B$')
    ax4.set_title('Color Evolution')
    ax4.grid(True, alpha=0.3)
    
    # 5. Equivalent width comparison
    ax5 = fig.add_subplot(gs[1, 1])
    for i, (results, label, color) in enumerate(zip(results_list, labels, colors)):
        time_log = np.log10(results.times)
        ax5.plot(time_log, results.ha_equivalent_width, 
                color=color, linewidth=2, label=label)
    
    ax5.set_xlim(6.0, 8.0)
    ax5.set_xlabel(r'$\log(t/{\rm yr})$')
    ax5.set_ylabel(r'$\log({\rm EW}_{{\rm H}\alpha}/\AA)$')
    ax5.set_title(r'H$\alpha$ Equivalent Width')
    ax5.grid(True, alpha=0.3)
    
    # 6. Spectral comparison at fixed time
    ax6 = fig.add_subplot(gs[1, 2])
    comparison_time_myr = 5.0  # Fixed time for spectral comparison
    
    for i, (results, label, color) in enumerate(zip(results_list, labels, colors)):
        time_idx = np.argmin(np.abs(results.times - comparison_time_myr * 1e6))
        flux = results.flux_spectra_with_nebular[time_idx]
        flux_norm = flux / np.max(flux)  # Normalize for comparison
        
        # Subsample for plotting
        wave_indices = np.arange(0, len(results.wavelength_grid), 10)
        ax6.plot(results.wavelength_grid[wave_indices], flux_norm[wave_indices], 
                color=color, linewidth=2, label=label, alpha=0.8)
    
    ax6.set_xlim(1000, 8000)
    ax6.set_xlabel(r'Wavelength (\AA)')
    ax6.set_ylabel('Normalized Flux')
    ax6.set_title(f'Spectra at {comparison_time_myr} Myr')
    ax6.grid(True, alpha=0.3)
    
    fig.suptitle('Model Comparison', fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Model comparison plot saved to {save_path}")
    
    return fig


def plot_quick_summary(results: StellarPopulationResults,
                      config: Optional[StellarPopulationConfig] = None,
                      figsize: Tuple[float, float] = (12, 8),
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a quick summary plot with the most important quantities.
    
    Parameters
    ----------
    results : StellarPopulationResults
        Results to plot
    config : StellarPopulationConfig, optional
        Configuration object
    figsize : Tuple[float, float]
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    time_log = np.log10(results.times)
    time_myr = results.times / 1e6
    
    # Colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. Ionizing photon rates
    ax1.plot(time_log, results.hi_ionizing_flux, color=colors[0], 
            linewidth=2.5, label=r'H\,{\sc i}')
    ax1.plot(time_log, results.hei_ionizing_flux, color=colors[1], 
            linewidth=2.5, label=r'He\,{\sc i}')
    ax1.plot(time_log, results.heii_ionizing_flux, color=colors[2], 
            linewidth=2.5, label=r'He\,{\sc ii}')
    
    ax1.set_xlim(6.0, 8.0)
    ax1.set_ylim(44.0, 54.0)
    ax1.set_xlabel(r'$\log(t/{\rm yr})$')
    ax1.set_ylabel(r'$\log(Q/{\rm s}^{-1})$')
    ax1.set_title('Ionizing Photon Rates')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Wind properties
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(time_log, results.wind_power, color=colors[0], 
                    linewidth=2.5, label='Power')
    line2 = ax2_twin.plot(time_log, results.wind_momentum, color=colors[1], 
                         linewidth=2.5, label='Momentum')
    
    ax2.set_xlim(6.0, 7.5)
    ax2.set_xlabel(r'$\log(t/{\rm yr})$')
    ax2.set_ylabel(r'$\log(\dot{E}_{\rm wind}/{\rm erg\,s}^{-1})$', color=colors[0])
    ax2_twin.set_ylabel(r'$\log(\dot{p}_{\rm wind}/{\rm dyne})$', color=colors[1])
    ax2.set_title('Stellar Wind Properties')
    
    ax2.tick_params(axis='y', labelcolor=colors[0])
    ax2_twin.tick_params(axis='y', labelcolor=colors[1])
    ax2.grid(True, alpha=0.3)
    
    # 3. Spectral properties
    valid_mask = ~np.isnan(results.uv_slope_beta)
    if np.any(valid_mask):
        ax3.plot(time_log[valid_mask], results.uv_slope_beta[valid_mask], 
                color=colors[0], linewidth=2.5, marker='o', markersize=4)
    
    ax3.set_xlim(6.0, 8.0)
    ax3.set_xlabel(r'$\log(t/{\rm yr})$')
    ax3.set_ylabel(r'$\beta_{\rm UV}$')
    ax3.set_title(r'UV Slope ($f_\lambda \propto \lambda^\beta$)')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3)
    
    # 4. Line equivalent widths
    ax4.plot(time_log, results.ha_equivalent_width, color=colors[0], 
            linewidth=2.5, label=r'H$\alpha$')
    
    if hasattr(results, 'hb_equivalent_width'):
        ax4.plot(time_log, results.hb_equivalent_width, color=colors[1], 
                linewidth=2.5, label=r'H$\beta$')
    
    ax4.set_xlim(6.0, 8.0)
    ax4.set_xlabel(r'$\log(t/{\rm yr})$')
    ax4.set_ylabel(r'$\log({\rm EW}/\AA)$')
    ax4.set_title('Hydrogen Line Equivalent Widths')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add overall title
    if config:
        title = (f"Z = {config.metallicity}")
    else:
        title = "Stellar Population Synthesis Results"
    
    fig.suptitle(title, fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Quick summary plot saved to {save_path}")
    
    return fig


# Convenience function for quick plotting
def quick_plot(results: StellarPopulationResults, 
               config: Optional[StellarPopulationConfig] = None,
               plot_type: str = 'summary',
               **kwargs) -> plt.Figure:
    """
    Quick plotting function for common plot types.
    
    Parameters
    ----------
    results : StellarPopulationResults
        Results to plot
    config : StellarPopulationConfig, optional
        Configuration object
    plot_type : str
        Type of plot: 'summary', 'ionizing', 'wind', 'uv', 'colors', 'comparison'
    **kwargs
        Additional arguments passed to plotting functions
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    plotter = PopulationPlotter(results, config)
    
    if plot_type == 'summary':
        return plotter.create_summary_plot(**kwargs)
    elif plot_type == 'quick':
        return plot_quick_summary(results, config, **kwargs)
    elif plot_type == 'ionizing':
        return plotter.plot_ionizing_fluxes(**kwargs)
    elif plot_type == 'wind':
        return plotter.plot_wind_properties(**kwargs)
    elif plot_type == 'uv':
        return plotter.plot_uv_slope(**kwargs)
    elif plot_type == 'colors':
        return plotter.plot_color_evolution(**kwargs)
    elif plot_type == 'hr':
        return plotter.plot_hertzsprung_russell_diagram(**kwargs)
    elif plot_type == 'mass_function':
        return plotter.plot_mass_function_evolution(**kwargs)
    elif plot_type == 'diagnostics':
        return plotter.create_diagnostic_plots(**kwargs)
    elif plot_type == 'sed':
        return plotter.plot_sed_evolution(**kwargs)
    elif plot_type == 'spectral_2d':
        return plotter.plot_spectral_evolution_2d(**kwargs)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}. "
                        f"Available types: summary, quick, ionizing, wind, uv, colors, "
                        f"hr, mass_function, diagnostics, sed, spectral_2d")
