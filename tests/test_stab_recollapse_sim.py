import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from pathlib import Path
import pytest
from toddlers.stab import recollapse as TORCH
from toddlers.stab.config import *

# Needs the generated recollapse HDF5 data; deselected by the default (fast) test run.
pytestmark = pytest.mark.data

def test_star_formation_simulation():
    """Test StarFormationSimulation following the established plotting pattern."""
    
    # Test parameters using values from the parameter grid
    test_params = {
        'Z': 0.02,         # Solar metallicity
        'epsilon': 0.01,  # Moderate star formation efficiency
        'n_cl': 640.0,     # Moderate cloud density
        'ver': 'v2'        # hdf5 version
    }
    
    # Initialize simulation
    sim = TORCH.StarFormationSimulation(
        Z=test_params['Z'],
        epsilon=test_params['epsilon'],
        n_cl=test_params['n_cl'],
        simulation_data=[],  # Use synthetic data
        num_particles=10000,
        version=test_params['ver']
    )
    
    # Calculate bin values following the original pattern
    M_star_particles_per_bin = [
        np.sum(sim.M_star_particles[sim.temporal_bins == bin_idx]) 
        for bin_idx in range(N_TEMPORAL_BINS)  # -1 for correct number of bins
    ]
    
    adjusted_M_star_particles_per_bin = [
        np.sum(
            sim.M_cloud_particles[sim.temporal_bins == bin_idx] * 
            sim.w_ki[sim.temporal_bins == bin_idx] * 
            sim.epsilon[sim.temporal_bins == bin_idx]
        ) 
        for bin_idx in range(N_TEMPORAL_BINS)
    ]
    
    M_tot_bin = np.array(adjusted_M_star_particles_per_bin) + sim.recollapse_contributions_per_time_bin
    
    # Create figures directory
    if not os.path.exists("figures"):
        os.makedirs("figures")
    
    # Create plot following the original pattern
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        plt.figure(figsize=(5, 5))
        plt.subplot(1, 1, 1)
        
        plt.plot(M_tot_bin/M_star_particles_per_bin, label='total')
        plt.plot(
            np.array(adjusted_M_star_particles_per_bin) / M_star_particles_per_bin, 
            label='pristine'
        )
        plt.plot(
            sim.recollapse_contributions_per_time_bin / M_star_particles_per_bin, 
            label='recollapse'
        )
        
        plt.xlabel("temporal bins")
        plt.ylabel("contributions")
        plt.legend()
        plt.tight_layout()
        
        save_figname = f"figures/adjusted_distribs_test_Z_{test_params['Z']}_sfe_{test_params['epsilon']}_density_{test_params['n_cl']}_{test_params['ver']}.pdf"
        plt.savefig(save_figname)
        plt.close()
        
    # Print basic statistics
    print(f"\nTest completed:")
    print(f"Parameters: Z={test_params['Z']}, ε={test_params['epsilon']}, n={test_params['n_cl']}")
    print(f"Plot saved as: {save_figname}")
    
    # Return simulation object for further analysis if needed
    return sim

if __name__ == "__main__":
    test_star_formation_simulation()