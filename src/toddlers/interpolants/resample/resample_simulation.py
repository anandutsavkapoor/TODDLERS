import numpy as np
import TODDLERS_recollapse_handling as TORCH
from cloud_sampling import CloudDistribution
from particle import Particle
import matplotlib.pyplot as plt
import os
import warnings
from names_and_constants import *

class GalaxySimulation:

    """ 
    This class takes the initial TODDLERS input file with columns x, y, z, h, t, Z, SFE, n_cl, init_M_particle
    length scales (x, y, z, h) must be in kpc or pc
    t must be in yr or Myr
    SFE needs to be a fraction within the range of the TODDLERS parameters (0.01 - 0.15)
    n_cl should be in the units of cm-3 within the range of TODDLERS parameters (10 - 320 cm-3)
    init_M_particle or the initial particle mass in the simulation should be in Msun
    Returns: A file which can be used with SEDs from TODDLERS in the form SED = weight * f(t, Z, SFE, n_cl, M_cl)
    The weight accounts for recollpase in the models 
    The output file has columns x, y, z, h, t, Z, SFE, n_cl, M_cl, weight
    """

    def __init__(self, filename, length_unit, age_unit, sfe_override=None, density_override=None, alpha=EXPONENT):
        self.particles = []
        self.length_unit = length_unit
        self.age_unit = age_unit
        self.sfe_override = sfe_override
        self.density_override = density_override
        self.alpha = alpha
        self._load_from_file(filename)
        self._compute_relative_mass_fractions()

    def _convert_length(self, value, unit):
        if unit == 'kpc':
            return value  
        elif unit == 'pc':
            return value / 1000  

    def _convert_age(self, value, unit):
        if unit == 'Myr':
            return value 
        elif unit == 'yr':
            return value / 1e6 

    def _load_from_file(self, filename):
        """The script uses kpc as length units and Myr as time units, 
        ensure the conversion is done if units are different """
        with open(filename, 'r') as f:
            print("converting units to kpc, Myr. Input units are: ", self.length_unit, self.age_unit)
            for line in f:
                if not line.startswith('#') and line.strip():
                    data = list(map(float, line.split()))
                    data[0] = self._convert_length(data[0], self.length_unit)  # x-coordinate
                    data[1] = self._convert_length(data[1], self.length_unit)  # y-coordinate
                    data[2] = self._convert_length(data[2], self.length_unit)  # z-coordinate
                    data[3] = self._convert_length(data[3], self.length_unit)  # smoothing length
                    data[4] = self._convert_age(data[4], self.age_unit)  # age

                    if self.sfe_override is not None:
                        data[6] = self.sfe_override
                    if self.density_override is not None:
                        data[7] = self.density_override
                    
                    particle = Particle(*data)
                    self.particles.append(particle)

    def _compute_relative_mass_fractions(self):
        cloud_distr = CloudDistribution(self.alpha, MASS_BIN_CENTERS, SAMPLE_SIZE)
        sampled_masses = cloud_distr.sample_and_filter()
        unique, counts = np.unique(sampled_masses, return_counts=True)
        total_mass_system = np.sum(unique * counts)
        self.relative_mass_fractions = dict(zip(unique, (unique * counts) / total_mass_system))

    def decompose_particles(self, r, smooth_age_scale):
        """Decompose all particles in the simulation
        r: spread decomposed particles in a sphere of this radius 
        smooth_age_scale: add Gaussian noise with this scale on top of the ages to smoothen them 
        Both r and smooth_age_scale should be in the same units as the simulation
        """
        r = self._convert_length(r, self.length_unit)
        smooth_age_scale = self._convert_age(smooth_age_scale, self.age_unit)

        decomposed_particles = []
        total_particles = len(self.particles)

        # Calculate step size for progress reporting
        # Use max(1, total_particles/100) to avoid division by zero
        percent_step = max(1, total_particles / 100)

        for i, particle in enumerate(self.particles):
            mass_fractions = [self.relative_mass_fractions[mass_value] for mass_value in MASS_BIN_CENTERS]
            decomposed_particles.extend(particle.decompose(r, smooth_age_scale, mass_fractions))
            
            # Check if we should print progress
            if total_particles < 100:  # For small numbers of particles
                if (i + 1) == total_particles:  # Only print at the end
                    print("Simulation particle decomposition into mass bins: 100% complete")
            else:  # For larger numbers of particles
                if (i + 1) % int(percent_step) == 0 or i == total_particles - 1:
                    percent_complete = (i + 1) / total_particles * 100
                    print(f"Simulation particle decomposition into mass bins, progress: {percent_complete:.1f}% complete")

        self.particles = decomposed_particles

    def to_numpy_array(self):
        num_particles = len(self.particles)
        data_array = np.zeros((num_particles, 7))
        for i, particle in enumerate(self.particles):
            data_array[i] = [particle.age, particle.metallicity, particle.sfe, particle.density, particle.mass, 
                             particle.cloud_mass, particle.decomposed_mass]
        
        return data_array
                
    def save_to_file(self, filename, cloud_mass, weight):
        with open(filename, 'w') as f:
                # Write headers
                f.write("# SKIRT 9 import format for SFR with TODDLERS model#\n")
                f.write("# Column 1: x-coordinate (kpc)\n")
                f.write("# Column 2: y-coordinate (kpc)\n")
                f.write("# Column 3: z-coordinate (kpc)\n")
                f.write("# Column 4: size h (kpc)\n")
                f.write("# Column 5: Age (Myr)\n")
                f.write("# Column 6: metallicity (1)\n")
                f.write("# Column 7: SFE (1)\n")
                f.write("# Column 8: Cloud Particle Density (1/cm3)\n")
                f.write("# Column 9: Associated Cloud mass (Msun)\n")
                f.write("# Column 10: Weight (1)\n")

                
                for i, particle in enumerate(self.particles):
                    line = f"{particle.x}   {particle.y}    {particle.z}    {particle.h}    {particle.age}  "
                    line += f"{particle.metallicity}    {particle.sfe}  {particle.density}  "
                    line += f"{cloud_mass[i]}   " 
                    line += f"{weight[i]}   "
                    f.write(line + "\n")


def process_simulation(input_filename, input_length_unit, input_age_unit, output_filename, sfe_override=None, density_override=None, r=0, smooth_age_scale=0):
    # Load the simulation from the file
    sim = GalaxySimulation(input_filename, input_length_unit, input_age_unit, sfe_override=sfe_override, density_override=density_override)

    # Split all particles
    sim.decompose_particles(r=r, smooth_age_scale=smooth_age_scale)

    # Extract age and mass data of split particles
    simulation_data = sim.to_numpy_array()

    # Call the toddlers recollapse handler
    TORCHsim   = TORCH.StarFormationSimulation(Z=np.nan, epsilon=np.nan, n_cl=np.nan, simulation_data=simulation_data)
    weight     = TORCHsim.w_ki
    cloud_mass = TORCHsim.M_cloud_particles


    # Plot data
    M_star_particles_per_bin = [np.sum(TORCHsim.M_star_particles[TORCHsim.temporal_bins == bin_idx]) for bin_idx in range(TORCH.N_TEMPORAL_BINS)]
    adjusted_M_star_particles_per_bin = [np.sum(TORCHsim.M_cloud_particles[TORCHsim.temporal_bins == bin_idx] * TORCHsim.w_ki[TORCHsim.temporal_bins == bin_idx] * TORCHsim.epsilon[TORCHsim.temporal_bins == bin_idx]) for bin_idx in range(TORCH.N_TEMPORAL_BINS)]
    M_tot_bin = adjusted_M_star_particles_per_bin + TORCHsim.recollapse_contributions_per_time_bin

 
    if not os.path.exists("recollapse_figures"):
        os.makedirs("recollapse_figures")
    
    warnings.simplefilter("ignore", category=RuntimeWarning)
    plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    plt.plot(M_tot_bin/M_star_particles_per_bin, label='total')
    plt.plot(np.array(adjusted_M_star_particles_per_bin) / M_star_particles_per_bin, label='pristine')
    plt.plot(TORCHsim.recollapse_contributions_per_time_bin / M_star_particles_per_bin, label='recollapse')
    plt.xlabel("temporal bins")
    plt.ylabel("contributions")
    plt.legend()
    plt.tight_layout()
    save_figname = f"recollapse_figures/adjusted_distribs_{os.path.basename(input_filename)}_sfe_{sfe_override}_density_{density_override}.pdf"
    plt.savefig(save_figname)
    # Save the updated simulation data
    sim.save_to_file(output_filename, cloud_mass, weight)
    return sim


if __name__ == '__main__':
    # Load the simulation from a file
    in_file  = 'halo1SFR.dat'
    out_file = 'halo1SFR_updated.dat'
    process_simulation(input_filename=in_file, input_length_unit='kpc', input_age_unit='Myr', output_filename=out_file, sfe_override=0.025, density_override=320., r=0.25, smooth_age_scale=0.1)