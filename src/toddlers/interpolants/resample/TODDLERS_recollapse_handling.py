import h5py
import numpy as np
import sys
sys.path.append('../')
from cloud_sampling import CloudDistribution
import matplotlib.pyplot as plt
import warnings
from particle import Particle
from names_and_constants import *


def find_closest_value(value, lst):
    """Find the closest value in lst to the given value."""
    return min(lst, key=lambda x: abs(x - value))

class RecollapseData:

    """ Reads an hdf5 file containing recollapse times for the TODDLERS parameters.
    hdf5 should already be present, created in interpolants/generate_interpolants.py """

    def __init__(self):

        self.Z_all = np.array([.001, .004, .008, .02, .04])
        self.epsilon_all = np.array([.01, .025, .05, .075, .1, .125, .15])
        self.n_cl_all = np.around(10**np.arange(1, 3.5, .30102), 0)
        self.M_cl_all = np.array([10**np.around(i, 2) for i in np.arange(5, 6.8, .25)])
        self.logM_cl_all = np.log10(self.M_cl_all)

    def create_v1_structure(self, v2_file: str) -> str:
        """Convert v2 HDF5 structure to v1 format in a new file for faster querying.
        
        This function restructures a v2 format HDF5 file (which uses nested groups and attributes)
        into a v1 format file (which uses flat structure with concatenated keys). This conversion
        enables faster data access by reducing the number of HDF5 group traversals needed.
        
        V2 structure (input):
            recollapse_data/
                simulations/
                    sim_1/
                        attrs (Z, eta_sf, n_cl, logM)
                        recollapse_times_myr
                    sim_2/
                        ...
        
        V1 structure (output):
            recollapse_data/
                Z_epsilon_n_cl_logM_cl/
                    times
        
        Args:
            v2_file: Path to the v2 format HDF5 file
            
        Returns:
            str: Path to the newly created v1 format HDF5 file
            
        Note:
            The output filename is derived from the input filename by adding '_v1_format'
            before the .hdf5 extension.
        """
        v1_filename = HDF5_FILENAME.replace('.hdf5', '_v1_format.hdf5')
        
        with h5py.File(v2_file, 'r') as f_v2, h5py.File(v1_filename, 'w') as f_v1:
            # Create main group
            recollapse_group = f_v1.create_group("recollapse_data")
            sims_group = f_v2['recollapse_data/simulations']
            
            # Convert each simulation to v1 format
            for sim_key in sims_group:
                sim = sims_group[sim_key]
                # Create v1-style key
                key = f"{sim.attrs['Z']}_{sim.attrs['eta_sf']}_{sim.attrs['n_cl']}_{sim.attrs['logM']}"
                
                # Create dataset with times
                if 'recollapse_times_myr' in sim:
                    times = sim['recollapse_times_myr'][:]
                    recollapse_group.create_dataset(key + '/times', data=times)
        
        return v1_filename

    def query_from_hdf5(
            self,
            Z: float,
            epsilon: float,
            n_cl: float,
            logM_cl: float,
            version: str = 'v1'
        ) -> dict:
        """Query recollapse data from HDF5 file, supporting both v1 and v2 formats.
        
        This function retrieves recollapse times data from an HDF5 file. For v2 format,
        it automatically creates and uses a v1 format copy for faster access. The v1
        format provides better performance by using a flat structure with direct key access.
        
        Args:
            Z: Metallicity value (dimensionless)
            epsilon: Star formation efficiency (dimensionless)
            n_cl: Cloud number density (cm^-3)
            logM_cl: Log of cloud mass (log M_sun)
            version: Format version to use, either 'v1' or 'v2'. Defaults to 'v1'
            
        Returns:
            dict: Contains key 'times' with numpy array of recollapse times.
                For v1: Original times array from file
                For v2: Times array from file prepended with t=0
                Returns empty array if no matching data is found
                
        Note:
            For v2 queries, the function creates a v1-format copy of the file if it
            doesn't exist. This conversion happens only once and subsequent queries
            use the converted file for faster access.
        """
        data = {'times': np.array([])}
        
        try:
            if version == 'v2':
                # Create v1 structure if it doesn't exist
                v1_file = HDF5_FILENAME.replace('.hdf5', '_v1_format.hdf5')
                if not os.path.exists(v1_file):
                    v1_file = self.create_v1_structure(HDF5_FILENAME)
                    
                # Use v1 file for query
                with h5py.File(v1_file, 'r') as f:
                    key = f"{Z}_{epsilon}_{n_cl}_{logM_cl}"
                    if key in f["recollapse_data"]:
                        times = f[f"recollapse_data/{key}/times"][:]
                        data['times'] = np.concatenate([[0], times])
                    else:
                        warnings.warn(f"The key {key} was not found in the HDF5 file.")
            else:
                # Original v1 code
                with h5py.File(HDF5_FILENAME, 'r') as f:
                    key = f"{Z}_{epsilon}_{n_cl}_{logM_cl}"
                    if key in f["recollapse_data"]:
                        sub_group = f[f"recollapse_data/{key}"]
                        if 'times' in sub_group:
                            data['times'] = sub_group['times'][:]
                    else:
                        warnings.warn(f"The key {key} was not found in the HDF5 file.")
                        
        except FileNotFoundError:
            print(f"Error: The file {HDF5_FILENAME} was not found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
        return data

class StarFormationSimulation:
    """ This class assigns a weight to the simulation particles to account for recollapse contributions.
    Supports both v1 (original) and v2 (new) HDF5 file formats.
    
    The simulation data should have the following columns, with decomposition done already:
    Col.1 : Age [Myr]
    Col.2 : Metallicity [1]
    Col.3 : SFE [fraction]
    Col.4 : Cloud density [cm-3]
    Col.5 : Original M* from the simulation [Msun], as each particle is decomposed into several cloud components, this value will repeat for each cloud component.
    Col.6 : M_cloud values [Msun]
    Col.7 : Decomposed M*, i.e., M* associated with each M_cloud above

    main output : weights associated with each particle with a consideration to recollapsed mass
    This weight value (w_ki) is initalized and modified in the function calculate_recollapse_contribution of this class
    """
    def __init__(self, Z, epsilon, n_cl, simulation_data=None, num_particles=2000, version='v1'):
        """num_particles is prior to decomposition, only applicable when simulation_data isnt provided"""
        self.data_obj = RecollapseData()
        self.simulation_data = simulation_data
        self.version = version
        
        if len(self.simulation_data) > 0:            
            self.leftover_time = AGE_LIMIT - self.simulation_data[:, 0]
            self.Z = self.simulation_data[:, 1]
            self.epsilon = self.simulation_data[:, 2]
            self.n_cl = self.simulation_data[:, 3]
            self.M_star_original = self.simulation_data[:, 4]
            self.M_cloud_particles = self.simulation_data[:, 5]
            self.M_star_particles = self.simulation_data[:, 6]
            self.num_particles = len(self.M_star_particles)
        else:
            self.num_particles = num_particles
            self.ages, _, self.M_cloud_particles, self.M_star_particles = self.generate_and_decompose_star_formation_data()
            self.leftover_time = AGE_LIMIT - self.ages
            self.Z = Z * np.ones_like(self.ages)
            self.epsilon = epsilon * np.ones_like(self.ages)
            self.n_cl = n_cl * np.ones_like(self.ages)
         
        self.temporal_bins = self.bin_temporal_data()
        self.w_ki = self.get_initial_stellar_mass_weights()
        self.recollapse_contributions_per_particle, self.recollapse_contributions_per_time_bin = self.calculate_recollapse_contribution()

    def generate_and_decompose_star_formation_data(self, r=SPREAD_RADIUS, smooth_age_scale=SMOOTH_AGE_SCALE):
        """ This generates synthetic data if simulation data isnt avaialable or for testing"""
        ages = np.linspace(AGE_START, AGE_LIMIT, self.num_particles, endpoint=True) # uniform sampling by definition
        M_star_variation = 0. * M_STAR_MEAN
        M_star_particles = np.random.uniform(M_STAR_MEAN - M_star_variation, 
                                            M_STAR_MEAN + M_star_variation, self.num_particles)
        
        # Compute relative mass fractions for the given power-law exponent
        cloud_distr = CloudDistribution(EXPONENT, MASS_BIN_CENTERS, SAMPLE_SIZE)
        sampled_masses = cloud_distr.sample_and_filter()
        unique, counts = np.unique(sampled_masses, return_counts=True)
        total_mass_system = np.sum(unique * counts)
        relative_mass_fractions = dict(zip(unique, (unique * counts) / total_mass_system))
        # Decompose particles
        decomposed_ages = []
        original_masses = []
        cloud_masses = []
        decomposed_mass = []
        power_law_mass_fractions = [relative_mass_fractions[mass_value] for mass_value in MASS_BIN_CENTERS]
        print(power_law_mass_fractions)
        for age, mass in zip(ages, M_star_particles):
            # Calculate mass fractions based on relative contribution and the particle's mass
            # x, y, z, h, age, metallicity, sfe, density, orig_mass, cloud_mass=None, decomposed_mass=None
            particle = Particle(0, 0, 0, 0, age, 0, 0, 0, mass)  # Only age and mass are relevant for decomposition
            decomposed_particles = particle.decompose(r, smooth_age_scale, power_law_mass_fractions)

            for dp in decomposed_particles:
                decomposed_ages.append(dp.age)
                original_masses.append(dp.mass)
                cloud_masses.append(dp.cloud_mass)
                decomposed_mass.append(dp.decomposed_mass)
    
        return np.array(decomposed_ages), np.array(original_masses), np.array(cloud_masses), np.array(decomposed_mass)


    def bin_temporal_data(self):
        return np.digitize(self.leftover_time, np.linspace(0, AGE_LIMIT, N_TEMPORAL_BINS))


    def get_initial_stellar_mass_weights(self):
        w_ki = self.M_star_particles / (self.epsilon * self.M_cloud_particles)
        return w_ki

    def calculate_recollapse_contribution(self):
        Z_all = METALLICITIES 
        epsilon_all = STAR_FORMATION_EFFICIENCIES 
        n_cl_all = CLOUD_DENSITIES
        recollapse_contributions_per_particle = np.zeros_like(self.M_star_particles)
        recollapse_contributions_per_bin = np.zeros(N_TEMPORAL_BINS)
        
        for bin_idx in range(N_TEMPORAL_BINS):
            indices_in_bin = np.where(self.temporal_bins == bin_idx)[0]

            if recollapse_contributions_per_bin[bin_idx] > 0:
                total_M_star_particles_ibin = np.sum(self.M_star_particles[self.temporal_bins == bin_idx])
                total_recollapse_ibin = recollapse_contributions_per_bin[bin_idx]
                ## avoid negative numbers, could be an issue with high recollapse models and falling SFRs
                g_ki = max((total_M_star_particles_ibin - total_recollapse_ibin) / total_M_star_particles_ibin, 1e-6)
                self.w_ki[self.temporal_bins == bin_idx] *= g_ki

            for idx in indices_in_bin:
                # allow for each particle to have its own parameters, but use nearest neighbour
                Z_closest = find_closest_value(self.Z[idx], Z_all)
                epsilon_closest = find_closest_value(self.epsilon[idx], epsilon_all)
                n_cl_closest = find_closest_value(self.n_cl[idx], n_cl_all)
                
                # Pass version to query_from_hdf5
                data = self.data_obj.query_from_hdf5(
                    Z_closest, 
                    epsilon_closest, 
                    n_cl_closest, 
                    np.log10(self.M_cloud_particles[idx]),
                    version=self.version
                )
                
                recollapse_times = data.get('times', [])
                if len(recollapse_times) == 0:
                    raise ValueError(
                    f"No recollapse times found for parameters: Z={Z_closest}, epsilon={epsilon_closest}, "
                    f"n_cl={n_cl_closest}, logM={np.log10(self.M_cloud_particles[idx])}.\n"
                    "The 'times' data retrieved from the HDF5 file is empty.\n"
                    "This list should at least contain 0.\n"
                    "Most likely the key names are incorrect, perhaps missing a decimal."
                    )
                
                if len(recollapse_times) > 1:  # Skip if only t=0 exists
                    n_gen = 2
                    for time in recollapse_times[1:]:  # Skip t=0
                        n = n_gen - 1
                        subsequent_bin_idx = np.digitize(self.leftover_time[idx] + time, 
                                                       np.linspace(0, AGE_LIMIT, N_TEMPORAL_BINS))
                        
                        if self.leftover_time[idx] + time < AGE_LIMIT:
                            contribution = ((1 - self.epsilon[idx])**n) * self.w_ki[idx] * self.epsilon[idx] * self.M_cloud_particles[idx]
                            recollapse_contributions_per_particle[idx] += contribution
                            recollapse_contributions_per_bin[subsequent_bin_idx] += contribution
                        n_gen += 1
            # Print completion percentage after each bin is processed
            completion_percentage = ((bin_idx + 1) / N_TEMPORAL_BINS) * 100
            print(f"weight calculation completion: {completion_percentage:.2f}%")

        return recollapse_contributions_per_particle, recollapse_contributions_per_bin


if __name__ == '__main__':

    sim = StarFormationSimulation(Z=0.02, epsilon=0.025, n_cl=320.0, simulation_data=[])
    SFR_without_adjustments = np.sum(sim.M_star_particles) / 30e6
    adjusted_M_star_particles = sim.M_cloud_particles * sim.w_ki * sim.epsilon
    SFR_with_new_adjustments = (np.sum(adjusted_M_star_particles) + np.sum(sim.recollapse_contributions_per_time_bin)) / 30e6


    # Compute the M_star_particles for each temporal bin
    M_star_particles_per_bin = [np.sum(sim.M_star_particles[sim.temporal_bins == bin_idx]) for bin_idx in range(N_TEMPORAL_BINS)]

    # Compute the adjusted M_star_particles for each temporal bin
    adjusted_M_star_particles_per_bin = [np.sum(sim.M_cloud_particles[sim.temporal_bins == bin_idx] * sim.w_ki[sim.temporal_bins == bin_idx] * sim.epsilon) for bin_idx in range(N_TEMPORAL_BINS)]

    M_tot_bin = adjusted_M_star_particles_per_bin + sim.recollapse_contributions_per_time_bin

    plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    plt.plot(M_tot_bin/M_star_particles_per_bin, label='total')
    plt.plot(np.array(adjusted_M_star_particles_per_bin) / M_star_particles_per_bin, label='pristine')
    plt.plot(sim.recollapse_contributions_per_time_bin / M_star_particles_per_bin, label='recollapse')
    plt.xlabel("temporal bins")
    plt.ylabel("contributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig('adjusted_distribs.pdf')


    plt.figure(figsize=(5, 5))
    plt.semilogy(M_tot_bin)
    plt.semilogy(M_star_particles_per_bin)
    plt.tight_layout()
    plt.savefig('total_Mstar.pdf')


    