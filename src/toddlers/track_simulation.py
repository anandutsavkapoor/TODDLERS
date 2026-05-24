from .imports import os, np, json, interp1d
from .constants import *
from .utils import identify_spurious_data

class TrackSimulation:
    def __init__(self, time_interval = MAX_STEP_PH1 * 2):
        self.time_interval = time_interval
        self.last_recorded_time = 0
        self.reset()

    def reset(self):
        """Reset all tracking dictionaries."""
        self.solution = {'t': [], 'y': []}
        self.shell_properties = {
            'n_shell_in': [],
            'n_shell_max': [],
            'n_shell_max_ionized': [],
            'M_shell': [],
            'f_esc_i': [],
            'f_esc_uv': [],
            'columnDensity_H1': [],
            'covering_fraction': []
        }
        self.cloud_properties = {
            'n_avg_cloud': [], 
            'f_esc_i_cloud': [],
            'R_cloud': []
        }
        self.feedback = {
            'Q_i': [],
            'L_i': [],
            'L_n': [],
            'L_mech': [],
            'F_ram': [],
            'Lya_multiplier': [],
            'eta_rad': []
        }
        self.forces = {
            'F_rad_UV_IR': [],
            'F_rad_Lya': [],
            'F_grav': [],
            'F_ext': [],
            'F_w_sn': [],
            'F_mass_gain': []
        }
        self.energetics = {
            'E_kinetic': [],
            'E_potential': []
        }

    def should_record(self, t):
        """Determine if the current time step should be recorded."""
        if t - self.last_recorded_time >= self.time_interval:
            self.last_recorded_time = t
            return True
        return False

    def update_solution(self, t, y):
        """Update solution tracking if it's time to record."""
        self.solution['t'].append(t)
        self.solution['y'].append(y.copy())

    def update_shell_properties(self, t, **kwargs):
        """Update shell properties tracking if it's time to record."""
        for key, value in kwargs.items():
            if key in self.shell_properties:
                self.shell_properties[key].append(value)

    def update_cloud_properties(self, t, **kwargs):
        """Update cloud properties tracking if it's time to record."""
        for key, value in kwargs.items():
            if key in self.cloud_properties:
                self.cloud_properties[key].append(value)

    def update_feedback(self, t, **kwargs):
        """Update feedback tracking if it's time to record."""
        for key, value in kwargs.items():
            if key in self.feedback:
                if isinstance(value, list):  # For per-generation data
                    if not self.feedback[key]:  # If it's the first entry
                        self.feedback[key] = [value]
                    else:
                        self.feedback[key].append(value)
                else:  # For non-generation-specific data
                    self.feedback[key].append(value)

    def update_forces(self, t, **kwargs):
        """Update force tracking if it's time to record."""
        for key, value in kwargs.items():
            if key in self.forces:
                    self.forces[key].append(value)

    def update_energetics(self, t, **kwargs):
        """Update energetics tracking if it's time to record."""
        for key, value in kwargs.items():
            if key in self.energetics:
                self.energetics[key].append(value)

    def get_results(self):
        """Get all tracked data as numpy arrays."""
        results = {
            'time': np.array(self.solution['t']),
            'radius': np.array([y[0] for y in self.solution['y']]),
            'velocity': np.array([y[1] for y in self.solution['y']]),
            'mass': np.array([y[3] if len(y) == 5 else y[2] for y in self.solution['y']]),
            'n_cloud_avg': np.array([y[4] if len(y) == 5 else y[3] for y in self.solution['y']]),
            'shell_properties': {
                key: np.array(value) for key, value in self.shell_properties.items()
            },
            'cloud_properties': {
                key: np.array(value) for key, value in self.cloud_properties.items()
            },
            'stellar_feedback': {
                key: np.array(value) for key, value in self.feedback.items()
            },
            'forces': {
                key: np.array(value) for key, value in self.forces.items()
            },
            'energetics': {
                key: np.array(value) for key, value in self.energetics.items()
            }
        }
        if len(self.solution['y'][0]) == 5:  # Check if energy is included
            results['energy'] = np.array([y[2] for y in self.solution['y']])
        return results
    
    def remove_last_datapoint(self):
        """Remove the last recorded data point from all tracked quantities."""
        if self.solution['t']:
            self.solution['t'].pop()
            self.solution['y'].pop()
        
        for properties in [self.shell_properties, self.cloud_properties, self.feedback, self.forces, self.energetics]:
            for key in properties:
                if properties[key]:
                    properties[key].pop()

    def trim_to_time(self, last_valid_time):
        """
        Trim all tracked data to the last valid time.
        
        Args:
            last_valid_time (float): The last valid time from the solver.
        """
        # Find the index of the last time that's less than or equal to last_valid_time
        last_valid_index = np.searchsorted(self.solution['t'], last_valid_time, side='right') - 1

        # Trim all data arrays to this index
        self.solution['t'] = self.solution['t'][:last_valid_index]
        self.solution['y'] = self.solution['y'][:last_valid_index]

        for key in self.shell_properties:
            self.shell_properties[key] = self.shell_properties[key][:last_valid_index]

        for key in self.cloud_properties:
            self.cloud_properties[key] = self.cloud_properties[key][:last_valid_index]

        for key in self.feedback:
            self.feedback[key] = self.feedback[key][:last_valid_index]

        for key in self.forces:
            self.forces[key] = self.forces[key][:last_valid_index]

        for key in self.energetics:
            self.energetics[key] = self.energetics[key][:last_valid_index]

    def write_summary(self, results, simulation_params, summary_file):
        """Write a summary of the simulation results."""
        summary = "\nEvolution run summary:\n"
        summary += f"Total number of generations: {len(results)}\n"
        summary += f"\nInitial cloud mass: {simulation_params['M_cl_init']}\n"
        summary += f"Initial cloud radius: {results[0]['cloud_radius']/PC_TO_CM:.2f} pc\n"
        if simulation_params.get('add_cover_frac', False):
            summary += f"Post-sweep covering fraction: {simulation_params.get('post_sweep_covering_fraction', COVERING_FRACTION_DEF)}\n"

        for i, generation in enumerate(results):
            summary += f"\nGeneration {i+1}:\n"
            start_time = generation['time'][0]
            end_time = generation['time'][-1]
            summary += f"  Start time: {start_time / MYR_TO_SEC:.2f} Myr\n"
            summary += f"  End time: {end_time / MYR_TO_SEC:.2f} Myr\n"
            summary += f"  Duration: {(end_time - start_time) / MYR_TO_SEC:.2f} Myr\n"
            summary += f"  Final status: {generation['status']}\n"
            summary += f"  Initial radius: {generation['radius'][0] / PC_TO_CM:.2f} pc\n"
            summary += f"  Final radius: {generation['radius'][-1] / PC_TO_CM:.2f} pc\n"
            summary += f"  Initial velocity: {generation['velocity'][0] / KM_TO_CM:.2f} km/s\n"
            summary += f"  Final velocity: {generation['velocity'][-1] / KM_TO_CM:.2f} km/s\n"
            summary += f"  Final shell mass: {generation['mass'][-1] / M_SUN:.2e} M_sun\n"
            summary += f"  Final inner shell density: {generation['shell_properties']['n_shell_in'][-1]:.2e} cm⁻³\n"
            summary += f"  Final max shell density: {generation['shell_properties']['n_shell_max'][-1]:.2e} cm⁻³\n"
            summary += f"  Initial cloud avg density: {generation['n_cloud_avg'][0]:.2e} cm⁻³\n"
            summary += f"  Final cloud avg density: {generation['n_cloud_avg'][-1]:.2e} cm⁻³\n"

            cf_data = generation['shell_properties']['covering_fraction']
            summary += f"  Initial covering fraction: {cf_data[0]:.2f}\n"
            summary += f"  Final covering fraction: {cf_data[-1]:.2f}\n"

            final_energetics = generation['energetics']
            summary += f"  Final energetics:\n"
            summary += f"    E_kinetic: {final_energetics['E_kinetic'][-1]:.2e} erg\n"
            summary += f"    E_potential: {final_energetics['E_potential'][-1]:.2e} erg\n"

            phase_transitions = generation.get('phase_transitions', [])
            if phase_transitions:
                summary += f"  Phase transitions:\n"
                for transition in phase_transitions:
                    transition_type, transition_time, transition_reason = transition
                    summary += f"    {transition_type} at {transition_time/MYR_TO_SEC:.2f} Myr, Reason: {transition_reason}\n"

        # Write summary to file
        with open(summary_file, 'w') as f:
            f.write(summary)

    def write_output_file(self, output_file, simulation_params, all_results, logger, correct_spurious=False):
        """
        Write simulation results to a JSON file.

        Args:
            output_file (str): Path to the output file.
            simulation_params (dict): Dictionary of simulation parameters.
            all_results (list): List of dictionaries, each containing a generation's results.
            logger: logger object
        """
        output_data = {
            "parameters": simulation_params,
            "generations": []
        }

        for gen_results in all_results:
            gen_data = {
                "status": gen_results["status"],
                "phase_transitions": gen_results["phase_transitions"],
                "cluster_formation_mode": gen_results["cluster_formation_mode"],
                "formation_timescale": gen_results["formation_timescale"],
                "star type": gen_results["star type"],
                "generation": gen_results["generation"],
                "cloud_mass": gen_results["cloud_mass"],
                "cloud_radius": gen_results["cloud_radius"],
                "post_sweep_covering_fraction": gen_results["post_sweep_covering_fraction"],
                "time": gen_results["time"].tolist(),
                "radius": gen_results["radius"].tolist(),
                "velocity": gen_results["velocity"].tolist(),
                "mass": gen_results["mass"].tolist(),
                "n_cloud_avg": gen_results["n_cloud_avg"].tolist(),
            }

            if correct_spurious:
                gen_results = correct_spurious_data(gen_results, logger) # fragmentation only

            if "energy" in gen_results:
                gen_data["energy"] = gen_results["energy"].tolist()

            gen_data["shell_properties"] = {
                key: value.tolist() for key, value in gen_results["shell_properties"].items()
            }
            gen_data["cloud_properties"] = {
                key: value.tolist() for key, value in gen_results["cloud_properties"].items()
            }
            gen_data["stellar_feedback"] = {
                key: value.tolist() for key, value in gen_results["stellar_feedback"].items()
            }
            gen_data["energetics"] = {
                key: value.tolist() for key, value in gen_results["energetics"].items()
            }
            gen_data["forces"] = {
                key: value.tolist() for key, value in gen_results["forces"].items()
            }
            output_data["generations"].append(gen_data)

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        # Generate and write summary
        summary_file = os.path.splitext(output_file)[0] + '.summary'
        self.write_summary(all_results, simulation_params, summary_file)
        

def load_output_file(file_path):
    """
    Load simulation results from a JSON file.

    Args:
        file_path (str): Path to the input JSON file.

    Returns:
        tuple: (simulation_params, all_results)
            simulation_params (dict): Dictionary of simulation parameters.
            all_results (list): List of dictionaries, each containing a generation's results.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    simulation_params = data["parameters"]
    path_params = extract_params_from_path(file_path)
    simulation_params.update(path_params)

    all_results = []

    for gen_data in data["generations"]:
        gen_results = {
            "status": gen_data["status"],
            "phase_transitions": gen_data["phase_transitions"],
            "cluster_formation_mode": gen_data["cluster_formation_mode"],
            "formation_timescale": gen_data["formation_timescale"],
            "star type": gen_data.get("star type", None),
            "generation": gen_data["generation"],
            "cloud_mass": gen_data["cloud_mass"],
            "cloud_radius": gen_data["cloud_radius"],
            "post_sweep_covering_fraction": gen_data.get("post_sweep_covering_fraction", None),
            "time": np.array(gen_data["time"]),
            "radius": np.array(gen_data["radius"]),
            "velocity": np.array(gen_data["velocity"]),
            "mass": np.array(gen_data["mass"]),
            "n_cloud_avg": np.array(gen_data["n_cloud_avg"]),
        }

        if "energy" in gen_data:
            gen_results["energy"] = np.array(gen_data["energy"])

        gen_results["shell_properties"] = {
            key: np.array(value) for key, value in gen_data["shell_properties"].items()
        }
        
        if "cloud_properties" in gen_data:
            gen_results["cloud_properties"] = {
                key: np.array(value) for key, value in gen_data["cloud_properties"].items()
            }
        
        gen_results["stellar_feedback"] = {
            key: np.array(value) for key, value in gen_data["stellar_feedback"].items()
        }

        gen_results["forces"] = {
            key: np.array(value) for key, value in gen_data["forces"].items()
        }
        
        gen_results["energetics"] = {
            key: np.array(value) for key, value in gen_data["energetics"].items()
        }

        all_results.append(gen_results)

    return simulation_params, all_results


def extract_params_from_path(file_path):
    """
    Extract parameters from any part of the directory path and filename.
    """
    path = os.path.normpath(file_path)
    directories = path.split(os.sep)
    filename = directories[-1]
    
    params = {}
    
    # Extract parameters from directories
    for dir_name in directories:
        if 'template_' in dir_name:
            params['template'] = dir_name.split('template_')[1]
        elif 'imf_' in dir_name:
            params['imf'] = dir_name.split('imf_')[1]
        elif 'star_type_' in dir_name:
            params['star_type'] = dir_name.split('star_type_')[1]
        elif 'cluster_mode_' in dir_name:
            params['cluster_formation_mode'] = dir_name.split('cluster_mode_')[1]
        elif 'profile_type_' in dir_name:
            params['profile_type'] = dir_name.split('profile_type_')[1]
    
    # Extract parameters from filename
    filename_w_suffix = filename.replace(".dat", "")
    parts = filename_w_suffix.split('_')
    
    # Set a default value of post_sweep_covering_fraction to unity
    params['post_sweep_covering_fraction'] = 1.0
    params['add_cover_frac'] = False
    for part in parts:
        if part.startswith('Z'):
            params['Z'] = float(part[1:])
        elif part.startswith('eta'):
            params['eta_sf'] = float(part[3:])
        elif part.startswith('n'):
            params['n_cl'] = float(part[1:])
        elif part.startswith('logM'):
            params['M_cl_init'] = 10**float(part[4:])
        elif part.startswith('cover'):
            params['post_sweep_covering_fraction'] = float(part[5:])
            params['add_cover_frac'] = True
        elif part.startswith('dynDens'):
            params['dynamic_cloud_density'] = True
        elif part.startswith('tform'):
            params['formation_timescale'] = float(part[5:]) * MYR_TO_SEC
    
    return params

def correct_spurious_data(gen_results, logger):
    """
    Correct spurious data points in the fragmentation phase by interpolation,
    using time as the basis for interpolation. Skips processing of stellar feedback data
    and logs detailed information about removed points for shell inner density only if spurious data is detected.

    Args:
        gen_results (dict): Dictionary containing a generation's results.
        logger (EvolutionLogger): The logger object for recording information.

    Returns:
        dict: Updated generation results with spurious data points corrected in the fragmentation phase.
    """
    # Identify the fragmentation phase
    phase_transitions = gen_results["phase_transitions"]
    frag_start = next((t[1] for t in phase_transitions if t[0] == 'phase1_to_fragmentation'), None)
    frag_end = next((t[1] for t in phase_transitions if t[0] == 'fragmentation_to_phase2'), None)
    
    if frag_start is not None and frag_end is not None:
        time = gen_results["time"]
        frag_mask = (time >= frag_start) & (time <= frag_end)
        frag_time = time[frag_mask]
        
        n_shell_in_frag = gen_results["shell_properties"]["n_shell_in"][frag_mask].copy()
        non_spurious_mask = identify_spurious_data(n_shell_in_frag)
        
        spurious_indices = np.where(~non_spurious_mask)[0]
        
        if len(spurious_indices) > 0:
            spurious_times = frag_time[spurious_indices]
            
            logger.info("Spurious Data Correction Report:")
            logger.info(f"Total points in fragmentation phase: {len(frag_time)}")
            logger.info(f"Number of points identified as spurious: {len(spurious_indices)}")
            logger.info(f"Times of removed points: {spurious_times.tolist()}")
            
            non_spurious_time = frag_time[non_spurious_mask]
            
            for key, value in gen_results.items():                
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, np.ndarray) and len(v) == len(time):
                            frag_data = v[frag_mask]
                            if len(frag_data) == len(non_spurious_mask):
                                try:
                                    f = interp1d(non_spurious_time, frag_data[non_spurious_mask], 
                                                 kind='linear', fill_value='extrapolate')
                                    interpolated_data = f(frag_time)
                                    v[frag_mask] = interpolated_data
                                    
                                    # Log detailed information only for shell inner density
                                    if key == "shell_properties" and k == "n_shell_in":
                                        original_values = frag_data[spurious_indices]
                                        interpolated_values = interpolated_data[spurious_indices]
                                        logger.info("Removed data for shell inner density (n_shell_in):")
                                        logger.info(f"  Original values: {original_values.tolist()}")
                                        logger.info(f"  Interpolated values: {interpolated_values.tolist()}")
                                    else:
                                        logger.info(f"Corrected spurious data for {key}.{k}")
                                except ValueError as e:
                                    logger.error(f"Interpolation error for {key}.{k}: {str(e)}")
                elif isinstance(value, np.ndarray) and len(value) == len(time):
                    frag_data = value[frag_mask]
                    if len(frag_data) == len(non_spurious_mask):
                        try:
                            f = interp1d(non_spurious_time, frag_data[non_spurious_mask], 
                                         kind='linear', fill_value='extrapolate')
                            interpolated_data = f(frag_time)
                            value[frag_mask] = interpolated_data
                            logger.info(f"Corrected spurious data for {key}")
                        except ValueError as e:
                            logger.error(f"Interpolation error for {key}: {str(e)}")
        else:
            logger.info("No spurious data detected in the fragmentation phase.")
    else:
        logger.info("No fragmentation phase found. Skipping spurious data correction.")
    
    return gen_results