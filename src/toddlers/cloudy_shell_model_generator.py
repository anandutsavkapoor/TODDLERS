from .imports import np
from .constants import *

class ShellModelGenerator:
    """
    Generates Cloudy input files for the shell model.

    This class is responsible for creating Cloudy input files specific to the
    shell model, which represents the shell of gas around young stellar populations.

    Attributes:
        base (BaseCloudyInputGenerator): The base input generator instance.
        model_prefix (str): Prefix for the shell model files.

    Methods:
        get_density_options(t): Generates density-related options for the shell model.
        get_geometry_options(t): Generates geometry-related options for the shell model.
        get_stopping_criteria(t): Generates stopping criteria for the Cloudy simulation.
        write_input_file(t): Writes the complete Cloudy input file for the shell model.
    """

    def __init__(self, base_generator, logger=None, use_density_turbulence=True):
        """
        Initialize the ShellModelGenerator.

        Args:
            base_generator (BaseCloudyInputGenerator): The base input generator instance.
            logger: Logger instance for recording progress.
            use_density_turbulence (bool, optional): Whether to use density-dependent
                turbulence scaling. Defaults to True.
        """
        self.logger = logger
        self.base = base_generator
        self.model_prefix = 'shell'
        self.use_density_turbulence = use_density_turbulence

    def get_density_options(self, t):
        n_shell_in = self.base.n_shell_in_interp(t)
        n_H = n_shell_in * self.base.nuclei_to_h  # convert total nuclei density to hydrogen number density
        return [
            'Constant pressure',
            f'hden {np.log10(n_H)}',
            self.get_turbulence_str(t)
        ]

    def get_geometry_options(self, t):
        R = self.base.radius_interp(t)
        covering_fraction = self.base.get_effective_covering_fraction(t)
        return [
            f'Covering factor = {covering_fraction:.4e}',
            'sphere',
            f'Radius inner = {R / PC_TO_CM:.6e} parsec linear'
        ]

    def get_stopping_criteria(self, t):
        M_sh = self.base.shell_mass_interp(t)
        return [
            "Stop temperature off",
            f'Stop mass {np.log10(M_sh)}'
        ]

    def get_turbulence_str(self, t):
        """
        Generate turbulence command string for shell model Cloudy input.
        
        The turbulence behavior depends on use_density_turbulence setting:

        - If True: Uses HIGH_DENSITY_TURB_FRAC (10%) of shell velocity with pressure
          in dense regions (log(n) > 4)
        - If False: Always uses DEFAULT_TURB_FRAC without pressure (original behavior)
        
        Args:
            t (float): Current time in seconds.
            
        Returns:
            str: Cloudy command for turbulence settings.
        """
        v_sh = self.base.velocity_interp(t)
        
        if self.use_density_turbulence:
            n_shell_in = self.base.n_shell_in_interp(t)  # total nuclei density, cm^-3
            if np.log10(n_shell_in) > LOG_DENSITY_THRESHOLD_TURB:
                return f'turbulence {HIGH_DENSITY_TURB_FRAC * abs(v_sh)/KM_TO_CM:.6f} km/sec'
        
        return f'turbulence {DEFAULT_TURB_FRAC * abs(v_sh)/KM_TO_CM:.6f} km/sec no pressure'

    def write_input_file(self, t):
        file_path = f"{self.model_prefix}_{t/MYR_TO_SEC:.2f}.in"
        with open(file_path, 'w') as f:
            f.write(f"# Shell model at t = {t/MYR_TO_SEC:.2f} Myr\n")
            
            for option in self.base.get_primary_source_options(t):
                f.write(f"{option}\n")
            
            for option in self.get_density_options(t):
                f.write(f"{option}\n")
            
            for option in self.get_geometry_options(t):
                f.write(f"{option}\n")
            
            for option in self.base.get_abundance_options():
                f.write(f"{option}\n")
            
            for option in self.base.get_grain_options():
                f.write(f"{option}\n")
            
            for option in self.base.get_cr_options(t):
                f.write(f"{option}\n")
            
            for option in self.base.get_other_physics_options():
                f.write(f"{option}\n")

            for option in self.base.get_speedup_options():
                f.write(f"{option}\n")
                
            for option in self.get_stopping_criteria(t):
                f.write(f"{option}\n")
            
            for option in self.base.get_output_options():
                f.write(f"{option}\n")