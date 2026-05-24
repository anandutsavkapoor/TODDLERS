from .imports import np
from .constants import *

class DissolvedModelGenerator:
    """
    Generates Cloudy input files for post-dissolution stellar spectra.
    
    This generator creates Cloudy models with negligible gas content to obtain
    effectively direct stellar spectra. It uses a very small mass stopping criterion
    and includes only stellar radiation without any old stellar background.
    
    Attributes:
        base (BaseCloudyInputGenerator): The base input generator instance.
        logger (Logger): Logger instance for recording progress.
        model_prefix (str): Prefix for the dissolved model files.
    """
    
    def __init__(self, base_generator, logger=None):
        """
        Initialize the DissolvedModelGenerator.

        Args:
            base_generator (BaseCloudyInputGenerator): The base input generator instance.
            logger (Logger, optional): Logger instance. Defaults to None.
        """
        self.logger = logger
        self.base = base_generator
        self.model_prefix = 'dissolved'

    def get_source_options(self, t):
        """
        Get source options - only direct stellar radiation.

        Args:
            t (float): Current time in seconds.

        Returns:
            list: Cloudy commands for source specification.
        """
        return self.base.get_primary_source_options(t, scaling=1)

    def get_density_options(self):
        """
        Get density options - uses very low ISM density.

        Returns:
            list: Cloudy commands for density specification.
        """
        return [
            f'hden {np.log10(N_ISM)}', 
            'constant density'
        ]

    def get_geometry_options(self, t):
        """
        Get geometry options - minimal sphere at the dissolution radius.

        Args:
            t (float): Current time in seconds.

        Returns:
            list: Cloudy commands for geometry specification.
        """
        R = self.base.radius_interp(t) # gives last radius beyond dissolution time
        return [
            'sphere',
            f'Radius inner = {R / PC_TO_CM:.6e} parsec linear'
        ]

    def get_stopping_criteria(self):
        """
        Get stopping criteria - negligible mass stopping.

        Returns:
            list: Cloudy commands for stopping criteria.
        """
        return [f"stop mass {np.log10(ZERO_GAS_MASS)}"]

    def get_output_options(self):
        """
        Get output options - only continuum output needed.

        Returns:
            list: Cloudy commands for output specification.
        """
        return ['save last continuum ".cont" units micron']

    def write_input_file(self, t):
        """
        Write the complete Cloudy input file for dissolved model.

        Args:
            t (float): Current time in seconds.
        """
        file_path = f"{self.model_prefix}_{t/MYR_TO_SEC:.2f}.in"
        
        with open(file_path, 'w') as f:
            f.write(f"# Dissolved model (direct stellar spectrum) at t = {t/MYR_TO_SEC:.2f} Myr\n")
            
            for option in self.get_source_options(t):
                f.write(f"{option}\n")

            for option in self.get_density_options():
                f.write(f"{option}\n")

            for option in self.get_geometry_options(t):
                f.write(f"{option}\n")
            
            for option in self.base.get_abundance_options():
                f.write(f"{option}\n")
            
            for option in self.base.get_cr_options(t, scale_with_Z=False):
                f.write(f"{option}\n")
                        
            for option in self.get_stopping_criteria():
                f.write(f"{option}\n")
            
            for option in self.get_output_options():
                f.write(f"{option}\n")