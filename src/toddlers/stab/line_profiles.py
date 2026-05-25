import numpy as np
from scipy.interpolate import interp1d

class LineProfileGenerator:
    """
    Handles creation and manipulation of spectral line profiles for high-resolution SEDs.
    """
    
    def __init__(self, n_points=37, n_sigma=4, clamp_fraction=1e-9):
        """
        Initialize the LineProfileGenerator.
        
        Args:
            n_points (int): Number of points per line profile
            n_sigma (float): Number of standard deviations for profile width
            clamp_fraction (float): Small fraction to clamp profile edges
        """
        self.n_points = n_points
        self.n_sigma = n_sigma
        self.clamp_fraction = clamp_fraction
    
    def create_line_profiles(self, line_lums, resolution):
        """
        Creates gaussian profiles for spectral lines.
        
        Args:
            line_lums (dict): Dictionary mapping line identifiers (element, ion, wavelength)
                           to luminosities
            resolution (float): Spectral resolution (lambda/delta_lambda)
            
        Returns:
            dict: Dictionary mapping wavelengths to (profile_x, profile_y) tuples\
        
        Note:
            This method is agnostic to the units, thus if Cloudy output is directly used, 
            the line profiles will have a unit erg/s/micron.
        """
        profiles = {}
        
        for line_id, Ltot in line_lums.items():
            # Extract wavelength from line identifier tuple
            cwl = line_id[2]  # Central wavelength
            
            # if luminosity is zero or very small, use a small value
            if Ltot <= 0 or Ltot < 1e-99:
                Ltot = 1e-99
            
            # Calculate profile points
            wl_gauss = np.linspace(
                cwl - ((1 - self.clamp_fraction)*(cwl/(2*resolution))),
                cwl + ((1 - self.clamp_fraction)*(cwl/(2*resolution))), 
                self.n_points, 
                endpoint=True
            )
            
            # Calculate sigma based on resolution
            sigma = ((1 - self.clamp_fraction)*cwl)/(2*self.n_sigma*resolution)
            
            # Create normalized Gaussian profile
            profile = np.exp(-0.5 * ((wl_gauss - cwl) / sigma)**2)
            
            # Normalize by numerical integration
            profile_area = np.trapz(profile, wl_gauss)
            if profile_area > 0:  # Avoid division by zero
                profile = profile / profile_area
            
            # Scale by total luminosity, area under profile should be Ltot
            profile = profile * Ltot
            
            # Add zero-flux endpoints
            final_wavelengths = np.concatenate([
                [cwl - cwl/(2*resolution)],
                wl_gauss,
                [cwl + cwl/(2*resolution)]
            ])
            
            final_luminosities = np.concatenate([
                [1e-99],
                profile,
                [1e-99]
            ])
            
            profiles[cwl] = (final_wavelengths, final_luminosities)

        self.check_profile_overlaps(profiles)

        return profiles

    def check_profile_overlaps(self, profiles):
        """
        Check if any line profiles overlap.
        
        Args:
            profiles (dict): Dictionary mapping wavelengths to (profile_x, profile_y) tuples
            
        Returns:
            bool: True if profiles overlap, False otherwise
            
        Raises:
            ValueError: If overlapping profiles are found, with details about which lines overlap
        """
        wavelengths = sorted(profiles.keys())
        
        for i in range(len(wavelengths)-1):
            cwl1 = wavelengths[i]  # Central wavelength of first line
            cwl2 = wavelengths[i+1]  # Central wavelength of next line
            
            # Get the wavelength ranges for both profiles
            x1, _ = profiles[cwl1]
            x2, _ = profiles[cwl2]
            
            # Get max of first profile and min of second profile
            max_x1 = x1[-1]  # Last point of first profile
            min_x2 = x2[0]   # First point of second profile
            
            # Check for overlap
            if max_x1 >= min_x2:
                overlap = max_x1 - min_x2
                raise ValueError(
                    f"Line profiles overlap by {overlap:.2e} meters:\n"
                    f"Line 1: central wavelength = {cwl1:.2e}m, extends to {max_x1:.2e}m\n"
                    f"Line 2: central wavelength = {cwl2:.2e}m, starts at {min_x2:.2e}m"
                )
        
        return False

    def merge_profiles(self, continuum_waves, continuum, line_profiles):
        """
        Merge continuum and line profiles onto a combined wavelength grid.
        Expects specific luminosity in the same units as the lines.
        
        Args:
            continuum_waves (np.ndarray): Original wavelength points for continuum
            continuum (np.ndarray): Continuum specific luminosity
            line_profiles (dict): Dictionary of line profiles from create_line_profiles
            
        Returns:
            tuple: (wavelength_grid, spectrum)
                wavelength_grid (np.ndarray): Combined wavelength grid including continuum and line points
                spectrum (np.ndarray): Combined spectrum on the wavelength grid
        
        Note:
            Expects input continuum to be in specific luminosity units, same as for the lines.
            For direct Cloudy output for TODDLERS models, this is erg/s/micron
        """
        # Start with continuum wavelengths
        all_wavelengths = list(continuum_waves)
        
        # Add wavelength points from each line profile
        for _, (profile_x, _) in line_profiles.items():
            all_wavelengths.extend(profile_x)
        
        # Convert to array and sort, reverse=true to keep original cloudy direction
        wavelength_grid = np.array(sorted(set(all_wavelengths), reverse=True))
        
        # Interpolate continuum onto full grid
        cont_interp = interp1d(
        continuum_waves, 
        np.log10(np.maximum(continuum, 1e-99)),
        kind='linear',
        bounds_error=False,
        fill_value=np.log10(1e-99)
        )
        
        # interp continua sans lines on common grid
        result = 10**cont_interp(wavelength_grid)
        
        # Add line profiles
        for _, (profile_x, profile_y) in line_profiles.items():
            # Check for valid profile
            if len(profile_x) != len(profile_y):
                continue
                
            # Interpolate profile onto combined grid
            profile_interp = interp1d(
                profile_x, np.log10(profile_y),
                kind='linear',
                bounds_error=False,
                fill_value=-99
            )
            
            # Add interpolated profile
            result += 10**profile_interp(wavelength_grid)

        return wavelength_grid, result