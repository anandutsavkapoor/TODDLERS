from .imports import np, Optional, Dict, dataclass, Any
from .constants import *

@dataclass
class VariationThresholds:
    """
    Thresholds for different physical parameters that trigger Cloudy runs.
    
    Attributes:
        n_H_shell (float): Fractional variation threshold for shell density
        r_shell (float): Fractional variation threshold for shell radius
        M_shell (float): Fractional variation threshold for shell mass
        L_i (float): Fractional variation threshold for ionizing luminosity
    """
    n_H_shell: float = TRIGGER_FRAC_NSHELL
    r_shell: float = TRIGGER_FRAC_RSHELL
    M_shell: float = TRIGGER_FRAC_MSHELL
    L_i: float = TRIGGER_FRAC_LION

class TimeGridGenerator:
    """
    Generates time grids for Cloudy simulations using sequential point selection.
    Uses interpolation functions from CloudySimulationManager for consistency.
    """
    
    def __init__(self, 
                 t_start: float,
                 dissolution_time: Optional[float] = None,
                 interpolants: Optional[Dict] = None,
                 logger: Any = None):
        """
        Initialize the TimeGridGenerator.

        Args:
            t_start (float): Initial time in seconds
            t_max (float): Maximum simulation time in seconds
            dissolution_time (float, optional): Time of shell dissolution
            interpolants (dict, optional): Dictionary of interpolation functions:
                - radius_interp: Shell radius interpolator
                - n_shell_in_interp: Shell inner face density interpolator (total nuclei, cm^-3)
                - shell_mass_interp: Shell mass interpolator
            logger: Logger instance for recording progress
        """
        self.t_start = t_start
        self.t_max = MAX_SIMULATION_TIME
        self.dissolution_time = dissolution_time
        self.interpolants = interpolants or {}
        self.logger = logger

    def _calculate_total_L_i(self, t: float, stellar_feedback) -> float:
        """Calculate total ionizing luminosity at a given time."""
        feedback_data = stellar_feedback.get_feedback_data_per_generation(t)
        return sum(feedback_data['L_i'])

    def _check_variation(self, 
                        current_value: float,
                        next_value: float, 
                        threshold: float) -> bool:
        """Check if variation between two values exceeds threshold."""
        if current_value == 0:
            return abs(next_value) > threshold
        return abs(next_value - current_value) / abs(current_value) > threshold

    def _find_next_significant_point(self,
                                   current_t: float,
                                   max_t: float,
                                   stellar_feedback,
                                   thresholds: VariationThresholds) -> Optional[float]:
        """
        Find the next time point where a significant variation occurs.
        Uses interpolants from CloudySimulationManager.
        """
        print_msg = False
        # Get current values using interpolants
        current_n = self.interpolants['n_shell_in_interp'](current_t)
        current_r = self.interpolants['radius_interp'](current_t)
        current_m = self.interpolants['shell_mass_interp'](current_t)
        current_L = self._calculate_total_L_i(current_t, stellar_feedback)
        
        # Search through future times until significant variation found
        dt_search = MIN_DT_ADAPTIVE
        t = current_t + dt_search
        
        while t <= min(current_t + MAX_DT_ADAPTIVE, max_t):
            # Get next values using interpolants
            next_n = self.interpolants['n_shell_in_interp'](t)
            next_r = self.interpolants['radius_interp'](t)
            next_m = self.interpolants['shell_mass_interp'](t)
            next_L = self._calculate_total_L_i(t, stellar_feedback)
            
            # Check variations
            if (self._check_variation(current_n, next_n, thresholds.n_H_shell) or
                self._check_variation(current_r, next_r, thresholds.r_shell) or
                self._check_variation(current_m, next_m, thresholds.M_shell) or
                self._check_variation(current_L, next_L, thresholds.L_i)):
                
                if self.logger and print_msg:
                    self.logger.debug(f"Found significant variation at t={t/MYR_TO_SEC:.2f} Myr:")
                    if self._check_variation(current_n, next_n, thresholds.n_H_shell):
                        variation_n = abs(next_n - current_n) / abs(current_n)
                        self.logger.debug(f"  Density variation: {variation_n:.2e} (relative change)")
                    if self._check_variation(current_r, next_r, thresholds.r_shell):
                        variation_r = abs(next_r - current_r) / abs(current_r)
                        self.logger.debug(f"  Radius variation: {variation_r:.2e} (relative change)")
                    if self._check_variation(current_m, next_m, thresholds.M_shell):
                        variation_m = abs(next_m - current_m) / abs(current_m)
                        self.logger.debug(f"  Mass variation: {variation_m:.2e} (relative change)")
                    if self._check_variation(current_L, next_L, thresholds.L_i):
                        variation_L = abs(next_L - current_L) / abs(current_L)
                        self.logger.debug(f"  L_i variation: {variation_L:.2e} (relative change)")

                return t
                
            t += dt_search
            
        return None

    def _generate_adaptive_grid(self,
                              stellar_feedback,
                              thresholds: VariationThresholds) -> np.ndarray:
        """
        Generate adaptive time grid by sequential point selection.
        Uses interpolants for physical quantities.
        """
        print_msg = False
        grid_points = [self.t_start]
        current_t = self.t_start
        
        max_time = self.dissolution_time if self.dissolution_time is not None else self.t_max
        
        while current_t < max_time:
            # Find next point where significant variation occurs
            next_t = self._find_next_significant_point(
                current_t, self.t_max, stellar_feedback, thresholds
            )
            
            # If no significant variation found, use maximum time step
            if next_t is None:
                next_t = current_t + MAX_DT_ADAPTIVE
            else:
                # Ensure we don't violate minimum time step
                next_t = max(current_t + MIN_DT_ADAPTIVE, 
                        min(next_t, current_t + MAX_DT_ADAPTIVE))
            
            # Add point if we haven't reached the end
            if next_t <= self.t_max:
                grid_points.append(next_t)
            
            current_t = next_t

            if self.logger and print_msg:
                self.logger.debug(f"Added point at t={current_t/MYR_TO_SEC:.2f} Myr")      

        return np.array(grid_points)

    def _generate_uniform_grid(self, 
                            n_points: Optional[int] = None,
                            timestep: Optional[float] = None,
                            t_start: Optional[float] = None,
                            t_end: Optional[float] = None) -> np.ndarray:
        """
        Generate a uniform time grid.

        Args:
            n_points (int, optional): Number of points in the grid. If None, calculated from timestep.
            timestep (float, optional): Time step between points. If None, uses n_points.
            t_start (float, optional): Start time. If None, uses self.t_start.
            t_end (float, optional): End time. If None, uses self.t_max.

        Returns:
            np.ndarray: Array of time points.
        """
        t_start = t_start if t_start is not None else self.t_start
        t_end = t_end if t_end is not None else self.t_max
        
        if timestep is not None:
            n_points = int(np.ceil((t_end - t_start) / timestep)) + 1
        elif n_points is None:
            raise ValueError("Either n_points or timestep must be provided.")
        if n_points > 0:
            return np.linspace(t_start, t_end, n_points, endpoint=True)
        else:
            return np.array(t_end)

    def generate_grid(
        self,
        method: str = 'toddlers_v1',
        n_points: Optional[int] = None,
        continue_after_dissolution: bool = False,
        stellar_feedback=None,
        thresholds: Optional[VariationThresholds] = None,
        t_max: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate time grid based on specified method and parameters.

        Args:
            method (str): Grid generation method: 'adaptive', 'uniform', or 'toddlers_v1'.
                - 'adaptive': Uses changes in quantities specified in VariationThresholds to trigger runs.
                - 'uniform': Adds a uniform number of points between start time and dissolution.
                - 'toddlers_v1': Same as TODDLERS, Kapoor+23.

            n_points (int, optional): Number of points for uniform grid. Defaults to None.

            continue_after_dissolution (bool): Whether to continue after dissolution. Defaults to False.
                If True, affects the methods as follows:

                - 'adaptive': Adds additional runs beyond dissolution with a specified time resolution.
                - 'uniform': If n_points is given, it splits equally between pre- and post-dissolution;
                  otherwise, uses fixed temporal resolution between start and t_max.
                - 'toddlers_v1': Uses the complete grid. If False, the grid is cut at dissolution time.

            stellar_feedback: Stellar feedback instance for adaptive sampling. Defaults to None.

            thresholds (VariationThresholds, optional): Thresholds for adaptive sampling.

            t_max (float, optional): Maximum time to generate grid up to. If provided, overrides 
                the default maximum time. Value should be in seconds.

        Returns:
            np.ndarray: Array of time points.
        """
        if self.logger:
            self.logger.info(f"Generating {method} time grid")
            if t_max is not None:
                self.logger.info(f"Using custom maximum time: {t_max/MYR_TO_SEC:.2f} Myr")

        # Use provided t_max if available, otherwise use default
        max_time = t_max if t_max is not None else self.t_max

        if method == 'toddlers_v1':
            # Calculate midpoint of the total time interval
            t_mid = (self.t_start + max_time) / 2

            # Generate points with different resolutions
            dt_early = 0.25 * MYR_TO_SEC  # 0.25 Myr resolution for early times
            dt_late = 0.50 * MYR_TO_SEC   # 0.50 Myr resolution for late times

            # Generate early and late time points, last point = max_time
            early_points = np.arange(self.t_start, t_mid, dt_early)
            late_points = np.arange(t_mid, max_time, dt_late)
            if late_points[-1] != max_time:
                late_points = np.append(late_points, max_time)

            # Combine and ensure uniqueness
            grid = np.unique(np.concatenate([early_points, late_points]))

            # Optionally cut the grid at the dissolution time
            if not continue_after_dissolution and self.dissolution_time is not None:
                grid = grid[grid <= self.dissolution_time]
                length_post_dissolution = 0
            else:
                length_post_dissolution = (
                    np.sum(grid >= self.dissolution_time) if self.dissolution_time else 0
                )

            if self.logger:
                n_early = len(early_points)
                n_late = len(late_points)
                self.logger.info("Generated TODDLERS v1 grid:")
                self.logger.info(f"  Early phase (dt=0.25 Myr): {n_early} points")
                self.logger.info(f"  Late phase (dt=0.50 Myr): {n_late} points")

        elif method == 'uniform':
            if n_points is None:
                if continue_after_dissolution and self.dissolution_time is not None:
                    pre_points = self._generate_uniform_grid(
                        t_start=self.t_start, t_end=self.dissolution_time, timestep=MAX_DT_UNIFORM
                    )
                    post_points = self._generate_uniform_grid(
                        t_start=self.dissolution_time, t_end=max_time, timestep=MAX_DT_UNIFORM
                    )
                    length_post_dissolution = len(post_points) if self.dissolution_time != MAX_SIMULATION_TIME else 0
                    grid = np.unique(np.concatenate([pre_points, post_points]))
                else:
                    grid = self._generate_uniform_grid(
                        t_start=self.t_start, t_end=self.dissolution_time, timestep=MAX_DT_UNIFORM
                    )
                    length_post_dissolution = 0
            else:
                if continue_after_dissolution and self.dissolution_time is not None:
                    pre_points = self._generate_uniform_grid(
                        t_start=self.t_start, t_end=self.dissolution_time, n_points=n_points // 2
                    )
                    post_points = self._generate_uniform_grid(
                        t_start=self.dissolution_time, t_end=max_time, n_points=n_points // 2
                    )
                    length_post_dissolution = len(post_points) if self.dissolution_time != MAX_SIMULATION_TIME else 0
                    grid = np.unique(np.concatenate([pre_points, post_points]))
                else:
                    grid = self._generate_uniform_grid(
                        t_start=self.t_start, t_end=self.dissolution_time, n_points=n_points
                    )
                    length_post_dissolution = 0

        else:  # adaptive
            if stellar_feedback is None:
                raise ValueError("stellar_feedback required for adaptive sampling")
            if thresholds is None:
                thresholds = VariationThresholds()

            grid = self._generate_adaptive_grid(stellar_feedback, thresholds)
            length_post_dissolution = 0

            if continue_after_dissolution and self.dissolution_time is not None:
                # Add uniform points after dissolution
                post_points = self._generate_uniform_grid(
                    t_start=self.dissolution_time, t_end=max_time, timestep=MAX_DT_ADAPTIVE
                )
                grid = np.unique(np.concatenate([grid, post_points]))
                length_post_dissolution = len(post_points) if self.dissolution_time != MAX_SIMULATION_TIME else 0
                
        if not continue_after_dissolution and self.dissolution_time is not None:
            grid = grid[grid <= self.dissolution_time]

        if self.logger:
            self.logger.info(f"Generated grid with {len(grid)} points, {length_post_dissolution} after dissolution")
            self.logger.info(f"Generated grid with t_init={np.min(grid) / MYR_TO_SEC:.2e} Myr and t_end={np.max(grid) / MYR_TO_SEC:.2e}")
            dt = np.diff(grid)
            self.logger.info("Time step statistics:")
            self.logger.info(f"  Min: {np.min(dt) / MYR_TO_SEC:.3e} Myr")
            self.logger.info(f"  Max: {np.max(dt) / MYR_TO_SEC:.3e} Myr")
            self.logger.info(f"  Mean: {np.mean(dt) / MYR_TO_SEC:.3e} Myr")

        return np.around(grid, 2)