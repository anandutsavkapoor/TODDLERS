from .imports import np, List, dataclass, field
from .constants import *

@dataclass
class EvolutionState:
    """
    Class representing the instantaneous state of the shell evolution.

    Tracks physical properties, cumulative times, and phase information for the expanding shell/cloud.
    This includes both instantaneous quantities (radius, velocity, etc.) and cumulative properties
    used to assess stability and phase transitions.

    Attributes:
        phase (str): Current evolution phase ('phase1', 'fragmentation', 'phase2', 'dissolution')
        R (float): Shell radius in cm
        V_sh (float): Shell velocity in cm/s
        P_b (float): Bubble pressure in dyne/cm^2
        M_sh (float): Shell mass in g
        t (float): Current time in s
        t_old (float): Previous timestep time in s
        E_b (float): Bubble energy in erg
        alpha (float): Shell acceleration parameter (t/R * V)
        beta (float): Shell pressure parameter (-(t/P) * dP/dt)
        current_generation (int): Current stellar generation number
        dissolved (bool): Flag indicating if shell has dissolved
        t_list_collapse (List[float]): Times of previous collapses
        Rin (float): Inner radius of shell in cm
        dt (float): Current timestep size in s
        dt_cum_dissolution (float): Cumulative time shell density below dissolution threshold in s
        dt_cum_stalled (float): Cumulative time shell has been stalled in s
        dt_cum_posAcc (float): Cumulative time shell has been accelerating in s
        dt_cum_beta (float): Cumulative time beta condition has been met in s
        dt_cum_density (float): Cumulative time density contrast condition has been met in s
        n_cloud_avg (float): Average cloud number density in cm^-3
        t_end (float): End time of current phase in s
        f_esc_i (float): Ionizing photon escape fraction
        f_esc_uv (float): UV photon escape fraction
        n_shell_in (float): Inner shell number density in cm^-3
        n_shell_max (float): Maximum shell number density in cm^-3
        n_shell_max_ionized (float): Maximum ionized shell number density in cm^-3
        eta_rad (float): Radiation force coupling efficiency
        shell_thickness (float): Thickness of the shell in cm
        columnDensity_H1 (float): Neutral hydrogen column density in cm^-2
        T_sh_ion (float): Shell ionized gas temperature in K

    Notes:
        - Cumulative times (dt_cum_*) are used to ensure physical transitions between phases
        - All times must exceed local sound crossing time for fragmentation conditions
    """
    phase: str = 'phase1'
    R: float = None
    V_sh: float = None
    P_b: float = None
    M_sh: float = None
    t: float = T_INIT_TEMPLATE
    t_old: float = T_INIT_TEMPLATE
    E_b: float = None
    alpha: float = None
    beta: float = None
    delta: float = None
    current_generation: int = 0
    dissolved: bool = False
    t_list_collapse: List[float] = field(default_factory=list)
    Rin: float = None
    dt: float = 0
    dt_cum_dissolution: float = 0
    dt_cum_stalled: float = 0
    dt_cum_posAcc: float = 0
    dt_cum_beta: float = 0
    dt_cum_density: float = 0
    n_cloud_avg: float = N_ISM + 1e-2
    t_end: float = 0
    f_esc_i: float = 1
    f_esc_uv: float = 1
    n_shell_in: float = 0
    n_shell_max: float = 0
    n_shell_max_ionized: float = 0
    eta_rad: float = 0
    shell_thickness: float = None
    columnDensity_H1: float = 0
    T_sh_ion: float = None
    
    T: float = None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"EvolutionState has no attribute '{key}'")

    def get_volume(self):
        return (4/3) * np.pi * (self.R**3 - self.Rin**3)

    def get_state_vector(self):
        """
        Returns the current state vector.

        Returns:
            numpy.ndarray: Current state vector [R, V_sh, M_sh, n_cloud_avg] for all phases.
                           For phase1 and fragmentation, E_b is inserted as the third element.
        """
        state_vector = np.array([self.R, self.V_sh, self.M_sh, self.n_cloud_avg])
        if self.phase in ['phase1', 'fragmentation']:
            state_vector = np.insert(state_vector, 2, self.E_b)
        return state_vector