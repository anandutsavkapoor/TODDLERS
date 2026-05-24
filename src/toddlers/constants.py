# constants.py
import os as _os
from .imports import np, const, u, platform

# Physical constants
G = const.G.cgs.value
C = const.c.cgs.value
M_SUN = const.M_sun.cgs.value
L_SUN = const.L_sun.cgs.value
M_P = const.m_p.cgs.value
K_BOLTZMANN = const.k_B.cgs.value

# Conversion factors
MYR_TO_SEC = u.Myr.to(u.s)
YR_TO_SEC  = u.yr.to(u.s)
PC_TO_CM = u.pc.to(u.cm)
KM_TO_CM = u.km.to(u.cm)
ELECTRON_VOLT_TO_ERG = u.eV.to(u.erg)
JYR_TO_ERGS = 0.317098  # Conversion factor for J/yr to erg/s

# Model-specific constants
GAMMA = 5/3  # Adiabatic index for ideal gas
GAMMA_2 = 7/5  # Gas adiabatic constant for diatomic gas
MU_N = (14/11) * M_P  # Mean mass per nucleus
MU_P = (14/23) * M_P  # Mean mass per particle


# Stellar and nebular parameters
T_ION = 1e4  # Temperature of ionized gas in K
T_NEUTRAL = 100  # Temperature of neutral gas in K
N_ISM = 0.1  # cm-3, ISM Beyond the cloud
ALPHA_B = 2.59e-13  # Case B recombination coefficient at T=10^4 K in cm^3 s^-1

# Dust properties
SIGMA_DUST_SOLAR = 1.5e-21  # Dust cross-section at solar metallicity in cm^2
KAPPA_IR_SOLAR = 4  # Rosseland mean dust opacity for self-absorption at solar metallicity in cm^2/g
DUST_ALBEDO = 0.46

# Lyman-alpha related constants
E_LYALPHA =  (10.198 * u.eV).to(u.erg).value # Lyman-alpha photon energy in erg
E_LYC = (13.6 * u.eV).to(u.erg).value # H ionizing photon energy in erg
ION_TO_LYALPHA = 0.68  # Conversion factor from ionizing photons to Lyman-alpha

# Time-related constants
T_SN = 3.0 * MYR_TO_SEC  # Time of first supernova explosion

# Cloud and shell properties
DISSOLUTION_DENSITY = 0.5  # Density threshold for shell dissolution in cm^-3
DISSOLUTION_PERIOD = 1 * MYR_TO_SEC  # Time period for dissolution condition in seconds
DISSOLUTION_RADIUS = 1000 * PC_TO_CM  # Radius threshold for dissolution in cm

# Numerical method parameters
if platform.system().lower() == 'darwin':  # MacOS
    SOLVER_PH1 = "BDF"    # Solver for phase-1
    SOLVER_FRAG = "BDF"   # Solver for fragmentation 
    SOLVER_PH2 = "BDF"    # Solver for phase-2
    ALT_SOLVER_PH1 = "BDF"    # Alternative solver for phase-1
    ALT_SOLVER_FRAG = "BDF"   # Alternative solver for fragmentation 
    ALT_SOLVER_PH2 = "Radau"    # Alternative solver for phase-2
else:  # Linux
    SOLVER_PH1 = "BDF"    # Solver for phase-1
    SOLVER_FRAG = "BDF"   # Solver for fragmentation 
    SOLVER_PH2 = "LSODA"  # Solver for phase-2, faster
    ALT_SOLVER_PH1 = "BDF"    # Alternative solver for phase-1
    ALT_SOLVER_FRAG = "BDF"   # Alternative solver for fragmentation 
    ALT_SOLVER_PH2 = "BDF"    # Alternative solver for phase-2
RTOL_MAIN_PH1  = 1e-6  # Relative tolerance for main solver, phase-1
RTOL_MAIN_FRAG = 1e-6  # Relative tolerance for main solver, fragmentation
RTOL_MAIN_PH2  = 1e-5  # Relative tolerance for main solver, phase-2
SIMULATION_TIMEOUT = 3600 * 2 # 2 hours in seconds
LOOSER_RTOL = RTOL_MAIN_PH1  * 1.25  # Looser relative tolerance for third attempt
MAX_STEP_PH1  = 3e-3*MYR_TO_SEC # max step allowed, phase-1
MAX_STEP_FRAG = 3e-3*MYR_TO_SEC # max step allowed, fragmentation
MAX_STEP_PH2  = 5e-3*MYR_TO_SEC # max step allowed, phase-2
MAX_SIMULATION_TIME = 30 * MYR_TO_SEC # seconds

# Stellar feedback parameters
M_LIB = 1e6  # BPASS/SB99 library burst mass in solar masses
T_INIT_TEMPLATE = 1e-1 * MYR_TO_SEC

# Miscellaneous
V_ZERO_STALL = 10000 # stall velocity in cm/s
LYMAN_ALPHA_VELOCITY_THRESHOLD = 100 * KM_TO_CM  # Velocity threshold for Lyman-alpha escape fraction in cm/s
FRAC_M_CL = 1.0 # when cloud density is dynamic, this fraction is used in cloud_condition for dissolution in phase-2
FAC_FRAG_R_CL_INIT = 0.1 # no fragmentation if shell under this cloud radial fraction
FAC_T_SOUND = 1. # numbers of sound crossing times in shell for RTI fragmentation
USE_BETA_CONDITION = False # dP/dt should be > 0 for fragmentation
USE_DENSITY_CONTRAST = True  # Set to True to enable density contrast criterion
DENSITY_CONTRAST_RT = 1.1  # Minimum density contrast for RT instability
MIN_ACCEL_TIME = 0.1 * MYR_TO_SEC  # 0.1 Myr in seconds, to avoid very small acceleration times
MAX_ACCEL_TIME = 0.5 * MYR_TO_SEC # 0.5 Myr in seconds, to avoid very long acceleration times

# inner radius and fragmentation end
FRAC_EB_INIT = 0.01 # fraction of bubble energy at which fragmentation is terminated
FRAC_R_IN_MAX = 1 - (2*FRAC_EB_INIT) # maximum fraction of the shell radius that can be assumed by Rin
XTOL_INNER_RADIUS = 1e-12 # inner radius implicit equation tolerance
MAX_ITER_INNER_RADIUS = 100 # inner radius implicit equation max iterations

# Metallicity values
Z_SOLAR = 0.02  # Solar metallicity, evolution
Z_MIN_BPASS = 1e-5
Z_MAX_BPASS = 0.04
Z_MIN_SB99 = 0.001
Z_MAX_SB99 = 0.04
Z_KROUPA120_ROT = 0.014  # Single metallicity for Kroupa 120 rotational model
Z_VALUES_BPASS = np.array([1e-5, 1e-4, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.010, 0.014, 0.020, 0.030, 0.040])
Z_VALUES_SB99 = np.array([0.001, 0.004, 0.008, 0.020, 0.040])

# bubble related
ZETA = 0.9 
MEAN_T_BUBBBLE = 1e6

# fudge factors
LYA_FF = 1
ETA_KE = 15e-4 / 2  # kinetic energy efficiency from Freyer+ 2003, paper-I, with wind
COVERING_FRACTION_DEF = 0.5 # default of add_cover_frac is true and post_sweep value is not defined
INFALL_RADIUS_RATIO_COVERFRAC = 0.5 # covering fraction becomes unity if shell falls below this radius

# Cloudy related
# Cloudy executable and data directory. Both are environment-overridable so the
# package runs on any machine without editing source: set CLOUDY_EXE / CLOUDY_DATA_DIR
# in the environment (the HPC submit scripts do this). The _DEFAULT_CLOUDY_DATA_DIR
# literal is what scripts/download_data.py rewrites to the local Cloudy data dir; it
# ships empty so no machine-specific path is baked in.
_DEFAULT_CLOUDY_DATA_DIR = ""
CLOUDY_EXE = _os.environ.get("CLOUDY_EXE", "cloudy.exe")


def _resolve_cloudy_data_dir():
    """Resolve Cloudy's data directory without baking in a machine path.

    Order: explicit ``$CLOUDY_DATA_DIR`` wins; otherwise derive it from the
    Cloudy executable (a standard install has ``<cloudy>/source/cloudy.exe``
    alongside ``<cloudy>/data``); otherwise fall back to the rewritable default
    (empty unless ``scripts/download_data.py`` has set it)."""
    env = _os.environ.get("CLOUDY_DATA_DIR")
    if env:
        return env
    import shutil as _shutil
    exe = CLOUDY_EXE if _os.path.isabs(CLOUDY_EXE) else (_shutil.which(CLOUDY_EXE) or "")
    if exe:
        cand = _os.path.join(_os.path.dirname(_os.path.dirname(exe)), "data")
        if _os.path.isdir(cand):
            return cand
    return _DEFAULT_CLOUDY_DATA_DIR


CLOUDY_DATA_DIR = _resolve_cloudy_data_dir()
SMALL_TO_LARGE_MASS_RATIO = 0.1 # modified orion grains' dist, bw 0.05 -- 0.40
A_0_GRAINS = 0.001  # Lower grain size limit, micron
A_L_GRAINS = 0.03 # Cutoff for exponential function, micron
Q_PAH = 4.6 # 4.6 % as used in TODDLERS-v0, Kapoor+23
MAX_RATIO = 0.43 # max small grain to large grain mass ratio in the grains generator
ISM_LIKE_RATIO = 0.40 # above this model starts to be like the "ISM" grains in cloudy
ORION_LIKE_RATIO = 0.01  # below this model starts to be like the "Orion" grains in cloudy
Z_SOLAR_GASS10 = 0.014 # new estimates, same as in TODDLERS v1
BPASS_REDUCE_RES_FAC = 1 # reduce wavelenth resolution by this factor, BPASS
SB99_REDUCE_RES_FAC = 1 # reduce wavelenth resolution by this factor, SB99
NUM_STELLAR_SPEC_CLOUDY = 75 # spectra bw 1e5 yr and tmax in the cloudy ascii

TRANSITION_FACTOR_SHELL_CLOUD = 1.1 # 10%
T_INIT_OBSERVABLES = 0.1 * MYR_TO_SEC
N_ISM_CLOUDY = 1  # ISM Beyond the cloud for Cloudy calcs
ESC_FRAC_ZERO = 1e-6 # use this to say that shell isnt leaky
MASS_COMPAR_TOL = 1e-2 # Msh < (1 - tol) * M_cl => within cloud (in cloudy_sim_manager / is_within_cloud_interp)

LOGU_BG_OLD_DEFAULT = -3.5 # The DIG logU
MEAN_E_FACTOR_OLD = 2.075 # <E_ion> (~all Z, 1e9 yr) is higher than E_LYC by this factor
TURBULENT_DISSIPATION_SCALE = 3e22 # ~100 kpc ==> uniform dissipation in the DIG
MACH_DIG_DEF = 0.5 # mach number of the dig if velocity is not defined and dissipation is on.
# Other Turbulence parameters
DEFAULT_TURB_FRAC = 0.01         # default fraction of shell velocity for turbulence (no pressure)
LOG_DENSITY_THRESHOLD_TURB = 4.0  # log(n_H/cm^-3) threshold for enhanced turbulence
HIGH_DENSITY_TURB_FRAC = 0.1     # fraction of shell velocity for turbulence in high density regions

# Runs
TRIGGER_FRAC_NSHELL = 0.225
TRIGGER_FRAC_RSHELL = 0.125
TRIGGER_FRAC_MSHELL = 0.225
TRIGGER_FRAC_LION = 0.225
MAX_DT_ADAPTIVE = 1.0 * MYR_TO_SEC # max time lag bw Cloudy runs when using adaptive
MAX_DT_UNIFORM  = 0.5 * MYR_TO_SEC # max time lag bw Cloudy runs when using uniform
MIN_DT_ADAPTIVE = 0.01 * MYR_TO_SEC # min time lag bw Cloudy runs when using adaptive
ZERO_GAS_MASS = 1e-16 * M_SUN # used for direct radiation output dig runs

# output dirs, same level as src
EVOLUTION_OUTPUT_DIR = 'evolution_output' 
EVOLUTION_LOGS_DIR = 'evolution_logs' 
CLOUDY_OUTPUT_DIR = 'cloudy_output' 
