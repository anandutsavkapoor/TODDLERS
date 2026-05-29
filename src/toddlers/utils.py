from .imports import np, shutil, solve_ivp, find_peaks, re, u, os, time, List, brentq, warnings
from .constants import *


def dtm_label(dtm):
    """Filename suffix for a dust-to-metal value: '' at the fiducial 1.0, else '_dtm<val>'.

    Uses a general (``:g``) format so small values are represented distinctly -- e.g.
    1e-3 -> '_dtm0.001'. A fixed 2-decimal format collapses 1e-3 to '_dtm0.00' (i.e. 0.0),
    which mislabels the model and breaks the log-scaled f_dust axis. Round-trips with parse_dtm.
    """
    return "" if abs(float(dtm) - 1.0) < 1e-9 else f"_dtm{float(dtm):g}"


_DTM_RE = re.compile(r"_dtm(\d*\.?\d+(?:[eE][+-]?\d+)?)")


def parse_dtm(name):
    """Dust-to-metal value parsed from a filename / stem; 1.0 if there is no _dtm token."""
    m = _DTM_RE.search(str(name))
    return float(m.group(1)) if m else 1.0


def generate_toddlers_banner():
    """Generates TODDLERS banner for logs etc."""
    banner = [
            " _____ ___  ____  ____  _     _____ ____  ____ ",
            "|_   _/ _ \\|  _ \\|  _ \\| |   | ____|  _ \\/ ___|",
            "  | || | | | | | | | | | |   |  _| | |_) \\___ \\",
            "  | || |_| | |_| | |_| | |___| |___|  _ < ___) |",
            "  |_| \\___/|____/|____/|_____|_____|_| \\_\\____/",
            "                                               ",
            "TODDLERS: Time evolution of Observables including Dust",
            "           Diagnostics and Line Emission from Regions",
            "           containing young Stars",
            "",
            "Author: Anand Utsav Kapoor",
            "Ghent University, Belgium",
            "Email: anandutsavkapoor@gmail.com",
            "",
            "--------------------------------------------------"
        ]
    # Find the width of the widest line
    max_width = max(len(line) for line in banner)
    
    # Center each line
    centered_banner = [line.center(max_width) for line in banner]
    
    # Add some vertical padding
    padding = "\n" * 2
    
    return padding + "\n".join(centered_banner) + padding

def add_banner_to_log(log_file):
    banner = generate_toddlers_banner()

    # Get terminal size
    terminal_size = shutil.get_terminal_size((80, 20))  # default to 80 if can't determine
    terminal_width = terminal_size.columns

    # Split the banner into lines
    banner_lines = banner.split('\n')

    # Center each line
    centered_banner = []
    for line in banner_lines:
        padding = ' ' * ((terminal_width - len(line)) // 2)
        centered_banner.append(padding + line)

    # Join the centered lines back into a single string
    centered_banner_str = '\n'.join(centered_banner)

    with open(log_file, 'a') as f:
        f.write(centered_banner_str + '\n\n')

def interpolate_with_tanh(x1, y1, x2, y2, x):
    """
    Create a smooth transition between two points using a hyperbolic tangent function.
    This function generates a curve that smoothly transitions from (x1, y1) to (x2, y2),
    with the steepest part of the transition occurring at the midpoint between x1 and x2.

    Parameters:
    x1 (float): The x-coordinate of the start point of the transition.
    y1 (float): The y-coordinate of the start point of the transition.
    x2 (float): The x-coordinate of the end point of the transition.
    y2 (float): The y-coordinate of the end point of the transition.
    x (float or array): The x-coordinate(s) at which to evaluate the transition.

    Returns:
    float or array: The y-coordinate(s) of the transition curve at the given x-value(s).
    Notes:
    - The steepness of the transition is determined by the distance between x1 and x2.
    - The transition is symmetric around the midpoint (x1 + x2) / 2.
    """
    midpoint = (x1 + x2) / 2
    return y1 + (y2 - y1) * 0.5 * (1 + np.tanh((x - midpoint) * (2 / (x2 - x1))))

def linear_interpolation(x1, y1, x2, y2, x):
    slope = (y2 - y1) / (x2 - x1)
    y_intercept = y1 - (slope * x1)
    return (slope * x) + y_intercept

def func_tanh_velocity(V):
    b = 20  # cm/s
    a = 5 * b  # 100 cm/s
    x1 = a - b  # Start of transition
    x2 = a + b  # End of transition
    y1 = 0
    y2 = 1
    return interpolate_with_tanh(x1, y1, x2, y2, V)

def func_tanh_extPressure_velocity(V):
    """
    Smoothly interpolates external pressure as a function of shell velocity using a tanh transition.

    Parameters:
    V (float or array): Shell velocity in cgs units.

    Returns:
    float or array: The interpolated pressure value(s) for the given velocity V.
    """
    x1 = 1e-4 # very small shell velocity, cgs
    x2 = 1e2  # small but positive, cgs
    y1 = -4   # 1e-4 x full pressure
    y2 = 0    # full pressure 
    return 10**interpolate_with_tanh(x1, y1, x2, y2, V)

def func_tanh_extPressure_fesc_ion(f_esc_ion, P_ion, P_neutral):
    """
    Smoothly interpolates external pressure as a function of f_esc_ion using a tanh transition.

    Parameters:
    f_esc_ion (float or array): The escape fraction of ionized gas.
    P_ion (float): The pressure in the ionized regime (higher pressure).
    P_neutral (float): The pressure in the neutral regime (lower pressure).

    Returns:
    float or array: The interpolated pressure value(s) for the given f_esc_ion in linear scale.
    """
    x1 = 0    # Minimum f_esc_ion
    x2 = .05  # Maximum f_esc_ion
    y1 = np.log10(P_neutral)  # Logarithmic pressure at x1
    y2 = np.log10(P_ion)      # Logarithmic pressure at x2
    return 10**interpolate_with_tanh(x1, y1, x2, y2, f_esc_ion) #return in linear scale

def func_tanh_fesc(fesc):
    b = 2e-2 
    a = 2.5 * b # 0.05
    x1 = a - b  # Start of transition
    x2 = a + b  # End of transition
    y1 = 0
    y2 = 1
    return interpolate_with_tanh(x1, y1, x2, y2, fesc)

def func_tanh_fragmentation_cooling_up(t_since_frag, t_sound_crossing):
    """
    Create a smooth transition for increasing cooling rate based on time
    """
    transition_period = t_sound_crossing
    x1 = 0
    x2 = transition_period
    y1 = -1
    y2 = 0 
    return 10**interpolate_with_tanh(x1, y1, x2, y2, t_since_frag)

def func_tanh_fragmentation_cooling_down(energy_ratio, energy_ratio_threshold=2*FRAC_EB_INIT):
    """
    Create a smooth transition for reducing cooling rate based on energy ratio.
    """
    y1 = 0  # Minimum reduction (full cooling)
    y2 = -1  # Maximum reduction (cooling reduced by factor of 10^-3)
    x1 = energy_ratio_threshold
    x2 = energy_ratio_threshold * 0.1
    return 10**interpolate_with_tanh(x1, y1, x2, y2, energy_ratio)

def func_tanh_sweep_coverfrac(t, t_transition, cover_frac_final, transition_duration=0.25*MYR_TO_SEC):
    """
    Create a smooth transition for the covering fraction using a hyperbolic tangent function based on time.

    Args:
        t (float): Current time.
        t_transition (float): Time at which to start the transition.
        cover_frac_final (float): Final covering fraction to transition to.
        transition_duration (float): Duration of the transition in seconds. Default is 0.25 Myr.

    Returns:
        float: Smoothly interpolated covering fraction between 1.0 and cover_frac_final.
    """
    x1 = t_transition
    x2 = t_transition + transition_duration
    y1 = 1.0  # Initial covering fraction
    y2 = cover_frac_final  # Final covering fraction
    return interpolate_with_tanh(x1, y1, x2, y2, t)

def func_tanh_infall_coverfrac(radius_ratio: float, post_sweep_cf: float = COVERING_FRACTION_DEF, 
                           transition_point: float = INFALL_RADIUS_RATIO_COVERFRAC, 
                           transition_width: float = 0.05) -> float:
    """
    Create a smooth transition for the covering fraction during infall based on radius ratio.
    
    When a shell falls back within a certain fraction of the cloud radius (transition_point),
    its covering fraction transitions smoothly back to 1.0 from its post-sweep value.
    
    Args:
        radius_ratio (float): Ratio of shell radius to cloud radius
        post_sweep_cf (float): Post-sweep covering fraction value to transition from.
                              Defaults to COVERING_FRACTION_DEF.
        transition_point (float, optional): Ratio at which transition is centered.
                                          Defaults to INFALL_RADIUS_RATIO_COVERFRAC.
        transition_width (float, optional): Relative width of the transition region.
                                          Defaults to 0.05.
    
    Returns:
        float: Covering fraction between post_sweep_cf and 1.0
    
    Notes:
        - When radius_ratio << transition_point, returns ~1.0
        - When radius_ratio >> transition_point, returns ~post_sweep_cf
        - The transition is smooth and centered at transition_point
        - The width of the transition region is transition_point * transition_width
    """
    x1 = transition_point * (1 - transition_width)  # Start of transition
    x2 = transition_point * (1 + transition_width)  # End of transition
    
    # Directly interpolate between post-sweep value and 1.0
    return interpolate_with_tanh(x1, 1.0,  # at small radius, go to 1.0
                               x2, post_sweep_cf,  # at large radius, go to post-sweep value
                               radius_ratio)

def get_fmol_Krum2013(flux_LyW, n_cnm, tau_dust):
    """Calculate molecular fraction using Krumholz & Gnedin 2011 model.
    
    Args:
        flux_LyW (array-like): Unattenuated Lyman Werner flux in erg/(s cm² Å)
        n_cnm (array-like): Local number density in cm⁻³
        tau_dust (array-like): Cumulative dust optical depth
        
    Returns:
        array-like: Molecular fraction f_H2 between 0 and 1
        
    Notes:
        Unit conversion from https://www.stsci.edu/~strolger/docs/UNITS.txt
    """
    # LyW photon energy and wavelength
    E_LyW = 12.39847 * 1.6021766339999e-12  # erg (h*nu at 1000 Å)
    wave_LyW_ang = 1000  # Å
    
    # Convert Draine field to same units as input flux
    LyW_mw_Draine = 3.43e-8  # Original Draine field
    LyW_mw_Draine_conv1 = LyW_mw_Draine * E_LyW
    LyW_mw_Draine_conv2 = LyW_mw_Draine_conv1 * 1e23 
    flux_LyW_mw = (LyW_mw_Draine_conv2 * 2.99792458e-5) / (wave_LyW_ang**2)
    
    # Calculate chi parameter (UV intensity scaled by density)
    U_mw = np.clip(flux_LyW / flux_LyW_mw, 0, None)  # Prevent negative ratios
    chi = (7.2 * U_mw) / (n_cnm/10)
    
    # Calculate s parameter with f_c = 1
    term1 = 0.6 * chi
    term2 = 0.01 * chi**2
    f_c = 1  # Clumping factor
    term3 = f_c * 0.6 * np.clip(tau_dust, 1e-16, None)  # Prevent division by 0
    s = np.log1p(term1 + term2) / term3  # log1p more numerically stable
    
    # Calculate molecular fraction
    f_mol = 1 - (3*s/(4 + s))
    f_mol = np.where(s >= 2, 0, f_mol)  # Atomic dominated regime
    return np.clip(f_mol, 0, 1)  # Ensure physical bounds

def monotonic(x):
    dx = np.diff(x)
    return np.all(dx < 0) or np.all(dx > 0)

def get_atols(
    R0: float, 
    E0: float = None, 
    phase: str = None, 
    dynamic_cloud_density: bool = False
) -> List[float]:
    """
    Calculate absolute tolerances for ODE solver based on initial conditions and phase.
    
    Args:
        R0 (float): Initial radius in cm.
        E0 (float, optional): Initial energy in erg. Required for 'phase1' or 'fragmentation'.
        phase (str, optional): Current evolution phase ('phase1', 'fragmentation', 'phase2').
        dynamic_cloud_density (bool): Whether cloud density is dynamically changing.
        
    Returns:
        list[float]: Absolute tolerances for each variable.
    
    Raises:
        ValueError: If E0 is required but not provided for certain phases,
                   or if phase is not recognized.
    """
    if phase not in [None, 'phase1', 'fragmentation', 'phase2', 'dissolution']:
        raise ValueError(f"Unrecognized phase: {phase}")

    R_atol = 1e-4 * R0
    V_atol = 1e-3  # cm/s
    M_atol = 1e-4 * M_SUN if phase in ('phase1', 'fragmentation') else 1e-3 * M_SUN
    n_cloud_avg_atol = 1e-4 if dynamic_cloud_density else 1e-2

    if phase in ('phase1', 'fragmentation') and E0 is None:
        raise ValueError(f"E0 must be provided for phase '{phase}'")
    
    if E0 is not None:
        E_atol = 1e-6 * E0 if phase != 'fragmentation' else 1e-8 * E0
        return [R_atol, V_atol, E_atol, M_atol, n_cloud_avg_atol]
    else:
        return [R_atol, V_atol, M_atol, n_cloud_avg_atol]
   
def get_scalar(value):
    """Convert numpy array to scalar if necessary."""
    return value.item() if isinstance(value, np.ndarray) and value.size == 1 else value

def format_value(value, format_spec):
    """Format a value, handling numpy arrays."""
    scalar_value = get_scalar(value)
    return format(scalar_value, format_spec)

def format_row(label, value, unit):
    """Format a row for tabular output."""
    if isinstance(value, (int, float)) or np.issubdtype(type(value), np.number):
        scalar_value = get_scalar(value)
        if abs(scalar_value) > 1000 or abs(scalar_value) < 0.01:
            formatted_value = format_value(scalar_value, '.2e')
        else:
            formatted_value = format_value(scalar_value, '.2f')
    elif isinstance(value, bool):
        formatted_value = str(value)
    elif isinstance(value, (list, np.ndarray)):
        formatted_value = "[" + ", ".join(format_value(get_scalar(v), '.2e') if abs(get_scalar(v)) > 1000 or abs(get_scalar(v)) < 0.01 else format_value(get_scalar(v), '.2f') for v in value) + "]"
    else:
        formatted_value = str(value)

    return f"| {label:<25} | {formatted_value:>15} | {unit:<25} |"

def calculate_injection_energy(f_esc, Q_i, dt, fudge_factor):
    """
    Calculate the energy injected into the cloud based on the escape fraction
    of ionizing photons and a fudge factor.
    
    Args:
        f_esc (float): Escape fraction of ionizing photons
        Q_i (float): Ionizing photon rate (photons/s)
        dt (float): Time step (s)
        fudge_factor (float): Adjustment factor for energy injection efficiency
    
    Returns:
        float: Injected energy in ergs
    """
    E_ph = (13.6 * u.eV).to(u.erg).value  # Energy of Lyman limit photon in erg
    E_inj = f_esc * Q_i * E_ph * dt * fudge_factor
    return E_inj

def calculate_cloud_escape_fraction(M_cloud, n_avg, Z, Q_i, r_in):
    """
    Calculate the escape fraction of ionizing photons from the cloud,
    considering both dust absorption and photoionization. Integration starts from
    an inner radius and continues until the specified cloud mass is reached.
    
    Args:
        M_cloud (float): Mass of the cloud in grams
        n_avg (float): Average number density of the cloud in cm^-3
        Z (float): Metallicity of the cloud in solar units
        Q_i (float): Ionizing photon rate in s^-1
        r_in (float): Inner radius of the cloud in cm
    
    Returns:
        float: Escape fraction of ionizing photons from the cloud
    """
    # Calculate the dust cross-section, scaled with metallicity
    sigma_dust = SIGMA_DUST_SOLAR * (Z / Z_SOLAR)
    
    def cloud_structure_ode(r, y):
        phi, tau_d, m = y
        n_H = n_avg  # Constant density throughout the cloud
        
        dm_dr = 4 * np.pi * r**2 * n_H * MU_N
        dphi_dr = -((4 * np.pi * r**2 * ALPHA_B * n_H**2) / Q_i) - (n_H * sigma_dust * phi)
        dtau_dr = n_H * sigma_dust
        
        return [dphi_dr, dtau_dr, dm_dr]

    def mass_event(r, y):
        return M_cloud - y[2]

    def phi_event(r, y):
        return y[0]
        
    phi_event.terminal = True
    phi_event.direction = -1
    
    mass_event.terminal = True
    mass_event.direction = -1
    
    # Initial conditions: phi(r_in) = 1, tau_d(r_in) = 0, m(r_in) = 0
    m_in = 0
    y0 = [1.0, 0.0, m_in]
    
    # Solve the ODE
    r_max = 1e9 * PC_TO_CM  # A very large radius that won't be reached before the mass condition
    sol = solve_ivp(cloud_structure_ode, [r_in, r_max], y0, events=(mass_event, phi_event), method='BDF', rtol=1e-8, atol=1e-10)
    
    # The escape fraction is the final value of phi
    f_esc_cloud = sol.y[0][-1]
    
    return f_esc_cloud if f_esc_cloud >= 1e-6 else 0.0

def process_buffers(get_buffers):
    """
    Process the output buffers and return their contents.

    Args:
        get_buffers (function): Function to retrieve the output buffers.

    Returns:
        tuple: A tuple containing (stdout_output, stderr_output).
    """
    stdout_buf, stderr_buf = get_buffers()
    stdout_buf.seek(0)
    stderr_buf.seek(0)
    stdout_output = stdout_buf.read()
    stderr_output = stderr_buf.read()
    return stdout_output, stderr_output

def rolling_mean(data, window_size):
    """
    Compute rolling mean using convolution.
    
    Args:
        data (np.array): The input data array.
        window_size (int): Size of the rolling window.
    
    Returns:
        np.array: Rolling mean array.
    """
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='same')

def rolling_std(data, window_size):
    """
    Compute rolling standard deviation using a sliding window.
    
    Args:
        data (np.array): The input data array.
        window_size (int): Size of the rolling window.
    
    Returns:
        np.array: Rolling standard deviation array.
    """
    rolling_mean_vals = rolling_mean(data, window_size)
    rolling_variance = rolling_mean((data - rolling_mean_vals)**2, window_size)
    return np.sqrt(rolling_variance)

def rolling_outlier_detection(data, window_size=3, threshold=3):
    """
    Detect outliers using a rolling window mean and standard deviation.

    Args:
        data (np.array): The input data array.
        window_size (int): Size of the rolling window.
        threshold (float): Number of standard deviations to define outliers.

    Returns:
        np.array: A boolean mask where True indicates non-outliers.
    """
    rolling_mean_vals = rolling_mean(data, window_size)
    rolling_std_vals = rolling_std(data, window_size)

    # Detect outliers as points that deviate significantly from the rolling mean
    mask = np.abs(data - rolling_mean_vals) < threshold * rolling_std_vals
    mask = np.nan_to_num(mask, nan=True)  # Replace NaN with True (non-outliers)

    return mask

def identify_spurious_data(data, log=False , prominence=0.5, window_size=3, threshold=3):
    """
    Identify spurious data points using combined peak detection and rolling outlier detection.

    Args:
        data (np.array): The input data array.
        log (bool): Whether to apply log10 to data before processing.
        prominence (float): Minimum prominence of peaks to consider them real.
        window_size (int): Size of the rolling window for outlier detection.
        threshold (float): Number of standard deviations to define outliers.
        
    Returns:
        np.array: A boolean mask where True indicates non-spurious data points.
    """
    if log:
        data = np.log10(data)

    # Find peaks with prominence
    peaks, _ = find_peaks(data, prominence=prominence)

    # Detect outliers using rolling window statistics
    outlier_mask = rolling_outlier_detection(data, window_size=window_size, threshold=threshold)

    # Create a mask where peaks and statistical outliers are considered spurious
    mask = np.ones(len(data), dtype=bool)
    mask[peaks] = False
    mask = mask & outlier_mask  # Combine the two masks
    
    return mask

def generate_spectral_table_filename(template, imf, star_type, formation_timescale=None, wavelength_resolution_factor=1):
    """
    Generate an appropriate filename for the spectral table based on its characteristics.

    Args:
        template (str): The stellar evolution model template (e.g., 'BPASSv2.2.1', 'SB99').
        imf (str): Initial Mass Function (e.g., 'chab100').
        star_type (str): Type of stars ('binary' or 'single').
        formation_timescale (float, optional): Formation timescale in years for constant SFR mode.
        wavelength_resolution_factor (int, optional): Factor by which wavelength resolution is degraded.

    Returns:
        str: The generated filename.
    """
    if template.startswith('BPASS'):  # Remove version number from BPASS
        template = re.sub(r'(BPASS)v\d+\.\d+\.\d+', r'\1', template)
    elif template.startswith('SB99'):
        template = 'SB99'  # using 'SB99' without any version

    # Determine the mode (burst or constant SFR)
    mode = "burst" if formation_timescale is None else "constantSFR"

    # Create the base filename
    filename = f"{template}_{imf}_{star_type}_{mode}"

    # Add formation timescale for constant SFR mode
    if formation_timescale is not None:
        formation_timescale_myr = formation_timescale / 1e6
        filename += f"_t{formation_timescale_myr:.1e}myr"

    # Add wavelength resolution factor if it's not 1
    if wavelength_resolution_factor != 1:
        filename += f"_resFac{wavelength_resolution_factor}"

    # Add file extension
    filename += ".ascii"

    return filename

def calculate_Q(spectrum, wavegrid, ionization_threshold=13.6):
    """
    Calculate the total ionizing photon rate in a spectrum for the luminosity case.
    
    Args:
        spectrum (nu * Lnu) in units of erg/s (cgs)
        wavegrid in micron
        ionization_threshold (float): Energy threshold for ionization in eV.
                                        Default is 13.6 eV (Lyman limit).

    Returns:
        Q value in photons/s.
    """
    # Convert wavelength from microns to energy in eV
    energy_eV = (const.h * const.c / (wavegrid * u.micron)).to(u.eV).value
    nu_Lnu = spectrum * u.erg / u.s  # erg/s

    ionizing_mask = energy_eV >= ionization_threshold
    
    # Convert wavelength to frequency
    nu = (const.c / (wavegrid[ionizing_mask] * u.micron)).to(u.Hz)
    
    # Convert nuLnu to Lnu
    Lnu = nu_Lnu[ionizing_mask] / nu

    # Convert from Lnu to photon luminosity as Lnu / h nu
    Lnu_photon = Lnu / (const.h * nu)

    return np.trapz(Lnu_photon.value, nu.value)

def calculate_mean_ionizing_energy(age, Z, L_i_interpolant, Q_i_interpolant):
    """
    Calculate the mean ionizing photon energy at a given age and metallicity
    using model predictions.

    Args:
        age (float): Age in years
        Z (float): Metallicity in solar units
        L_i_interpolant (callable): Interpolation function for ionizing luminosity
        Q_i_interpolant (callable): Interpolation function for ionizing photon rate

    Returns:
        float: Mean ionizing photon energy in ergs

    Notes:
        - age should be in years
        - This is more accurate than using the Lyman limit energy (13.6 eV)
        - Returns E_LYC if interpolation fails as a fallback
    """
    try:
        # Convert age to Myr for interpolants
        age_myr = age / 1e6
        
        # Get ionizing luminosity and photon rate
        L_i = 10**L_i_interpolant(age_myr, Z)  # erg/s
        Q_i = 10**Q_i_interpolant(age_myr, Z)  # photons/s
        
        # Calculate mean energy per photon
        if Q_i > 0:
            return L_i / Q_i
        else:
            return E_LYC  # Fallback to Lyman limit energy
            
    except Exception as e:
        print(f"Warning: Error calculating mean ionizing energy: {str(e)}")
        print("Falling back to Lyman limit energy")
        return E_LYC

def is_file_stable(file_path, stability_duration=12, max_wait_time=120):
    """
    Wait for file size to stabilize and be non-zero, indicating writing is complete.
    
    Args:
        file_path (str): Path to the file to check
        stability_duration (int): Time in seconds file should remain unchanged
        max_wait_time (int): Maximum time to wait for stability
        
    Returns:
        bool: True if non-zero file size stabilizes, False if timeout reached
    """
    start_time = time.time()
    last_size = -1
    stable_time = 0
    dt = 3
    
    while time.time() - start_time < max_wait_time:
        try:
            # Verify file exists and is non-zero in size
            current_size = os.path.getsize(file_path)
            if current_size > 0:
                # Check if size has stabilized
                if current_size == last_size:
                    stable_time += dt  # Increase stability counter
                    if stable_time >= stability_duration:
                        return True
                else:
                    # Reset stable time if size changes
                    stable_time = 0
                last_size = current_size
            else:
                stable_time = 0  # Reset if file size is zero or fluctuates unexpectedly
            
            time.sleep(dt)
        except OSError:
            # Wait if file isn't accessible
            time.sleep(dt)
            continue
    
    return False

def get_shell_inner_face_density(F, R, Z, Q_i, temp_interpolator=None, fix_n=False):
	"""
    Calculate shell inner face density based on the ionized gas temperature.

    When using the Cloudy interpolant, enforces minimum temperature of 1000K. A minimum 
    ionizing flux of 3e4 photons/cm^2/s is used as a safeguard due to table limits.

    Args:
        F (float): Total force on shell in dynes
        R (float): Shell radius in cm  
        Z (float): Gas metallicity in solar units
        Q_i (float): Ionizing photon rate in s^-1
        temp_interpolator (callable, optional): Function that returns gas temperature from 
            density, metallicity and ionizing flux
        fix_n (bool, optional): If True, skip root finding and use N_FALLBACK. Default False.

    Returns:
        tuple: (n_sh_R, T) where:
            n_sh_R (float): Shell inner face density in cm^-3 
            T (float): Ionized gas temperature in K
	"""
	T_MIN = 1000  # Minimum temperature in K
	N_FALLBACK = 100  # Fallback density in cm^-3
	PHI_MIN = 3e4  # Minimum ionizing flux in photons/cm^2/s
	P = F / (4 * np.pi * R**2)
	
	if temp_interpolator is None:
		# Direct calculation for non-interpolator case
		T_final = T_ION
		n_final = (MU_P * P) / (MU_N * K_BOLTZMANN * T_ION)
	else:
		# Use root finding with interpolator
		def density_equation(n):
			phi = max(Q_i / (4 * np.pi * R**2), PHI_MIN)
			temp_point = np.array([[
				np.log10(Z),
				np.log10(n),
				np.log10(phi)
			]])
			T = 10**temp_interpolator(temp_point)[0]
			n_calc = (MU_P * P) / (MU_N * K_BOLTZMANN * T)
			return n_calc - n
		
		if fix_n:
			n_final = N_FALLBACK
		else:
			try:
				n_final = brentq(density_equation, 1e-3, 1e6)
			except (ValueError, RuntimeError):
				n_final = N_FALLBACK
				warnings.warn(f"Root finding failed. Using fallback density of {N_FALLBACK} cm^-3",
							RuntimeWarning)
		
		# Calculate final temperature
		phi = max(Q_i / (4 * np.pi * R**2), PHI_MIN)  # Apply same minimum here
		temp_point = np.array([[
			np.log10(Z),
			np.log10(n_final),
			np.log10(phi)
		]])
		T_final = 10**temp_interpolator(temp_point)[0]
		T_final = max(T_MIN, T_final)  # Enforce minimum
	
	# Sanity checks
	if n_final <= 0:
		raise ValueError(f"Calculated negative or zero density: {n_final}")
	if T_final < T_MIN:
		raise ValueError(f"Temperature below minimum: {T_final} < {T_MIN}")
		
	return n_final, T_final