from .imports import np, os, re, sys, json, RegularGridInterpolator, interp1d
from .constants import *
from .utils import generate_spectral_table_filename
from ._fileutils import unique_tmp_path

# Bump when the table written for an unchanged request changes; tables with an older
# version in their sidecar are then regenerated.
STAR_TABLE_CODE_VERSION = 1
# Version of the sidecar's own layout (<table>.meta.json); 1 adds 'units'.
STAR_TABLE_SIDECAR_SCHEMA_VERSION = 1

# md5 of the star tables in the released Cloudy-resources bundle (download_data.py,
# GOOGLE_DRIVE_ID). A table without a sidecar is reused only if it is one of these.
RELEASED_TABLE_MD5 = {
    'BPASS_chab100_bin_burst_resFac10.ascii': '9275d20d0f72ceb0e155d47783339c5a',
    'BPASS_chab100_bin_constantSFR_t2.0e+00myr_resFac10.ascii': '741854b94d1b739fc8f1203bdfbffbdf',
    'BPASS_chab100_bin_constantSFR_t5.0e+00myr_resFac10.ascii': '1eb566617edeb1b427b27be44534271a',
    'BPASS_chab100_sin_burst_resFac10.ascii': '5807901cde769c8241e4035c5a2d3b8d',
    'BPASS_chab100_sin_constantSFR_t2.0e+00myr_resFac10.ascii': 'ea939cfdb54957d05fa73473ddeb3d65',
    'BPASS_chab100_sin_constantSFR_t5.0e+00myr_resFac10.ascii': 'd42bf70baa52ae7ffae400445f3234c3',
    'SB99_kroupa100_sin_burst.ascii': '7f51036e8c5d5db1fbafdee05916217e',
    'SB99_kroupa100_sin_constantSFR_t2.0e+00myr.ascii': '0a6d3cabfd01e040e1863b7784925abd',
    'SB99_kroupa100_sin_constantSFR_t5.0e+00myr.ascii': 'cca16ec1245f04bda4d1f7cac76c104d',
}


def bpass_burst_source_path(imf, star_type):
    """Where the full-resolution BPASS v2.2.1 burst table (from data.nublado.org) is read
    from: <src>/others/Cloudy Spectra/BPASSv2.2.1_imf_<imf>_burst_<single|binary>.ascii.
    scripts/download_data.py writes it here."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_name = f"BPASSv2.2.1_imf_{imf}_burst_{'binary' if star_type == 'bin' else 'single'}.ascii"
    return os.path.join(base_path, "others", "Cloudy Spectra", file_name)


def _file_md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(obj):
    """obj with numpy scalars/arrays and tuples turned into plain JSON types."""
    return json.loads(json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))


def _write_json_atomic(path, obj):
    tmp = unique_tmp_path(path)
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# Ionization edges [Angstrom] that no averaged block may straddle: He II, He I, H I.
IONIZATION_EDGES_ANGSTROM = (227.84, 504.26, 911.75)


def reduce_resolution(wavelengths, fluxes, factor, resample='decimate'):
    """
    Reduce the wavelength resolution of a spectral grid by ``factor``.

    'decimate' keeps every factor-th point (``[::factor]``): the method the released v2
    tables used, kept as the default so they stay reproducible. 'average' replaces each
    block of ``factor`` consecutive points by its cell-weighted mean, with the same
    weights for the wavelength, where a point's cell spans the midpoints to its
    neighbours; this keeps the integral of the spectrum, which decimation moves by about
    1%. Blocks are formed separately on each side of the He II, He I
    and H I edges (:data:`IONIZATION_EDGES_ANGSTROM`), so no block mixes ionizing and
    non-ionizing flux; the last block of each segment may be shorter.

    Args:
        wavelengths (np.ndarray): ascending wavelength grid, shape (n,).
        fluxes (np.ndarray): fluxes with wavelength on the last axis, shape (..., n).
        factor (int): reduction factor; 1 returns the inputs unchanged.
        resample (str): 'decimate' or 'average'.

    Returns:
        tuple: (wavelengths, fluxes) at the reduced resolution.
    """
    if resample not in ('decimate', 'average'):
        raise ValueError(f"resample must be 'decimate' or 'average', got {resample!r}")
    if np.any(np.diff(np.asarray(wavelengths, dtype=float)) <= 0):
        raise ValueError("wavelengths must be strictly ascending")
    if factor == 1:
        return wavelengths, fluxes
    if resample == 'decimate':
        return wavelengths[::factor], fluxes[..., ::factor]
    lam = np.asarray(wavelengths, dtype=float)
    edges = np.concatenate(([lam[0]], 0.5 * (lam[1:] + lam[:-1]), [lam[-1]]))
    w = np.diff(edges)
    cuts = np.unique(np.concatenate(([0], np.searchsorted(lam, IONIZATION_EDGES_ANGSTROM), [lam.size])))
    starts = np.concatenate([np.arange(c0, c1, factor) for c0, c1 in zip(cuts[:-1], cuts[1:]) if c1 > c0])
    W = np.add.reduceat(w, starts)
    new_lam = np.add.reduceat(w * lam, starts) / W
    new_flux = np.add.reduceat(np.asarray(fluxes, dtype=float) * w, starts, axis=-1) / W
    return new_lam, new_flux


def parse_pysb99_population_metadata(metadata):
    """
    Read the population definition from a pySB99 interpolant metadata dict.

    Older metadata files used imf_type/exponents/mass_limits instead of
    population_type/imf_exponents/imf_mass_limits; both are read, so a custom
    or non-Kroupa population never falls back to the Kroupa defaults.

    Returns:
        tuple: (population_type, custom_star_numbers, imf_exponents, imf_mass_limits)
    """
    population_type = metadata.get('population_type')
    if population_type is None:
        legacy_type = metadata.get('imf_type')
        if legacy_type == 'custom' or 'custom_star_numbers' in metadata:
            population_type = 'custom'
        else:
            population_type = 'imf'

    if population_type == 'custom':
        # JSON converts dict keys to strings - convert back to floats
        custom_star_numbers_raw = metadata.get('custom_star_numbers', None)
        if custom_star_numbers_raw is not None:
            custom_star_numbers = {float(k): v for k, v in custom_star_numbers_raw.items()}
        else:
            custom_star_numbers = None
        return population_type, custom_star_numbers, None, None

    imf_exponents = metadata.get('imf_exponents', metadata.get('exponents', [1.3, 2.3]))
    imf_mass_limits = tuple(metadata.get('imf_mass_limits',
                                         metadata.get('mass_limits', (0.1, 0.5, 100.0))))
    return population_type, None, imf_exponents, imf_mass_limits


def cloudy_logZ_text(Z):
    """log10 Z as every star table stores it and every ``table star`` line requests it.

    One formatter for both sides: Cloudy refuses a request outside the table's log Z range,
    and a full-precision request lies just outside a one-node table (or an edge node)."""
    return f"{np.log10(Z):.6f}"


def _grid_lines(time_points, Z_values):
    """The age/logZ lines of a star table, exactly as :func:`write_cloudy_star_table` writes them."""
    return [f"{t:.6e} {cloudy_logZ_text(Z)}" for Z in Z_values for t in time_points]


def write_cloudy_star_table(full_path, time_points, Z_values, wavelengths, spectra, sidecar):
    """
    Write a Cloudy star table (ascii, 2 axes Age / log(Z)) and its sidecar ``<table>.meta.json``.

    The one writer for every Cloudy star table (SpectralTableGenerator and the stochastic
    pySB99 tables). The grid is checked before anything is written; the table goes to a
    temporary file renamed into place, so a partial table never carries the final name.

    Args:
        full_path (str): destination of the table.
        time_points (array-like): ages [yr], strictly ascending, >= 0.
        Z_values (array-like): metallicities (mass fractions), > 0 (the table stores log10 Z).
        wavelengths (array-like): wavelengths [Angstrom], strictly ascending.
        spectra (array-like): shape (len(Z_values), len(time_points), len(wavelengths)),
            finite and >= 0.
        sidecar (dict): fields stored in the sidecar next to 'md5' and 'filename', e.g.
            'origin', 'key', 'paired_feedback_md5'.

    Returns:
        str: full_path.
    """
    t = np.asarray(time_points, dtype=float)
    Zs = np.asarray(Z_values, dtype=float)
    lam = np.asarray(wavelengths, dtype=float)
    spec = np.asarray(spectra, dtype=float)
    if t.ndim != 1 or t.size < 1 or np.any(t < 0) or np.any(np.diff(t) <= 0):
        raise ValueError("star table ages must be >= 0 and strictly ascending")
    if Zs.ndim != 1 or Zs.size < 1 or np.any(~np.isfinite(Zs)) or np.any(Zs <= 0):
        raise ValueError(f"star table metallicities must be > 0 (log10 Z is written); got {list(Zs)}")
    if lam.ndim != 1 or np.any(np.diff(lam) <= 0):
        raise ValueError("star table wavelengths must be strictly ascending")
    if spec.shape != (Zs.size, t.size, lam.size):
        raise ValueError(f"spectra have shape {spec.shape}; expected (n_Z, n_t, n_wl) = "
                         f"{(Zs.size, t.size, lam.size)}")
    if not np.all(np.isfinite(spec)):
        raise ValueError("spectra must be finite")
    if np.any(spec < 0):
        raise ValueError("spectra must be non-negative")

    tmp_path = unique_tmp_path(full_path)
    with open(tmp_path, 'w') as f:
        f.write("20060612\n")
        f.write("2\n2\n")
        f.write("Age\nlog(Z)\n")
        f.write(f"{len(time_points) * len(Z_values)}\n")
        f.write(f"{len(wavelengths)}\n")
        f.write("lambda\n1.00000000e+00\n")
        f.write("F_lambda\n1.00000000e+00\n")

        # Write age-metallicity combinations
        for line in _grid_lines(time_points, Z_values):
            f.write(line + "\n")

        # Write wavelengths
        for i, w in enumerate(wavelengths):
            f.write(f"{w:.6e}")
            if (i + 1) % 5 == 0 or i == len(wavelengths) - 1:
                f.write("\n")
            else:
                f.write("  ")

        # Write spectra
        for Z_spectra in spectra:
            for spectrum in Z_spectra:
                for i, flux in enumerate(spectrum):
                    f.write(f"{flux:.6e}")
                    if (i + 1) % 5 == 0 or i == len(spectrum) - 1:
                        f.write("\n")
                    else:
                        f.write("  ")
    # an index Cloudy compiled from an earlier table of this name would be read against
    # the new one at the wrong offsets: it goes before the new table takes the name
    move_cloudy_index_aside(full_path)
    os.replace(tmp_path, full_path)
    meta = {'filename': os.path.basename(full_path), 'md5': _file_md5(full_path),
            'sidecar_schema_version': STAR_TABLE_SIDECAR_SCHEMA_VERSION,
            'units': {'age': 'yr', 'Z': '', 'lambda': 'Angstrom', 'F_lambda': 'unspecified'}}
    meta.update(sidecar)
    _write_json_atomic(full_path + '.meta.json', _jsonable(meta))
    return full_path


def star_table_is_current(full_path, key, time_points, Z_values):
    """True if the table at full_path holds what a request with content ``key`` on this
    age/Z grid would write.

    With a sidecar: its md5 must match the file, its key must equal ``key`` (or, for a
    released table, the file must still have the released md5), and the table's own grid
    must equal the requested one. Without a sidecar: only a released table (md5 in
    RELEASED_TABLE_MD5) is accepted, and a sidecar is written for it.
    """
    filename = os.path.basename(full_path)
    if _read_grid_lines(full_path) != _grid_lines(time_points, Z_values):
        return False
    meta_path = full_path + '.meta.json'
    md5 = _file_md5(full_path)
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            return False  # unreadable sidecar: the table's content is unknown
        if meta.get('md5') != md5:
            return False  # the table changed after its sidecar was written
        if meta.get('origin') == 'released-registry':
            return md5 == RELEASED_TABLE_MD5.get(filename)
        return meta.get('key') == key
    if RELEASED_TABLE_MD5.get(filename) == md5:
        _write_json_atomic(meta_path, {'origin': 'released-registry', 'md5': md5, 'filename': filename})
        return True
    return False


def _read_grid_lines(path):
    """The age/logZ lines of an existing table (None if it is not a readable table)."""
    try:
        with open(path) as f:
            head = [next(f) for _ in range(11)]
            n_mod = int(head[5])
            return [next(f).strip() for _ in range(n_mod)]
    except (StopIteration, ValueError, IndexError, OSError):
        return None


def cloudy_index_files(full_path):
    """The files Cloudy compiles from a star table <name>.ascii: <name>.idx (Cloudy
    stars.cpp:2841 at 013dc0bf, 2402 at 01c4cfc6) and, if present, <name>.mod."""
    base = full_path[:-len('.ascii')] if full_path.endswith('.ascii') else full_path
    return [base + '.idx', base + '.mod']


def _move_files_aside(paths, dest_root):
    from datetime import datetime, timezone
    present = [p for p in paths if os.path.exists(p)]
    if not present:
        return None
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    dest_dir = os.path.join(dest_root, 'stale_tables', stamp)
    os.makedirs(dest_dir, exist_ok=True)
    for p in present:
        os.replace(p, os.path.join(dest_dir, os.path.basename(p)))
    return dest_dir


def move_star_table_aside(full_path, data_dir=None):
    """Move a table, its sidecar and its Cloudy index (.idx/.mod) to
    <data_dir>/stale_tables/<UTC time>/; never delete. data_dir defaults to the table's
    own directory. Cloudy reads an index it finds without checking that it was built
    from the table next to it, so the index must leave with its table."""
    dest_root = data_dir or os.path.dirname(os.path.abspath(full_path))
    dest_dir = _move_files_aside([full_path, full_path + '.meta.json'] + cloudy_index_files(full_path),
                                 dest_root)
    if dest_dir:
        print(f"  Moved the outdated '{os.path.basename(full_path)}' to {dest_dir}")


def move_cloudy_index_aside(full_path, data_dir=None):
    """Move a table's Cloudy index (.idx/.mod), if any, to <data_dir>/stale_tables/<UTC time>/."""
    dest_root = data_dir or os.path.dirname(os.path.abspath(full_path))
    dest_dir = _move_files_aside(cloudy_index_files(full_path), dest_root)
    if dest_dir:
        print(f"  Moved the Cloudy index of '{os.path.basename(full_path)}' to {dest_dir}")


def files_md5(paths):
    """md5 over the concatenated bytes of ``paths`` (the rule feedback_table_md5 uses)."""
    import hashlib
    h = hashlib.md5()
    for p in paths:
        with open(p, 'rb') as f:
            h.update(f.read())
    return h.hexdigest()


class SpectralTableGenerator:
    def __init__(self, template, imf, star_type, max_age=None, wavelength_resolution_factor=1,
                 resample='decimate'):
        """
        Initialize the SpectralTableGenerator.

        Args:
            template (str): The stellar evolution model template (e.g., 'BPASSv2.2.1', 'SB99', 'pySB99').
            imf (str): Initial Mass Function (e.g., 'chab100', 'kroupa100').
                      For pySB99: Must match a generated interpolant name.
            star_type (str): Type of stars ('binary' or 'single').
                           For pySB99: Always 'sin' (single stars only).
            max_age (float, optional): Maximum age in years. If None, use the default for the template.
            wavelength_resolution_factor (int, optional): Factor by which to degrade wavelength resolution.
            resample (str, optional): How the resolution is reduced: 'decimate' (default, as the
                released tables) or 'average' (keeps the integral; files get a _resAvgN suffix).
                See :func:`reduce_resolution`.
        """
        if resample not in ('decimate', 'average'):
            raise ValueError(f"resample must be 'decimate' or 'average', got {resample!r}")
        self.template = template
        self.imf = imf
        self.star_type = star_type
        self.wavelength_resolution_factor = wavelength_resolution_factor
        self.resample = resample
        
        if self.template.startswith('BPASS'):
            self.max_age = max_age if max_age is not None else 2e9 #yrs
            self.data = self._load_bpass_burst_data()
        elif self.template.startswith('SB99'):
            self.data = self._load_sb99_burst_data()
            self.max_age = np.max(self.data['ages']) # input overridden here => data is short
        elif self.template == 'pySB99':
            self.max_age = max_age if max_age is not None else 50e6  # 50 Myr default
            self.data = self._load_pysb99_burst_data()
        else:
            raise ValueError(f"Unsupported template for Cloudy star tables: {self.template}. Its "
                             f"feedback table can drive the evolution, but the Cloudy stage needs a "
                             f"spectral loader for the template (spectra per age and Z); built-in "
                             f"loaders: BPASS, SB99, pySB99.")
        
        self._setup_interpolator()
        self.convolution_dt = 0.025 * MYR_TO_SEC

    def _get_burst_table_path(self):
        """Get the path to the burst table file."""
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if self.template.startswith('BPASS'):
            return bpass_burst_source_path(self.imf, self.star_type)
        
        elif self.template.startswith('SB99'):
            # SB99 files are organized differently, with separate files for each metallicity
            sb99_dir = os.path.join(base_path, "others", "SB99", self.imf)
            
            # return a dictionary of paths for each metallicity
            metallicities = [0.001, 0.004, 0.008, 0.020, 0.040]
            file_paths = {}
            
            for Z in metallicities:
                Z_str = f"{int(Z*1000):03d}"
                file_name = f"lim_100_Z_.{Z_str}.spectrum1"
                file_path = os.path.join(sb99_dir, f"Z_{Z_str}", file_name)
                file_paths[Z] = file_path
            
            return file_paths
        
        else:
            raise ValueError(f"Unsupported template: {self.template}")

    def _load_bpass_burst_data(self):
        """
        Load the burst mode data from the BPASS file, treating age and metallicity as a grid.
        Ensures consistency between ages, metallicities, and fluxes.
        The ascii files were downloaded from Cloudy website, so units are consistent 
        with Cloudy.
        """
        file_path = self._get_burst_table_path()

        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Read header information
        n_models = int(lines[5].strip())
        n_wavelengths = int(lines[6].strip())
        wavelength_factor = float(lines[8].strip())
        flux_factor = float(lines[10].strip())

        # Read ages and metallicities
        ages_and_Z = []
        current_line = 11
        while len(ages_and_Z) < n_models:
            values = list(map(float, lines[current_line].split()))
            ages_and_Z.extend(zip(values[::2], values[1::2]))
            current_line += 1
        
        ages_and_Z = np.array(ages_and_Z)
        
        # Read wavelengths
        wavelengths = []
        while len(wavelengths) < n_wavelengths:
            wavelengths.extend(map(float, lines[current_line].split()))
            current_line += 1
        wavelengths = np.array(wavelengths)
        
        # Read fluxes
        fluxes = []
        for _ in range(n_models):
            flux_row = []
            while len(flux_row) < n_wavelengths:
                flux_row.extend(map(float, lines[current_line].split()))
                current_line += 1
            fluxes.append(flux_row)
        fluxes = np.array(fluxes)

        # Sort ages_and_Z and fluxes as a grid
        sorted_indices = np.lexsort((ages_and_Z[:, 1], ages_and_Z[:, 0]))
        ages_and_Z = ages_and_Z[sorted_indices]
        fluxes = fluxes[sorted_indices]

        # Extract unique ages and metallicities
        unique_ages = np.unique(ages_and_Z[:, 0])
        unique_metallicities = np.unique(10**ages_and_Z[:, 1]) # Convert from log10(Z) to Z

        # Create a grid of ages and metallicities
        n_ages = len(unique_ages)
        n_metallicities = len(unique_metallicities)

        # Reshape fluxes to match the grid structure (metallicities, ages, wavelengths)
        fluxes_grid = np.zeros((n_metallicities, n_ages, n_wavelengths))
        for i, (age, log_Z) in enumerate(ages_and_Z):
            age_index = np.where(unique_ages == age)[0][0]
            Z_index = np.where(unique_metallicities == 10**log_Z)[0][0]
            fluxes_grid[Z_index, age_index, :] = fluxes[i]

        # Create the burst_data dictionary
        burst_data = {
            'ages': unique_ages,
            'metallicities': unique_metallicities,
            'wavelengths': wavelengths * wavelength_factor,
            'fluxes': fluxes_grid * flux_factor
        }

        # Add t=0 entry (assuming it's the same as the spectrum at the earliest age)
        burst_data['ages'] = np.concatenate(([0], burst_data['ages']))
        burst_data['fluxes'] = np.concatenate((burst_data['fluxes'][:, :1, :], burst_data['fluxes']), axis=1)

        # Apply max_age filter if specified
        if self.max_age is not None:
            age_mask = burst_data['ages'] <= self.max_age
            burst_data['ages'] = burst_data['ages'][age_mask]
            burst_data['fluxes'] = burst_data['fluxes'][:, age_mask, :]

        # Apply wavelength resolution degradation
        burst_data['wavelengths'], burst_data['fluxes'] = reduce_resolution(
            burst_data['wavelengths'], burst_data['fluxes'], self.wavelength_resolution_factor, self.resample)

        # Assign the burst_data to self.burst_data
        self.burst_data = burst_data

        return self.burst_data

    def _load_sb99_burst_data(self):
        """Load SB99 data from files downloaded from the SB99 website, unit conversion is required
        for use in Cloudy"""
        flux_fac = 4 * np.pi * (10 * PC_TO_CM)**2
        file_paths = self._get_burst_table_path()
        
        log_Z_list = []
        log_ages_list = []
        wavelengths_list = []
        log_fluxes_list = []

        for Z, file_path in file_paths.items():
            with open(file_path, 'r') as f:
                lines = f.readlines()[6:]  # Skip header lines
            
            ages = []
            wavelengths = []
            fluxes = []
            
            for line in lines:
                data = line.split()
                age = float(data[0])
                wavelength = float(data[1])
                flux = float(data[3]) - np.log10(flux_fac) # Stellar, not total, flux at 10pc 
                
                if len(ages) == 0 or age != ages[-1]:
                    if len(ages) > 0:
                        log_Z_list.append(np.log10(Z))
                        log_ages_list.append(np.log10(ages[-1]))
                        wavelengths_list.append(wavelengths)
                        log_fluxes_list.append(fluxes)
                    ages.append(age)
                    wavelengths = []
                    fluxes = []
                
                wavelengths.append(wavelength)
                fluxes.append(flux)
            
            # Append the last age group
            log_Z_list.append(np.log10(Z))
            log_ages_list.append(np.log10(ages[-1]))
            wavelengths_list.append(wavelengths)
            log_fluxes_list.append(fluxes)

        # Ensure all wavelength grids are the same
        assert all(np.array_equal(wavelengths_list[0], w) for w in wavelengths_list[1:])
        wavelengths = np.array(wavelengths_list[0])
        
        # Create unique sorted arrays for log_Z and log_ages
        log_Z = np.unique(log_Z_list)
        log_ages = np.unique(log_ages_list)

        # Create a 3D grid: (log_Z, log_age, wavelength)
        log_fluxes = np.array(log_fluxes_list).reshape(len(log_Z), len(log_ages), len(wavelengths))

        # Convert to linear scale for consistency with BPASS output
        burst_data = {
            'ages': 10**log_ages,
            'metallicities': 10**log_Z,
            'wavelengths': wavelengths,
            'fluxes': 10**log_fluxes
        }

        # Add t=0 entry (assuming it's the same as the spectrum at the earliest age)
        burst_data['ages'] = np.concatenate(([0], burst_data['ages']))
        burst_data['fluxes'] = np.concatenate((burst_data['fluxes'][:, :1, :], burst_data['fluxes']), axis=1)

        # Apply wavelength resolution degradation (the filename carries the factor)
        burst_data['wavelengths'], burst_data['fluxes'] = reduce_resolution(
            burst_data['wavelengths'], burst_data['fluxes'], self.wavelength_resolution_factor, self.resample)

        # Assign the burst_data to self.burst_data
        self.burst_data = burst_data

        return self.burst_data

    def _load_pysb99_burst_data(self):
        """
        Load burst mode data from pySB99 by running population synthesis.
        
        This method:
        1. Verifies that pySB99 interpolants exist for this IMF
        2. Loads metadata to get metallicities and spectral library
        3. Runs pySB99 for each metallicity used in interpolant generation
        4. Extracts wavelength-resolved spectral data
        5. Organizes into the standard burst_data format
        
        Returns:
            dict: Dictionary with keys 'ages', 'metallicities', 'wavelengths', 'fluxes'
        """
        from toddlers.pysb99.pysb99_core import StellarPopulationConfig, run_population_synthesis
        
        print(f"Generating pySB99 spectral data for IMF: {self.imf}")
        
        # The generator configuration of the table: the unified HDF5 table's 'source'
        # attribute (generator_config), else a 2.0 table's _metadata.json (deprecated).
        from ._paths import get_data_dir
        from .feedback_tables import find_table, read_table_file
        database_dir = get_data_dir()  # $TODDLERS_DATA or <pkg>/database
        h5 = find_table('pySB99', self.imf)
        interpolant_file = os.path.join(database_dir, f'pySB99interpolation_{self.imf}.obj')
        metadata_file = os.path.join(database_dir, f'pySB99interpolation_{self.imf}_metadata.json')
        if h5 is not None:
            metadata = read_table_file(h5).metadata['source'].get('generator_config')
            if metadata is None:
                raise ValueError(f"{h5} records no pySB99 generator_config in its source attribute; "
                                 f"regenerate it with toddlers.pysb99.generate_pysb99_interpolants.")
            interpolant_file = h5
        elif not os.path.exists(interpolant_file):
            raise FileNotFoundError(
                f"pySB99 feedback table not found: {os.path.join(database_dir, f'pySB99_{self.imf}_feedback.h5')}\n"
                f"Generate it first with toddlers.pysb99.generate_pysb99_interpolants"
            )
        elif os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            # Metadata is required - cannot safely assume defaults
            raise FileNotFoundError(
                f"Metadata file required but not found: {metadata_file}\n"
                f"Generate interpolants with metadata using:\n"
                f"  - generate_kroupa_like_interpolants() for IMF populations\n"
                f"  - generate_custom_population_interpolants() for custom populations\n"
                f"(see examples/11_custom_population_pipeline.py)."
            )
        self._pysb99_metadata = metadata
        self._pysb99_interpolant_file = interpolant_file
        metallicity_strings = metadata['metallicities']
        Z_values = metadata['Z_values']
        spectral_library = metadata['spectral_library']
        rotation = metadata.get('rotation', False)
        population_type, custom_star_numbers, imf_exponents, imf_mass_limits = \
            parse_pysb99_population_metadata(metadata)
        print(f"  Loaded metadata: {len(metallicity_strings)} metallicities, library={spectral_library}")
        print(f"  Population type: {population_type}")

        metallicity_strings = list(metallicity_strings)
        Z_values = list(Z_values)

        # Build the table from the same population run as the interpolant: same time
        # grid, speed mode and IMF extension, read from the metadata. The table's age
        # axis is the run's own time nodes.
        run_speed_mode = metadata.get('run_speed_mode', 'DEFAULT')  # the only mode older generators used
        auto_extend_imf = metadata.get('auto_extend_imf')
        if population_type == 'custom' and auto_extend_imf is None:
            auto_extend_imf = True  # custom-population interpolants have always been generated with it on
        if auto_extend_imf is None:
            raise ValueError(
                f"Metadata {metadata_file} does not record 'auto_extend_imf'. The Cloudy table "
                f"must use the interpolant's IMF upper limit, and older metadata cannot tell "
                f"whether the IMF was extended to the track maximum. Add "
                f"\"auto_extend_imf\": true or false (as used when generating the interpolant), "
                f"or regenerate the interpolant."
            )
        time_start = metadata['time_start_myr'] * 1e6
        time_end = metadata['time_end_myr'] * 1e6
        time_step = metadata['time_step_myr'] * 1e6
        if self.max_age > time_end:
            print(f"  Note: max_age {self.max_age/1e6:.1f} Myr exceeds the interpolant's "
                  f"{time_end/1e6:.1f} Myr; the spectrum is held at its last age beyond it, "
                  f"as the feedback interpolant is.")

        print(f"  Metallicities: {metallicity_strings}")
        print(f"  Time: {time_start/1e6:.3f} - {time_end/1e6:.1f} Myr, step {time_step/1e6:.3f} Myr, "
              f"mode {run_speed_mode}, auto_extend_imf={auto_extend_imf}")

        # Run pySB99 for each metallicity
        all_spectra = []
        wavelength_grid = None
        time_points = None

        for met_string in metallicity_strings:
            print(f"  Running pySB99 for {met_string}...")

            config = StellarPopulationConfig(
                metallicity=met_string,
                time_start=time_start,
                time_end=time_end,
                time_step=time_step,
                spectral_library=spectral_library,
                rotation=rotation,
                run_speed_mode=run_speed_mode,
                auto_extend_imf=bool(auto_extend_imf)
            )

            # Set population parameters
            if custom_star_numbers is not None:
                # Custom stellar population
                config.custom_star_numbers = custom_star_numbers
                config.total_mass = None  # Not used with custom populations
            else:
                # IMF-based population
                config.total_mass = 1e6
                config.imf_exponents = imf_exponents
                config.imf_mass_limits = imf_mass_limits

            # Run synthesis
            results = run_population_synthesis(config)

            # Get actual stellar mass for normalization
            if custom_star_numbers is not None:
                # Custom population: use actual mass
                initial_masses = np.array(results.stellar_masses[0])
                actual_mass = np.sum(results.number_of_stars * initial_masses)
                normalization_factor = 1e6 / actual_mass
                print(f"    Custom population: {actual_mass:.2e} Msun --> normalizing to 1e6 Msun (x{normalization_factor:.3f})")
            else:
                # IMF population: already 1e6 Msun
                normalization_factor = 1.0

            if wavelength_grid is None:
                wavelength_grid = results.wavelength_grid
                time_points = np.asarray(results.times, dtype=float)
            elif not np.array_equal(time_points, results.times):
                raise RuntimeError("pySB99 runs at different metallicities returned different time grids")

            # pySB99 units (10^-20 erg/s/A), make it even smaller
            # this doesnt matter as Cloudy renormalizes anyway
            # but Cloudy complains if numbers are too large, adding a 1e-20 factor
            met_spectra = np.maximum(np.asarray(results.flux_spectra, dtype=float), 1e-50) \
                * 1e-20 * normalization_factor
            all_spectra.append(met_spectra)

        # Keep the ages up to max_age (plus the first node beyond it, so max_age is bracketed)
        n_keep = min(len(time_points), int(np.searchsorted(time_points, self.max_age)) + 1)
        time_points = time_points[:n_keep]
        all_spectra = [sp[:n_keep] for sp in all_spectra]

        # Organize into standard format
        fluxes_grid = np.array(all_spectra)
        
        burst_data = {
            'ages': time_points,
            'metallicities': np.array(Z_values),
            'wavelengths': wavelength_grid,
            'fluxes': fluxes_grid
        }
        
        print(f"  Spectral data ready: {burst_data['fluxes'].shape}")
        
        # Apply wavelength resolution degradation if needed
        if self.wavelength_resolution_factor > 1:
            print(f"  Reducing wavelength resolution by factor {self.wavelength_resolution_factor} ({self.resample})...")
            burst_data['wavelengths'], burst_data['fluxes'] = reduce_resolution(
                burst_data['wavelengths'], burst_data['fluxes'], self.wavelength_resolution_factor, self.resample)
            print(f"    New wavelength grid: {len(burst_data['wavelengths'])} points")

        self.burst_data = burst_data
        return self.burst_data

    def _setup_interpolator(self):
        """Set up 2D interpolator for age and metallicity in log-space."""
        log_metallicities = np.log10(self.burst_data['metallicities'])
        log_ages = np.log10(self.burst_data['ages'] + 1e-10)
        log_fluxes = np.log10(self.burst_data['fluxes'])

        self.flux_interpolator = RegularGridInterpolator(
            (log_metallicities, log_ages),
            log_fluxes,
            method='linear',
            bounds_error=False,
            fill_value=None
        )

    def _get_burst_spectrum(self, t, Z):
        """Get the spectrum for a given time and metallicity in burst mode using 2D interpolation in log-space.

        For pySB99 the age is clamped to the run's age range (the spectrum is held at the
        edge, as the feedback interpolant is), never extrapolated."""
        log_Z = np.log10(Z)
        log_t = np.log10((max(t, 1e-10)))
        if self.template == 'pySB99':
            ages = self.burst_data['ages']
            log_t = np.clip(log_t, np.log10(ages[0] + 1e-10), np.log10(ages[-1] + 1e-10))
        log_spectrum = self.flux_interpolator((log_Z, log_t))
        return 10**log_spectrum
    
    def _get_constant_sfr_spectrum(self, t, Z, formation_timescale):
        """Get the spectrum for a given time and metallicity in constant SFR mode.
        formation time-scale in years"""
        spectrum = np.zeros_like(self.burst_data['wavelengths'])
        dt = self.convolution_dt / YR_TO_SEC  # Convert to years
        
        for tau in np.arange(0, min(t, formation_timescale), dt):
            weight = dt / formation_timescale
            burst_spectrum = self._get_burst_spectrum(t - tau, Z)
            spectrum += weight * burst_spectrum

        return spectrum

    def _generate_output_filename(self, formation_timescale=None):
        """
        Generate an appropriate filename for the spectral table based on its characteristics
        (the same name the Cloudy input generator expects; see
        :func:`toddlers.utils.generate_spectral_table_filename`).

        Args:
            formation_timescale (float, optional): Formation timescale in years for constant SFR mode.

        Returns:
            str: The generated filename.
        """
        return generate_spectral_table_filename(
            self.template, self.imf, self.star_type, formation_timescale=formation_timescale,
            wavelength_resolution_factor=self.wavelength_resolution_factor, resample=self.resample)

    def _source_fingerprint(self):
        """What the table is built from: md5 of the source files (BPASS, SB99) or, for
        pySB99, the generator configuration and the md5 of the feedback table it describes
        (the unified HDF5 table, or a 2.0 .obj)."""
        if self.template == 'pySB99':
            return {'pysb99_metadata': self._pysb99_metadata,
                    'interpolant_md5': _file_md5(self._pysb99_interpolant_file)}
        paths = self._get_burst_table_path()
        if isinstance(paths, dict):
            return {str(Z): _file_md5(p) for Z, p in sorted(paths.items())}
        return {'source_md5': _file_md5(paths)}

    def _content_key(self, formation_timescale):
        """Everything the table content depends on, apart from its age/logZ grid (which
        is read back from the table's own header). Returned in its JSON form (numpy
        scalars converted), so it compares equal to what a sidecar stores."""
        if formation_timescale is not None:
            formation_timescale = float(formation_timescale)
        return _jsonable({
            'template': self.template, 'imf': self.imf, 'star_type': self.star_type,
            'formation_timescale': formation_timescale,
            'wavelength_resolution_factor': self.wavelength_resolution_factor,
            'resample': self.resample, 'max_age': float(self.max_age),
            'table_code_version': STAR_TABLE_CODE_VERSION,
            'source': self._source_fingerprint(),
        })

    @staticmethod
    def _grid_lines(time_points, Z_values):
        """The age/logZ lines exactly as _write_spectral_table writes them."""
        return _grid_lines(time_points, Z_values)

    @staticmethod
    def _read_grid_lines(path):
        """The age/logZ lines of an existing table (None if it is not a readable table)."""
        return _read_grid_lines(path)

    def _table_is_current(self, full_path, key, time_points, Z_values):
        """True if the table at full_path holds what this request would write
        (see :func:`star_table_is_current`)."""
        return star_table_is_current(full_path, key, time_points, Z_values)

    @staticmethod
    def _move_aside(full_path):
        """Move a table (and its sidecar) to CLOUDY_DATA_DIR/stale_tables/<UTC time>/; never delete."""
        move_star_table_aside(full_path, CLOUDY_DATA_DIR)

    def _paired_feedback_md5(self):
        """md5 of the feedback table this star table pairs with (Cloudy takes the SED shape
        from the star table and the normalization from the feedback table), or None if that
        table is not present."""
        from .stellar_feedback import feedback_table_md5
        family = 'BPASS' if self.template.startswith('BPASS') else (
            'SB99' if self.template.startswith('SB99') else self.template)
        star_type = self.star_type if family == 'BPASS' else None
        try:
            return feedback_table_md5(family, self.imf, star_type)
        except (FileNotFoundError, ValueError):
            return None

    def _sidecar_units(self):
        """Units of the table as written (Cloudy uses the spectral shape only)."""
        if self.template.startswith('SB99'):
            f = 'erg s-1 cm-2 Angstrom-1'
            note = 'Starburst99 stellar spectrum at 10 pc, per 1e6 Msun'
        elif self.template == 'pySB99':
            f = '1e40 erg s-1 Angstrom-1'
            note = 'pySB99 flux_spectra (1e20 erg s-1 Angstrom-1) x 1e-20, per 1e6 Msun'
        else:
            f = 'unspecified'
            note = 'as in the BPASS star table distributed with Cloudy (scale not recorded in that file)'
        return {'age': 'yr', 'Z': '', 'lambda': 'Angstrom', 'F_lambda': f, 'F_lambda_note': note}

    def _write_spectral_table(self, time_points, Z_values, wavelengths, spectra, formation_timescale=None,
                              sidecar=None):
        """
        Write the spectral table in Cloudy-compatible format through
        :func:`write_cloudy_star_table`. Reuse decisions are made by generate_spectral_table.

        Args:
            time_points (array-like): Array of time points.
            Z_values (array-like): Array of metallicity values.
            wavelengths (array-like): Array of wavelengths.
            spectra (list): List of spectra for each metallicity and time point.
            formation_timescale (float, optional): Formation timescale for constant SFR mode.
            sidecar (dict, optional): sidecar fields (default: none beyond md5 and filename).

        Returns:
            str: Path to the spectral table file.
        """
        filename = self._generate_output_filename(formation_timescale)
        full_path = os.path.join(CLOUDY_DATA_DIR, filename)
        return write_cloudy_star_table(full_path, time_points, Z_values, wavelengths, spectra,
                                       sidecar or {})

    def generate_spectral_table(self, time_points, Z_values, formation_timescale=None):
        """
        Generate a spectral table for Cloudy, supporting both burst and constant SFR modes for multiple metallicities.

        An existing table of the same name is reused only if it holds what this request
        would write (see _table_is_current); otherwise it is moved to
        CLOUDY_DATA_DIR/stale_tables/ and the table is regenerated. Each table written
        gets a sidecar <table>.meta.json with its content key.

        Args:
            time_points (np.array): Array of time points (in years) for which to generate spectra.
            Z_values (list or np.array): List of metallicities in solar units.
            formation_timescale (float, optional): Formation timescale in years for constant SFR mode.
                                                   If None, burst mode is assumed.

        Returns:
            None. Writes the spectral table (and its sidecar) into CLOUDY_DATA_DIR.
        """
        # Generate the filename
        filename = self._generate_output_filename(formation_timescale)
        full_path = os.path.join(CLOUDY_DATA_DIR, filename)
        key = self._content_key(formation_timescale)

        if os.path.exists(full_path):
            if self._table_is_current(full_path, key, time_points, Z_values):
                print(f"Spectral table '{filename}' is current. Reusing it.")
                return
            self._move_aside(full_path)

        spectra = []
        for Z in Z_values:
            Z_spectra = []
            for t in time_points:
                if formation_timescale is None:
                    spectrum = self._get_burst_spectrum(t, Z)
                else:
                    spectrum = self._get_constant_sfr_spectrum(t, Z, formation_timescale)
                Z_spectra.append(spectrum)
            spectra.append(Z_spectra)

        self._write_spectral_table(time_points, Z_values, self.burst_data['wavelengths'], spectra,
                                   formation_timescale,
                                   sidecar={'origin': 'generated', 'key': key,
                                            'paired_feedback_md5': self._paired_feedback_md5(),
                                            'units': self._sidecar_units()})
        print(f"Spectral table '{filename}' has been generated and saved.")
        return