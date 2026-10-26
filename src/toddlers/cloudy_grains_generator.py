import hashlib
import json
import shlex
import shutil
import subprocess
import time

from ._fileutils import DirLock, unique_tmp_path
from .imports import os, np, quad, brentq, Dict, Optional, warnings
from .constants import *

# Below this small-to-large ratio Cloudy's own Orion grains are used (with a warning).
ORION_FALLBACK_RATIO = 0.05


def _ratio_tag(ratio: float) -> str:
    """The 3-digit file-name tag of a small-to-large ratio (0.10 -> '100').

    Names resolve the ratio to 1e-3. A ratio off that grid would share a name (and so a
    .szd and .opc) with its neighbour, so it is refused instead of truncated.
    """
    x = float(ratio) * 1000.0
    k = int(round(x))
    if abs(x - k) > 1e-6:
        raise ValueError(
            f"small_to_large_ratio={ratio!r} is not a multiple of 0.001; grain file names "
            "resolve the ratio to 1e-3, so this value would share files with "
            f"{k / 1000.0!r}. Use a ratio with at most 3 decimals.")
    return f"{k:03d}"


# Version of the grain sidecar layout (<opc>.meta.json).
GRAIN_SIDECAR_SCHEMA_VERSION = 1


def _file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _md5_or_none(path: str) -> Optional[str]:
    return _file_md5(path) if os.path.isfile(path) else None


_EXE_MD5_CACHE: Dict[tuple, str] = {}


def _cloudy_exe_md5(cloudy_path: str) -> Optional[str]:
    """md5 of the Cloudy executable (identifies the build); None if it cannot be found.

    ``cloudy_path`` may be a bare name on PATH or carry arguments (first token is the
    executable). Cached per (path, size, mtime) so parallel input writing hashes once.
    """
    first = shlex.split(cloudy_path)[0] if cloudy_path and cloudy_path.strip() else ''
    exe = first if os.path.isabs(first) else (shutil.which(first) or first)
    if not exe or not os.path.isfile(exe):
        return None
    st = os.stat(exe)
    k = (os.path.abspath(exe), st.st_size, st.st_mtime_ns)
    if k not in _EXE_MD5_CACHE:
        _EXE_MD5_CACHE[k] = _file_md5(exe)
    return _EXE_MD5_CACHE[k]


def _opc_version_line(opc_path: str) -> Optional[str]:
    """First header line of an .opc ('# this file was created by Cloudy ...')."""
    try:
        with open(opc_path, errors='replace') as f:
            return f.readline().rstrip('\n')
    except OSError:
        return None


def _opc_is_current(opc_path: str, key: dict) -> bool:
    """True if the .opc exists, its sidecar is readable, the sidecar md5 matches the file
    and the sidecar key equals ``key``."""
    meta_path = opc_path + '.meta.json'
    if not (os.path.isfile(opc_path) and os.path.isfile(meta_path)):
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return False
    return meta.get('md5') == _file_md5(opc_path) and meta.get('key') == key


def _write_opc_sidecar(opc_path: str, key: dict) -> None:
    from datetime import datetime, timezone
    meta = {
        'filename': os.path.basename(opc_path),
        'md5': _file_md5(opc_path),
        'sidecar_schema_version': GRAIN_SIDECAR_SCHEMA_VERSION,
        'key': key,
        'cloudy_version_line': _opc_version_line(opc_path),
        'compiled_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    _write_text_if_changed(opc_path + '.meta.json', json.dumps(meta, indent=2) + '\n')


def _move_aside(path: str, data_dir: str) -> None:
    """Move a file and its sidecar to <data_dir>/stale_tables/<UTC time>/; never delete."""
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    dest_dir = os.path.join(data_dir, 'stale_tables', stamp)
    os.makedirs(dest_dir, exist_ok=True)
    for p in (path, path + '.meta.json'):
        if os.path.exists(p):
            os.replace(p, os.path.join(dest_dir, os.path.basename(p)))
    print(f"  Moved the outdated '{os.path.basename(path)}' to {dest_dir}")


class _DirLock(DirLock):
    """The grain-compile lock (toddlers._fileutils.DirLock: owner-stamped, abandoned locks
    of dead processes on this host are broken)."""

    def __init__(self, path, timeout=3600.0, poll=0.2):
        super().__init__(path, timeout=timeout, poll=poll, what='grain-compile lock')


def _write_text_if_changed(path: str, text: str) -> bool:
    """Write ``text`` to ``path`` unless the file already holds exactly that text.

    The write goes to a temporary file in the same directory, renamed into place, so a
    reader (a parallel worker, a running Cloudy) never sees a partial file. Returns True
    if the file was written.
    """
    try:
        with open(path) as f:
            if f.read() == text:
                return False
    except (OSError, UnicodeDecodeError):
        pass
    tmp_path = unique_tmp_path(path)
    with open(tmp_path, 'w') as f:
        f.write(text)
    os.replace(tmp_path, path)
    return True


class CloudyGrainsGenerator:
    def __init__(self, cloudy_path: str, data_dir: str):
        """
        Initialize the CloudyGrainsGenerator.

        Args:
            cloudy_path (str): Path to the Cloudy executable.
            data_dir (str): Directory containing grain data files.
        """
        self.cloudy_path = cloudy_path
        self.data_dir = data_dir
        # seconds a worker waits for another worker's compile of the same .opc
        self.lock_timeout = 3600.0
        # ratios below ORION_FALLBACK_RATIO that were replaced by Cloudy's Orion grains
        self.orion_fallbacks = []

    def create_modified_orion_distribution(self, small_to_large_ratio: float, n: int = 3, a1: float = 0.25) -> Dict[str, float]:
        """
        Create a modified Orion-like distribution with exponential cutoff.

        Args:
            small_to_large_ratio (float): Desired ratio of mass in small grains (< 0.03 micron) to large grains.
            n (int): Power of the exponential cutoff (determines exp1, exp2, or exp3).
            a1 (float): Upper grain size limit in microns.

        Returns:
            Dict[str, float]: Parameters defining the modified Orion-like distribution.

        Raises:
            ValueError: If small_to_large_ratio is greater than 0.43.
        """

        if small_to_large_ratio > MAX_RATIO:
            raise ValueError(f"small_to_large_ratio must be less than or equal to {MAX_RATIO}. "
                            f"Values above {MAX_RATIO} lead to errors in root finding.")

        if small_to_large_ratio > ISM_LIKE_RATIO:
            warnings.warn(f"small_to_large_ratio is close to the maximum allowed value. "
                        f"The distribution may be unstable.")

        if abs(small_to_large_ratio - ISM_LIKE_RATIO) < 0.05:
            print(f"Note: A ratio of {small_to_large_ratio:.2f} produces a distribution similar to "
                f"the ISM distribution in Cloudy.")
        elif small_to_large_ratio <= ORION_LIKE_RATIO:
            print(f"Note: A ratio of {small_to_large_ratio:.2f} produces a distribution similar to "
                f"the default Orion distribution in Cloudy.")

        a0 = A_0_GRAINS  # Lower grain size limit
        a_l = A_L_GRAINS  # Cutoff for exponential function
        sigma_l = self.infer_sigma(small_to_large_ratio, a0, a1, a_l, n)
        
        dist_type = f"exp{min(n, 3)}"  # exp1, exp2, or exp3 based on n
        
        return {
            "dist_type": dist_type,
            "a_l": a_l,        # lower cutoff radius (micron)
            "a_u": a1,         # upper cutoff radius (micron)
            "alpha": -3.5,     # power law index
            "beta": 0.0,       # curvature parameter (micron^-1)
            "sigma_l": sigma_l,  # lower cutoff width (micron)
            "sigma_u": 0.0,    # upper cutoff width (micron)
            "a0": a0,          # minimum grain radius (micron)
            "a1": a1           # maximum grain radius (micron)
        }

    def power_law(self, a: float, alpha: float = -3.5) -> float:
        """
        Compute the power law function for grain size distribution.

        Args:
            a (float): Grain size in microns.
            alpha (float): Power law index.

        Returns:
            float: Value of the power law function.
        """
        return (4 * np.pi / 3) * (a**3) * (a**alpha)

    def truncation(self, a: float, sigma: float, n: int, a_l: float) -> float:
        """
        Compute the truncation function for grain size distribution.

        Args:
            a (float): Grain size in microns.
            sigma (float): Width of the exponential cutoff.
            n (int): Power of the exponential cutoff.
            a_l (float): Lower cutoff radius.

        Returns:
            float: Value of the truncation function.
        """
        return np.exp(-((a_l - a) / sigma)**n) if a < a_l else 1

    def calculate_mass_ratio(self, sigma: float, a0: float, a1: float, a_l: float, n: int, small_grain_threshold: float = A_L_GRAINS) -> float:
        """
        Calculate the ratio of mass in small grains to mass in large grains.

        Args:
            sigma (float): Width of the exponential cutoff.
            a0 (float): Lower grain size limit in microns.
            a1 (float): Upper grain size limit in microns.
            a_l (float): Lower cutoff radius.
            n (int): Power of the exponential cutoff.
            small_grain_threshold (float): Size threshold for small grains in microns.

        Returns:
            float: Ratio of mass in small grains to mass in large grains.
        """
        def integrand(a):
            return self.power_law(a) * self.truncation(a, sigma, n, a_l)

        mass_small, _ = quad(integrand, a0, small_grain_threshold)
        mass_large, _ = quad(integrand, small_grain_threshold, a1)

        return mass_small / mass_large

    def infer_sigma(self, target_ratio: float, a0: float, a1: float, a_l: float, n: int) -> float:
        """
        Infer sigma based on the desired small to large grain ratio.

        Args:
            target_ratio (float): Desired ratio of mass in small grains to large grains
                (small: below the 0.03 micron threshold; large: above it).
            a0 (float): Lower grain size limit in microns.
            a1 (float): Upper grain size limit in microns.
            a_l (float): Lower cutoff radius.
            n (int): Power of the exponential cutoff.

        Returns:
            float: Inferred sigma value.
        """
        def ratio_difference(sigma):
            return self.calculate_mass_ratio(sigma, a0, a1, a_l, n) - target_ratio

        # Use brentq to find the root of the ratio difference function
        sigma = brentq(ratio_difference, 1e-6, 1.0)
        return sigma

    def create_size_distribution_file(self, small_to_large_ratio: float, n: int = 3) -> str:
        """
        Create a size distribution (.szd) file with the given parameters.

        A file that already holds the same content is left untouched; otherwise the
        file is written to a temporary name and renamed into place (parallel workers call
        this for every Cloudy input they write).

        Args:
            small_to_large_ratio (float): Desired ratio of mass in small grains (< 0.03 micron) to large grains.
            n (int): Power of the exponential cutoff (determines exp1, exp2, or exp3).

        Returns:
            str: Name of the created .szd file.
        """
        dist = self.create_modified_orion_distribution(small_to_large_ratio, n)
        
        ratio_str = _ratio_tag(small_to_large_ratio)
        filename = f"ratio{ratio_str}_exp{n}.szd"
        file_path = os.path.join(self.data_dir, filename)
        
        lines = [
            "# Size distribution file for modified Orion grains\n",
            "# Based on user-defined small to large grain ratio\n",
            "# Generated using TODDLERS CloudyGrainsGenerator\n",
            "2010403 # magic number for version control\n",
            f"{dist['dist_type']}  # distribution type\n",
            f"{dist['a_l']:.6e}  # lower cutoff radius (micron)\n",
            f"{dist['a_u']:.6e}  # upper cutoff radius (micron)\n",
            f"{dist['alpha']:.6e}  # power law index\n",
            f"{dist['beta']:.6e}  # curvature parameter (micron^-1)\n",
            f"{dist['sigma_l']:.6e}  # lower cutoff width (micron)\n",
            f"{dist['sigma_u']:.6e}  # upper cutoff width (micron)\n",
            f"{dist['a0']:.6e}  # minimum grain radius (micron)\n",
            f"{dist['a1']:.6e}  # maximum grain radius (micron)\n",
        ]
        _write_text_if_changed(file_path, "".join(lines))

        return filename

    def compile_grains(self, szd_file: str, material: str, n_bins: int = 10,
                       small_to_large_ratio: Optional[float] = None) -> str:
        """
        Compile a grain opacity file (.opc) with Cloudy, or reuse a current one.

        An existing .opc is reused only when its sidecar ``<opc>.meta.json`` matches: the
        sidecar md5 equals the file's, and its key (material, szd and rfi md5, n_bins,
        ratio, md5 of the Cloudy executable) equals this request's. Anything else (no
        sidecar, an edited file, a changed recipe or Cloudy build) is moved to
        ``<data_dir>/stale_tables/<UTC time>/`` (never deleted) and compiled again. The
        check and the compile run under a lock directory ``<opc>.lock``, so parallel
        workers compile a given file once.

        Args:
            szd_file (str): Name of the .szd file to use (in the data directory).
            material (str): Grain material (e.g., 'silicate', 'graphite'); reads <material>.rfi.
            n_bins (int): Number of size bins for the distribution.
            small_to_large_ratio (float, optional): the ratio the .szd was built for,
                recorded in the sidecar key at full precision.

        Returns:
            str: Name of the compiled .opc file.
        """
        # Grain compilation runs Cloudy in its data directory. Fail early and clearly
        # if that directory is unset/missing (the usual cause: Cloudy is not installed
        # on this machine), rather than dying later with a cryptic error.
        if not self.data_dir or not os.path.isdir(self.data_dir):
            raise RuntimeError(
                f"Cloudy data directory not found (CLOUDY_DATA_DIR={self.data_dir!r}). "
                "Compiling grains requires a working Cloudy installation: install Cloudy "
                "and set CLOUDY_EXE (and/or CLOUDY_DATA_DIR) to point at it. "
                "TODDLERS' dynamics-only API runs without Cloudy."
            )

        # Construct the expected .opc filename based on Cloudy's naming convention
        opc_file = f"{material}_{szd_file[:-4]}_{n_bins}.opc"
        opc_path = os.path.join(self.data_dir, opc_file)
        key = self._opc_key(szd_file, material, n_bins, small_to_large_ratio)

        with _DirLock(opc_path + '.lock', timeout=self.lock_timeout):
            if _opc_is_current(opc_path, key):
                return opc_file
            if os.path.exists(opc_path) or os.path.exists(opc_path + '.meta.json'):
                _move_aside(opc_path, self.data_dir)

            # n_bins in the name: the lock is per .opc, so compiles of other bin counts
            # of this material and .szd may run at the same time
            input_file = f"compile_{material}_{szd_file[:-4]}_{n_bins}.in"
            input_path = os.path.join(self.data_dir, input_file)
            with open(input_path, 'w') as f:
                f.write(f"compile grain \"{material + '.rfi'}\" \"{szd_file}\" {n_bins}\n")

            # Run Cloudy in its data directory, as the child's working directory only: the
            # process cwd is never changed (the local runner's threads share it).
            returncode = self.run_cloudy(input_file, cwd=self.data_dir)

            if returncode not in (0, None):
                raise RuntimeError(
                    f"Cloudy grain compile failed (exit status {returncode}): "
                    f"'{self.cloudy_path} {input_file}' in {self.data_dir}")
            if not os.path.isfile(opc_path):
                raise RuntimeError(
                    f"Cloudy grain compile wrote no {opc_file} in {self.data_dir} "
                    f"('{self.cloudy_path} {input_file}')")
            _write_opc_sidecar(opc_path, key)
        return opc_file

    def _opc_key(self, szd_file: str, material: str, n_bins: int,
                 small_to_large_ratio: Optional[float]) -> dict:
        """The content key of a compiled grain file (see :meth:`compile_grains`)."""
        rfi_file = material + '.rfi'
        return {
            'material': material,
            'szd_file': szd_file,
            'szd_md5': _md5_or_none(os.path.join(self.data_dir, szd_file)),
            'rfi_file': rfi_file,
            'rfi_md5': _md5_or_none(os.path.join(self.data_dir, rfi_file)),
            'n_bins': int(n_bins),
            'small_to_large_ratio': (None if small_to_large_ratio is None
                                     else float(small_to_large_ratio)),
            'cloudy_exe_md5': _cloudy_exe_md5(self.cloudy_path),
        }

    def run_cloudy(self, input_file: str, cwd: Optional[str] = None) -> int:
        """
        Run Cloudy with the given input file.

        Args:
            input_file (str): Path to the Cloudy input file (relative to ``cwd``).
            cwd (str, optional): working directory of the Cloudy process; the current
                directory when None. The calling process's cwd is not changed.

        Returns:
            int: Cloudy's exit status (0 on success).
        """
        return subprocess.run(f"{self.cloudy_path} {input_file}", shell=True, cwd=cwd).returncode

    def generate_grain_commands(self, metallicity_scaling: float, silicate_ratio: float, 
                            graphite_ratio: float, pah_ratio: Optional[float] = None,
                            n: int = 3, n_bins: int = 10, skip_pah: bool = False) -> list:
        """
        Generate Cloudy commands for grains in the main input file.

        Args:
            metallicity_scaling (float): Metallicity scaling factor.
            silicate_ratio (float): Small-to-large grain mass ratio for silicates.
            graphite_ratio (float): Small-to-large grain mass ratio for graphites.
            pah_ratio (Optional[float]): Small-to-large grain mass ratio for PAHs (not tested). 
                                    If None, use default PAH.
            n (int): Power of the exponential cutoff (determines exp1, exp2, or exp3).
            n_bins (int): Number of size bins for the distribution.
            skip_pah (bool): If True, PAHs will not be included in the grain commands.

        Returns:
            list: Cloudy-compatible grain commands.
        """
        commands = []
        
        def process_grain(ratio: float, material: str) -> str:
            if ratio < ORION_FALLBACK_RATIO:
                # Below 0.05 the modified-Orion recipe is not used: Cloudy's own Orion
                # grains stand in, with a warning and a record in orion_fallbacks.
                opc = f"{material}_orion_10.opc"
                record = {'material': material, 'small_to_large_ratio': float(ratio), 'opc': opc}
                self.orion_fallbacks.append(record)
                warnings.warn(
                    f"small_to_large_ratio={ratio!r} < {ORION_FALLBACK_RATIO} for {material}: "
                    f"using Cloudy's Orion grains ({opc}) instead of a modified-Orion "
                    "size distribution.")
                commands.append(f"# TODDLERS: {material} small_to_large_ratio {ratio!r} < "
                                f"{ORION_FALLBACK_RATIO}, Cloudy Orion grains {opc} used")
                return opc
            else:
                szd_file = self.create_size_distribution_file(ratio, n)
                return self.compile_grains(szd_file, material, n_bins,
                                           small_to_large_ratio=ratio)

        # Compile and add silicate grains
        silicate_file = process_grain(silicate_ratio, "silicate")
        commands.append(f'grain "{silicate_file}" {np.log10(metallicity_scaling)} log function sublimation')

        # Compile and add graphite grains
        graphite_file = process_grain(graphite_ratio, "graphite")
        commands.append(f'grain "{graphite_file}" {np.log10(metallicity_scaling)} log function sublimation')
        
        # Add PAH if not skipped
        if not skip_pah:
            if pah_ratio is not None:
                pah_file = process_grain(pah_ratio, "pah")
                commands.append(f'grain "{pah_file}" {np.log10(2.125*Q_PAH*metallicity_scaling)} log')
            else:
                commands.append(f"grains PAH function {2.125*Q_PAH*metallicity_scaling}")

        return commands