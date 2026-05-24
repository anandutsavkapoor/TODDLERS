from .imports import os, np, quad, brentq, Dict, Optional, warnings
from .constants import *

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
        Calculate the ratio of mass in small grains to total grain mass.

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
            target_ratio (float): Desired ratio of mass in small grains to total grain mass.
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

        Args:
            small_to_large_ratio (float): Desired ratio of mass in small grains (< 0.01 micron) to large grains.
            n (int): Power of the exponential cutoff (determines exp1, exp2, or exp3).

        Returns:
            str: Name of the created .szd file.
        """
        dist = self.create_modified_orion_distribution(small_to_large_ratio, n)
        
        ratio_str = f"{int(small_to_large_ratio * 1000):03d}"
        filename = f"ratio{ratio_str}_exp{n}.szd"
        file_path = os.path.join(self.data_dir, filename)
        
        with open(file_path, 'w') as f:
            f.write("# Size distribution file for modified Orion grains\n")
            f.write("# Based on user-defined small to large grain ratio\n")
            f.write("# Generated using TODDLERS CloudyGrainsGenerator\n")
            f.write("2010403 # magic number for version control\n")
            f.write(f"{dist['dist_type']}  # distribution type\n")
            f.write(f"{dist['a_l']:.6e}  # lower cutoff radius (micron)\n")
            f.write(f"{dist['a_u']:.6e}  # upper cutoff radius (micron)\n")
            f.write(f"{dist['alpha']:.6e}  # power law index\n")
            f.write(f"{dist['beta']:.6e}  # curvature parameter (micron^-1)\n")
            f.write(f"{dist['sigma_l']:.6e}  # lower cutoff width (micron)\n")
            f.write(f"{dist['sigma_u']:.6e}  # upper cutoff width (micron)\n")
            f.write(f"{dist['a0']:.6e}  # minimum grain radius (micron)\n")
            f.write(f"{dist['a1']:.6e}  # maximum grain radius (micron)\n")
        
        return filename

    def compile_grains(self, szd_file: str, material: str, n_bins: int = 10) -> str:
        """
        Generate Cloudy input file for grain compilation and run Cloudy.

        Args:
            szd_file (str): Name of the .szd file to use.
            material (str): Grain material (e.g., 'silicate', 'graphite').
            n_bins (int): Number of size bins for the distribution.

        Returns:
            str: Name of the compiled .opc file (or existing file if already present).
        """
        # Construct the expected .opc filename based on Cloudy's naming convention
        opc_file = f"{material}_{szd_file[:-4]}_{n_bins}.opc"
        opc_path = os.path.join(self.data_dir, opc_file)
        
        # Check if the .opc file already exists
        if os.path.exists(opc_path):
            return opc_file
        
        # If not, proceed with compilation
        input_file = f"compile_{material}_{szd_file[:-4]}.in"
        input_path = os.path.join(self.data_dir, input_file)
        
        with open(input_path, 'w') as f:
            f.write(f"compile grain \"{material + '.rfi'}\" \"{szd_file}\" {n_bins}\n")
        
        # Change to the Cloudy data directory before running Cloudy
        current_dir = os.getcwd()
        os.chdir(self.data_dir)
        
        self.run_cloudy(input_file)
        
        # Change back to the original directory
        os.chdir(current_dir)
        
        return opc_file

    def run_cloudy(self, input_file: str) -> None:
        """
        Run Cloudy with the given input file.

        Args:
            input_file (str): Path to the Cloudy input file.
        """
        os.system(f"{self.cloudy_path} {input_file}")

    def generate_grain_commands(self, metallicity_scaling: float, silicate_ratio: float, 
                            graphite_ratio: float, pah_ratio: Optional[float] = None,
                            n: int = 3, n_bins: int = 10, skip_pah: bool = False) -> str:
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
            if ratio < 0.05:
                # Use Orion distribution for ratios below 0.05
                return f"{material}_orion_10.opc"
            else:
                szd_file = self.create_size_distribution_file(ratio, n)
                return self.compile_grains(szd_file, material, n_bins)

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