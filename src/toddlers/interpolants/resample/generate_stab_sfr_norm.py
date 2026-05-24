import numpy as np
from pathlib import Path
import pts.storedtable as stab
import logging
from enum import Enum
from typing import Tuple, Dict, Optional, Set
from names_and_constants import *
import re

class SEDType(Enum):
    DUST = "Dust"
    NODUST = "noDust"

class Resolution(Enum):
    LOW = "lr"
    HIGH = "hr"

class TODDLERSStabGenerator:
    """Generates STAB files for SKIRT from TODDLERS SED data.

    The STAB axes are:

    - ``lambda``: wavelength (m, log-scaled)
    - ``Z``: metallicity (dimensionless, log-scaled)
    - ``SFE``: star formation efficiency (dimensionless, linear)
    - ``n_cl``: cloud density (cm^-3, log-scaled)
    - ``DTM`` (optional, 5D only): dust-to-metal scaling factor relative to
      solar (dimensionless, log-scaled). This is a multiplicative factor
      applied to the grain abundance per hydrogen atom in the Cloudy input:
      ``grain_scaling = DTM * Z / Z_solar``. A value of 1.0 means full dust
      content as prescribed by the grain opacity files; 0.5 means half the
      dust. This quantity is Z-independent by construction and should not be
      confused with D/G divided by Z, which varies ~20% across metallicities
      due to helium dilution of the gas mass.
    """
    
    def __init__(self, out_base_dir: Path, sed_base_dir: Path):
        self.out_base_dir = Path(out_base_dir)
        self.sed_base_dir = Path(sed_base_dir)
        self.outdir_stab = Path("stab_output")
        self.outdir_stab.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Parameter space definition
        self.Z = METALLICITIES
        self.etaSF = STAR_FORMATION_EFFICIENCIES
        self.n_cl = CLOUD_DENSITIES
        
        # Fixed model parameters (current TODDLERS version)
        self.stellar_template = STELLAR_TEMPLATE
        self.imf = IMF_TYPE
        self.star_type = STAR_TYPE
        
    def get_sed_folder(self, sed_type: SEDType, resolution: Resolution) -> Path:
        """Get the appropriate SED output folder."""
        model_prefix = f"{self.stellar_template}_{self.imf}_{self.star_type}"
        self.logger.info(f"Reading from SED directory: {self.sed_base_dir} / {model_prefix}_sed_output_{sed_type.value}_{resolution.value}")
        return self.sed_base_dir / f"{model_prefix}_sed_output_{sed_type.value}_{resolution.value}"

    def parse_header_units(self, filename: Path) -> Tuple[str, str]:
        """
        Parse the header of the SED file to extract wavelength and SED units.
        """
        try:
            with open(filename, 'r') as f:
                header = f.readline().strip()
            wavelength_unit_match = re.search(r'Wavelength\[([^\]]+)\]', header)
            sed_unit_match = re.search(r'SED_per_SFR\[([^\]]+)\]', header)
            wavelength_unit = wavelength_unit_match.group(1) if wavelength_unit_match else "m"
            sed_unit = sed_unit_match.group(1) if sed_unit_match else "W/m/Msun/yr"
            return wavelength_unit, sed_unit
        except Exception as e:
            self.logger.error(f"Could not parse header units from {filename}: {e}")

    def load_sed_file(self, filename: Path) -> tuple:
        """
        Load wavelength and SED data from file, converting wavelength to meters based on header.
        """
        try:
            # Parse the header to determine units
            wavelength_unit, _ = self.parse_header_units(filename)
            data = np.loadtxt(filename, skiprows=1)
            wavelength = data[:, 0]
            sed_per_sfr = data[:, 1]  # W/m/Msun/yr
            # Convert wavelength to meters based on the unit specified in the header
            if wavelength_unit.lower() == "micron" or wavelength_unit.lower() == "microns":
                wavelength = wavelength * 1e-6  # microns to meters
            elif wavelength_unit.lower() == "nm" or wavelength_unit.lower() == "nanometer" or wavelength_unit.lower() == "nanometers":
                wavelength = wavelength * 1e-9  # nanometers to meters
            elif wavelength_unit.lower() == "angstrom" or wavelength_unit.lower() == "angstroms":
                wavelength = wavelength * 1e-10  # angstroms to meters
            # If already in meters, no conversion needed
            self.logger.debug(f"Loaded {filename} with wavelength unit: {wavelength_unit}")
            self.logger.debug(f"Wavelength range: {wavelength.min()} to {wavelength.max()} meters")
            return wavelength, sed_per_sfr    
        except Exception as e:
            self.logger.error(f"Error loading SED file {filename}: {e}")
            raise
            
    def get_sed_filename(self, folder: Path, Z: float, eta: float, n: float, dtm: float = None) -> Path:
        """Construct SED filename based on parameters."""
        name = f"sed_sfr_scaled_{self.stellar_template}_{self.imf}_{self.star_type}_Z_{Z:.3f}_eta_{eta:.3f}_n_{n:.1f}"
        if dtm is not None and dtm != 1.0:
            name += f"_dtm{dtm:.2f}"
        return folder / f"{name}.txt"

    def get_stab_filename(self, sed_type: SEDType, resolution: Resolution, has_dtm: bool = False) -> Path:
        """Construct STAB filename based on parameters."""
        model_str = f"ToddlersSFRNormalizedSEDFamily_{self.stellar_template}_{self.imf}_{self.star_type}"
        if sed_type == SEDType.NODUST:
            model_str += "_noDust"
        if has_dtm:
            model_str += "_DTM"
        return self.outdir_stab / f"{model_str}_{resolution.value}.stab"

    def discover_dtm_values(self, folder: Path) -> np.ndarray:
        """Discover available DTM values from SED filenames in a folder.

        Returns sorted array of DTM values found. Files without _dtm suffix
        are treated as DTM=1.0.
        """
        dtm_values = set()
        pattern = f"sed_sfr_scaled_{self.stellar_template}_{self.imf}_{self.star_type}_*.txt"
        for f in folder.glob(pattern):
            match = re.search(r'_dtm(\d+\.\d+)', f.stem)
            if match:
                dtm_values.add(float(match.group(1)))
            else:
                dtm_values.add(1.0)
        return np.sort(np.array(list(dtm_values)))
            
    def create_sfr_scaled_stabs(self,
                              sed_types: Optional[Set[SEDType]] = None,
                              resolutions: Optional[Set[Resolution]] = None,
                              parameters: Optional[Dict] = None):
        """
        Create selected STAB files for SKIRT.

        Automatically discovers DTM values from SED filenames. If multiple DTM
        values are found, creates a 5D STAB with a DTM axis. Otherwise creates
        the standard 4D STAB.

        Args:
            sed_types: Set of SEDType to process. If None, processes all types.
            resolutions: Set of Resolution to process. If None, processes all resolutions.
            parameters: Dict with optional parameter subsets:
                'Z': List of metallicities
                'etaSF': List of star formation efficiencies
                'n_cl': List of cloud densities
        """
        sed_types = sed_types or set(SEDType)
        resolutions = resolutions or set(Resolution)

        if parameters:
            if 'Z' in parameters:
                self.Z = np.array(parameters['Z'])
            if 'etaSF' in parameters:
                self.etaSF = np.array(parameters['etaSF'])
            if 'n_cl' in parameters:
                self.n_cl = np.array(parameters['n_cl'])

        try:
            for sed_type in sed_types:
                for resolution in resolutions:
                    sed_folder = self.get_sed_folder(sed_type, resolution)
                    if not sed_folder.exists():
                        self.logger.warning(f"Skipping {sed_type.value}_{resolution.value} - folder not found: {sed_folder}")
                        continue

                    self.logger.info(f"Processing {sed_type.value} {resolution.value} SEDs...")

                    # Discover DTM values from filenames
                    dtm_values = self.discover_dtm_values(sed_folder)
                    has_dtm = len(dtm_values) > 1
                    if has_dtm:
                        self.logger.info(f"Found DTM values: {dtm_values}")
                    else:
                        self.logger.info(f"Single DTM value: {dtm_values[0]} (no DTM axis)")

                    # Initialize using first available file
                    first_dtm = dtm_values[0] if has_dtm else None
                    first_file = self.get_sed_filename(sed_folder, self.Z[0], self.etaSF[0], self.n_cl[0], first_dtm)
                    if not first_file.exists():
                        self.logger.warning(f"Skipping {sed_type.value} {resolution.value} - no files found")
                        continue

                    wl_grid, _ = self.load_sed_file(first_file)

                    # Validate all files exist before proceeding
                    self._validate_completeness(sed_folder, dtm_values if has_dtm else None)

                    if has_dtm:
                        # 5D: (wavelength, Z, etaSF, n_cl, DTM)
                        L_sed = np.zeros((len(wl_grid), len(self.Z), len(self.etaSF), len(self.n_cl), len(dtm_values)))

                        for i, Z in enumerate(self.Z):
                            for j, eta in enumerate(self.etaSF):
                                for k, n in enumerate(self.n_cl):
                                    for l, dtm in enumerate(dtm_values):
                                        filename = self.get_sed_filename(sed_folder, Z, eta, n, dtm)
                                        if filename.exists():
                                            _, sed = self.load_sed_file(filename)
                                            L_sed[:, i, j, k, l] = sed
                                        else:
                                            self.logger.warning(f"Missing SED file: {filename}")

                        stab_file = self.get_stab_filename(sed_type, resolution, has_dtm=True)
                        self.logger.info(f"Creating 5D STAB file with DTM axis: {stab_file}")
                        self._log_grid_info(wl_grid, dtm_values, L_sed)
                        stab.writeStoredTable(
                            str(stab_file),
                            ['lambda', 'Z', 'SFE', 'n_cl', 'DTM'],
                            ['m', '1', '1', '1/cm3', '1'],
                            ['log', 'log', 'lin', 'log', 'log'],
                            [wl_grid, self.Z, self.etaSF, self.n_cl, dtm_values],
                            ['Llambda'],
                            ['W/m'],
                            ['log'],
                            [L_sed]
                        )
                    else:
                        # 4D: standard (wavelength, Z, etaSF, n_cl)
                        L_sed = np.zeros((len(wl_grid), len(self.Z), len(self.etaSF), len(self.n_cl)))

                        for i, Z in enumerate(self.Z):
                            for j, eta in enumerate(self.etaSF):
                                for k, n in enumerate(self.n_cl):
                                    filename = self.get_sed_filename(sed_folder, Z, eta, n)
                                    if filename.exists():
                                        _, sed = self.load_sed_file(filename)
                                        L_sed[:, i, j, k] = sed
                                    else:
                                        self.logger.warning(f"Missing SED file: {filename}")

                        stab_file = self.get_stab_filename(sed_type, resolution, has_dtm=False)
                        self.logger.info(f"Creating 4D STAB file: {stab_file}")
                        self._log_grid_info(wl_grid, None, L_sed)
                        stab.writeStoredTable(
                            str(stab_file),
                            ['lambda', 'Z', 'SFE', 'n_cl'],
                            ['m', '1', '1', '1/cm3'],
                            ['log', 'log', 'lin', 'log'],
                            [wl_grid, self.Z, self.etaSF, self.n_cl],
                            ['Llambda'],
                            ['W/m'],
                            ['log'],
                            [L_sed]
                        )

            self.logger.info("Requested STAB files created successfully")

        except Exception as e:
            self.logger.error(f"Error creating STAB files: {e}")
            raise

    def _validate_completeness(self, sed_folder, dtm_values=None):
        """Check that all (Z, eta, n, DTM) combinations have SED files.

        Raises:
            ValueError: If any files are missing, listing all missing combinations.
        """
        missing = []
        if dtm_values is not None and len(dtm_values) > 1:
            for Z in self.Z:
                for eta in self.etaSF:
                    for n in self.n_cl:
                        for dtm in dtm_values:
                            f = self.get_sed_filename(sed_folder, Z, eta, n, dtm)
                            if not f.exists():
                                missing.append(f"Z={Z}, eta={eta}, n={n}, dtm={dtm}")
        else:
            for Z in self.Z:
                for eta in self.etaSF:
                    for n in self.n_cl:
                        f = self.get_sed_filename(sed_folder, Z, eta, n)
                        if not f.exists():
                            missing.append(f"Z={Z}, eta={eta}, n={n}")

        if missing:
            msg = f"Missing {len(missing)} SED files:\n" + "\n".join(missing[:20])
            if len(missing) > 20:
                msg += f"\n... and {len(missing) - 20} more"
            raise ValueError(msg)

        total = len(self.Z) * len(self.etaSF) * len(self.n_cl)
        if dtm_values is not None and len(dtm_values) > 1:
            total *= len(dtm_values)
        self.logger.info(f"Validated: all {total} SED files present")

    def _log_grid_info(self, wl_grid, dtm_values, L_sed):
        """Log grid information for diagnostics."""
        self.logger.info(f"min-max in w_grid: {np.amin(wl_grid)}, {np.amax(wl_grid)}")
        self.logger.info(f"min-max in Z_grid: {np.amin(self.Z)}, {np.amax(self.Z)}")
        self.logger.info(f"min-max in etaSF_grid: {np.amin(self.etaSF)}, {np.amax(self.etaSF)}")
        self.logger.info(f"min-max in n_cl_grid: {np.amin(self.n_cl)}, {np.amax(self.n_cl)}")
        if dtm_values is not None:
            self.logger.info(f"min-max in DTM_grid: {np.amin(dtm_values)}, {np.amax(dtm_values)}")
        self.logger.info(f"Number of NaN values in L_sed: {np.isnan(L_sed).sum()}")
        self.logger.info(f"10-90 ptile in L_sed: {np.percentile(L_sed, 10)}, {np.percentile(L_sed, 90)}")

if __name__ == "__main__":
    generator = TODDLERSStabGenerator(out_base_dir=Path(""), 
                                    sed_base_dir=Path(""))
        
    generator.create_sfr_scaled_stabs(
        sed_types={SEDType.NODUST},
        resolutions={Resolution.HIGH}
    )

    generator.create_sfr_scaled_stabs(
        sed_types={SEDType.DUST},
        resolutions={Resolution.HIGH}
    )

    generator.create_sfr_scaled_stabs(
        sed_types={SEDType.DUST},
        resolutions={Resolution.LOW}
    )

    generator.create_sfr_scaled_stabs(
        sed_types={SEDType.NODUST},
        resolutions={Resolution.LOW}
    )
    