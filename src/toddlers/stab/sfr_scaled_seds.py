# sed_generator.py
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import logging
from pathlib import Path
from tqdm import tqdm
import psutil
import pickle
from astropy import units as u
from .sfr_scaling import SEDmanipulator, SimulationParameters
from .config import *
from ..utils import dtm_label

class SEDGenerator:
    """Generate SFR-scaled SEDs across the TODDLERS parameter space.

    Args:
        output_dir: Directory for SED output files.
        max_workers: Maximum parallel workers. Auto-determined from memory if None.
        dust_to_metal_values: List of DTM scaling factors for a dust sweep. Each
            value is a multiplicative factor applied to the grain abundance per
            hydrogen atom (1.0 = full dust, 0.5 = half). This is Z-independent
            and should not be confused with D/G divided by Z.
        interpolator_file: Path to the SED interpolator file.
    """
    def __init__(self, output_dir=SED_OUTPUT_DIR, max_workers=None, dust_to_metal_values=None,
                 interpolator_file=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dust_to_metal_values = dust_to_metal_values
        self.interpolator_file = interpolator_file or SED_INTERPOLATOR_FILE
            
        # Determine resolution from output directory
        is_hr = 'hr' in str(output_dir)
        
        # Memory-based worker calculation
        # Use 40GB for high resolution, 19GB for low resolution
        base_memory = 40e9 if is_hr else 19e9
        memory_per_process = base_memory * (SAMPLE_SIZE_TIME / 1000)  # Estimate per process
        available_memory = psutil.virtual_memory().available
        suggested_workers = max(1, int(available_memory / memory_per_process))
        self.max_workers = min(suggested_workers, os.cpu_count()) if max_workers is None else max_workers
        
        self.setup_logging()
        self.log_system_info()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'sed_generation_{MODEL_PREFIX}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_system_info(self):
        """Log system information and configuration parameters."""
        self.logger.info(
            f"Initializing SED Generator\n"
            f"System Configuration:\n"
            f"- Available Memory: {psutil.virtual_memory().available / 1e9:.2f} GB\n"
            f"- CPU Cores: {os.cpu_count()}\n"
            f"- Workers: {self.max_workers}\n"
            f"Input Parameters:\n"
            f"- SED Interpolator File: {self.interpolator_file}\n"
            f"- Output Directory: {self.output_dir}\n"
            f"- Sample Size: {SAMPLE_SIZE_TIME}\n"
            f"Parameter Ranges:\n"
            f"- Metallicities: {METALLICITIES}\n"
            f"- Star Formation Efficiencies: {STAR_FORMATION_EFFICIENCIES}\n"
            f"- Cloud Densities: {CLOUD_DENSITIES}"
        )

    def _get_sed_filename(self, Z, eta, n_cl, dtm=None):
        """Construct SED output filename, optionally including DTM."""
        name = f"sed_sfr_scaled_{MODEL_PREFIX}_Z_{Z:.3f}_eta_{eta:.3f}_n_{n_cl:.1f}"
        if dtm is not None:
            name += dtm_label(dtm)
        return self.output_dir / f"{name}.txt"

    def _get_interpolator_file(self, dtm=None):
        """Get interpolator file path, with DTM suffix if needed."""
        if dtm is None or dtm == 1.0:
            return self.interpolator_file
        base, ext = os.path.splitext(self.interpolator_file)
        return f"{base}{dtm_label(dtm)}{ext}"

    def process_parameter_combination(self, params):
        """Process a single parameter combination."""
        if len(params) == 4:
            Z, eta, n_cl, dtm = params
        else:
            Z, eta, n_cl = params
            dtm = None

        try:
            output_file = self._get_sed_filename(Z, eta, n_cl, dtm)

            if output_file.exists():
                self.logger.info(f"Skipping existing file: {output_file}")
                return True

            # Load interpolator in each process to avoid memory sharing issues
            interp_file = self._get_interpolator_file(dtm)
            with open(interp_file, 'rb') as f:
                sed_interpolator = pickle.load(f)

            # Create simulation parameters
            sim_params = SimulationParameters(
                Z=Z * u.dimensionless_unscaled,
                eta=eta,
                n=n_cl * u.cm**-3
            )

            # Initialize SEDmanipulator
            manip = SEDmanipulator(
                sed_interpolator=sed_interpolator,
                recollapse_sim_dir=RECOLLAPSE_SIM_DIR,
                wavelength_unit=WAVELENGTH_UNIT,
                sed_unit=SED_UNIT
            )

            # Compute SED
            _, _, sed_per_sfr = manip.compute_sfr_scaled_sed(sim_params, load_sim=True)

            # Ensure wavelength is in microns for saving
            wavelength_microns = manip.wavelength_grid.to(u.micron).value

            # Save wavelength grid and SED values
            save_data = np.column_stack((wavelength_microns, sed_per_sfr.value))
            np.savetxt(output_file, save_data, fmt='%.18e',
                      header='Wavelength[microns] SED_per_SFR[W/m/Msun/yr]')

            dtm_str = f", dtm={dtm}" if dtm is not None else ""
            self.logger.info(f"Successfully generated SED for Z={Z}, eta={eta}, n_cl={n_cl}{dtm_str}")
            return True

        except Exception as e:
            dtm_str = f", dtm={dtm}" if dtm is not None else ""
            self.logger.error(f"Error processing Z={Z}, eta={eta}, n_cl={n_cl}{dtm_str}: {str(e)}")
            return False

    def generate_all_seds(self):
        """Generate SEDs for all parameter combinations, including DTM if specified."""
        if self.dust_to_metal_values is not None:
            parameter_combinations = [
                (Z, eta, n_cl, dtm)
                for Z in METALLICITIES
                for eta in STAR_FORMATION_EFFICIENCIES
                for n_cl in CLOUD_DENSITIES
                for dtm in self.dust_to_metal_values
            ]
        else:
            parameter_combinations = [
                (Z, eta, n_cl)
                for Z in METALLICITIES
                for eta in STAR_FORMATION_EFFICIENCIES
                for n_cl in CLOUD_DENSITIES
            ]

        total_combinations = len(parameter_combinations)
        self.logger.info(f"Processing {total_combinations} parameter combinations using {self.max_workers} workers")

        failures = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(tqdm(
                executor.map(self.process_parameter_combination, parameter_combinations),
                total=total_combinations,
                desc="Generating SEDs"
            ))

            # Track failures
            for params, success in zip(parameter_combinations, results):
                if not success:
                    failures.append(params)

        if failures:
            self.logger.error(f"Failed parameter combinations: {failures}")
        else:
            self.logger.info("All SEDs generated successfully")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate SFR-scaled SEDs')
    parser.add_argument('--dust-to-metal', type=float, nargs='+', default=None,
                      help='DTM value(s) for sweep. If omitted, no DTM axis.')
    parser.add_argument('--sed-type', type=str, choices=['Dust', 'noDust'], default=None,
                      help='SED type. Overrides names_and_constants.py if given.')
    parser.add_argument('--resolution', type=str, choices=['lr', 'hr'], default=None,
                      help='Resolution. Overrides names_and_constants.py if given.')
    parser.add_argument('--max-workers', type=int, default=None,
                      help='Max parallel workers (default: auto)')
    args = parser.parse_args()

    # Determine output dir and interpolator from CLI args or names_and_constants.py
    if args.sed_type is not None and args.resolution is not None:
        output_dir = f"{MODEL_PREFIX}_sed_output_{args.sed_type}_{args.resolution}"
        is_nodust = args.sed_type == 'noDust'
        is_hr = args.resolution == 'hr'
        interp_base = f"{MODEL_PREFIX}_interp_tables"
        if is_hr:
            interp_file = os.path.join(interp_base,
                f"TODDLERS_tot_hr_{MODEL_PREFIX}_lines_emergent={'False' if is_nodust else 'True'}.pkl")
        else:
            if is_nodust:
                interp_file = os.path.join(interp_base, f"TODDLERS_inciSED_lr_{MODEL_PREFIX}.pkl")
            else:
                interp_file = os.path.join(interp_base, f"TODDLERS_totSED_lr_{MODEL_PREFIX}.pkl")
    else:
        output_dir = SED_OUTPUT_DIR
        interp_file = SED_INTERPOLATOR_FILE

    generator = SEDGenerator(
        output_dir=output_dir,
        max_workers=args.max_workers,
        dust_to_metal_values=args.dust_to_metal,
        interpolator_file=interp_file
    )
    generator.generate_all_seds()