#!/usr/bin/env python3
"""
Runner script for TODDLERS interpolant generation.
"""

import sys
import traceback
import pickle
import argparse
from pathlib import Path
import numpy as np
from generate_interpolants import TODDLERSInterpolantGenerator

def save_interpolant(interpolant, filename):
    """Save interpolant to pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(interpolant, f)

def interpolant_exists(filepath):
    try:
        with open(filepath, 'rb') as f:
            pickle.load(f)
        return True
    except:
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate TODDLERS interpolants')
    parser.add_argument('--evolution-dir', type=str, required=True,
                      help='Directory containing evolution simulation .dat files')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Directory to save interpolants')
    parser.add_argument('--dust-to-metal', type=float, default=1.0,
                      help='Dust-to-metal ratio relative to solar (default: 1.0)')

    args = parser.parse_args()

    evolution_dir = Path(args.evolution_dir)
    output_dir = Path(args.output_dir)

    # Verify evolution directory exists and contains .dat files
    if not evolution_dir.exists():
        print(f"Error: Evolution directory {evolution_dir} does not exist")
        sys.exit(1)
        
    sim_files = list(evolution_dir.glob("*.dat"))
    if not sim_files:
        print(f"Error: No .dat files found in {evolution_dir}")
        sys.exit(1)
        
    # Derive model prefix from evolution directory structure
    evo_parts = evolution_dir.parts
    template = imf = star_type = None
    for part in evo_parts:
        if part.startswith('template_'): template = part[len('template_'):]
        elif part.startswith('imf_'): imf = part[len('imf_'):]
        elif part.startswith('star_type_'): star_type = part[len('star_type_'):]

    if not all([template, imf, star_type]):
        print("Warning: Could not extract template/imf/star_type from path, using generic names")
        model_prefix = "TODDLERS"
    else:
        model_prefix = f"{template}_{imf}_{star_type}"

    dtm = args.dust_to_metal
    dtm_suffix = f"_dtm{dtm:.2f}" if dtm != 1.0 else ""

    print(f"\nFound {len(sim_files)} simulation files in {evolution_dir}")
    print(f"Model prefix: {model_prefix}")
    print(f"Dust-to-metal ratio: {dtm}")
    print("\nInitializing interpolant generator...")
    generator = TODDLERSInterpolantGenerator(evolution_dir, dust_to_metal=dtm)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Generate and save low-res SED interpolants
        # Naming follows names_and_constants.py convention for SFR STAB pipeline
        print("\nGenerating low-resolution SED interpolants...")

        sed_low_dust = output_dir / f'TODDLERS_totSED_lr_{model_prefix}{dtm_suffix}.pkl'
        if not interpolant_exists(sed_low_dust):
            sed_interp = generator.generate_sed_interpolant(resolution='low', dust=True)
            save_interpolant(sed_interp, sed_low_dust)
        else:
            print("Low-res dust SED interpolant already exists, skipping...")

        sed_low_nodust = output_dir / f'TODDLERS_inciSED_lr_{model_prefix}{dtm_suffix}.pkl'
        if not interpolant_exists(sed_low_nodust):
            sed_interp = generator.generate_sed_interpolant(resolution='low', dust=False)
            save_interpolant(sed_interp, sed_low_nodust)
        else:
            print("Low-res no-dust SED interpolant already exists, skipping...")

        generator.data_manager.clear_cache()

        print("\nGenerating high-resolution SED interpolants...")

        sed_high_dust = output_dir / f'TODDLERS_tot_hr_{model_prefix}_lines_emergent=True{dtm_suffix}.pkl'
        if not interpolant_exists(sed_high_dust):
            sed_interp = generator.generate_sed_interpolant(resolution='high', dust=True)
            save_interpolant(sed_interp, sed_high_dust)
        else:
            print("High-res dust SED interpolant already exists, skipping...")
        
        sed_high_nodust = output_dir / f'TODDLERS_tot_hr_{model_prefix}_lines_emergent=False{dtm_suffix}.pkl'
        if not interpolant_exists(sed_high_nodust):
            sed_interp = generator.generate_sed_interpolant(resolution='high', dust=False)
            save_interpolant(sed_interp, sed_high_nodust)
        else:
            print("High-res no-dust SED interpolant already exists, skipping...")
        
        generator.data_manager.clear_cache()
        
        print("\nGenerating line luminosity interpolant...")
        line_interp = output_dir / 'line_luminosities_interp.pkl'
        line_map = output_dir / 'line_map.pkl'
        if not (interpolant_exists(line_interp) and interpolant_exists(line_map)):
            line_output = generator.generate_line_interpolant()
            save_interpolant(line_output['interpolant'], line_interp)
            save_interpolant({
                'line_map': line_output['line_map'], 
                'line_info': line_output['line_info']
            }, line_map)
        else:
            print("Line interpolants already exist, skipping...")
        
        generator.data_manager.clear_cache()
        
        print("\nGenerating dissolution time interpolant...")
        diss_time = output_dir / 'dissolution_time_interp.pkl'
        if not interpolant_exists(diss_time):
            diss_interp = generator.generate_dissolution_interpolant()
            save_interpolant(diss_interp, diss_time)
        else:
            print("Dissolution time interpolant already exists, skipping...")
        
        generator.data_manager.clear_cache()
        
        print("\nSaving recollapse data...")
        recollapse_file = output_dir / 'recollapse_data.h5'
        if not recollapse_file.exists():
            generator.save_recollapse_data(recollapse_file)
        else:
            print("Recollapse data already exists, skipping...")
        
        generator.data_manager.clear_cache()

        # Generate and save property interpolants last
        print("\nGenerating property interpolants...")
        properties = [
            'velocity',      
            'radius',        
            'density',       
            'temperature',   
            'f_esc_i'       
        ]
        
        for prop in properties:
            prop_file = output_dir / f'{prop}_interp.pkl'
            if not interpolant_exists(prop_file):
                print(f"  {prop}...")
                interp = generator.generate_property_interpolant(prop)
                save_interpolant(interp, prop_file)
            else:
                print(f"  {prop} interpolant already exists, skipping...")
        
        generator.data_manager.clear_cache()
        
        print("\nSuccessfully generated all interpolants!")
        print(f"Output files saved to: {output_dir}")
    except Exception as e:
        print(f"\nError generating interpolants: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()