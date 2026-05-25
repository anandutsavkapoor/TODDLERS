#!/usr/bin/env python3
"""
Script to rename TODDLERS STAB files to be compatible with ToddlersSEDFamily naming conventions.

The ToddlersSEDFamily class in SKIRT expects resource names in a specific format based on the
configured parameters (sedMode, stellarTemplate, includeDust, resolution, sfrPeriod).

This script renames existing STAB files to match the expected pattern.
"""

import os
import shutil
from pathlib import Path
import argparse

def get_new_filename(old_filename, sfr_period="Period10Myr"):
    """
    Convert old filename format to the new format expected by ToddlersSEDFamily.
    
    Args:
        old_filename: Path object for the original filename
        sfr_period: String specifying SFR period ("Period10Myr" or "Period30Myr")
        
    Returns:
        Path object with the new filename
    """
    stem = old_filename.stem
    suffix = old_filename.suffix
    
    # Parse components from the old filename
    parts = stem.split('_')
    
    # Example old format: ToddlersSFRNormalizedSEDFamily_BPASS_chab300_bin_noDust_hr
    is_sfr_normalized = "SFRNormalized" in stem
    
    # Extract stellar model info (BPASS/SB99, IMF, stellar type)
    model_index = -1
    for i, part in enumerate(parts):
        if part in ["BPASS", "SB99"]:
            model_index = i
            break
    
    if model_index >= 0:
        stellar_template = f"{parts[model_index]}_{parts[model_index+1]}_{parts[model_index+2]}"
    else:
        # Fallback if we can't find the model parts
        stellar_template = "SB99_kroupa100_sin"
    
    # Check if noDust is specified
    include_dust = "noDust" not in stem
    
    # Get resolution
    resolution = "lr" if "lr" in stem else "hr"
    
    # Check if an existing time period is specified in the filename
    existing_period = None
    if "10Myr" in stem:
        existing_period = "Period10Myr"
    elif "30Myr" in stem:
        existing_period = "Period30Myr"
    
    # Construct new filename
    # Format: ToddlersSEDFamily_SFRNormalized_BPASS_chab300_bin_Dust_hr_30Myr
    new_name = "ToddlersSEDFamily"
    
    # Add SED mode
    if is_sfr_normalized:
        new_name += "_SFRNormalized"
    else:
        new_name += "_Cloud"
    
    # Add stellar template
    new_name += f"_{stellar_template}"
    
    # Add dust option
    new_name += "_Dust" if include_dust else "_noDust"
    
    # Add resolution
    new_name += f"_{resolution}"
    
    # Add SFR period if SFRNormalized mode
    if is_sfr_normalized:
        # Use existing period if found in the filename, otherwise use provided period
        period_to_use = existing_period if existing_period else sfr_period
        if period_to_use == "Period30Myr":
            new_name += "_30Myr"
        elif period_to_use == "Period10Myr":
            new_name += "_10Myr"
    
    return old_filename.with_name(f"{new_name}{suffix}")

def is_already_correct_format(filename):
    """
    Check if a filename is already in the correct format expected by ToddlersSEDFamily.
    
    Args:
        filename: Path object for the filename to check
        
    Returns:
        Boolean indicating if the filename is already in the correct format
    """
    stem = filename.stem
    
    # Check if it starts with the proper prefix
    if not stem.startswith("ToddlersSEDFamily_"):
        return False
    
    # Check other components
    parts = stem.split('_')
    
    # Minimum expected parts: ToddlersSEDFamily_[SFRNormalized/Cloud]_[BPASS/SB99]_[imf]_[type]_[Dust/noDust]_[resolution]
    if len(parts) < 7:
        return False
        
    # Check if mode is valid
    if parts[1] not in ["SFRNormalized", "Cloud"]:
        return False
        
    # Check if stellar model components exist
    if parts[2] not in ["BPASS", "SB99"]:
        return False
        
    # Check if dust option is specified
    if "Dust" not in stem and "noDust" not in stem:
        return False
        
    # Check if resolution is specified
    if "lr" not in stem and "hr" not in stem:
        return False
    
    # Additional check for SFRNormalized files: they should have a period suffix
    # But we don't enforce a specific one, just check that it has either 10Myr or 30Myr
    if parts[1] == "SFRNormalized" and "10Myr" not in stem and "30Myr" not in stem:
        return False
    
    return True

def rename_files(input_dir, output_dir=None, dry_run=False, sfr_period="Period10Myr"):
    """
    Rename STAB files in the input directory to match ToddlersSEDFamily conventions.
    
    Args:
        input_dir: Directory containing STAB files to rename
        output_dir: Directory to place renamed files (if None, uses input_dir)
        dry_run: If True, print changes but don't actually rename/copy files
        sfr_period: SFR period to use in filenames ("Period10Myr" or "Period30Myr")
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
    
    # Create output directory if needed
    if output_dir and not output_path.exists():
        if not dry_run:
            output_path.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_path}")
    
    # Find all STAB files
    stab_files = list(input_path.glob("*.stab"))
    
    if not stab_files:
        print(f"No STAB files found in {input_dir}")
        return
        
    print(f"Found {len(stab_files)} STAB files to process")
    
    # Track statistics
    files_skipped = 0
    files_renamed = 0
    
    # Process each file
    for stab_file in stab_files:
        # Check if the file is already in the correct format
        if is_already_correct_format(stab_file):
            print(f"Skipping: {stab_file.name} (already in correct format)")
            files_skipped += 1
            continue
            
        new_name = get_new_filename(stab_file, sfr_period)
        
        # Use output directory if specified
        if output_dir:
            new_path = output_path / new_name.name
        else:
            new_path = new_name
        
        # Check if the destination file already exists
        if new_path.exists() and stab_file.resolve() != new_path.resolve():
            print(f"Warning: Destination file already exists, skipping: {new_path}")
            files_skipped += 1
            continue
            
        print(f"Renaming: {stab_file.name} -> {new_path.name}")
        files_renamed += 1
        
        if not dry_run:
            if output_dir:
                # Copy to new location with new name
                shutil.copy2(stab_file, new_path)
                print(f"  Copied to {new_path}")
            else:
                # Rename in place
                stab_file.rename(new_path)
                print(f"  Renamed to {new_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Files processed: {len(stab_files)}")
    print(f"  Files renamed: {files_renamed}")
    print(f"  Files skipped: {files_skipped}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename TODDLERS STAB files to match ToddlersSEDFamily conventions.")
    parser.add_argument("input_dir", help="Directory containing STAB files to rename")
    parser.add_argument("--output-dir", help="Directory to place renamed files (if not specified, rename in place)")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without actually renaming files")
    parser.add_argument("--sfr-period", choices=["Period10Myr", "Period30Myr"], default="Period10Myr",
                        help="SFR period to use in filenames (default: Period10Myr)")
    
    args = parser.parse_args()
    
    rename_files(args.input_dir, args.output_dir, args.dry_run, args.sfr_period)