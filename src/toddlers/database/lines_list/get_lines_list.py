#!/usr/bin/env python3
"""
Cloudy Line List Processor

This script processes Cloudy line list files (LineList_HII.dat and LineList_PDR_H2.dat)
to create a unified, deduplicated list of spectral lines while maintaining exact wavelengths
and formatting. It includes optional filtering of high-ionization species.

The script reads the official line lists, combines them, removes duplicates,
and optionally filters out species with ionization levels above a specified threshold.
Removed lines are saved to a separate file for reference.

Usage:
    Place this script in the same directory as the input files and run:
    python process_linelist.py

Output files:
    - cloudy_lines_v2_with_comments.dat: Full listing with comments
    - cloudy_lines_v2.dat: Clean version without comments
    - cloudy_lines_v2_sorted.dat: Wavelength-sorted version in microns
    - cloudy_lines_v2_removed.dat: Lines removed due to high ionization (if filtering enabled)
"""

def get_ionization_level(species):
    """
    Extract ionization level from species label.
    
    Args:
        species (str): Species identifier (e.g., 'H  1', 'Fe14', 'O  3')
        
    Returns:
        int: Ionization level (0 for neutrals, positive for ions, -1 if not applicable)
    """
    # Handle special cases like blended lines
    if species.lower().startswith('blnd'):
        return -1
        
    # Extract numeric part
    parts = species.split()
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1]) - 1  # Convert to ionization level (e.g., O III is +2)
    elif parts[0][-1:].isdigit() and len(parts) == 1:
        # Handle cases like Fe14
        num = ''
        for char in reversed(parts[0]):
            if char.isdigit():
                num = char + num
            else:
                break
        if num:
            return int(num) - 1
    return -1

def convert_to_micron(wavelength):
    """
    Convert wavelength to microns preserving Cloudy's precision.
    
    Args:
        wavelength (str): Wavelength with optional unit:
            - No unit or 'A' for Angstrom (Angstroms is default)
            - 'm' for micron
            - 'c' for centimeter
        
    Returns:
        tuple: (float value in microns, str formatted value preserving original precision)
    """
    # Get the numeric part and unit, preserving original precision
    if wavelength[-1] in ['A', 'm', 'c']:
        value_str = wavelength[:-1]
        unit = wavelength[-1]
    else:
        value_str = wavelength
        unit = 'A'  # Default unit is Angstroms
    
    # Convert while preserving original decimal places
    value = float(value_str)
    decimals = len(value_str.split('.')[-1]) if '.' in value_str else 0
    
    if unit == 'A':  # Including default case
        micron_value = value / 10000.0
        # Adjust precision for Angstrom conversion
        formatted = f"{micron_value:.6f}"
    elif unit == 'c':
        micron_value = value * 10000.0
        formatted = f"{micron_value:.{decimals}f}"
    else:  # microns
        micron_value = value
        formatted = value_str
        
    return micron_value, formatted

def format_species(species, level=None):
    """
    Format species identifier according to Cloudy syntax while preserving original case.
    
    Args:
        species (str): Species identifier (e.g., 'H', 'He', 'O', 'blnd', 'Blnd')
        level (str, optional): Ionization level (e.g., '1', '2')
        
    Returns:
        str: Properly formatted species string with correct spacing
    """
    # Preserve original case
    species = species.strip()
    
    # Handle blended lines with original case
    if species.lower() in ['blnd', 'bac', 'inci']:
        return species
    
    # For regular species with ionization level
    if level is not None:
        # Ensure the total length is 4 characters (including space)
        space_needed = 4 - (len(species) + len(level))
        if space_needed > 0:
            return f"{species}{' ' * space_needed}{level}"
        return f"{species}{level}"  # no species are > 2, or levels > 2
        
    return f"{species:<4}"

def parse_line(line):
    """
    Parse a single line and extract components.
    
    Args:
        line (str): Input line from line list file
        
    Returns:
        tuple: (species, wavelength, comment) or None if line should be skipped
    """
    # Split for comment
    parts = line.split('#', 1)
    main_part = parts[0].strip()
    comment = parts[1].strip() if len(parts) > 1 else ""
    
    words = main_part.split()
    if not words:
        return None
        
    # Skip unwanted patterns
    first_word_lower = words[0].lower()
    if first_word_lower in ['inci', 'bac', 'grat', 'fir', 'tir', 'nmir', 'tfir', 
                           'inwd', 'nira', 'nirb', 'mira', 'mirb']:
        return None
    if len(words) >= 2 and ' '.join(words[:2]).lower() in ['ba c', 'ca b']:
        return None
        
    # For blended lines, only keep if they have comments
    if first_word_lower == 'blnd':
        if not comment:
            return None
            
    # Format the species and level properly
    if len(words) >= 3 and words[1].isdigit():
        species = format_species(words[0], words[1])
        wavelength = words[2]
    else:
        species = format_species(words[0])
        wavelength = words[1]
        
    return species, wavelength, comment

def process_file(filename, include_h2=False):
    """
    Process a single line list file.
    
    Args:
        filename (str): Path to input file
        include_h2 (bool): Whether to include H2 lines
        
    Returns:
        list: Parsed line data (species, wavelength, comment)
    """
    lines = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            result = parse_line(line)
            if result:
                species, wavelength, comment = result
                
                # Skip H2 lines if not included
                if not include_h2 and species.lower().startswith('h2'):
                    continue
                    
                lines.append((species, wavelength, comment))
    return lines

def filter_high_ionization(lines, max_ionization):
    """
    Filter out species above specified ionization level.
    
    Args:
        lines (list): List of tuples (micron_value, formatted_micron, species, wavelength, comment)
        max_ionization (int): Maximum allowed ionization level
        
    Returns:
        tuple: (kept_lines, removed_lines) Lists of lines that pass and fail the filter
    """
    kept_lines = []
    removed_lines = []
    
    for line in lines:
        micron_value, formatted_micron, species, wavelength, comment = line
        ion_level = get_ionization_level(species)
        
        if ion_level > max_ionization and ion_level != -1:
            removed_lines.append(line)
        else:
            kept_lines.append(line)
            
    return kept_lines, removed_lines

def write_removed_lines(removed_lines, filename):
    """
    Write removed lines to a separate file.
    
    Args:
        removed_lines (list): List of removed line tuples
        filename (str): Output filename
    """
    with open(filename, 'w') as f:
        f.write("# Lines removed due to high ionization\n")
        f.write("# Format: <species> <wavelength> [#comment]\n")
        f.write("#\n")
        
        # Group by ionization level for better organization
        removed_by_level = {}
        for line in removed_lines:
            _, _, species, wavelength, comment = line
            ion_level = get_ionization_level(species)
            if ion_level not in removed_by_level:
                removed_by_level[ion_level] = []
            removed_by_level[ion_level].append(line)
        
        # Write summary
        f.write(f"# Total lines removed: {len(removed_lines)}\n")
        f.write("# Breakdown by ionization level:\n")
        for level in sorted(removed_by_level.keys()):
            count = len(removed_by_level[level])
            f.write(f"# Ionization +{level}: {count} lines\n")
        f.write("#\n")
        
        # Write the actual lines
        for level in sorted(removed_by_level.keys()):
            f.write(f"\n# --- Ionization +{level} ---\n")
            for _, _, species, wavelength, comment in sorted(removed_by_level[level]):
                line = f"{species}    {wavelength}"
                if comment:
                    line += f"  #{comment}"
                f.write(line + "\n")

def extract_addnl_he_lines(filename="LineList_HeH.dat"):
    """
    Extract strong He lines from LineList_HeH.dat.
    Returns list of tuples (species, wavelength, comment).
    """
    target_wavelengths = {
        "3187.74", "3964.73", "4026.21A", 
        "4921.93", "5015.68", "6678.15"
    }
    
    he_lines = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split('#', 1)
            main_part = parts[0].strip()
            comment = parts[1].strip() if len(parts) > 1 else ""
            
            words = main_part.split()
            if len(words) < 2:
                continue
                
            if words[-1] in target_wavelengths:
                species = format_species(words[0], words[1]) if len(words) >= 3 else format_species(words[0])
                wavelength = words[-1]
                he_lines.append((species, wavelength, comment))
                
    return he_lines

def write_output_files(lines, output_base, max_ionization=None):
    """
    Write processed lines to output files.
    
    Args:
        lines (list): Processed line data
        output_base (str): Base name for output files
        max_ionization (int, optional): Maximum allowed ionization level
    """
    # Convert wavelengths to microns for sorting while preserving original format
    lines_with_microns = []
    for species, wavelength, comment in lines:
        try:
            micron_value, formatted_micron = convert_to_micron(wavelength)
            lines_with_microns.append((micron_value, formatted_micron, species, wavelength, comment))
        except ValueError as e:
            print(f"WARNING: Skipping invalid line: {species} {wavelength} - {str(e)}")
    
    # Sort by wavelength in microns
    sorted_lines = sorted(lines_with_microns, key=lambda x: x[0])
    
    # Optionally filter high ionization lines
    if max_ionization is not None:
        kept_lines, removed_lines = filter_high_ionization(sorted_lines, max_ionization)
        # Write removed lines to separate file
        write_removed_lines(removed_lines, f"{output_base}_removed.dat")
        print(f"Removed {len(removed_lines)} high-ionization lines (saved to {output_base}_removed.dat)")
        sorted_lines = kept_lines
    
    # Write full version with comments
    with open(f"{output_base}_with_comments.dat", 'w') as f:
        f.write("# Full line list with comments\n")
        f.write("# Generated from official Cloudy line lists\n")
        f.write("#\n")
        f.write("# Wavelength units:\n")
        f.write("# - No unit specified: Assumed to be Angstroms\n")
        f.write("# - 'A': Angstroms\n")
        f.write("# - 'm': microns\n")
        f.write("# - 'c': centimeters\n")
        f.write("#\n")
        f.write("# Format: <species> <wavelength> [#comment]\n")
        f.write("# Species format examples:\n")
        f.write("#   H  1  - Neutral hydrogen\n")
        f.write("#   He 2  - Singly ionized helium\n")
        f.write("#   O  3  - Doubly ionized oxygen\n")
        f.write("#   Blnd  - Blend of several lines\n")
        f.write("#\n")
        
        for _, _, species, wavelength, comment in sorted_lines:
            line = f"{species}    {wavelength}"
            if comment:
                line += f"  #{comment}"
            f.write(line + "\n")
            
    # Write clean version without comments
    with open(f"{output_base}.dat", 'w') as f:
        f.write("# Clean line list without comments\n")
        f.write("# Note: wavelengths without units are in Angstroms\n")
        f.write("# Species are formatted according to Cloudy syntax\n")
        
        for _, _, species, wavelength, _ in sorted_lines:
            f.write(f"{species}    {wavelength}\n")
            
    # Write wavelength sorted version
    with open(f"{output_base}_sorted.dat", 'w') as f:
        f.write("# Wavelength sorted line list\n")
        f.write("# All wavelengths converted to microns for sorting\n")
        f.write("# Original units preserved in output\n")
        f.write("#\n")
        f.write("# Columns:\n")
        f.write("# 1. wavelength in microns (for sorting)\n")
        f.write("# 2. wavelength in original format\n")
        f.write("# 3. species identification (Cloudy format)\n")
        f.write("# 4. original wavelength with unit\n")
        f.write("# 5. comment (if any)\n")
        f.write("#\n")
        
        for micron_value, formatted_micron, species, orig_wl, comment in sorted_lines:
            comment_str = f"\t# {comment}" if comment else ""
            f.write(f"{micron_value:.6e}\t{formatted_micron}\t{species:<6}\t{orig_wl}{comment_str}\n")

def main():
    """Main function to process line lists."""
    # Process HII region lines
    print("Processing HII region lines...")
    hii_lines = process_file("LineList_HII.dat")
    
    # Process PDR lines
    print("Processing PDR lines...")
    pdr_lines = process_file("LineList_PDR_H2.dat", include_h2=False)
    
    # Add missing strong He lines
    addnl_he_lines = extract_addnl_he_lines()

    # Combine and deduplicate lines
    print("Combining and deduplicating lines...")
    seen_lines = {}
    all_lines = []
    
    for species, wavelength, comment in hii_lines + pdr_lines + addnl_he_lines:
        line_id = f"{species.lower()}_{wavelength}"
        if line_id not in seen_lines:
            seen_lines[line_id] = True
            all_lines.append((species, wavelength, comment))
            
    # Write output files with optional ionization filtering
    print("Writing output files...")
    # Set max_ionization=None to disable filtering
    write_output_files(all_lines, "cloudy_lines_TODDLERS_v2", max_ionization=6)
    
    print("\nDone! Generated output files:")
    print("1. cloudy_lines_v2_with_comments.dat - Full version with comments")
    print("2. cloudy_lines_v2.dat - Clean version without comments") 
    print("3. cloudy_lines_v2_sorted.dat - Wavelength sorted version in microns")
    print("4. cloudy_lines_v2_removed.dat - Removed high-ionization lines")

if __name__ == "__main__":
    main()