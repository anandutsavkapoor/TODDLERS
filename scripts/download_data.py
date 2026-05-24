"""
Download required data files for TODDLERS observables generation.

This script handles:
1. Downloading and organizing base data files from Google Drive
2. Downloading full-resolution BPASS stellar model tables from data.nublado.org
3. Updating the CLOUDY_DATA_DIR in constants.py
4. Managing the Cloudy data directory structure

The script can:
- Download all required base resources
- Download all full-resolution BPASS tables
- Download specific BPASS tables by IMF and star type
"""

import os
import sys
import subprocess
import shutil
import zipfile
import argparse
import requests
import lzma
import numpy as np
from urllib.parse import urljoin
from typing import Optional, Tuple
from toddlers.cloudy_stellar_spectra_generator import SpectralTableGenerator
from toddlers.constants import *

GOOGLE_DRIVE_ID = '12Gk6wJAjtft5huRbl9RJyiD8vscOIjT_'
MARKER_FILE = 'BPASS_chab100_bin_constantSFR_t2.0e+00myr_resFac10.ascii'

def get_cloudy_data_dir() -> Tuple[Optional[str], bool]:
    """
    Get the Cloudy data directory path.
    
    Returns:
        tuple: (data_dir, should_update_constant)
            - data_dir: Path to Cloudy data directory or None if not found
            - should_update_constant: Boolean indicating if constants.py should be updated
            
    Notes:
        - Looks for cloudy.exe in system PATH
        - Data directory is expected to be ../data relative to executable
    """
    try:
        cloudy_path = subprocess.check_output(['which', 'cloudy.exe'], universal_newlines=True).strip()
        cloudy_data_dir = os.path.abspath(os.path.join(os.path.dirname(cloudy_path), '..', 'data'))
        if os.path.exists(cloudy_data_dir):
            return cloudy_data_dir, True
        print(f"Warning: Expected Cloudy data directory {cloudy_data_dir} not found.")
        return None, False
    except subprocess.CalledProcessError:
        print("Error: Could not find Cloudy executable. Please ensure Cloudy is installed.")
        return None, False

def check_resources_exist(data_dir: str) -> bool:
    """
    Check if TODDLERS resources are already installed.
    
    Args:
        data_dir (str): Path to Cloudy data directory
        
    Returns:
        bool: True if marker file exists, indicating resources are installed
    """
    marker_path = os.path.join(data_dir, MARKER_FILE)
    return os.path.exists(marker_path)

def update_constants_file(data_dir: str) -> bool:
    """
    Update the CLOUDY_DATA_DIR constant in constants.py.
    
    Args:
        data_dir (str): New data directory path to set
        
    Returns:
        bool: True if update was successful
        
    Notes:
        - Looks for constants.py in src directory
        - Creates CLOUDY_DATA_DIR if it doesn't exist
        - Updates existing value if it does exist
    """
    try:
        # scripts/ lives at the project root; constants is src/toddlers/constants.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        constants_path = os.path.join(project_root, 'src', 'toddlers', 'constants.py')

        if not os.path.exists(constants_path):
            print(f"Error: Could not find constants.py at {constants_path}")
            return False

        with open(constants_path, 'r') as f:
            lines = f.readlines()

        # Rewrite the rewritable default literal; CLOUDY_DATA_DIR itself stays
        # environment-overridable (it reads $CLOUDY_DATA_DIR with this as fallback).
        updated = False
        for i, line in enumerate(lines):
            if line.startswith('_DEFAULT_CLOUDY_DATA_DIR'):
                lines[i] = f'_DEFAULT_CLOUDY_DATA_DIR = "{data_dir}"\n'
                updated = True
                break

        if not updated:
            lines.append(f'\n_DEFAULT_CLOUDY_DATA_DIR = "{data_dir}"\n')
            
        with open(constants_path, 'w') as f:
            f.writelines(lines)
            
        print(f"Updated CLOUDY_DATA_DIR in constants.py to: {data_dir}")
        return True
        
    except Exception as e:
        print(f"Error updating constants.py: {str(e)}")
        return False

def download_bpass_table(imf: str, star_type: str, 
            cloudy_data_dir: str, cleanup: bool = False) -> bool:
    """
    Download and process a single full-resolution BPASS table.
    
    Args:
        imf (str): Initial mass function ('chab100' or 'chab300')
        star_type (str): Star type ('sin' for single or 'bin' for binary)
        cloudy_data_dir (str): Path to Cloudy data directory
        cleanup (bool, optional): Whether to remove temporary files and directories 
                                after processing. Defaults to True.
        
    Returns:
        bool: True if successful, False otherwise
        
    Notes:
        - Downloads compressed file from data.nublado.org to others/Cloudy Spectra
        - Uses SpectralTableGenerator to process and format the file
        - Automatically saves to Cloudy data directory with correct naming
        - Cleans up temporary files and directories if cleanup=True
    """
    # Map our naming convention to BPASS naming
    star_type_map = {'sin': 'single', 'bin': 'binary'}
    if star_type not in star_type_map:
        print(f"Invalid star type: {star_type}. Must be 'sin' or 'bin'")
        return False
        
    if imf not in ['chab100', 'chab300']:
        print(f"Invalid IMF: {imf}. Must be 'chab100' or 'chab300'")
        return False

    # Get base project directory
    src_dir = os.path.abspath(__file__)
    project_root = os.path.dirname(src_dir)
        
    # Construct URL and filename
    base_url = "https://data.nublado.org/stars/bpass/v2.2.1/"
    download_filename = f"BPASSv2.2.1_imf_{imf}_burst_{star_type_map[star_type]}.ascii"
    url = urljoin(base_url, download_filename + ".xz")
    
    # Construct paths
    cloudy_spectra_dir = os.path.join(project_root, "others", "Cloudy Spectra")
    os.makedirs(cloudy_spectra_dir, exist_ok=True)
    temp_file = os.path.join(cloudy_spectra_dir, download_filename)
    temp_compressed = temp_file + ".xz"
    
    # Check if final file already exists in Cloudy directory
    final_filename = f"BPASS_{imf}_{star_type}_burst.ascii"
    final_path = os.path.join(cloudy_data_dir, final_filename)
    
    if os.path.exists(final_path):
        print(f"Full resolution table already exists: {final_filename}")
        return True
        
    print(f"Downloading {url}...")
    
    try:
        # Download compressed file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create temporary file for compressed data
        with open(temp_compressed, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        # Decompress file to Cloudy Spectra directory
        print(f"Decompressing to {download_filename}...")
        with lzma.open(temp_compressed) as compressed:
            with open(temp_file, 'wb') as decompressed:
                decompressed.write(compressed.read())
        
        # Always remove the compressed file
        os.remove(temp_compressed)
        
        # Process using SpectralTableGenerator
        print("Processing with SpectralTableGenerator...")
        generator = SpectralTableGenerator(
            template="BPASSv2.2.1",
            imf=imf,
            star_type=star_type,
            wavelength_resolution_factor=1  # Full resolution
        )
        
        # Generate spectral table using full age range
        ages = np.logspace(5, np.log10(generator.max_age), NUM_STELLAR_SPEC_CLOUDY)
        generator.generate_spectral_table(ages, Z_VALUES_BPASS)
        
        # Cleanup if requested
        if cleanup:
            try:
                # Remove temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                # Remove Cloudy Spectra directory if empty
                if os.path.exists(cloudy_spectra_dir) and not os.listdir(cloudy_spectra_dir):
                    os.rmdir(cloudy_spectra_dir)
                # Remove others directory if empty
                others_dir = os.path.dirname(cloudy_spectra_dir)
                if os.path.exists(others_dir) and not os.listdir(others_dir):
                    os.rmdir(others_dir)
            except Exception as e:
                print(f"Warning: Cleanup partially failed: {str(e)}")
        
        print(f"Successfully processed {final_filename}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {str(e)}")
        return False
    except lzma.LZMAError as e:
        print(f"Error decompressing file: {str(e)}")
        if os.path.exists(temp_compressed):
            os.remove(temp_compressed)
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        # Clean up on error
        if os.path.exists(temp_compressed):
            os.remove(temp_compressed)
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False

def download_all_bpass_tables(cloudy_data_dir: str) -> None:
    """
    Download all full-resolution BPASS tables.
    
    Args:
        cloudy_data_dir (str): Path to Cloudy data directory
        
    Notes:
        - Downloads all combinations of IMF and star type
        - Reports overall success/failure count
    """
    combinations = [
        ('chab100', 'sin'),
        ('chab100', 'bin'),
        ('chab300', 'sin'),
        ('chab300', 'bin')
    ]
    
    success_count = 0
    for imf, star_type in combinations:
        if download_bpass_table(imf, star_type, cloudy_data_dir):
            success_count += 1
            
    print(f"\nDownloaded {success_count} out of {len(combinations)} BPASS tables")

def download_file(destination: str) -> Optional[str]:
    """
    Download the main TODDLERS resource file from Google Drive.
    
    Args:
        destination (str): Path where the downloaded file should be saved
        
    Returns:
        Optional[str]: Path to downloaded file if successful, None otherwise
        
    Notes:
        - Requires gdown package for Google Drive downloads
        - Will exit if gdown is not installed
    """
    try:
        import gdown
    except ImportError:
        print("Please install gdown: pip install gdown")
        sys.exit(1)

    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"
    return gdown.download(url, destination, quiet=False)

def extract_and_organize(zip_path: str, extract_path: str) -> bool:
    """
    Extract and organize files from the downloaded zip.
    
    Args:
        zip_path (str): Path to downloaded zip file
        extract_path (str): Directory where files should be extracted
        
    Returns:
        bool: True if extraction and organization was successful
        
    Notes:
        - Handles cleanup of temporary and macOS-specific files
        - Removes the zip file after extraction
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        resources_folder = os.path.join(extract_path, 'TODDLERS Cloudy resources')
        
        if os.path.exists(resources_folder):
            for item in os.listdir(resources_folder):
                src = os.path.join(resources_folder, item)
                dst = os.path.join(extract_path, item)
                
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                        
                shutil.move(src, extract_path)
            
            os.rmdir(resources_folder)
            
            macosx_dir = os.path.join(extract_path, '__MACOSX')
            if os.path.exists(macosx_dir):
                shutil.rmtree(macosx_dir)
                
            return True
            
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        return False
        
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)

def main():
    """
    Main function to download and organize TODDLERS data files.
    
    This function handles:
    1. Parsing command line arguments
    2. Checking Cloudy installation
    3. Updating constants.py if needed
    4. Downloading requested resources
    
    Command line options:
    --full-bpass: Download all full-resolution BPASS tables
    --bpass-imf: Download specific BPASS table - specify IMF
    --bpass-stars: Download specific BPASS table - specify star type
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(description='Download TODDLERS data files')
    parser.add_argument('--full-bpass', action='store_true',
                      help='Download all full-resolution BPASS tables')
    parser.add_argument('--bpass-imf', choices=['chab100', 'chab300'],
                      help='Download specific BPASS table: choose IMF')
    parser.add_argument('--bpass-stars', choices=['sin', 'bin'],
                      help='Download specific BPASS table: choose star type')
    args = parser.parse_args()
    
    print("Starting TODDLERS data installation check...")
    
    cloudy_data_dir, should_update = get_cloudy_data_dir()
    if not cloudy_data_dir:
        sys.exit(1)
        
    if should_update:
        if not update_constants_file(cloudy_data_dir):
            print("Warning: Failed to update constants.py")

    # Handle BPASS table downloads
    if args.full_bpass:
        print("\nDownloading all full-resolution BPASS tables...")
        download_all_bpass_tables(cloudy_data_dir)
        print("Full-resolution BPASS table download complete")
        if not check_resources_exist(cloudy_data_dir):
            print("\nProceeding with main resource download...")
        else:
            return 0
    elif args.bpass_imf and args.bpass_stars:
        print(f"\nDownloading single BPASS table: {args.bpass_imf} IMF with {args.bpass_stars} stars...")
        success = download_bpass_table(args.bpass_imf, args.bpass_stars, cloudy_data_dir)
        if success:
            print("BPASS table download complete")
            return 0
        else:
            print("Failed to download BPASS table")
            return 1
    elif args.bpass_imf or args.bpass_stars:
        print("Error: Must provide both --bpass-imf and --bpass-stars for single table download")
        return 1

    if check_resources_exist(cloudy_data_dir):
        print("TODDLERS Cloudy resources are already installed.")
        return 0

    zip_path = os.path.join(cloudy_data_dir, 'toddlers_temp.zip')

    print(f"Downloading data to {cloudy_data_dir}...")
    if download_file(zip_path):
        print("Download complete. Extracting and organizing files...")
        if extract_and_organize(zip_path, cloudy_data_dir):
            print(f"Data successfully installed in {cloudy_data_dir}")
            return 0
    
    print("Failed to download or extract data files.")
    return 1

if __name__ == '__main__':
    sys.exit(main())