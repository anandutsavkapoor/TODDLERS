"""
Download required data files for TODDLERS observables generation.

This script handles:
1. Downloading and organizing base data files from Google Drive
2. Downloading full-resolution BPASS stellar model tables from data.nublado.org
3. Updating the CLOUDY_DATA_DIR in constants.py
4. Managing the Cloudy data directory structure
5. Downloading the single-star feedback database (single_star_tracks.h5) for the
   stochastic IMF sampler (Google Drive; does not require Cloudy)

The script can:
- Download all required base resources
- Download all full-resolution BPASS tables
- Download specific BPASS tables by IMF and star type
- Download single_star_tracks.h5 (--stochastic-tracks)
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
from toddlers.cloudy_stellar_spectra_generator import (SpectralTableGenerator,
                                                       bpass_burst_source_path,
                                                       RELEASED_TABLE_MD5)
from toddlers.constants import *

from toddlers._paths import data_resources

# Drive ids and md5s come from the package's versioned resources file (data_resources.json):
# a new data version is a new Drive file and a new entry, so released code keeps its own data.
RESOURCES = data_resources()
GOOGLE_DRIVE_ID = RESOURCES['cloudy_resources']['drive_id']
MARKER_FILE = 'BPASS_chab100_bin_constantSFR_t2.0e+00myr_resFac10.ascii'

# md5 of the full-resolution BPASS v2.2.1 burst tables on data.nublado.org: the .xz as
# downloaded and the decompressed .ascii the generator reads (recorded
# for 2.1.0; the chab100 binary .ascii also matches the copy used for the released
# resFac10 tables). A download that does not match is refused.
BPASS_SOURCE_MD5 = {
    'BPASSv2.2.1_imf_chab100_burst_single.ascii': {'xz': '4c34614c35632671297a164272daeeec',
                                                   'ascii': 'da04ed1cb822c2511054894cdc8a2c2a'},
    'BPASSv2.2.1_imf_chab100_burst_binary.ascii': {'xz': 'c75f9188abf14710f2739a4285ca550a',
                                                   'ascii': '0601d41d78e533fd99450556c2a536c3'},
    'BPASSv2.2.1_imf_chab300_burst_single.ascii': {'xz': '50914ce3635f105e75006c16e52d8c45',
                                                   'ascii': 'a64444fa7189e77fe9cf9635b9a4dd5b'},
    'BPASSv2.2.1_imf_chab300_burst_binary.ascii': {'xz': '6bf6b5cf326e03042e0aba208de6b417',
                                                   'ascii': 'f8667d3525317e7ddd96e125c11fbc08'},
}


def _md5(path: str) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# The single-star feedback database (needed by the stochastic IMF sampler): Drive id and md5
# of this version from data_resources.json (a null id would point users at the local build,
# examples/build_stochastic_database.py).
SINGLE_STAR_TRACKS_DRIVE_ID = RESOURCES['single_star_tracks']['drive_id']
SINGLE_STAR_TRACKS_MD5 = RESOURCES['single_star_tracks']['md5']
SINGLE_STAR_TRACKS_VERSION = RESOURCES['single_star_tracks']['version']
# Below this many MB a "downloaded" file is treated as a failure (e.g. an HTML error page).
_TRACKS_MIN_MB = 50

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
        - Downloads compressed file from data.nublado.org to src/others/Cloudy Spectra
          (bpass_burst_source_path: where SpectralTableGenerator reads it)
        - Checks the md5 of the .xz and of the decompressed table (BPASS_SOURCE_MD5);
          a mismatch is refused and the downloaded files are removed
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

    # The generator reads the source table from bpass_burst_source_path; write it there.
    temp_file = bpass_burst_source_path(imf, star_type)
    download_filename = os.path.basename(temp_file)
    cloudy_spectra_dir = os.path.dirname(temp_file)
    os.makedirs(cloudy_spectra_dir, exist_ok=True)
    temp_compressed = temp_file + ".xz"

    # Construct URL
    base_url = "https://data.nublado.org/stars/bpass/v2.2.1/"
    url = urljoin(base_url, download_filename + ".xz")
    expected = BPASS_SOURCE_MD5.get(download_filename)
    if expected is None:
        print(f"No pinned md5 for {download_filename}; refusing an unchecked download.")
        return False
    
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
                
        got = _md5(temp_compressed)
        if got != expected['xz']:
            print(f"Error: MD5 mismatch for {download_filename}.xz (got {got}, expected "
                  f"{expected['xz']}); download refused.")
            os.remove(temp_compressed)
            return False

        # Decompress file to Cloudy Spectra directory
        print(f"Decompressing to {download_filename}...")
        with lzma.open(temp_compressed) as compressed:
            with open(temp_file, 'wb') as decompressed:
                decompressed.write(compressed.read())
        
        # Always remove the compressed file
        os.remove(temp_compressed)

        got = _md5(temp_file)
        if got != expected['ascii']:
            print(f"Error: MD5 mismatch for {download_filename} (got {got}, expected "
                  f"{expected['ascii']}); refused.")
            os.remove(temp_file)
            return False
        
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

def _move_aside(path: str) -> str:
    """Move a data file that is not this version's into stale_data/<UTC time>/ next to it
    (never delete); return the new path."""
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    dest_dir = os.path.join(os.path.dirname(path), 'stale_data', stamp)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(path))
    os.replace(path, dest)
    return dest


def list_resources() -> None:
    """Print the versioned resources of this code version and whether each is installed."""
    from toddlers._paths import get_database_path
    for name, r in sorted(RESOURCES.items()):
        state = ''
        if name == 'single_star_tracks':
            p = get_database_path()
            state = ('current' if os.path.exists(p) and _md5(p) == r['md5']
                     else 'other version' if os.path.exists(p) else 'missing')
        print(f"{name:20s} v{r['version']:8s} md5 {r['md5']}  drive "
              f"{r['drive_id'] or '(not uploaded yet)'}  {state}")


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
    out = gdown.download(url, destination, quiet=False)
    want = RESOURCES['cloudy_resources']['md5']
    if out and want and _md5(destination) != want:
        print(f"Error: the resource bundle md5 is not {want}; download is corrupt. Removing it.")
        os.remove(destination)
        return None
    return out

def download_single_star_tracks(overwrite: bool = False) -> bool:
    """Download the single-star feedback database (``single_star_tracks.h5``) from Google Drive.

    This track library (~583 MB) is what the stochastic IMF sampler needs. It is not shipped
    in the repository (too large) and is not part of the Cloudy-resources bundle; this
    fetches it into the resolved data directory (``get_database_path()``, i.e.
    ``$TODDLERS_DATA`` or ``<package>/database``). It does NOT require Cloudy.

    As a reproducible alternative, build it locally with
    ``examples/build_stochastic_database.py`` (slower, but no download).

    Args:
        overwrite (bool): Refetch even if a real file is already present.

    Returns:
        bool: True on success (or if a real file is already present), False otherwise.
    """
    try:
        from toddlers._paths import get_database_path
    except Exception as e:
        print(f"Error: cannot resolve the database path ({e}).")
        return False

    dest = get_database_path()
    if os.path.exists(dest):
        if not overwrite and _md5(dest) == SINGLE_STAR_TRACKS_MD5:
            print(f"single_star_tracks.h5 v{SINGLE_STAR_TRACKS_VERSION} already present at {dest}; "
                  f"skipping (use --overwrite to refetch).")
            return True
        if not SINGLE_STAR_TRACKS_DRIVE_ID:
            print(f"{dest} is not the v{SINGLE_STAR_TRACKS_VERSION} database (md5 "
                  f"{SINGLE_STAR_TRACKS_MD5}), and that version is not on Drive yet; kept as is.")
            print("Build it locally instead:  python examples/build_stochastic_database.py")
            return False
        moved = _move_aside(dest)
        print(f"{os.path.basename(dest)} is not the v{SINGLE_STAR_TRACKS_VERSION} database; "
              f"moved it to {moved} and downloading the current one.")

    if not SINGLE_STAR_TRACKS_DRIVE_ID:
        print(f"single_star_tracks.h5 v{SINGLE_STAR_TRACKS_VERSION} is not on Drive yet "
              f"(drive_id unset in toddlers/data_resources.json).")
        print("Build it locally instead:  python examples/build_stochastic_database.py")
        return False

    try:
        import gdown
    except ImportError:
        print("Please install gdown: pip install gdown")
        return False

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"https://drive.google.com/uc?id={SINGLE_STAR_TRACKS_DRIVE_ID}"
    print(f"Downloading single_star_tracks.h5 (~{RESOURCES['single_star_tracks']['size_mb']} MB) to {dest} ...")
    # NB: no fuzzy= kwarg -- it was dropped in gdown 6.x and isn't needed for an
    # explicit uc?id= URL (gdown still handles the large-file confirm token).
    out = gdown.download(url, dest, quiet=False)
    if not out or not os.path.exists(dest):
        print("Error: download failed.")
        return False
    size_mb = os.path.getsize(dest) / 1e6
    if size_mb < _TRACKS_MIN_MB:
        print(f"Warning: downloaded file is only {size_mb:.1f} MB (expected ~{RESOURCES['single_star_tracks']['size_mb']} MB); "
              f"the Drive link may be wrong or rate-limited. Inspect {dest}.")
        return False
    got = _md5(dest)
    if got != SINGLE_STAR_TRACKS_MD5:
        print(f"Error: MD5 mismatch (got {got}, expected {SINGLE_STAR_TRACKS_MD5}); "
              f"download is corrupt. Removing {dest}.")
        os.remove(dest)
        return False
    print(f"single_star_tracks.h5 v{SINGLE_STAR_TRACKS_VERSION} installed ({size_mb:.0f} MB).")
    return True


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

def check_released_star_tables(data_dir: str) -> list:
    """Names of the released star tables (RELEASED_TABLE_MD5) that are missing from
    data_dir or differ from their released md5; empty if all match."""
    bad = []
    for name, md5 in sorted(RELEASED_TABLE_MD5.items()):
        p = os.path.join(data_dir, name)
        if not os.path.exists(p) or _md5(p) != md5:
            bad.append(name)
    return bad


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
    --stochastic-tracks: Download single_star_tracks.h5 (no Cloudy needed)
    --overwrite: with --stochastic-tracks, refetch even if present

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
    parser.add_argument('--stochastic-tracks', action='store_true',
                      help='Download single_star_tracks.h5 (stochastic sampler DB); does not need Cloudy')
    parser.add_argument('--overwrite', action='store_true',
                      help='With --stochastic-tracks: refetch even if the file is already present')
    parser.add_argument('--list', action='store_true',
                      help='List the versioned data resources of this code version and exit')
    args = parser.parse_args()
    if args.list:
        list_resources()
        return 0

    print("Starting TODDLERS data installation check...")

    # The stochastic track database lives in the package data dir, not the Cloudy data
    # dir, and fetching it must not require a Cloudy install -> handle it before that check.
    if args.stochastic_tracks:
        return 0 if download_single_star_tracks(overwrite=args.overwrite) else 1

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
            bad = check_released_star_tables(cloudy_data_dir)
            if bad:
                print("Error: these installed star tables do not have the released md5 "
                      "(RELEASED_TABLE_MD5): " + ", ".join(bad))
                return 1
            print(f"Data successfully installed in {cloudy_data_dir}")
            return 0
    
    print("Failed to download or extract data files.")
    return 1

if __name__ == '__main__':
    sys.exit(main())