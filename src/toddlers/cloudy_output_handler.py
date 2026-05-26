from .imports import np, u, os, re
from .constants import *
from .imports import Dict, List, Optional, Set
from .imports import os, re, np, dataclass, logging, traceback
from .utils import is_file_stable

class CloudyOutputHandler:
    """
    Handler for Cloudy output files with optional absolute path support.

    If ``absolute_path`` is given, changes to that directory on initialization
    (recording the previous working directory in ``self.orig_dir``). It does NOT
    change back automatically; the caller is responsible for restoring the
    working directory if needed.
    """
    def __init__(self, model_prefix, time, absolute_path=None, check_file_stability=False, parse_data=True):
        """
        Initialize the CloudyOutputHandler.

        Args:
            model_prefix (str): Prefix for model files (e.g., 'shell', 'unified', 'dig')
            time (float): Simulation time in seconds
            absolute_path (str, optional): If provided, changes to this directory for all operations
            check_file_stability (bool, optional): check file stability when checking if run successful
            parse_data (bool, optional): parse output files if run successful.
        """
        self.orig_dir = None
        
        # Change to absolute_path if provided
        if absolute_path:
            self.orig_dir = os.getcwd()
            os.chdir(os.path.abspath(absolute_path))

        self.model_prefix = model_prefix
        self.time = time
        self.base_filename = f"{model_prefix}_{time/MYR_TO_SEC:.2f}"
        self.out_file = f"{self.base_filename}.out"
        self.check_file_stability = check_file_stability
        self.parse_data = parse_data
        self.success_message = "Cloudy exited OK"
        self.outputs = {}
        self.skip_lines = [('mg', 7, 9.03094)] # wavelength has to be in micron
        
        # Process outputs
        if self.check_cloudy_success(self.check_file_stability) and self.parse_data:
            self.parse_output()

    def check_input_exists(self):
        """
        Check if a COMPLETE input file exists for this model and time.

        Beyond being non-empty, the .in must contain the ``save last continuum`` command
        that every phase's input carries (it is the one save common to all of them,
        including the dissolved phase). This rejects two mid-write truncation modes that
        otherwise never self-heal because the file is non-empty:

        * 0-byte input (disk-full mid-write): Cloudy "No incident radiation field".
        * a few-line stub from a worker killed at walltime during input generation
          (title + ``table star`` + ``luminosity`` only, no density law / inner radius):
          Cloudy "Hydrogen density MUST be specified" / inner radius not set.

        Either way the caller regenerates the input rather than handing Cloudy a file it
        cannot run. The ``.in`` is a few KB, so reading it here is cheap.

        Returns:
            bool: True if a complete input file exists, False otherwise
        """
        try:
            path = self.get_file_path("in")
            if os.path.getsize(path) == 0:
                return False
            with open(path, errors="ignore") as f:
                return "save last continuum" in f.read()
        except OSError:
            return False

    def check_cloudy_success(self, print_traceback=False, validate_outputs=True):
        """
        Check if Cloudy run was successful by searching for 'Cloudy exited OK' near the end.
        Much faster than reading the entire file since success message is always at the end.

        When ``validate_outputs`` is True (default), a run that exited OK must ALSO have
        produced usable save files: the essential outputs for its phase must exist, be
        non-empty, and carry their ``#`` header line. This rejects truncated or
        header-clobbered saves (e.g. a model that "exited OK" but whose ``save continuum``
        wrote nothing, or whose ``.cum``/``.ovr`` header was blanked, when the disk filled
        mid-write). Without this, such a model is treated as complete and is skipped by
        both the resume logic and ``run_simulation``, and the STAB build silently consumes
        empty continuum / line-less HR SEDs.

        Args:
            print_traceback (bool): Whether to print debug info if check fails
            validate_outputs (bool): Also require non-empty, header-valid essential outputs

        Returns:
            bool: True if run successful, False otherwise
        """
        if not os.path.exists(self.out_file):
            #print(f"Output file {self.out_file} does not exist => Pending or failed run.")
            return False
        
        try:
            # Read last 8KB of file - success message is always near end
            chunk_size = 8192  # 8KB is usually sufficient
            with open(self.out_file, 'r') as f:
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                
                # Read only last chunk
                chunk_start = max(0, file_size - chunk_size) 
                f.seek(chunk_start)
                last_chunk = f.read()

            # Use regex pattern for robust matching
            pattern = re.compile(r"\bCloudy\s+exited\s+OK\b")
            success = bool(pattern.search(last_chunk))
                    
            if print_traceback and not success:
                print(f"Success message '{self.success_message}' not found in {self.base_filename}")
                caller = traceback.extract_stack()[-2]
                print(f"\nChecking Cloudy success from {caller.name} at line {caller.lineno}")
                print(f"Current working directory: {os.getcwd()}")

            # Check file stability if requested
            if success and self.check_file_stability:
                required_files = ['cont', 'phy', 'rad']
                for ext in required_files:
                    file_path = self.get_file_path(ext)
                    if not is_file_stable(file_path):
                        print(f"{file_path} is not stable for {self.base_filename}")
                        return False

            # Reject "exited OK" runs whose save files are empty or header-clobbered.
            if success and validate_outputs and not self._outputs_valid():
                if print_traceback:
                    print(f"Essential output missing/empty/headerless for {self.base_filename}")
                return False

            return success

        except Exception as e:
            if print_traceback:
                print(f"Error checking success for {self.out_file}: {str(e)}")
                traceback.print_exc()
            return False

    def _outputs_valid(self):
        """Essential save files exist, are non-empty, and have an intact first line.

        The downstream SED interpolant + STAB build reads, per phase:
          dissolved        -> .cont
          shell / unified  -> .cont, .ovr, .phy, .rad, .cum, .cumEmer
        A disk-full mid-write can leave a model that "exited OK" but with one of these
        truncated to 0 bytes or with its first (header) row clobbered to NUL bytes (a file
        hole) or whitespace; both make the build produce empty continuum / line-less HR
        SEDs with no error. We strip NUL and whitespace from the first line and require
        real content, plus a leading '#' for the files Cloudy gives a '#...' header
        (.cont/.ovr/.phy/.cum/.cumEmer; the line readers and parsers need that header).
        .rad can begin with a data row, so it is only required to be non-empty/intact.
        """
        if self.model_prefix == 'dissolved':
            required = ('cont',)
        else:
            required = ('cont', 'ovr', 'phy', 'rad', 'cum', 'cumEmer')
        hashed = {'cont', 'ovr', 'phy', 'cum', 'cumEmer'}
        for ext in required:
            path = self.get_file_path(ext)
            try:
                if os.path.getsize(path) == 0:
                    return False
                with open(path, 'r', errors='replace') as f:
                    first_line = f.readline()
            except OSError:
                return False
            cleaned = first_line.replace('\x00', '').strip()
            if not cleaned:
                return False
            if ext in hashed and not cleaned.startswith('#'):
                return False
        return True

    def get_file_path(self, extension):
        return f"{self.base_filename}.{extension}"

    def parse_output(self):
        """
        Parse output files with error handling.
        
        For dissolved models, only parses the continuum file.
        For all other models, parses overview, continuum, physical, and radius files.
        """
        try:
            # Determine if this is a dissolved model
            is_dissolved = self.model_prefix == 'dissolved'
            
            # Define required files based on model type
            if is_dissolved:
                required_files = {
                    'continuum': ('cont', 9)  # Only parse continuum for dissolved models
                }
            else:
                required_files = {
                    'overview': ('ovr', None),
                    'continuum': ('cont', 9),
                    'physical': ('phy', None),
                    'radius': ('rad', None)
                }
            
            for output_type, (extension, n_cols) in required_files.items():
                try:
                    self.outputs[output_type] = self._parse_file(extension, n_cols)
                except FileNotFoundError:
                    print(f"Output file .{extension} not found for {self.base_filename}")
                except Exception as e:
                    print(f"Error parsing {extension} file: {str(e)}")
                    raise
                                        
        except Exception as e:
            print(f"Error in parse_output for {self.base_filename}: {str(e)}")
            raise

    def _parse_file(self, extension, n_cols=None):
        """
        Parse Cloudy output file and return data as a dictionary.

        Args:
            extension (str): File extension (e.g., 'ovr', 'cont', 'phy').
            n_cols (int, optional): Number of columns to read. If None, read all columns.

        Returns:
            dict: Dictionary with column names as keys and numpy arrays as values.
        """
        file_path = self.get_file_path(extension)
        
        with open(file_path, 'r') as f:
            header = f.readline().strip()
        
        header_parts = header.split()

        if header_parts[0].startswith('#'):
            if extension in ['cont', 'phy']:
                # For .cont and .phy files, skip the header string next to #
                names = header_parts[1:]
                
                # Special handling for 'net trans' in continuum file
                if extension == 'cont':
                    # Create a corrected header list
                    corrected_names = []
                    i = 0
                    while i < len(names):
                        if i < len(names) - 1 and names[i] == 'net' and names[i + 1] == 'trans':
                            corrected_names.append('net_trans')
                            i += 2
                        else:
                            corrected_names.append(names[i])
                            i += 1
                    names = corrected_names
            else:
                # For all other files, use all parts including the one next to #
                names = [header_parts[0].replace("#", "")] + header_parts[1:]
        else:
            raise ValueError(f"Header does not start with '#' in {file_path}")
    
        if n_cols:
            names = names[:n_cols]
            data = np.genfromtxt(file_path, names=names, usecols=range(n_cols), skip_header=1)
        else:
            data = np.genfromtxt(file_path, names=names, skip_header=1)    
        
        if len(names) != len(data.dtype.names):
            print(f"Warning: Mismatch in number of columns for {extension} file.")
            print(f"Header names: {names}")
            print(f"Data columns: {data.dtype.names}")

        return {name: data[name] for name in data.dtype.names}

    def _parse_wavelength(self, wavelength_str):
        """Parse wavelength string to float in microns."""
        try:
            if wavelength_str.endswith('A'):
                return float(wavelength_str[:-1]) * 1e-4  # Angstroms to microns
            elif wavelength_str.endswith('m'):
                return float(wavelength_str[:-1])   # microns to microns
            else:
                # Assume Angstroms if no unit specified
                return float(wavelength_str) * 1e-4 # Angstroms to microns
        except ValueError:
            return None

    def get_density_structure(self):
        return self.outputs['physical']['nH']

    def get_radial_structure(self):
        return self.outputs['radius']['radius'], self.outputs['radius']['depth']

    def get_temperature_structure(self):
        return self.outputs['physical']['Te']

    def get_ionization_structure(self):
        ovr_data = self.outputs['overview']
        return {
            'H': {
                'H2': ovr_data['2H_2H'],
                'H0': ovr_data['HI'],
                'H+': ovr_data['HII']
            },
            'He': {
                'He0': ovr_data['HeI'],
                'He+': ovr_data['HeII'],
                'He++': ovr_data['HeIII']
            },
            'C': {
                'CO': ovr_data['COC'],
                'C0': ovr_data['C1'],
                'C+': ovr_data['C2'],
                'C++': ovr_data['C3'],
                'C3+': ovr_data['C4']
            },
            'O': {
                'O0': ovr_data['O1'],
                'O+': ovr_data['O2'],
                'O++': ovr_data['O3'],
                'O3+': ovr_data['O4'],
                'O4+': ovr_data['O5'],
                'O5+': ovr_data['O6']
            }
        }

    def get_continuum(self):
        """Get continua from .cont files. See Hazy 1 for definitions.

        Returns:
            dict containing wavelength and continua (incident, transmitted,
            observed, observed_no_lines) in Cloudy units. Also includes
            'nebular_unatt' (the unattenuated diffuse nebular continuum from the
            patched ".diffContUnatt" save) when that file is present and was
            written by a patched Cloudy; the key is absent otherwise.
        """
        cont_data = self.outputs['continuum']
        out = {
            'nu': cont_data['nu'], # col 1
            'incident': cont_data['incident'], # col 2
            'transmitted': cont_data['trans'], # col 3 => transmitted incident, no effect of covering fraction
            'observed': cont_data['net_trans'], # col 5 =>  observed continua, affected by covering fraction
            'observed_no_lines': cont_data['net_trans'] - cont_data['outlin'] # col 9 - col 5 =>  observed continua without line emission
        }
        # Unattenuated diffuse nebular continuum (gas-only: free-bound, free-free,
        # two-photon), from the patched Cloudy ".diffContUnatt" save. Same Cloudy
        # frequency mesh and units as the .cont columns, so it is index-aligned with
        # 'incident'. None when the file is absent or the Cloudy build is unpatched.
        neb = self.get_nebular_continuum_unatt()
        if neb is not None:
            out['nebular_unatt'] = neb
        return out

    def get_nebular_continuum_unatt(self):
        """Read the unattenuated diffuse nebular continuum (".diffContUnatt", col 2).

        Returns the ``DiffContUnatt`` column aligned to the continuum grid, or ``None``
        if the file is missing or was written by an unpatched Cloudy (whose
        ``save diffuse continuum unattenuated`` silently degrades to the standard
        diffuse continuum, header ``ConEmitLocal ...``). See cloudy_patches/.
        """
        path = self.get_file_path('diffContUnatt')
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            header = f.readline()
        if 'DiffContUnatt' not in header:
            print(f"Warning: {path} lacks DiffContUnatt column (unpatched Cloudy?); "
                  f"nebular continuum not added.")
            return None
        neb = np.genfromtxt(path, skip_header=1, usecols=(1,))   # col 2 = DiffContUnatt
        cont_n = self.outputs['continuum']['nu'].size
        if neb.size != cont_n:
            print(f"Warning: .diffContUnatt rows ({neb.size}) != continuum rows "
                  f"({cont_n}); nebular continuum not added.")
            return None
        return neb

    def get_line_luminosities(self, use_emergent=False):
        """
        Get line luminosities from cumulative luminosity files.
        
        Handles both ionic and molecular lines from Cloudy output. For ionic lines,
        returns a dictionary with ion information. For molecular lines, returns the
        species name directly.
        
        The method skips certain lines defined in self.skip_lines list. Each entry
        in self.skip_lines is a tuple of (element, ion, wavelength) where:

        - element: lowercase element name (e.g., 'mg' for magnesium)
        - ion: ionization level as integer
        - wavelength: wavelength in microns

        Currently skipped lines: [('mg', 7, 9.03094)]

        Args:
            use_emergent (bool): Whether to use emergent line intensities.
                            Defaults to False for intrinsic intensities.
        
        Returns:
            dict: Dictionary mapping line identifiers to luminosities where:

                - For ions: key is (element, ion, wavelength / micron)
                - For molecules: key is (species, 0, wavelength / micron)

                Values are the line luminosities in linear scale.
                Wavelength is in microns based on TODDLERS input files.
        """

        if self.model_prefix == 'dissolved':
            return self._get_minimum_line_luminosities()
            
        # Determine which file to read
        suffix = 'cumEmer' if use_emergent else 'cum'
        file_path = self.get_file_path(suffix)
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Read and parse line luminosities from file
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            # Get headers (first line after #)
            header_line = next(line for line in lines if line.startswith('#'))
            headers = header_line.strip('#').strip().split('\t')
            
            # Get last row for total luminosities
            last_row = lines[-1].strip().split('\t')
            
            # Create line luminosity dictionary
            line_lums = {}
            for header, value in zip(headers[1:], last_row[1:]):  # Skip depth column
                try:
                    # Parse header into components
                    parts = header.strip().split()
                    if len(parts) < 2:
                        continue
                        
                    wavelength = self._parse_wavelength(parts[-1]) # micron
                    if wavelength is None:
                        continue

                    # Check if this is a molecular line
                    is_molecular = any(species in ['CO', 'CN', 'CS', 'HCN', 'HNC', 'NH3'] 
                                    for species in parts[:-1])
                    
                    # Create line identifier tuple based on type
                    try:
                        if is_molecular or parts[0].lower() == 'blnd':
                            line_id = (parts[0], 0, wavelength)
                        else:
                            ion_level = int(parts[1]) if len(parts) > 1 else 1
                            check_id = (parts[0].lower(), ion_level, wavelength)
                            
                            # Skip if line is in skip_lines list
                            if check_id in self.skip_lines:
                                continue
                                
                            line_id = (parts[0], ion_level, wavelength)
                    except ValueError:
                        continue

                    try:
                        line_lums[line_id] = float(value)
                    except ValueError:
                        continue
                        
                except (ValueError, IndexError):
                    continue
                    
            if not line_lums:
                raise ValueError(
                    f"No line luminosities parsed from {file_path} (empty or "
                    f"header-clobbered file?); refusing to return empty lines silently.")
            return line_lums

        except Exception as e:
            # Do NOT swallow: a missing or corrupt .cum/.cumEmer must fail loudly rather
            # than return None and let the HR STAB build produce line-less SEDs silently.
            print(f"Error reading line luminosities from {file_path}: {str(e)}")
            raise

    def _get_minimum_line_luminosities(self):
        """
        Get dictionary of minimum line luminosities for dissolved model.
        Used for dissolved model.
        
        Returns:
            dict: Dictionary mapping line identifiers to minimum luminosities
            Wavelength is in microns based on TODDLERS input files.
        """
        min_value = 1e-99  # Small but non-zero value
        
        # Read line list to get all possible lines
        line_list = self._read_line_list()
        
        # Create dictionary with minimum values
        line_lums = {}
        for line_info in line_list:
            try:
                wavelength = self._parse_wavelength(line_info['wavelength']) # micron
                if wavelength is not None:
                    # Create immutable tuple key
                    line_id = (line_info['element'], line_info['ion'], wavelength)
                    line_lums[line_id] = min_value
            except (KeyError, ValueError):
                continue
                
        return line_lums

    def _read_line_list(self):
        """
        Read TODDLERS line list file.
        
        Handles both ionic and molecular lines. For ionic lines, returns dictionary 
        with ion information. For molecular lines, returns the species name directly.
        
        Skips certain lines defined in self.skip_lines list.
        
        Returns:
            list: List of dictionaries containing line information
            Wavelengths are returned in microns
        """
        src_dir = os.path.dirname(os.path.abspath(__file__))
        line_list_path = os.path.join(src_dir, 'database', 'lines_list', 
                                    'cloudy_lines_TODDLERS_v2.dat')
                
        try:
            lines = []
            molecular_species = {'CO', 'CN', 'CS', 'HCN', 'HNC', 'NH3'}
            
            with open(line_list_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.strip().startswith('#'):
                        try:
                            # Parse line into components
                            parts = line.strip().split()
                            if len(parts) < 2:
                                continue
                                
                            # Get wavelength in microns for comparison with skip list                            
                            wavelength = self._parse_wavelength(parts[-1]) # wavelength float in micron
                            wavelength_str = str(wavelength) + 'm'
                            
                            # Check if this is a molecular line
                            is_molecular = any(species in molecular_species 
                                            for species in parts[:-1])
                            
                            # Create line information based on type
                            if is_molecular or parts[0].lower() == 'blnd':
                                lines.append({
                                    'element': parts[0],
                                    'ion': 0,
                                    'wavelength': wavelength_str
                                })
                            else:
                                ion_level = int(parts[1]) if len(parts) > 1 else 1
                                check_id = (parts[0].lower(), ion_level, wavelength)
                                
                                # Skip if line is in skip_lines list
                                if check_id in self.skip_lines:
                                    continue
                                    
                                lines.append({
                                    'element': parts[0],
                                    'ion': ion_level,
                                    'wavelength': wavelength_str
                                })
                                
                        except (ValueError, IndexError):
                            continue
                            
            return lines
            
        except Exception as e:
            print(f"Error reading line list: {str(e)}")
            traceback.print_exc()
            return []

    def get_wavelength_grid(self):
        """
        Get wavelength grid from continuum file.
        
        Returns:
            np.ndarray: Array of wavelengths in meters
        """
        if not 'continuum' in self.outputs:
            return np.array([])
            
        return self.outputs['continuum']['nu'] # check units

    def calculate_escape_fraction(self):
        cont_data = self.outputs['continuum']
        energy_eV = (const.h * const.c / (cont_data['nu'] * u.micron)).to(u.eV).value

        E_H = 13.6
        ionizing_mask = (energy_eV >= E_H) & (energy_eV < 4 * E_H)
        
        incident = cont_data['incident'][ionizing_mask]
        transmitted = cont_data['trans'][ionizing_mask] # always considers full cover, so this works
        
        total_transmitted = np.sum(transmitted)
        total_incident = np.sum(incident)
        
        # Calculate the escape fraction, avoiding division by zero
        escape_fraction = total_transmitted / total_incident
        
        return escape_fraction

    def calculate_Q(self, kind='transmitted'):
        cont_data = self.get_continuum()

        # Convert wavelength to energy
        wavelength = cont_data['nu'] * u.micron
        energy = (const.h * const.c / wavelength).to(u.eV)
        
        # Get the spectral luminosity
        nu_Lnu = cont_data[kind] * u.erg / u.s

        # Define ionizing photon energy range
        E_H = 13.6 * u.eV
        ionizing_mask = (energy >= E_H) & (energy < 4 * E_H)
        
        # Calculate frequency for ionizing photons
        nu = (const.c / wavelength[ionizing_mask]).to(u.Hz)
        
        # Calculate Lnu (spectral luminosity per unit frequency)
        Lnu = nu_Lnu[ionizing_mask] / nu

        # Calculate number of photons per unit frequency
        n_photons = Lnu / (const.h * nu)

        # Integrate to get total number of ionizing photons per second
        Q = np.trapz(n_photons.value, nu.value)

        return Q  # This is in units of photons/s

class ReadCloudyMainOutput:
    """
    A parser for Cloudy main output files (.out).
    """

    def __init__(self, file_name, keyword, output_dir="cloudy_main_output_parsed"):
        self.keyword = keyword
        self.file_name = file_name
        self.output_dir = output_dir
        self.all_headers = ["general properties", "continua", "RRC", "grains", "H-like iso-sequence", "He-like iso-sequence",
                            "extra Lyman", "database lines", "level 1 lines", "recombination", "level2 lines", 
                            "hyperfine structure", "inner shell", "molecules", "bands", "miscellaneous"]
        self.selected_headers = ["continua", "database lines", "recombination", "H-like iso-sequence"]
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    def _get_last_iteration_block(self, keyword):
        """Get the block for the last iteration."""
        with open(self.file_name, 'r') as file:
            lines = file.readlines()
        block = []
        in_block = False
        for i, line in enumerate(lines):
            if keyword in line and keyword not in ' '.join(lines[i+1:]):
                in_block = True
                continue
            elif in_block and line.strip() == "":
                break
            elif in_block:
                block.append(line)
        return block

    def _unravel_block(self, block):
        """Unravel the block."""
        unraveled = []
        section_width = len(block[0]) // 3
        for section_start in range(0, len(block[0]), section_width):
            for line in block:
                section_end = section_start + section_width
                section_content = line[section_start:section_end].strip()
                if "..............." in section_content:
                    unraveled.append(section_content.split("...............")[0].strip() + "...............")
                else:
                    unraveled.append(section_content)
        return unraveled

    def extract_tables(self, unraveled_data, selected_headers=None):
        """Extract tables from the unraveled data, optionally keeping only selected headers."""
        tables = {}
        current_header = None
        
        for line in unraveled_data:
            if "..............." in line:
                current_header = line.replace("...............", "").strip()
                # Only add the header if it's in selected_headers or if selected_headers is None
                if selected_headers is None or current_header in selected_headers:
                    tables[current_header] = []
                else:
                    current_header = None  # Reset if not in selected headers
            elif current_header:
                tables[current_header].append(line)
        
        return tables

    def parse(self, keyword="Intrinsic line intensities"):
        """Main parsing function."""
        block = self._get_last_iteration_block(keyword)
        unraveled_data = self._unravel_block(block)
        return self.extract_tables(unraveled_data, selected_headers=self.selected_headers)

    def _sort_by_wavelength(self, table):
        """Sort the table based on the wavelength."""
        header = table[0] if any(isinstance(i, str) and "..............." in i for i in table) else None
        data = table[1:] if header else table

        def extract_wavelength(line):
            match = re.search(r'(\d+\.\d+)([A|m])', line)
            if match:
                value, unit = float(match.group(1)), match.group(2)
                if unit == "A":
                    return value
                elif unit == "m":
                    return value * 1e4
            return 0

        sorted_data = sorted(data, key=extract_wavelength)
        return [header] + sorted_data if header else sorted_data

    def save_tables(self, tables):
        """Save each table to a separate file."""
        for header, lines in tables.items():
            filename = os.path.join(self.output_dir, header.replace(" ", "_") + "_keyword_" + self.keyword + ".txt")
            with open(filename, 'w') as file:
                file.write(header + "\n")
                file.write("\n".join(lines))

    def parse_and_save(self, sort_data=True):
        """Main parsing function that also saves the data to files."""
        tables = self.parse(self.keyword)
        if sort_data:
            for header, lines in tables.items():
                if any("A" in line or "m" in line for line in lines):
                    tables[header] = self._sort_by_wavelength(lines)
            self.save_tables(tables)
        else:
            self.save_tables(tables)

    def __str__(self):
        """Output the processed content as a formatted string."""
        tables = self.parse()
        output = []
        for header, lines in tables.items():
            output.append(header)
            output.extend(lines)
            output.append("")
        return "\n".join(output)

    def consolidate_outputBlocks_and_save(self, keys=["H-like iso-sequence", "He-like iso-sequence", "level2 lines", "database lines"],
                             exclude=["Ca A", "Ca B", "Blnd", "CaBo", "Cool", "Crst", "CION", "Q(H)", "Dest", "Pho+"],
                             fileprefix="consolidated_outputBlocks", sort_data=True):
        """Consolidate lines from tables based on specific keys and save to a file."""

        filename = fileprefix + '_' + self.keyword + '.txt'
        tables = self.parse(self.keyword)
        consolidated_data = []
        for key in keys:
            if key in tables:
                for line in tables[key]:
                    if not any(line.startswith(ex) for ex in exclude):
                        consolidated_data.append(line)
            else:
                print(f"Key '{key}' not found in the tables!")
        if sort_data:
            consolidated_data = self._sort_by_wavelength(consolidated_data)
        with open(os.path.join(self.output_dir, filename), 'w') as file:
            file.write("\n".join(consolidated_data))


    def wavelength_value(wavelength_string):
        """Extract the numeric value of the wavelength from a string."""
        try:
            # Extract the numeric value and unit (either 'm' or 'A')
            value, unit = re.findall(r'([\d.]+)([mA])', wavelength_string)[0]
            value = float(value)
            
            # Convert microns to Angstroms
            if unit == 'm':
                value *= 1e4
            return value
        except IndexError:
            return 0

    def _get_outputFile_data(self):
        """Load the consolidated data from the .out file into a dictionary."""
        filename = os.path.join(self.output_dir, "consolidated.txt")
        with open(filename, 'r') as file:
            lines = file.readlines()

        data_dict = {}
        for line in lines:
            parts = line.split()
            # Combine name and wavelength for the key
            combined_key = f"{parts[0]} {parts[1]} {parts[2]}"
            data_dict[combined_key] = parts[-2:]

        return data_dict

@dataclass
class LineTracker:
    """
    Tracks line appearances across different models, tables, and blocks.
    """
    dig_only: Dict[str, Dict[str, Set[str]]] = None  # table_type -> block -> line_ids
    inner_only: Dict[str, Dict[str, Set[str]]] = None
    common: Dict[str, Dict[str, Set[str]]] = None
    logger: logging.Logger = None
    
    def __post_init__(self):
        self.dig_only = {}
        self.inner_only = {}
        self.common = {}

    def _log_or_print(self, message: str, level: str = 'info'):
        """
        Log or print a message depending on logger availability.
        
        Args:
            message (str): Message to output
            level (str): Log level ('info' or 'warning')
        """
        if self.logger:
            if level == 'warning':
                self.logger.warning(message)
            else:
                self.logger.info(message)
        else:
            if level == 'warning':
                print(f"WARNING: {message}")
            else:
                print(message)

    def _ensure_sets_exist(self, table_type: str, block: str):
        """Ensure set structures exist for given table type and block."""
        for collection in [self.dig_only, self.inner_only, self.common]:
            if table_type not in collection:
                collection[table_type] = {}
            if block not in collection[table_type]:
                collection[table_type][block] = set()

    def track_line(self, line_id: str, is_dig: bool, table_type: str, block: str):
        """Track a line's appearance with table and block information."""
        self._ensure_sets_exist(table_type, block)
        
        if is_dig:
            if line_id in self.inner_only[table_type][block]:
                self.inner_only[table_type][block].remove(line_id)
                self.common[table_type][block].add(line_id)
            elif line_id not in self.common[table_type][block]:
                self.dig_only[table_type][block].add(line_id)
        else:
            if line_id in self.dig_only[table_type][block]:
                self.dig_only[table_type][block].remove(line_id)
                self.common[table_type][block].add(line_id)
            elif line_id not in self.common[table_type][block]:
                self.inner_only[table_type][block].add(line_id)

    def log_statistics(self):
        """Log detailed statistics by table type and block."""
        total_dig_only = sum(len(lines) for table in self.dig_only.values() 
                           for lines in table.values())
        total_inner_only = sum(len(lines) for table in self.inner_only.values() 
                             for lines in table.values())
        total_common = sum(len(lines) for table in self.common.values() 
                         for lines in table.values())

        self._log_or_print("\nOverall Line Statistics:")
        self._log_or_print(f"Lines in both models: {total_common}")
        self._log_or_print(f"Lines only in DIG: {total_dig_only}")
        self._log_or_print(f"Lines only in inner model: {total_inner_only}")

        self._log_or_print("\nDetailed Statistics by Table Type:")
        for table_type in sorted(self.common.keys()):
            self._log_or_print(f"\n{table_type}:")
            for block in sorted(self.common[table_type].keys()):
                common = len(self.common[table_type][block])
                dig_only = len(self.dig_only[table_type][block])
                inner_only = len(self.inner_only[table_type][block])
                if common + dig_only + inner_only > 0:  # Only show if there are lines
                    self._log_or_print(f"  {block}:")
                    self._log_or_print(f"    Both models: {common}")
                    self._log_or_print(f"    DIG only: {dig_only}")
                    self._log_or_print(f"    Inner only: {inner_only}")

"""
This module provides functionality for consolidating Cloudy output tables from different models
(shell, unified, and DIG) based on cloud conditions. It handles the combination of line and 
continuum data by adding their linear values and converting back to logarithmic form.

The module is designed to work with the Cloudy output parser and specifically handles:
- Continuum entries (filtered for specific types)
- Database lines
- Recombination lines

All values in the tables are relative to Hbeta line intensity.
"""

@dataclass
class TableEntry:
    """
    Represents a single entry in a Cloudy output table.
    
    Attributes:
        name (str): Name of the species/line
        wavelength (str): Wavelength value with unit
        log_luminosity (float): Log of the line/continuum luminosity relative to Hbeta
    """
    name: str
    wavelength: str
    log_luminosity: float
    
    @property
    def linear_value(self) -> float:
        """Convert log value to linear for combining entries."""
        return 10**self.log_luminosity

    @property
    def unique_id(self) -> str:
        """Create a unique identifier combining name and wavelength."""
        return f"{self.name}_{self.wavelength}"
    
    @classmethod
    def from_line(cls, line: str) -> Optional['TableEntry']:
        """
        Create a TableEntry from a line in the Cloudy output.
        
        The line format is:
        "N  3     452.227A   35.884    0.0031"
        "CH    180.425m   35.884    0.0031"
        "PAHC   180.425m   35.884    0.0031"
        
        Args:
            line (str): Line from Cloudy output table
            
        Returns:
            Optional[TableEntry]: Parsed entry or None if parsing fails
        """     
        exclude = ["Ca A", "Ca B", "Blnd", "CaBo", "Cool", "Crst", "CION",
                "Q(H)", "Dest", "Pho+", "LA X", "Strk", "Hrst", "M1"]
        
        if any(term in line for term in exclude):
            return None
        
        try:
            parts = line.strip().split()
            
            # Handle special case for continuum entries (like PAHC, IRAC, etc.)
            if len(parts[0]) > 2:  # for continuum lines like => PAHC   180.425m
                name = parts[0]
                wavelength = parts[1]
            else:
                try: # for lines like => N  3     452.227A 
                    element = parts[0].strip()
                    ion_level = int(parts[1])
                    name = f"{element}{ion_level}"
                    wavelength = parts[2]
                except ValueError: # for lines like => CH    180.425m 
                    name = parts[0]
                    wavelength = parts[1]
            
            log_luminosity = float(parts[-2])  # Use log value directly from file
            return cls(name, wavelength, log_luminosity)
        
        except (IndexError, ValueError) as e:
            print(f"Error parsing line: {line.strip()}")
            print(f"Error details: {str(e)}")
            return None

    def combine_with(self, other: 'TableEntry') -> 'TableEntry':
        """
        Combine this entry with another entry by adding their linear values.
        
        Args:
            other (TableEntry): Another entry to combine with
            
        Returns:
            TableEntry: New entry with combined value in log form
        """
        total = self.linear_value + other.linear_value
        return TableEntry(
            name=self.name,
            wavelength=self.wavelength,
            log_luminosity=np.log10(total)
        )
    
class CloudyTableConsolidator:
    """
    Consolidates Cloudy output tables from different models based on cloud conditions.
    
    This class handles the combination of output tables from shell, unified, and dig models,
    combining their values by converting to linear space, adding, and converting back to 
    logarithmic form.
    
    Attributes:
        continua_filter (set): Set of continuum entries to include in consolidation
        parser_class: Class used to parse Cloudy output files
        output_dir (str): Directory for consolidated output files
    """

    # Filter for relevant continuum entries
    CONTINUA_FILTER = {
        'PAHC', 'PAH', 'IRAC', 'MIRa', 'NMIR', 'MIRb', 'MIPS', 
        'F12', 'F25', 'F60', 'F100', 'PAC1', 'PAC2', 'PAC3', 
        'FIR', 'SPR1', 'SPR2', 'SPR3', 'TFIR', 'TIR'
    }

    TABLE_CONFIGS = {
        "continua": [ "Emergent line intensities"],  # Only emergent for continua
        "database lines": ["Intrinsic line intensities",  "Emergent line intensities"],  # Both blocks for lines
        "recombination": ["Intrinsic line intensities",  "Emergent line intensities"],   # Both blocks for recombination
        "H-like iso-sequence": ["Intrinsic line intensities",  "Emergent line intensities"] 
    }

    def __init__(self, parser_class, output_dir: str = "consolidated_output"):
        """
        Initialize the CloudyTableConsolidator.

        Args:
            parser_class: Class used to parse Cloudy output files
            output_dir (str): Directory for output files
        """
        self.parser_class = parser_class
        self.output_dir = output_dir
        self.line_tracker = LineTracker()
        os.makedirs(output_dir, exist_ok=True)

    def _parse_model_tables(self, model_file: str) -> Dict[str, Dict[str, List[TableEntry]]]:
        """
        Parse tables from a single model output file.

        Args:
            model_file (str): Path to model output file

        Returns:
            Dict[str, Dict[str, List[TableEntry]]]: Dictionary of tables by type and block
        """
        parsed_tables = {}
        
        for table_type, blocks in self.TABLE_CONFIGS.items():
            parsed_tables[table_type] = {}
            
            for block in blocks:
                parser = self.parser_class(model_file, keyword=block)
                tables = parser.parse(keyword=block)
                
                if table_type in tables:
                    entries = []
                    for line in tables[table_type]:
                        entry = TableEntry.from_line(line)
                        if entry:
                            if table_type == "continua" and entry.name not in self.CONTINUA_FILTER:
                                continue
                            entries.append(entry)
                    parsed_tables[table_type][block] = entries
        
        return parsed_tables

    def _combine_entries(self, entries: List[TableEntry]) -> TableEntry:
        """
        Combine multiple entries of the same line/continuum.
        
        Args:
            entries (List[TableEntry]): List of entries to combine
            
        Returns:
            TableEntry: Combined entry
        """
        if not entries:
            raise ValueError("Cannot combine empty list of entries")
            
        result = entries[0]
        for entry in entries[1:]:
            result = result.combine_with(entry)
        return result

    def _combine_tables(self, tables_list: List[Dict[str, Dict[str, List[TableEntry]]]], model_files: List[str]) -> Dict[str, Dict[str, List[TableEntry]]]:
        """
        Combine multiple tables, tracking lines by model source.
        
        Args:
            tables_list: List of tables to combine
            model_files: List of Cloudy output files these tables came from
        """
        combined = {}
        is_dig_table = ['dig_' in os.path.basename(f) for f in model_files]
        
        for table_type, blocks in self.TABLE_CONFIGS.items():
            combined[table_type] = {}
            
            for block in blocks:
                entries_dict = {}
                
                for i, tables in enumerate(tables_list):
                    if table_type in tables and block in tables[table_type]:
                        for entry in tables[table_type][block]:
                            # Track with table and block information
                            self.line_tracker.track_line(
                                entry.unique_id, 
                                is_dig_table[i],
                                table_type,
                                block
                            )
                            
                            if entry.unique_id in entries_dict:
                                entries_dict[entry.unique_id].append(entry)
                            else:
                                entries_dict[entry.unique_id] = [entry]
                
                combined_entries = []
                for entries in entries_dict.values():
                    combined_entries.append(self._combine_entries(entries))
                
                combined[table_type][block] = combined_entries
        
        self.line_tracker.log_statistics()
        return combined

    def _convert_wavelength_to_microns(self, wavelength: str) -> float:
        """
        Convert wavelength string to microns for sorting.
        
        Args:
            wavelength (str): Wavelength string with unit (e.g., "1234.56A", "12.34m", "1.25558c")
                - 'A': Angstroms
                - 'm': microns
                - 'c': centimeters
                
        Returns:
            float: Wavelength in microns
        """
        value = float(wavelength.rstrip('mAc'))
        if wavelength.endswith('A'):
            return value / 1e4  # Angstroms to microns
        elif wavelength.endswith('c'):
            return value * 1e4  # centimeters to microns
        return value  # Already in microns if ends with 'm'
    
    def consolidate_tables(self, 
                        timestep: float,
                        shell_file: Optional[str], 
                        unified_file: Optional[str], 
                        dig_file: Optional[str], 
                        is_within_cloud: bool,
                        add_dig: bool = True) -> Dict[str, Dict[str, List[TableEntry]]]:
        """
        Consolidate tables based on model conditions.
        
        Args:
            timestep (float): Current simulation timestep.
            shell_file (str, optional): Path to shell model output.
            unified_file (str, optional): Path to unified model output.
            dig_file (str, optional): Path to DIG model output.
            is_within_cloud (bool): Whether the shell is within the cloud.
            add_dig (bool): Whether to include DIG in consolidation. Defaults to True.
        
        Returns:
            dict: Consolidated tables.
        
        Raises:
            ValueError: If no valid model files are provided.
        """
        tables_to_combine = []
        model_files = []  # Track which files the tables came from
        
        # Check required files exist before proceeding
        if is_within_cloud and unified_file and os.path.exists(unified_file):
            tables_to_combine.append(self._parse_model_tables(unified_file))
            model_files.append(unified_file)
        elif shell_file and os.path.exists(shell_file):
            tables_to_combine.append(self._parse_model_tables(shell_file))
            model_files.append(shell_file)
                    
        if add_dig and dig_file and os.path.exists(dig_file):
            tables_to_combine.append(self._parse_model_tables(dig_file))
            model_files.append(dig_file)
        
        if not tables_to_combine:
            self.logger.warning(f"No valid model files found for timestep {timestep}")
            return {}
        
        self.line_tracker = LineTracker()  # Reset tracker
        
        # Pass both tables and their source files
        combined_tables = self._combine_tables(tables_to_combine, model_files)
        self._save_tables(combined_tables, timestep)
        return combined_tables

    def _save_tables(self, tables: Dict[str, Dict[str, List[TableEntry]]], timestep: float):
        """
        Save consolidated tables to files with source information.

        Args:
            tables: Tables to save
            timestep: Current simulation timestep
        """
        for table_type, blocks_data in tables.items():
            for block, entries in blocks_data.items():
                # Create filename including block type if table has multiple blocks
                if len(self.TABLE_CONFIGS[table_type]) > 1:  
                    filename = os.path.join(self.output_dir, 
                        f"{table_type.replace(' ', '_')}_{block.lower()}_{timestep:.2f}.txt")
                else:
                    filename = os.path.join(self.output_dir, 
                        f"{table_type.replace(' ', '_')}_{timestep:.2f}.txt")
                
                with open(filename, 'w') as f:
                    f.write(f"# Consolidated {table_type} ({block}) for timestep {timestep} Myr\n")
                    f.write("# Name            Wavelength    log(L)      Source\n")
                    f.write("#" + "-"*65 + "\n")
                    
                    sorted_entries = sorted(entries, 
                                        key=lambda x: self._convert_wavelength_to_microns(x.wavelength))
                    
                    for entry in sorted_entries:
                        # Determine source based on line tracking
                        line_id = entry.unique_id
                        if line_id in self.line_tracker.common[table_type][block]:
                            source = "both"
                        elif line_id in self.line_tracker.dig_only[table_type][block]:
                            source = "dig"
                        else:
                            source = "inner"
                        
                        # Write entry with source column
                        f.write(f"{entry.name:15} {entry.wavelength:10} {entry.log_luminosity:10.3f} {source:>10}\n")
