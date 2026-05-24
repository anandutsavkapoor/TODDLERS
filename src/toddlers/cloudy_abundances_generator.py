from .imports import np, InterpUS

class CloudyAbundancesGenerator:
    """
    A flexible abundance generator for Cloudy simulations.
    
    This class is largely based on the functionality of nebAbundTools 
    (A part of the Byler-FSPS modules) for generating abundance sets.
    """

    def __init__(self):

        self.abundance_sets = {
            'dopita': self._load_dopita,
            'newdopita': self._load_newdopita,
            'UVbyler': self._load_UVbyler,
            'gutkin': self._load_gutkin,
            'GASS10': self._load_GASS10,
        }
        self.depletion_sets = {
            'dopita': self._load_dopita_depletion,
            'newdopita': self._load_newdopita_depletion,
            'UVbyler': self._load_UVbyler_depletion,
            'gutkin': self._load_gutkin_depletion,
            'GASS10': self._load_cloudy_default_depletion,
        }
        self.solar = {
            'dopita': 'old solar 84',
            'newdopita': 'GASS10',
            'UVbyler': 'GASS10',
            'gutkin': 'GASS10',
            'GASS10': 'GASS10',
        }

    def generate_abundance_set(self, set_name, logZbyZsun, re_z=False, **kwargs):
        """
        Generate an abundance set based on the specified parameters.

        Args:
            set_name (str): Name of the abundance set to use.
            logZbyZsun (float): Logarithm of the metallicity relative to solar.
            re_z (bool or float): Whether to renormalize abundances.
            **kwargs: Additional parameters specific to certain abundance sets.

        Returns:
            dict: A dictionary containing the generated abundances and Cloudy commands.
        """
        if set_name not in self.abundance_sets:
            allowed_sets = ', '.join(self.abundance_sets.keys())
            raise ValueError(f"Unsupported abundance set: '{set_name}'. Allowed sets are: {allowed_sets}")

        base_abundances = self.abundance_sets[set_name]()
        depletions = self.depletion_sets[set_name]()
        
        abundances = self._calculate_abundances(base_abundances, depletions, set_name, logZbyZsun, re_z, **kwargs)
        
        cloudy_commands = self.get_cloudy_abundance_string(abundances)
        
        return {
            'abundances': abundances,
            'cloudy_commands': cloudy_commands,
            'solar': self.solar[set_name]
        }

    def _calculate_abundances(self, base_abundances, depletions, set_name, logZbyZsun, re_z=False):
        """
        Calculate final abundances based on base abundances, depletions, and metallicity.
        
        Args:
            base_abundances (dict): Base (reference) abundances for each element.
            depletions (dict): Depletion factors for each element.
            set_name (str): Name of abundance set ('newdopita', 'UVbyler', 'GASS10', 'gutkin').
            logZbyZsun (float): Log of metallicity relative to solar.
            re_z (bool): Whether to renormalize abundances. Defaults to False.
        
        Returns:
            dict: Final calculated abundances for each element.
            
        Notes:
            - Special elements (He, C, N, O) use set-specific formulas
            - All other elements are scaled from base values with metallicity 
            - Depletions are applied to all elements that have depletion factors
        """
        # Copy base abundances to avoid modifying the original
        abundances = base_abundances.copy()
        
        # Calculate special element abundances based on abundance set
        special_abundances = {}
        
        if set_name in ['newdopita', 'UVbyler', 'GASS10']:
            He = self._calc_He(logZbyZsun)
            C, N, O = self._calc_CNO(logZbyZsun, base_abundances['O'], set_name)
            special_abundances.update({'He': He, 'C': C, 'N': N, 'O': O})
        elif set_name == 'gutkin':
            He = self._calc_He_gutkin(logZbyZsun)
            C, N, O = self._calc_CNO_gutkin(logZbyZsun, base_abundances['O'], base_abundances['C'])
            special_abundances.update({'He': He, 'C': C, 'N': N, 'O': O})
            
        # Calculate abundances for other elements
        special_elements = {'He', 'C', 'N', 'O'}
        for elem, base_abund in base_abundances.items():
            if elem not in special_elements:
                # Scale with metallicity
                abundances[elem] = base_abund + logZbyZsun
                
        # Update with special element abundances
        abundances.update(special_abundances)
                
        # Apply depletions to all elements
        for elem in abundances:
            if elem in depletions:
                abundances[elem] += depletions[elem]
                
        # Optional renormalization to maintain total metallicity
        if re_z:
            self._renormalize_abundances(abundances, logZbyZsun)
        
        return abundances

    def _calc_He(self, logZbyZsun):
        """
        Calculate helium abundance for standard abundance sets.
        
        Uses a linear scaling relation between helium mass fraction (Y) and 
        metallicity (Z) derived from chemical evolution models.
        
        Args:
            logZbyZsun (float): Log of metallicity relative to solar.
            
        Returns:
            float: Log of He/H abundance.
            
        Note:
            Uses the relation: He/H = 0.0737 + 0.024 * Z/Z_sun (Dopita et al. 2000).
        """
        # Calculate He/H number density ratio
        He_H = 0.0737 + (0.024 * 10.0**logZbyZsun)
        
        # Convert to log abundance
        return np.log10(He_H)

    def _calc_He_gutkin(self, logZbyZsun):
        """
        Calculate helium abundance for Gutkin abundance set.
        
        Uses helium mass fraction (Y) relation calibrated to match 
        observations of metal-poor galaxies.
        
        Args:
            logZbyZsun (float): Log of metallicity relative to solar.
            
        Returns:
            float: Log of He/H abundance.
            
        Note:
            Uses Y = 0.2485 + 1.7756 * Z and X + Y + Z = 1
            where Z = Z_☉ * (Z/Z_☉) and Z_☉ = 0.01524
        """
        Z = (10.0**logZbyZsun) * 0.01524  # Convert to mass fraction
        Y = 0.2485 + 1.7756 * Z  # Helium mass fraction
        X = 1.0 - Y - Z  # Hydrogen mass fraction
        
        # Convert mass fractions to number density ratio and then to log
        return np.log10(Y/X/4.0)

    def _calc_CNO(self, logZbyZsun, O_base, set_name):
        """
        Calculate C, N, O abundances for standard abundance sets.
        
        Args:
            logZbyZsun (float): Log of metallicity relative to solar.
            O_base (float): Base oxygen abundance.
            set_name (str): Name of abundance set.
            
        Returns:
            tuple: (C, N, O) abundances in log scale.
        """
        # Calculate oxygen abundance
        O = O_base + logZbyZsun
        logOH = O + 12.0  # Convert to 12 + log(O/H) scale
        
        if set_name in ['newdopita', 'GASS10']:
            # Define abundance relations from observations
            oxy = np.array([7.39, 7.50, 7.69, 7.99, 8.17, 8.39, 8.69, 8.80, 8.99, 9.17, 9.39])
            nit = np.array([-6.61, -6.47, -6.23, -5.79, -5.51, -5.14, -4.60, -4.40, -4.04, -3.67, -3.17])
            car = np.array([-5.58, -5.44, -5.20, -4.76, -4.48, -4.11, -3.57, -3.37, -3.01, -2.64, -2.14])
            
            # Interpolate to get C and N abundances
            C = float(InterpUS(oxy, car, k=1)(logOH))
            N = float(InterpUS(oxy, nit, k=1)(logOH))
        
        elif set_name == 'UVbyler':
            # Calculate C/O and N/O ratios using smooth functions
            logCO = -0.8 + 0.14*(logOH - 8.0) + (0.192 * np.log(1. + np.exp((logOH - 8.0)/0.2)))
            logNO = -1.5 + (0.1 * np.log(1. + np.exp((logOH - 8.3)/0.1)))
            
            # Convert to absolute abundances
            C = logCO + O
            N = logNO + O
        
        return C, N, O

    def _calc_CNO_gutkin(self, logZbyZsun, O_base, C_base):
        """
        Calculate C, N, O abundances for Gutkin abundance set.
        
        Uses a combination of primary and secondary nitrogen production
        to determine N/O ratio.
        
        Args:
            logZbyZsun (float): Log of metallicity relative to solar.
            O_base (float): Base oxygen abundance.
            C_base (float): Base carbon abundance.
            
        Returns:
            tuple: (C, N, O) abundances in log scale.
            
        Note:
            N/O ratio includes both primary (constant) and secondary 
            (metallicity-dependent) components.
        """
        # Calculate oxygen and carbon abundances
        O = O_base + logZbyZsun
        C = C_base + logZbyZsun
        
        # Calculate nitrogen using primary + secondary components
        # N/O = 10^-1.6 (primary) + 10^(2.33 + log(O/H)) (secondary)
        # Factor of 0.41 accounts for oxygen depletion
        N = np.log10((0.41 * 10.**O) * (10.**-1.6 + 10.**(2.33 + O)))
        
        return C, N, O

    def _load_GASS10(self):
        return {
            'He': -1.07, 'Li': -10.95, 'Be': -10.62, 'B': -9.3, 'C': -3.57,
            'N': -4.17, 'O': -3.31, 'F': -7.44, 'Ne': -4.07, 'Na': -5.76,
            'Mg': -4.40, 'Al': -5.55, 'Si': -4.49, 'P': -6.59, 'S': -4.88,
            'Cl': -6.50, 'Ar': -5.60, 'K': -6.97, 'Ca': -5.66, 'Sc': -8.85,
            'Ti': -7.05, 'V': -8.07, 'Cr': -6.36, 'Mn': -6.57, 'Fe': -4.50,
            'Co': -7.01, 'Ni': -5.78, 'Cu': -7.81, 'Zn': -7.44
        }

    def _load_dopita(self):
        return {
            'He': -1.01, 'C': -3.44, 'N': -3.95, 'O': -3.07, 'Ne': -3.91,
            'Mg': -4.42, 'Si': -4.45, 'S': -4.79, 'Ar': -5.44, 'Ca': -5.64,
            'Fe': -4.33, 'F': -7.52, 'Na': -5.69, 'Al': -5.53, 'P': -6.43,
            'Cl': -6.73, 'K': -6.87, 'Ti': -6.96, 'Cr': -6.32, 'Mn': -6.47,
            'Co': -7.08, 'Ni': -5.75, 'Cu': -7.73, 'Zn': -7.34
        }

    def _load_newdopita(self):
        return {
            'He': -1.01, 'C': -3.57, 'N': -4.60, 'O': -3.31, 'Ne': -4.07,
            'Na': -5.75, 'Mg': -4.40, 'Al': -5.55, 'Si': -4.49, 'S': -4.86,
            'Cl': -6.63, 'Ar': -5.60, 'Ca': -5.66, 'Fe': -4.50, 'Ni': -5.78,
            'F': -7.44, 'P': -6.59, 'K': -6.97, 'Cr': -6.36, 'Ti': -7.05,
            'Mn': -6.57, 'Co': -7.01, 'Cu': -7.81, 'Zn': -7.44
        }

    def _load_UVbyler(self):
        return {
            'He': -1.01, 'C': -3.57, 'N': -4.17, 'O': -3.31, 'Ne': -4.07,
            'Na': -5.75, 'Mg': -4.40, 'Al': -5.55, 'Si': -4.49, 'S': -4.86,
            'Cl': -6.63, 'Ar': -5.60, 'Ca': -5.66, 'Fe': -4.50, 'Ni': -5.78,
            'F': -7.44, 'P': -6.59, 'K': -6.97, 'Cr': -6.36, 'Ti': -7.05,
            'Mn': -6.57, 'Co': -7.01, 'Cu': -7.81, 'Zn': -7.44
        }

    def _load_gutkin(self):
        return {
            'He': -1.01, 'C': -3.53, 'N': -4.32, 'O': -3.17, 'F': -7.47,
            'Ne': -4.01, 'Na': -5.70, 'Mg': -4.45, 'Al': -5.56, 'Si': -4.48,
            'P': -6.57, 'S': -4.87, 'Cl': -6.53, 'Ar': -5.63, 'K': -6.92,
            'Ca': -5.67, 'Sc': -8.86, 'Ti': -7.01, 'V': -8.03, 'Cr': -6.36,
            'Mn': -6.64, 'Fe': -4.51, 'Co': -7.11, 'Ni': -5.78, 'Cu': -7.82,
            'Zn': -7.43
        }

    def _load_dopita_depletion(self):
        return {
            'C': -0.30, 'N': -0.22, 'O': -0.22, 'Ne': 0.0, 'Mg': -0.70,
            'Si': -1.0, 'S': 0.0, 'Ar': 0.0, 'Ca': -2.52, 'Fe': -2.0,
            'F': 0.0, 'Na': 0.0, 'Al': 0.0, 'P': 0.0, 'Cl': 0.0,
            'K': 0.0, 'Ti': 0.0, 'Cr': 0.0, 'Mn': 0.0, 'Co': 0.0,
            'Ni': 0.0, 'Cu': 0.0, 'Zn': 0.0
        }

    def _load_newdopita_depletion(self):
        return {
            'He': 0.00, 'C': -0.30, 'N': -0.05, 'O': -0.07, 'Ne': 0.00,
            'Na': -1.00, 'Mg': -1.08, 'Al': -1.39, 'Si': -0.81, 'S': 0.00,
            'Cl': -1.00, 'Ar': 0.00, 'Ca': -2.52, 'Fe': -1.31, 'Ni': -2.00,
            'F': 0.0, 'P': 0.0, 'K': 0.0, 'Cr': 0.0, 'Ti': 0.0,
            'Mn': 0.0, 'Co': 0.0, 'Cu': 0.0, 'Zn': 0.0
        }

    def _load_UVbyler_depletion(self):
        return self._load_newdopita_depletion()

    def _load_gutkin_depletion(self):
        return {
            'He': 0.00, 'Li': -0.8, 'C': -0.30, 'O': -0.15, 'Na': -0.60,
            'Mg': -0.70, 'Al': -1.70, 'Si': -1.00, 'Cl': -0.30, 'Ca': -2.52,
            'Fe': -2.00, 'Ni': -1.40
        }

    def _load_cloudy_default_depletion(self):
        """Table 7.8 in Hazy, given as log10(depletion value in table 7.8)"""
        return {
            'He': np.log10(1.00),    'Li': np.log10(0.16),    'Be': np.log10(0.6),     'B':  np.log10(0.13),
            'C':  np.log10(0.4),     'N':  np.log10(1.0),     'O':  np.log10(0.6),     'F':  np.log10(0.3),
            'Ne': np.log10(1.0),     'Na': np.log10(0.2),     'Mg': np.log10(0.2),     'Al': np.log10(0.01),
            'Si': np.log10(0.03),    'P':  np.log10(0.25),    'S':  np.log10(1.0),     'Cl': np.log10(0.4),
            'Ar': np.log10(1.0),     'K':  np.log10(0.3),     'Ca': np.log10(1e-4),    'Sc': np.log10(5e-3),
            'Ti': np.log10(8e-3),    'V':  np.log10(6e-3),    'Cr': np.log10(6e-3),    'Mn': np.log10(5e-2),
            'Fe': np.log10(1e-2),    'Co': np.log10(1e-2),    'Ni': np.log10(1e-2),    'Cu': np.log10(0.1),
            'Zn': np.log10(0.25)
        }

    def _renormalize_abundances(self, abundances, logZbyZsun):
        """
        Renormalize abundances to maintain the same total metallicity.
        """
        total_Z = sum(10**abund for elem, abund in abundances.items() if elem != 'He')
        target_Z = 10**logZbyZsun
        factor = target_Z / total_Z
        for elem in abundances:
            if elem != 'He':
                abundances[elem] = np.log10(10**abundances[elem] * factor)

    def get_cloudy_abundance_string(self, abundances):
        """
        Generate a Cloudy-compatible abundance string from the calculated abundances.

        Args:
            abundances (dict): Dictionary of calculated abundances.

        Returns:
            list: Cloudy-compatible abundance commands.
        """
        commands = []

        # Element abundance commands
        for elem, abund in abundances.items():
            cmd = f'abundances {elem} {abund:.2f}'
            commands.append(cmd)
        
        return commands