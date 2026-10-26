# stellar_feedback.py

import re

from .imports import np, os, pickle, interp1d
from .constants import *
from .exceptions import ConfigurationError

#: Step [Myr] of the feedback precalculation (and of the numerical derivatives and the
#: constant-SFR convolution). The evolution's maximum solver steps must stay below it, so
#: a supernova point event (a hat one step wide either side) is never stepped over.
PRECALC_DT_MYR = 0.025


KNOWN_TEMPLATES = ('BPASS', 'SB99', 'pySB99')


def _pre_drawn_refusal(label, note=""):
    """The refusal of a stochastic table named as input (unified or 2.0)."""
    return ConfigurationError(
        f"{label}: a pre-drawn stochastic table is not read as input. Stochastic runs draw their "
        f"stars at run time, each generation from its own stream (a recollapse forms a new "
        f"population), so no file drawn in advance can serve a run.{note} Use "
        f"stochastic=StochasticSpec(IMFSpec(name, m_low, m_up), base_seed=...) with imf=None; the "
        f"same base_seed reproduces a run.")


# The 2.0 naming of pre-drawn stochastic pySB99 sets (stochastic_seed<N>_M<mass>, e.g.
# stochastic_seed17_M1000), which 2.1.0 does not ship or read.
_LEGACY_PRE_DRAWN_NAME = re.compile(r'stochastic_seed\d+_M\d+$')


def refuse_legacy_pre_drawn_name(template, imf):
    """Refuse a 2.0 pre-drawn stochastic IMF name by name, before any file is read (2.1.0
    does not ship its files, so a file-based check would give a bare FileNotFoundError)."""
    if template == 'pySB99' and isinstance(imf, str) and _LEGACY_PRE_DRAWN_NAME.match(imf):
        raise _pre_drawn_refusal(f"IMF '{imf}'", note=" This is the name of a TODDLERS 2.0 pre-drawn set; "
                                 "2.1.0 does not ship or read "
                                 "these sets.")


def _file_names(template, imf, star_type):
    """(family, imf, star_type) of the table file names; see toddlers.feedback_tables.resolve_names."""
    from .feedback_tables import resolve_names
    return resolve_names(template, imf, star_type)


def feedback_table_path(template, imf=None, star_type=None):
    """Path of the unified feedback table (``<template>_<imf>[_<star_type>]_feedback.h5`` in
    the data dir) for these names; the file may not exist."""
    from ._paths import get_data_dir
    from .feedback_tables import table_filename
    return os.path.join(get_data_dir(), table_filename(*_file_names(template, imf, star_type)))


def feedback_table_files(template, imf, star_type=None):
    """Paths of a TODDLERS 2.0 table (main .obj, Lyman-Werner .obj); deprecated layout, read by
    toddlers.feedback_tables.legacy."""
    from .feedback_tables.legacy import legacy_files
    return legacy_files(template, imf, star_type)


def feedback_table_md5(template, imf=None, star_type=None):
    """Fingerprint of the feedback table a StellarFeedback of this kind reads (keys the
    evolution track cache): the md5 of the unified HDF5 table if present, else the 2.0 md5
    over the two .obj files."""
    p = feedback_table_path(template, imf, star_type)
    if os.path.exists(p):
        from .feedback_tables import file_md5
        return file_md5(p)
    family, imf, star_type = _file_names(template, imf, star_type)
    from .feedback_tables.legacy import legacy_md5
    return legacy_md5(family, imf, star_type)


class StellarFeedback:
    """
    A class to calculate various stellar feedback quantities for different star formation scenarios.

    This class can handle single burst, constant star formation rate, and multiple generations
    of star formation. It reads the template's unified feedback table
    (toddlers.feedback_tables, one HDF5 file per table) and interpolates it in age.

    Attributes:
        template (str): The stellar evolution model template to use.
        Z (float): Metallicity of the stellar population.
        M_cl_init (float): Initial cloud mass in solar masses.
        eta_sf (float): Star formation efficiency.
        t_list_collapse (list): List of collapse times for multiple generations.
        mode (str): Star formation mode ('burst' or 'constant_sfr').
        formation_timescale (float): Timescale for star formation in constant SFR mode.
        imf (str): Initial Mass Function to use. Options depend on the template.
        star_type (str): For BPASS models, specifies 'sin' (single) or 'bin' (binary) stars.
        feedback_table (FeedbackTable or None): the table read (None for a 2.0 .obj table or a
            run-time stochastic population, whose per-generation tables are in generation_tables).
    """

    def __init__(self, template, Z, M_cl_init, eta_sf, t_list_collapse, mode='burst', 
                 formation_timescale=None, imf=None, star_type=None, stochastic=None,
                 populations=None):
        """
        Initialize the StellarFeedback object.

        Args:
            template (str): The stellar population template: 'BPASS', 'SB99', 'pySB99', or any
                name a unified feedback table was written under (toddlers.feedback_tables);
                the table's metadata, not the name, defines it.
            Z (float): Metallicity of the stellar population (a node of the table).
            M_cl_init (float): Initial cloud mass in grams.
            eta_sf (float): Star formation efficiency.
            t_list_collapse (list): List of collapse times for multiple generations.
            mode (str, optional): Star formation mode ('burst' or 'constant_sfr'). Defaults to 'burst'.
            formation_timescale (float, optional): Timescale for star formation in constant SFR mode.
                Required if mode is 'constant_sfr'.
            imf (str, optional): IMF name the table was written under. Defaults: BPASS
                'chab100', SB99 'kroupa100'; required for pySB99 and any other template.
            star_type (str, optional): BPASS 'sin' or 'bin' (default 'bin'); not used for SB99
                and pySB99 (single stars); for another template, the star type its table was
                written under, if any.
            stochastic (StochasticSpec or dict, optional): pySB99 only, with imf=None. Each
                generation i draws its own stars from the named IMF over its full range, to
                the mass eta_sf (1 - eta_sf)^i M_cl_init, with a stream derived from
                (base seed, i), and gets the feedback of exactly those stars
                (toddlers.pysb99.stochastic.population). In constant_sfr mode the stars are
                born sequentially at the generation's SFR and the table is not convolved again.
            populations (list of GenerationPopulation, optional): generations already drawn
                (e.g. by the previous StellarFeedback of the same run, or read from an evolution
                output); each must equal a fresh draw from its seed. The rest are drawn.

        Raises:
            ValueError: If an invalid mode, IMF, star_type or metallicity is given, if a table's
                units or metadata do not match, or if formation_timescale is missing for
                constant SFR mode.
            FileNotFoundError: If no table exists for (template, imf, star_type).

        Note:
            Every table is normalized per 1e6 Msun of its reference population (the full IMF,
            a custom star list, or a drawn realization), and feedback is scaled by
            M_stellar = eta_sf * M_cl for each generation.
        """
        self.template = template
        self.Z = Z
        self.M_cl_init = M_cl_init / M_SUN
        self.eta_sf = eta_sf
        self.t_list_collapse = t_list_collapse
        self.mode = mode
        self.formation_timescale = formation_timescale
        self.generations = len(self.t_list_collapse) + 1
        self.n_collapse = self.generations - 1
        self.derivative_dt = PRECALC_DT_MYR * MYR_TO_SEC
        self.convolution_dt = PRECALC_DT_MYR * MYR_TO_SEC
        self.stochastic = None
        self.populations = None
        self.feedback_table = None
        self.table_metadata = None
        self._ages_included = [False] * self.generations
        self._stellar_mass_override = None
        if stochastic is not None:
            from .pysb99.stochastic.population import StochasticSpec
            if template != 'pySB99':
                raise ValueError("stochastic populations are drawn from the pySB99 single-star database; "
                                 "use template='pySB99'")
            if imf is not None:
                raise ValueError("With stochastic=..., pass imf=None: the IMF is named in the stochastic spec")
            self.stochastic = StochasticSpec.coerce(stochastic)

        # Validate mode
        if mode not in ['burst', 'constant_sfr']:
            raise ValueError("Mode must be either 'burst' or 'constant_sfr'")
        if mode == 'constant_sfr' and formation_timescale is None:
            raise ValueError("Formation timescale must be provided for constant SFR mode")

        # Names (defaults for the built-in templates); the table itself defines Z nodes,
        # units and the population.
        if self.template == 'pySB99' and self.stochastic is not None:
            self._setup_pysb99_stochastic(star_type)
        else:
            refuse_legacy_pre_drawn_name(self.template, imf)
            self._setup_names(imf, star_type)

        # Load interpolants and calculate stellar properties
        if self.stochastic is not None:
            self._build_stochastic_generations(populations)
        else:
            self.load_interpolants()
        self.calculate_stellar_mass()
        self.calculate_generation_start_times()
        self.precalculate_feedback_quantities()

    def _setup_names(self, imf, star_type):
        """IMF and star type of a deterministic table: built-in defaults (BPASS chab100/bin,
        SB99 kroupa100), single stars for SB99 and pySB99, imf required otherwise."""
        if self.template.startswith('BPASS'):
            self.imf = imf or "chab100"
            self.star_type = star_type or "bin"
            if self.star_type not in ['sin', 'bin']:
                raise ValueError("BPASS star_type must be either 'sin' or 'bin'")
        elif self.template.startswith('SB99'):
            if star_type == 'bin':
                raise ValueError("SB99 does not support binary stars. Use star_type='sin' or None.")
            self.imf = imf or "kroupa100"
            self.star_type = 'sin'
        elif self.template == 'pySB99':
            if imf is None:
                raise ValueError("IMF must be specified for pySB99 template. "
                                 "This should match the IMF name used when generating interpolants.")
            if star_type == 'bin':
                raise ValueError("pySB99 does not support binary stars. Use star_type='sin' or None.")
            self.imf = imf
            self.star_type = 'sin'
            if self.Z < 0 or self.Z > 0.1:
                raise ValueError(f"Metallicity Z={self.Z} seems unreasonable. "
                                 f"Expected range: 0 <= Z <= 0.1 (0 to ~5 Z_solar)")
        else:
            if imf is None:
                raise ValueError(f"Unsupported template: {self.template}. Built-in templates are "
                                 f"'BPASS', 'SB99', 'pySB99'; any other needs imf= and a table "
                                 f"written by toddlers.feedback_tables.")
            self.imf = imf
            self.star_type = star_type

    def _file_star_type(self):
        """Star type as it appears in the table file name (BPASS and other templates only)."""
        return None if (self.template.startswith('SB99') or self.template == 'pySB99') else self.star_type

    def _setup_pysb99_stochastic(self, star_type):
        """Setup for run-time stochastic pySB99 populations (one draw per generation)."""
        from .pysb99.stochastic.population import metallicity_label
        if star_type == 'bin':
            raise ValueError("pySB99 does not support binary stars. Use star_type='sin' or None.")
        self.star_type = 'sin'
        self.imf = self.stochastic.label
        metallicity_label(self.Z)        # Z must be a node of the single-star database

    def _target_masses(self):
        """Stellar mass each generation forms, eta_sf (1 - eta_sf)^i M_cl_init [Msun]."""
        return [self.eta_sf * self.M_cl_init * ((1 - self.eta_sf)**i) for i in range(self.generations)]

    def _build_stochastic_generations(self, populations):
        """Draw (or reuse) every generation's stars and build its feedback table."""
        from .pysb99.stochastic.population import (draw_generation, build_generation_table,
                                                    check_redraw)
        spec = self.stochastic
        sf_mode = self.mode
        tau = None if self.mode == 'burst' else self.formation_timescale / MYR_TO_SEC
        targets = self._target_masses()
        given = list(populations or [])
        if len(given) > self.generations:
            raise ConfigurationError(f"{len(given)} populations given for {self.generations} generation(s)")
        pops = []
        for i in range(self.generations):
            if i < len(given):
                p = given[i]
                if p.generation_index != i or p.base_seed != spec.base_seed:
                    raise ConfigurationError(f"population {i} belongs to generation {p.generation_index}, "
                                     f"seed {p.base_seed}; expected generation {i}, seed {spec.base_seed}")
                if not np.isclose(p.target_mass_msun, targets[i], rtol=1e-12, atol=0.0):
                    raise ConfigurationError(f"population {i} was drawn to {p.target_mass_msun:.6e} Msun; "
                                     f"this run's generation {i} forms {targets[i]:.6e} Msun")
                if p.sf_mode != sf_mode or not np.isclose(p.metallicity_Z, self.Z, rtol=1e-6, atol=0.0):
                    raise ConfigurationError(f"population {i} was drawn for ({p.sf_mode}, Z={p.metallicity_Z}), "
                                     f"this run is ({sf_mode}, Z={self.Z})")
                check_redraw(p, spec)
            else:
                p = draw_generation(spec, i, targets[i], sf_mode, self.Z, tau_sf_myr=tau)
            pops.append(p)
        self.populations = pops
        self.generation_tables = [build_generation_table(p, spec) for p in pops]
        self._ages_included = [t.ages_included for t in self.generation_tables]
        self._stellar_mass_override = [p.drawn_mass_msun for p in pops]
        # In stochastic mode the interpolant handles are quantity names, resolved per
        # generation by _interp_for.
        self.L_mech_interpolant = 'L_mech'
        self.F_ram_interpolant = 'F_ram'
        self.L_bolo_interpolant = 'L_bolo'
        self.L_ion_interpolant = 'L_ion'
        self.Q_ion_interpolant = 'Q_ion'
        self.L_LyW_interpolant = 'L_LyW'
        self.F_ram_dot_interpolant = None
        self.L_mech_dot_interpolant = None
        self._gen_interpolants = [{k: t.interpolant(k) for k in t.log} for t in self.generation_tables]
        for t in self.generation_tables:
            if t.feedback_table is not None:
                self._check_point_event_grid(t.feedback_table)
        print(f"Built stochastic feedback for {self.generations} generation(s) "
              f"(IMF {spec.imf.label}, base seed {spec.base_seed}, {spec.interpolation_mode})")

    def _check_point_event_grid(self, table):
        """A point-event table (SN channel linear in t) delivers each event's energy
        exactly through the precalculation only if its time steps are whole multiples of
        derivative_dt; refuse any other grid. A pySB99 table with supernovae and no recorded
        deposition mode (written before 2.1.0) is read in log10, with a warning."""
        conv = table.metadata.get('conventions', {})
        if 'sn_time_deposition_mode' not in conv:
            if conv.get('include_supernovae'):
                import warnings
                warnings.warn(f"{table.metadata.get('name')}: no sn_time_deposition_mode recorded; "
                              f"L_mech and F_ram are interpolated in log10 (supernovae written as "
                              f"point events lose energy between nodes). Rebuild the table with "
                              f"this version's pySB99 generator.", UserWarning)
            return
        if not table.sn_point_events:
            return
        step = self.derivative_dt / MYR_TO_SEC
        ratio = np.diff(table.t) / step
        if not np.allclose(ratio, np.round(ratio), rtol=0.0, atol=1e-6) or np.any(np.round(ratio) < 1):
            raise ConfigurationError(
                f"{table.metadata.get('name')}: supernova point events need table time steps that "
                f"are whole multiples of {step:g} Myr (the feedback precalculation step); this "
                f"table's steps are {np.unique(np.round(np.diff(table.t), 9))[:5]} Myr")

    def _interp_for(self, generation, interpolant):
        """The interpolant of one generation: its own table in stochastic mode, the shared
        table otherwise."""
        if self.stochastic is None:
            return interpolant
        return self._gen_interpolants[generation][interpolant]

    def _validate_metallicity(self, Z_min, Z_max, Z_values):
        """
        Validate metallicity against allowed range and discrete values.
        
        Args:
            Z_min (float): Minimum allowed metallicity
            Z_max (float): Maximum allowed metallicity
            Z_values (np.ndarray): Array of pre-computed metallicity values
        """
        if not (Z_min <= self.Z <= Z_max):
            raise ValueError(f"Metallicity (Z={self.Z}) is outside the allowed range [{Z_min}, {Z_max}]")
        match = np.isclose(Z_values, self.Z, rtol=1e-6, atol=0.0)
        if not match.any():
            raise ValueError(f"Metallicity Z={self.Z} is not a node of the {self.template} grid "
                             f"{list(Z_values)}. Use one of these values; templates are not interpolated in Z.")
        self.Z = Z_values[match][0]

    def _validate_pysb99_metallicity(self, interpolant):
        """
        Validate metallicity for pySB99 against the interpolant's own Z axis.

        The same rule as SB99/BPASS: Z must be a node of the grid the interpolants
        were generated on (no interpolation in Z). Interp2DLinear clamps to its edges,
        so probing it never fails; the nodes are read from the grid instead.

        Args:
            interpolant: Interp2DLinear over (time [Myr], Z)

        Raises:
            ValueError: If Z is not a node of the interpolant's Z axis
        """
        Z_nodes = np.unique(np.asarray(interpolant._y, dtype=float))
        match = np.isclose(Z_nodes, self.Z, rtol=1e-6, atol=0.0)
        if not match.any():
            raise ValueError(
                f"Metallicity Z={self.Z} is not a node of the pySB99 grid for IMF "
                f"'{self.imf}' {list(Z_nodes)}. Use one of these values; templates are not "
                f"interpolated in Z."
            )
        self.Z = Z_nodes[match][0]


    def load_interpolants(self):
        """
        Read the feedback table and set the interpolants the class uses (log10 of L_mech,
        F_ram, L_bol, L_ion, Q_HI and L_LyW per 1e6 Msun, callable as f(t_myr, Z)).

        The unified HDF5 table (``<template>_<imf>[_<star_type>]_feedback.h5`` in the data
        dir) is read through toddlers.feedback_tables (units checked against the schema, Z
        nodes from the file); routing is by the file and its metadata, for any template
        name. Without it, a TODDLERS 2.0 ``.obj`` table of a built-in template is read by the
        deprecated loader (FutureWarning).

        Raises:
            FileNotFoundError: If neither table exists.
            ValueError: On a table that fails its checks, or a metallicity off its Z nodes.
        """
        from ._paths import get_data_dir
        from .feedback_tables import read_table_file, find_table, table_filename, check_table_names
        family, imf, star_type = _file_names(self.template, self.imf, self._file_star_type())
        path = find_table(family, imf, star_type)
        if path is not None:
            table = read_table_file(path)
            check_table_names(table, family, imf, star_type)
            if table.population is not None:
                raise _pre_drawn_refusal(os.path.basename(path))
            self._validate_metallicity(float(table.Z.min()), float(table.Z.max()), table.Z)
            self.feedback_table = table
            self.table_metadata = table.metadata
            self._check_point_event_grid(table)
            self.L_mech_interpolant = table.feedback_interpolant('L_mech')
            self.F_ram_interpolant = table.feedback_interpolant('F_ram')
            self.L_bolo_interpolant = table.log_interpolant('L_bol')
            self.L_ion_interpolant = table.log_interpolant('L_ion')
            self.Q_ion_interpolant = table.log_interpolant('Q_HI')
            self.L_LyW_interpolant = table.log_interpolant('L_LyW')
            # not stored in 2.1 (the evolution differentiates F_ram and L_mech on the fly)
            self.F_ram_dot_interpolant = None
            self.L_mech_dot_interpolant = None
            print(f"Successfully loaded feedback table {os.path.basename(path)} "
                  f"(template {self.template}, IMF {self.imf}, star type {self.star_type})")
            return
        if family not in KNOWN_TEMPLATES:
            raise ValueError(f"Unsupported template: {self.template}. No feedback table "
                             f"{table_filename(family, imf, star_type)} in {get_data_dir()}; built-in "
                             f"templates are 'BPASS', 'SB99', 'pySB99'.")
        self._load_legacy(family, imf, star_type)

    def _load_legacy(self, family, imf, star_type):
        """The TODDLERS 2.0 .obj table (deprecated, read-only)."""
        from ._paths import get_data_dir
        from .feedback_tables.legacy import load_legacy_obj, legacy_files
        from .feedback_tables import table_filename
        main_file, lw_file = legacy_files(family, imf, star_type)
        file_prefix = os.path.basename(main_file)[:-len('.obj')]
        try:
            loaded = load_legacy_obj(family, imf, star_type)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"No feedback table for template {self.template}, IMF {self.imf}"
                + (f", star type {star_type}" if star_type else "") + ".\n"
                f"Expected {table_filename(family, imf, star_type)} (or the 2.0 file {e.filename}) in "
                f"{get_data_dir()}.\n"
                + ("For pySB99, generate the table with toddlers.pysb99.generate_pysb99_interpolants "
                   "under this IMF name." if self.template == 'pySB99' else ""))
        self.L_mech_interpolant = loaded['L_mech']
        self.F_ram_interpolant = loaded['F_ram']
        self.L_bolo_interpolant = loaded['L_bolo']
        self.L_ion_interpolant = loaded['L_ion']
        self.Q_ion_interpolant = loaded['Q_ion']
        self.F_ram_dot_interpolant = loaded['F_ram_dot']
        self.L_mech_dot_interpolant = loaded['L_mech_dot']
        self.L_LyW_interpolant = loaded['L_LyW']

        # Refuse non-finite nodes (e.g. log10(0) = -inf): they turn into NaN feedback or
        # a failed run far from their cause.
        check = {'L_mech': self.L_mech_interpolant, 'F_ram': self.F_ram_interpolant,
                 'L_bolo': self.L_bolo_interpolant, 'L_ion': self.L_ion_interpolant,
                 'Q_ion': self.Q_ion_interpolant, 'F_ram_dot': self.F_ram_dot_interpolant,
                 'L_mech_dot': self.L_mech_dot_interpolant, 'L_LyW': self.L_LyW_interpolant}
        for name, interp in check.items():
            nodes = getattr(interp, '_z', None)
            if nodes is not None and not np.all(np.isfinite(nodes)):
                bad = np.argwhere(~np.isfinite(np.asarray(nodes)))
                raise ValueError(f"{file_prefix}: {name} has {len(bad)} non-finite node(s), first at "
                                 f"(Z index, time index) {tuple(bad[0])}. Floor the quantity before "
                                 f"taking log10 when building the table.")

        # Z must be a node of the table's own axis (no interpolation in Z)
        if self.template == 'pySB99':
            self._validate_pysb99_metallicity(self.Q_ion_interpolant)
            self._check_legacy_pre_drawn_table(main_file)
        else:
            Z_nodes = np.unique(np.asarray(self.Q_ion_interpolant._y, dtype=float))
            self._validate_metallicity(float(Z_nodes.min()), float(Z_nodes.max()), Z_nodes)

        print(f"Successfully loaded interpolation data for {self.template} "
              f"(IMF: {self.imf}, Star Type: {self.star_type if self.template.startswith('BPASS') else 'sin'})"
              f" [2.0 .obj, deprecated]")

    def _check_legacy_pre_drawn_table(self, main_file):
        """2.0 pre-drawn stochastic pySB99 tables (a 'stochastic*' name without metadata, or
        metadata marking it stochastic) are refused: stochastic runs draw at run time."""
        import json
        meta_path = main_file[:-len('.obj')] + '_metadata.json'
        meta = None
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        if (meta is None and self.imf.startswith('stochastic')) or (meta or {}).get('stochastic', False):
            raise _pre_drawn_refusal(os.path.basename(main_file),
                                     note=" 2.1.0 does not read the 2.0 pre-drawn stochastic sets.")


    def calculate_stellar_mass(self):
        """
        Calculate the stellar mass for each generation and the final cloud mass.

        This method populates the M_stellar_list attribute with the stellar mass formed
        in each generation and calculates the final cloud mass after all star formation events.
        """
        self.M_stellar_list = np.empty(self.generations)
        for i in range(self.generations):
            M_cl_i = self.M_cl_init * ((1 - self.eta_sf)**i)
            self.M_stellar_list[i] = self.eta_sf * M_cl_i
        if self._stellar_mass_override is not None:
            # stochastic: each generation's stellar mass is the mass actually drawn (within
            # one star of eta_sf M_cl_i); the gas bookkeeping (M_cl) stays analytic.
            self.M_stellar_list = np.array(self._stellar_mass_override, dtype=float)
        self.M_cl = ((1 - self.eta_sf)**self.generations) * self.M_cl_init
        self.M_stellar = np.sum(self.M_stellar_list) # Total stellar mass
        self.print_mass_distribution(is_recollapse=(self.generations > 1)) # Print the mass distribution after calculation

    def get_stellar_mass_cgs(self):
        """Return the total stellar mass in CGS units (grams)."""
        return self.M_stellar * M_SUN

    def get_cloud_mass_cgs(self):
        """Return the current cloud mass in CGS units (grams)."""
        return self.M_cl * M_SUN

    def calculate_generation_start_times(self):
        """
        Calculate the start times for each generation of star formation.

        This method creates a list of start times (t_start_list) for each generation,
        using 0 as the start time for the first generation and the provided collapse
        times for subsequent generations.
        """
        self.t_start_list = [0] + self.t_list_collapse

    def calculate_derivative(self, t, quantity_func):
        """
        Calculate the numerical derivative of a quantity at a given time.

        Args:
            t (float): The time at which to calculate the derivative.
            quantity_func (callable): A function that returns the quantity value given a time.

        Returns:
            float: The numerical derivative of the quantity at time t.
        """
        q1 = quantity_func(t)
        q2 = quantity_func(t + self.derivative_dt)
        return (q2 - q1) / self.derivative_dt

    def get_feedback_quantity(self, t, interpolant, generation=None, is_derivative=False):
        """
        Get a feedback quantity at a given time, optionally for a specific generation or as a derivative.

        Args:
            t (float): The time at which to calculate the feedback quantity.
            interpolant (callable): The interpolation function for the desired quantity.
            generation (int, optional): The specific generation to calculate for. If None, calculates for all generations.
            is_derivative (bool, optional): Whether to calculate the derivative of the quantity.

        Returns:
            float or np.ndarray: The feedback quantity (or its derivative) at time t.
            Returns a scalar float if the result is a single value, otherwise returns a numpy array.
        """
        if is_derivative:
            if generation is not None:
                raise ValueError("Derivative calculation is not supported for individual generations.")
            result = self.calculate_derivative(t, lambda x: self.get_feedback_quantity(x, interpolant, None, False))
        elif self.mode == 'burst':
            result = self._get_quantity_burst(t, interpolant, generation)
        elif self.mode == 'constant_sfr':
            result = self._get_quantity_constant_sfr(t, interpolant, generation)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # Sanitize the result
        if isinstance(result, (list, tuple, np.ndarray)):
            result = np.array(result)
            return result.item() if result.size == 1 else result
        elif isinstance(result, (int, float, np.number)):
            return float(result)
        else:
            raise ValueError(f"Unexpected result type: {type(result)}")

    def _get_quantity_burst(self, t, interpolant, generation=None):
        """
        Calculate a feedback quantity for the burst star formation mode.

        Args:
            t (float): The time at which to calculate the feedback quantity.
            interpolant (callable): The interpolation function for the desired quantity.
            generation (int, optional): The specific generation to calculate for. If None, calculates for all generations.

        Returns:
            float: The total feedback quantity at time t for the burst mode.
        """
        if generation is not None:
            # Calculate for a specific generation, t should be relative in this case
            t_myr = t / MYR_TO_SEC
            interp_result = self._interp_for(generation, interpolant)(t_myr, self.Z)
            return 10**interp_result * (self.M_stellar_list[generation] / M_LIB)
        else:
            # Calculate for all generations
            total_quantity = 0
            for i in range(self.generations):
                t_start = self.t_start_list[i]
                if t >= t_start:
                    t_myr = (t - t_start) / MYR_TO_SEC
                    interp_result = self._interp_for(i, interpolant)(t_myr, self.Z)
                    total_quantity += 10**interp_result * (self.M_stellar_list[i] / M_LIB)
            return total_quantity


    def _get_quantity_constant_sfr(self, t, interpolant, generation=None):
        """
        Calculate a feedback quantity for the constant star formation rate mode.

        Args:
            t (float): The time at which to calculate the feedback quantity.
            interpolant (callable): The interpolation function for the desired quantity.
            generation (int, optional): The specific generation to calculate for. If None, calculates for all generations.

        Returns:
            float: The total feedback quantity at time t for the constant SFR mode.
        """
        if generation is not None and self._ages_included[generation]:
            # the table already holds the generation's birth times: read it like a burst
            return self._get_quantity_burst(t, interpolant, generation)
        if generation is not None:
            # Calculate for a specific generation, t should be relative in this case
            t_myr = t / MYR_TO_SEC
            formation_time_myr = self.formation_timescale / MYR_TO_SEC

            sfr = self.M_stellar_list[generation] / self.formation_timescale

            impulse_times = np.arange(0, min(t_myr, formation_time_myr), self.convolution_dt / MYR_TO_SEC)
            dm = sfr * self.convolution_dt

            responses = []
            for tau in impulse_times:
                delay = t_myr - tau
                responses.append(10**interpolant(delay, self.Z) * dm)

            result = np.sum(responses) if responses else 0
            return result / M_LIB
        else:
            # Calculate for all generations
            total_quantity = 0
            for i in range(self.generations):
                t_start = self.t_start_list[i]
                t_rel = t - t_start
                
                if t_rel <= 0:
                    continue

                if self._ages_included[i]:
                    total_quantity += self._get_quantity_burst(t_rel, interpolant, i) * M_LIB
                    continue
                
                t_rel_myr = t_rel / MYR_TO_SEC
                formation_time_myr = self.formation_timescale / MYR_TO_SEC

                sfr = self.M_stellar_list[i] / self.formation_timescale

                impulse_times = np.arange(0, min(t_rel_myr, formation_time_myr), self.convolution_dt / MYR_TO_SEC)

                dm = sfr * self.convolution_dt

                responses = []
                for tau in impulse_times:
                    delay = t_rel_myr - tau
                    responses.append(10**interpolant(delay, self.Z) * dm)

                result = np.sum(responses) if responses else 0
                total_quantity += result

            return total_quantity / M_LIB


    def precalculate_feedback_quantities(self):
        """
        Precalculate feedback quantities for efficiency.

        This method precalculates feedback quantities for a range of times, storing
        the results to avoid redundant calculations during runtime.
        """
        self.time_range = np.arange(0, 50 * MYR_TO_SEC, self.derivative_dt)
        
        # Create arrays to hold the calculated values
        L_mech_values = np.zeros_like(self.time_range)
        F_ram_values = np.zeros_like(self.time_range)
        L_bolo_values = np.zeros_like(self.time_range)
        L_ion_values = np.zeros_like(self.time_range)
        Q_ion_values = np.zeros_like(self.time_range)
        L_LyW_values = np.zeros_like(self.time_range)
        
        # Fill the arrays with values calculated using the feedback quantity methods
        for i, t in enumerate(self.time_range):
            L_mech_values[i] = self.get_feedback_quantity(t, self.L_mech_interpolant)
            F_ram_values[i] = self.get_feedback_quantity(t, self.F_ram_interpolant)
            L_bolo_values[i] = self.get_feedback_quantity(t, self.L_bolo_interpolant)
            L_ion_values[i] = self.get_feedback_quantity(t, self.L_ion_interpolant)
            Q_ion_values[i] = self.get_feedback_quantity(t, self.Q_ion_interpolant)
            L_LyW_values[i] = self.get_feedback_quantity(t, self.L_LyW_interpolant)
        
        # Apply floor to prevent cubic interpolation artifacts from producing negative values
        # Physical quantities must be non-negative
        FLOOR = 1e-50
        L_mech_values = np.maximum(L_mech_values, FLOOR)
        F_ram_values = np.maximum(F_ram_values, FLOOR)
        L_bolo_values = np.maximum(L_bolo_values, FLOOR)
        L_ion_values = np.maximum(L_ion_values, FLOOR)
        Q_ion_values = np.maximum(Q_ion_values, FLOOR)
        L_LyW_values = np.maximum(L_LyW_values, FLOOR)
        
        # Create interpolants for each quantity using the time range and calculated values
        # Use linear interpolation (matching SB99) to prevent cubic artifacts and negative values
        def make_positive_interp(values, name):
            """Create linear interpolant that never returns negative values."""
            base_interp = interp1d(self.time_range, values, kind='linear', fill_value='extrapolate')
            def safe_interp(t):
                val = base_interp(t)
                return np.maximum(val, 1e-50)  # Floor at 1e-50
            return safe_interp
        
        # the sample arrays behind L_mech and F_ram, for their exact derivatives (_precalc_slope)
        self._precalc_values = {'L_mech': L_mech_values, 'F_ram': F_ram_values}
        self.precalculated_quantities = {
            'L_mech': make_positive_interp(L_mech_values, 'L_mech'),
            'F_ram': make_positive_interp(F_ram_values, 'F_ram'),
            'L_bolo': make_positive_interp(L_bolo_values, 'L_bolo'),
            'L_ion': make_positive_interp(L_ion_values, 'L_ion'),
            'Q_ion': make_positive_interp(Q_ion_values, 'Q_ion'),
            'L_LyW': make_positive_interp(L_LyW_values, 'L_LyW')
        }

    def get_precalculated_quantity(self, t, quantity_name):
        """
        Retrieve a precalculated feedback quantity at a given time.

        Args:
            t (float): The time at which to retrieve the precalculated quantity.
            quantity_name (str): The name of the feedback quantity.

        Returns:
            float: The precalculated feedback quantity at time t.
        """
        return self.precalculated_quantities[quantity_name](t)

    def get_mechanical_luminosity(self, t):
        """
        Get the mechanical luminosity at a given time.

        Args:
            t (float): The time at which to calculate the mechanical luminosity.

        Returns:
            float: The mechanical luminosity in erg/s.
        """
        return self.get_precalculated_quantity(t, 'L_mech')

    def get_ram_force(self, t):
        """
        Get the ram force at a given time.

        Args:
            t (float): The time at which to calculate the ram force.

        Returns:
            float: The ram force in dynes.
        """
        return self.get_precalculated_quantity(t, 'F_ram')

    def get_bolometric_luminosity(self, t):
        """
        Get the bolometric luminosity at a given time.

        Args:
            t (float): The time at which to calculate the bolometric luminosity.

        Returns:
            float: The bolometric luminosity in erg/s.
        """
        return self.get_precalculated_quantity(t, 'L_bolo')

    def get_ionizing_luminosity(self, t):
        """
        Get the ionizing luminosity at a given time.

        Args:
            t (float): The time at which to calculate the ionizing luminosity.

        Returns:
            float: The ionizing luminosity in erg/s.
        """
        return self.get_precalculated_quantity(t, 'L_ion')

    def get_ionizing_photon_rate(self, t):
        """
        Get the ionizing photon rate at a given time.

        Args:
            t (float): The time at which to calculate the ionizing photon rate.

        Returns:
            float: The ionizing photon rate in photons/s.
        """
        return self.get_precalculated_quantity(t, 'Q_ion')

    @property
    def _sn_point_events(self):
        """True if a table in use holds point-event supernovae."""
        tables = [self.feedback_table] if self.feedback_table is not None else []
        tables += [g.feedback_table for g in (getattr(self, 'generation_tables', None) or [])
                   if g.feedback_table is not None]
        return any(t.sn_point_events for t in tables)

    def _precalc_slope(self, t, quantity):
        """Exact derivative of the precalculated (piecewise-linear) L_mech or F_ram at time t:
        the slope of the segment holding t (the forward segment at a node)."""
        v = self._precalc_values[quantity]
        dt = self.derivative_dt
        k = int(np.clip(np.floor(float(t) / dt + 1e-9), 0, len(v) - 2))
        return (v[k + 1] - v[k]) / dt

    def get_ram_force_derivative(self, t):
        """
        Get the derivative of the ram force at a given time.

        For point-event supernova tables it is the slope of
        the precalculated F_ram the evolution uses, so the two agree at every time; otherwise
        a forward difference of the table interpolant over derivative_dt.

        Args:
            t (float): The time at which to calculate the ram force derivative.

        Returns:
            float: The ram force derivative in dynes/s.
        """
        if self._sn_point_events:
            return self._precalc_slope(t, 'F_ram')
        return self.calculate_derivative(t, lambda x: self.get_feedback_quantity(x, self.F_ram_interpolant))

    def get_mechanical_luminosity_derivative(self, t):
        """
        Get the derivative of the mechanical luminosity at a given time (for point-event
        supernova tables the slope of the precalculated L_mech, as get_ram_force_derivative).

        Args:
            t (float): The time at which to calculate the mechanical luminosity derivative.

        Returns:
            float: The mechanical luminosity derivative in erg/s^2.
        """
        if self._sn_point_events:
            return self._precalc_slope(t, 'L_mech')
        return self.calculate_derivative(t, lambda x: self.get_feedback_quantity(x, self.L_mech_interpolant))

    def get_lyman_werner_specific_luminosity(self, t):
        """
        Get the Lyman-Werner specific luminosity at a given time.

        Args:
            t (float): The time at which to calculate the Lyman-Werner specific luminosity.

        Returns:
            float: The Lyman-Werner specific luminosity in erg/s/Hz.
        """
        return self.get_precalculated_quantity(t, 'L_LyW')

    def get_feedback_data_per_generation(self, t):
        """Get feedback data for all generations at time t."""
        feedback_data = {
            'Q_i': [],
            'L_i': [],
            'L_n': [],
            'L_mech': [],
            'F_ram': [],
        }
        
        for gen in range(self.generations):
            t_start = self.t_start_list[gen]
            if t >= t_start:
                t_rel = t - t_start # very important, the methods being called below don't do this
                feedback_data['Q_i'].append(self.get_feedback_quantity(t_rel, self.Q_ion_interpolant, generation=gen))
                feedback_data['L_i'].append(self.get_feedback_quantity(t_rel, self.L_ion_interpolant, generation=gen))
                L_bolo = self.get_feedback_quantity(t_rel, self.L_bolo_interpolant, generation=gen)
                feedback_data['L_n'].append(L_bolo - feedback_data['L_i'][-1])
                feedback_data['L_mech'].append(self.get_feedback_quantity(t_rel, self.L_mech_interpolant, generation=gen))
                feedback_data['F_ram'].append(self.get_feedback_quantity(t_rel, self.F_ram_interpolant, generation=gen))
            else:
                for key in ['Q_i', 'L_i', 'L_n', 'L_mech', 'F_ram']:
                    feedback_data[key].append(0)

        return feedback_data

    def print_mass_distribution(self, is_recollapse=False):
        """
        Print the current mass distribution of the stellar generations and remaining cloud mass.
        
        Args:
            is_recollapse (bool): If True, indicates this is a recollapse event.
        """
        if is_recollapse:
            print("\nRecollapse occurred! Updated Stellar Mass Distribution:")
        else:
            print("\nInitial Stellar Mass Distribution:")
        
        print("=" * 60)
        print(f"Initial Cloud Mass: {self.M_cl_init:.2e} M☉")
        print(f"Star Formation Efficiency: {self.eta_sf:.2%}")
        print(f"Stellar Population Model: {self.template}")
        print(f"IMF: {self.imf}")
        if hasattr(self, 'star_type'):
            print(f"Star Type: {self.star_type}")
        print(f"Cluster Formation Mode: {self.mode}")
        if self.mode == 'constant_sfr':
            print(f"Formation Timescale: {self.formation_timescale/MYR_TO_SEC:.2f} Myr")
        print(f"Number of Generations: {len(self.M_stellar_list)}")
        print("-" * 60)
        for i, mass in enumerate(self.M_stellar_list, 1):
            fraction = mass / self.M_cl_init
            print(f"Generation {i:<2}: {mass:.2e} M☉ ({fraction:.2%} of initial cloud)")
        print("-" * 60)
        print(f"Total Stellar Mass: {self.M_stellar:.2e} M☉ ({self.M_stellar/self.M_cl_init:.2%} of initial cloud)")
        print(f"Remaining Cloud Mass: {self.M_cl:.2e} M☉ ({self.M_cl/self.M_cl_init:.2%} of initial cloud)")
        print("=" * 60)

