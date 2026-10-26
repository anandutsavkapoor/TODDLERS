"""
stochastic/population.py
========================

Run-time stochastic stellar populations: one independent IMF draw per star-forming
generation.

A generation i of a cloud forms the stellar mass M_i = eps (1 - eps)^i M_cl,init. Its
stars are drawn from an explicitly named IMF over its FULL mass range (stop-after: stars
are drawn until their mass reaches M_i, the crossing star is kept), with a random
stream derived reproducibly from (base seed, i). Stars below the lowest mass of the
single-star track grid count toward the mass and stay in the star list but give no
feedback. Feedback tables are normalized per 1e6 Msun of the full drawn mass, so the
StellarFeedback scaling by the drawn mass returns exactly the sum over the drawn stars.

Star-formation modes
--------------------
burst
    every star is born at the start of the generation.
constant_sfr
    stars are born in draw order at SFR = M_i / tau_SF; star k is born at
    t_k = (mass drawn before k + m_k / 2) / SFR (midpoint convention), so the mass
    formed by time t equals SFR * t to within half a star. The resulting table
    already contains the age spread (``ages_included``); StellarFeedback must not
    convolve it again.

Interpolation in mass
---------------------
'snap' (default): each star takes the nearest grid track, interpolated linearly in age in
linear space (the rule of stochastic.database.query_database and of the spectrum lookup,
read here from memory through EEPFeedbackInterpolator). 'eep': phase-aligned interpolation between the bracketing
tracks; feedback only (no EEP spectra exist, so the Cloudy stage refuses it).
Supernovae follow the PCHIP death-time rule of stochastic.interpolants.supernova_feedback
in both modes.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import hashlib
import os

import numpy as np

from .sampling import AVAILABLE_IMFS
from ...exceptions import ConfigurationError

SAMPLER_VERSION = 1
BIT_GENERATOR = 'PCG64'
DEFAULT_TIME_STEP_MYR = 0.025
# beyond 30 Myr, the feedback-table time-grid rule (toddlers.feedback_tables.time_grid_checks)
DEFAULT_T_MAX_MYR = 30.1
NORMALIZATION = 'per 1e6 Msun of full drawn mass'
# Star lists with fewer stars than this are also written inline in the evolution .dat (and
# inside a generation's feedback table); the same limit as toddlers.feedback_tables.
INLINE_STAR_LIMIT = 10000
#: Version of the rule that turns a star list into a feedback table (part of the run
#: fingerprint). Version 2: unified feedback table (L_ion = the < 912 A luminosity, 1e-50
#: floor per 1e6 Msun, grid to 30.1 Myr).
TABLE_RULE_VERSION = 2

# Database metallicity labels by mass fraction Z (the groups single_star_tracks.h5 holds).
_Z_LABELS = {0.002: 'SMC', 0.006: 'LMC', 0.014: 'MW', 0.020: 'MWC'}


def metallicity_label(Z):
    """Database metallicity label ('SMC', 'LMC', 'MW', 'MWC') for a mass fraction Z."""
    for z, lab in _Z_LABELS.items():
        if np.isclose(Z, z, rtol=1e-6, atol=0.0):
            return lab
    raise ValueError(f"Z={Z} is not a metallicity of the single-star database "
                     f"{sorted(_Z_LABELS)}; stochastic populations are not interpolated in Z.")


# ---------------------------------------------------------------------------
# IMF specification and random streams
# ---------------------------------------------------------------------------
# Lower end of the two-segment 'kroupa' form (Kroupa 2001 adds a slope-0.3 segment below).
KROUPA_M_LOW_MIN = 0.08


@dataclass(frozen=True)
class IMFSpec:
    """An IMF named together with its full mass range [Msun]. There is no default range:
    the range drawn is always the one named.
    'kroupa' needs m_low >= 0.08 (KROUPA_M_LOW_MIN); a draw needs m_up <= the top track
    mass of the single-star database (check_imf_within_tracks)."""
    name: str
    m_low: float
    m_up: float

    def __post_init__(self):
        if self.name not in AVAILABLE_IMFS:
            raise ValueError(f"Unknown IMF '{self.name}'. Available: {sorted(AVAILABLE_IMFS)}")
        if not (0.0 < float(self.m_low) < float(self.m_up)):
            raise ValueError(f"IMF range must satisfy 0 < m_low < m_up, got [{self.m_low}, {self.m_up}]")
        if self.name == 'kroupa' and float(self.m_low) < KROUPA_M_LOW_MIN:
            raise ValueError(f"IMF 'kroupa' here is the two-segment form (slopes 1.3 and 2.3, break "
                             f"0.5 Msun); below {KROUPA_M_LOW_MIN} Msun Kroupa (2001) has a third "
                             f"segment it does not carry, so m_low must be >= {KROUPA_M_LOW_MIN} "
                             f"(got {self.m_low}).")
        m = np.geomspace(float(self.m_low), float(self.m_up), 2001)
        xi = AVAILABLE_IMFS[self.name](m)
        if not np.all(xi > 0):
            bad = m[~(xi > 0)]
            raise ValueError(f"IMF '{self.name}' is zero on part of [{self.m_low}, {self.m_up}] Msun "
                             f"(e.g. {bad[0]:.3g} Msun); choose a range inside its support.")

    @property
    def label(self):
        return f"{self.name}_{self.m_low:g}-{self.m_up:g}"

    def pdf(self, m):
        return AVAILABLE_IMFS[self.name](np.asarray(m, dtype=float))


def generation_rng(base_seed, generation):
    """Random stream of one generation, reproducible from (base_seed, generation) and
    independent of every other generation's stream (numpy SeedSequence spawn keys).
    Never touches numpy's global random state."""
    ss = np.random.SeedSequence(entropy=int(base_seed), spawn_key=(int(generation),))
    return np.random.Generator(np.random.PCG64(ss))


def _inverse_cdf_table(imf, n_grid=20001):
    """Number CDF of the IMF on a log-mass grid (trapezoid in ln m of m * xi(m))."""
    lnm = np.linspace(np.log(imf.m_low), np.log(imf.m_up), n_grid)
    m = np.exp(lnm)
    integrand = m * imf.pdf(m)                     # dN/dln m
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(lnm))])
    cdf /= cdf[-1]
    mean_mass = np.trapz(m * integrand, lnm) / np.trapz(integrand, lnm)
    return cdf, lnm, mean_mass


def draw_stop_after(imf, target_mass, rng):
    """Draw stars from ``imf`` until their total mass reaches ``target_mass`` [Msun].

    Stop-after: the star that crosses the target is kept, so
    0 <= sum(masses) - target_mass < masses[-1] <= imf.m_up. Masses are continuous
    (never snapped to grid masses) and returned in draw order, which is the birth
    order in constant-SFR mode."""
    if not target_mass > 0:
        raise ConfigurationError(f"target_mass must be positive, got {target_mass}")
    cdf, lnm, mean_mass = _inverse_cdf_table(imf)
    chunks, total = [], 0.0
    while total < target_mass:
        n = int((target_mass - total) / mean_mass * 1.2) + 16
        batch = np.exp(np.interp(rng.random(n), cdf, lnm))
        chunks.append(batch)
        total += float(np.sum(batch))
    masses = np.concatenate(chunks)
    cut = int(np.searchsorted(np.cumsum(masses), target_mass, side='left'))
    return masses[:cut + 1]


def birth_times(masses, sf_mode, target_mass, tau_sf_myr=None):
    """Birth times [Myr, since the generation started]. burst: zeros. constant_sfr:
    t_k = (mass drawn before k + m_k / 2) / SFR with SFR = target_mass / tau_sf_myr."""
    masses = np.asarray(masses, dtype=float)
    if sf_mode == 'burst':
        return np.zeros_like(masses)
    if sf_mode != 'constant_sfr':
        raise ConfigurationError(f"sf_mode must be 'burst' or 'constant_sfr', got {sf_mode!r}")
    if tau_sf_myr is None or not tau_sf_myr > 0:
        raise ConfigurationError("constant_sfr needs a positive tau_sf_myr")
    sfr = target_mass / tau_sf_myr
    before = np.concatenate([[0.0], np.cumsum(masses)[:-1]])
    return (before + 0.5 * masses) / sfr


# ---------------------------------------------------------------------------
# Specification of a stochastic run and the per-generation record
# ---------------------------------------------------------------------------
_MD5_CACHE = {}


def file_md5(path):
    """md5 of a file, cached per process on (path, size, mtime)."""
    st = os.stat(path)
    key = (os.path.realpath(path), st.st_size, st.st_mtime)
    if key not in _MD5_CACHE:
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 24), b''):
                h.update(chunk)
        _MD5_CACHE[key] = h.hexdigest()
    return _MD5_CACHE[key]


@dataclass(frozen=True)
class StochasticSpec:
    """What a stochastic run needs besides the cloud: the IMF, the base seed, the
    single-star database and how stars take their feedback."""
    imf: IMFSpec
    base_seed: int
    database_path: Optional[str] = None
    interpolation_mode: str = 'snap'
    n_processes: int = 1
    time_step_myr: float = DEFAULT_TIME_STEP_MYR

    def __post_init__(self):
        if isinstance(self.imf, dict):
            object.__setattr__(self, 'imf', IMFSpec(**self.imf))
        if self.interpolation_mode not in ('snap', 'eep'):
            raise ValueError("interpolation_mode must be 'snap' or 'eep'")
        if self.database_path is None:
            from ..._paths import get_database_path
            object.__setattr__(self, 'database_path', get_database_path())
        object.__setattr__(self, 'base_seed', int(self.base_seed))

    @classmethod
    def coerce(cls, value):
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, dict):
            d = {k: v for k, v in value.items() if k in
                 ('imf', 'base_seed', 'database_path', 'interpolation_mode', 'n_processes',
                  'time_step_myr')}
            return cls(**d)
        raise TypeError(f"stochastic must be a StochasticSpec or a dict, got {type(value)}")

    @classmethod
    def from_run_record(cls, rec, database_path=None):
        """The spec of a recorded run (the .dat 'stochastic' record). The single-star database
        is ``database_path`` if given, else the recorded path; when that path does not exist on
        this machine (a run moved between machines), the local default database, accepted only
        if its md5 is the recorded one."""
        from ..._paths import get_database_path
        rec = dict(rec)
        if database_path is not None:
            rec['database_path'] = database_path
        elif rec.get('database_path') and not os.path.exists(rec['database_path']):
            local, want = get_database_path(), rec.get('database_md5')
            if not (want and os.path.exists(local) and file_md5(local) == want):
                raise FileNotFoundError(
                    f"The run's single-star database {rec['database_path']} does not exist here, and "
                    f"the local {local} is not it (md5 {want}); pass database_path.")
            rec['database_path'] = local
        return cls.coerce(rec)

    @property
    def label(self):
        """Label used in output directory names (no seed; the seed goes in the file name)."""
        return f"stoch_{self.imf.label}_{self.interpolation_mode}"

    def database_md5(self):
        if not os.path.exists(self.database_path):
            raise ConfigurationError(
                f"Stochastic runs need the single-star database, not found at {self.database_path}. "
                "Download it with 'python scripts/download_data.py --stochastic-tracks', or pass "
                "StochasticSpec(..., database_path=...).")
        return file_md5(self.database_path)

    def fingerprint(self):
        """md5 identifying the feedback a run with this spec gets (database, IMF, seed,
        sampler, interpolation, time step); keys the evolution output cache."""
        h = hashlib.md5()
        h.update(repr((self.database_md5(), self.imf.name, float(self.imf.m_low),
                       float(self.imf.m_up), self.base_seed, SAMPLER_VERSION, BIT_GENERATOR,
                       self.interpolation_mode, float(self.time_step_myr), TABLE_RULE_VERSION)).encode())
        return h.hexdigest()

    def to_dict(self):
        return {'imf': asdict(self.imf), 'base_seed': self.base_seed,
                'database_path': self.database_path, 'database_md5': self.database_md5(),
                'interpolation_mode': self.interpolation_mode, 'n_processes': self.n_processes,
                'time_step_myr': self.time_step_myr, 'sampler_version': SAMPLER_VERSION,
                'bit_generator': BIT_GENERATOR, 'normalization': NORMALIZATION,
                'table_rule_version': TABLE_RULE_VERSION}


@dataclass
class GenerationPopulation:
    """The stars of one generation and how they were drawn. Scalar fields are named so
    they map one to one onto HDF5 attributes; the two arrays onto datasets."""
    masses_msun: np.ndarray
    birth_times_myr: np.ndarray
    imf_name: str
    imf_m_low_msun: float
    imf_m_up_msun: float
    base_seed: int
    generation_index: int
    target_mass_msun: float
    sf_mode: str
    tau_sf_myr: Optional[float]
    metallicity_Z: float
    track_min_mass_msun: float
    interpolation_mode: str
    database_md5: str
    seed_spawn_key: List[int] = field(default_factory=list)
    bit_generator: str = BIT_GENERATOR
    sampler_version: int = SAMPLER_VERSION
    normalization: str = NORMALIZATION

    @property
    def drawn_mass_msun(self):
        return float(np.sum(self.masses_msun))

    @property
    def ages_included(self):
        return self.sf_mode == 'constant_sfr'

    def scalars(self):
        """Scalar record (no arrays), with the derived counts."""
        below = self.masses_msun < self.track_min_mass_msun
        d = {k: v for k, v in asdict(self).items() if k not in ('masses_msun', 'birth_times_myr')}
        d.update(drawn_mass_msun=self.drawn_mass_msun,
                 drawn_minus_target_msun=self.drawn_mass_msun - self.target_mass_msun,
                 n_stars=int(len(self.masses_msun)),
                 n_stars_below_track_min=int(np.sum(below)),
                 mass_below_track_min_msun=float(np.sum(self.masses_msun[below])),
                 ages_included=self.ages_included)
        return d

    def arrays_md5(self):
        h = hashlib.md5()
        h.update(np.ascontiguousarray(self.masses_msun, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(self.birth_times_myr, dtype=np.float64).tobytes())
        return h.hexdigest()

    @classmethod
    def from_record(cls, scalars, masses, birth_times):
        keep = {k: scalars[k] for k in cls.__dataclass_fields__ if k in scalars
                and k not in ('masses_msun', 'birth_times_myr')}
        return cls(masses_msun=np.asarray(masses, dtype=float),
                   birth_times_myr=np.asarray(birth_times, dtype=float), **keep)


def draw_generation(spec, generation, target_mass, sf_mode, Z, tau_sf_myr=None,
                    track_min_mass=None):
    """Draw generation ``generation`` of a run with ``spec``: full-IMF stop-after draw to
    ``target_mass`` [Msun] on the generation's own stream, then birth times."""
    check_imf_within_tracks(spec, Z)
    rng = generation_rng(spec.base_seed, generation)
    masses = draw_stop_after(spec.imf, target_mass, rng)
    if track_min_mass is None:
        track_min_mass = _track_min_mass(spec.database_path, metallicity_label(Z))
    return GenerationPopulation(
        masses_msun=masses,
        birth_times_myr=birth_times(masses, sf_mode, target_mass, tau_sf_myr),
        imf_name=spec.imf.name, imf_m_low_msun=float(spec.imf.m_low),
        imf_m_up_msun=float(spec.imf.m_up), base_seed=spec.base_seed,
        generation_index=int(generation), target_mass_msun=float(target_mass),
        sf_mode=sf_mode, tau_sf_myr=None if tau_sf_myr is None else float(tau_sf_myr),
        metallicity_Z=float(Z), track_min_mass_msun=float(track_min_mass),
        interpolation_mode=spec.interpolation_mode, database_md5=spec.database_md5(),
        seed_spawn_key=[int(generation)])


def check_redraw(pop, spec):
    """Raise unless the population's masses equal a fresh draw from its seed (the saved
    star list and the seed must tell the same story)."""
    fresh = draw_stop_after(spec.imf, pop.target_mass_msun,
                            generation_rng(pop.base_seed, pop.generation_index))
    if not np.array_equal(fresh, pop.masses_msun):
        raise ConfigurationError(f"Generation {pop.generation_index}: the stored star list does not equal "
                         f"the draw from base seed {pop.base_seed} (n {len(pop.masses_msun)} vs "
                         f"{len(fresh)}); the list or the sampler has changed.")


# ---------------------------------------------------------------------------
# Feedback of one generation
# ---------------------------------------------------------------------------
_FEEDBACK_QUANTITIES = ['wind_power', 'wind_momentum', 'L_bol', 'Q_HI', 'Q_HeI', 'Q_HeII',
                        'L_LyW', 'L_ion_HI', 'L_ion_HeI', 'L_ion_HeII']
_INTERP_CACHE = {}


def _interpolator(database_path, label):
    key = (os.path.realpath(database_path), label)
    if key not in _INTERP_CACHE:
        from ..eep_interpolation import EEPFeedbackInterpolator
        _INTERP_CACHE[key] = EEPFeedbackInterpolator(label, database_path,
                                                     quantities=_FEEDBACK_QUANTITIES)
    return _INTERP_CACHE[key]


def _track_min_mass(database_path, label):
    return float(np.min(_interpolator(database_path, label)._db_masses))


def check_imf_within_tracks(spec, Z):
    """Raise ConfigurationError if the IMF reaches above the top track mass of the
    single-star database at Z: such stars would take the top track's feedback (stars are
    snapped to the nearest track), so the draw would not be the IMF it names."""
    top = float(np.max(_interpolator(spec.database_path, metallicity_label(Z))._db_masses))
    if float(spec.imf.m_up) > top * (1 + 1e-9):
        raise ConfigurationError(
            f"IMF m_up = {spec.imf.m_up} Msun is above the top track of the single-star database "
            f"({top} Msun at Z = {Z}); choose m_up <= {top}.")


class TimeTable:
    """log10 feedback on a time grid at one metallicity node, callable like the 2D
    feedback interpolants as f(t_myr, Z). Linear in t, held at the grid ends."""

    def __init__(self, t_myr, log_values, Z):
        self._x = np.asarray(t_myr, dtype=float)
        self._z = np.asarray(log_values, dtype=float)
        self._y = np.array([float(Z)])
        if self._z.shape != self._x.shape:
            raise ValueError("log_values must match the time grid")

    def __call__(self, t_myr, Z):
        if not np.isclose(Z, self._y[0], rtol=1e-6, atol=0.0):
            raise ValueError(f"table built at Z={self._y[0]}, queried at Z={Z}")
        return np.interp(t_myr, self._x, self._z)


# StellarFeedback handle -> unified table quantity (totals summed over the channels)
_SF_NAMES = {'L_mech': 'L_mech', 'F_ram': 'F_ram', 'L_bolo': 'L_bol', 'L_ion': 'L_ion',
             'Q_ion': 'Q_HI', 'L_LyW': 'L_LyW'}


@dataclass
class GenerationTable:
    """Feedback of one generation per 1e6 Msun of its full drawn mass.

    ``feedback_table`` is the unified table (toddlers.feedback_tables.FeedbackTable, one Z
    node, with the population record); ``log`` holds its log10 nodes under the
    StellarFeedback handles (L_mech, F_ram, L_bolo, L_ion, Q_ion, L_LyW), and
    ``linear_components`` the absolute (not normalized) per-star sums."""
    time_grid_myr: np.ndarray
    Z: float
    log: Dict[str, np.ndarray]            # StellarFeedback quantities, log10 per 1e6 Msun
    linear_components: Dict[str, np.ndarray]   # absolute (not normalized) wind/SN parts
    drawn_mass_msun: float
    ages_included: bool
    population_md5: str
    feedback_table: object = None

    def interpolant(self, name):
        """f(t_myr, Z) of log10 of a StellarFeedback quantity; L_mech and F_ram of a table with
        point-event supernovae read the SN channel linearly in t."""
        ft = self.feedback_table
        if ft is not None and name in ('L_mech', 'F_ram') and ft.sn_point_events:
            return ft.feedback_interpolant(name)
        return TimeTable(self.time_grid_myr, self.log[name], self.Z)


def realization_metadata(pop, spec=None, include_supernovae=True):
    """Unified-table metadata of one generation's draw (IMF, normalization, source)."""
    n_db = {'database_md5': pop.database_md5}
    if spec is not None:
        n_db['database_path'] = os.path.basename(spec.database_path)
    return {
        'template': 'pySB99', 'imf': f"stoch_{pop.imf_name}_{pop.imf_m_low_msun:g}-{pop.imf_m_up_msun:g}"
                                     f"_{pop.interpolation_mode}_seed{pop.base_seed}_gen{pop.generation_index}",
        'star_type': None,
        'normalization': {'rule': NORMALIZATION, 'per_mass_msun': 1e6,
                          'population_mass_msun': pop.drawn_mass_msun,
                          'm_low_msun': float(pop.imf_m_low_msun), 'm_up_msun': float(pop.imf_m_up_msun)},
        'imf_definition': {'kind': 'stochastic_draw', 'name': pop.imf_name,
                           'm_low_msun': float(pop.imf_m_low_msun), 'm_up_msun': float(pop.imf_m_up_msun),
                           'sampling': 'stop_after: stars drawn until their mass reaches the target; '
                                       'the crossing star is kept',
                           'target_mass_msun': pop.target_mass_msun, 'drawn_mass_msun': pop.drawn_mass_msun},
        'source': {'code': 'toddlers.pysb99.stochastic.population', 'sampler_version': pop.sampler_version,
                   'bit_generator': pop.bit_generator, 'table_rule_version': TABLE_RULE_VERSION,
                   'single_star_database': n_db},
        'conventions': {'mass_assignment': pop.interpolation_mode,
                        'L_ion': 'sum over stars of the < 912 A luminosity (L_ion_HI)',
                        'Q_HI': 'sum over stars of the < 912 A photon rate (nested bands)',
                        'supernovae': 'PCHIP death-time rule of stochastic.interpolants.supernova_feedback'
                                      if include_supernovae else 'none',
                        'sn_time_deposition_mode': 'point' if include_supernovae else 'none',
                        'sub_track_stars': 'count toward the mass, no feedback',
                        'ages_included': bool(pop.ages_included)},
    }


def population_record(pop):
    """The /population record of a unified table: scalars, masses and birth times."""
    sc = pop.scalars()
    sc['population_md5'] = pop.arrays_md5()
    return {'scalars': sc, 'masses_msun': pop.masses_msun, 'birth_times_myr': pop.birth_times_myr}


def generation_table_from_quantities(pop, t_grid, quantities, spec=None, include_supernovae=True,
                                     linear_components=None):
    """GenerationTable of a population from its linear per-1e6-Msun quantities (unified
    names, toddlers.feedback_tables.QUANTITIES), through the one table builder."""
    from ...feedback_tables import build_feedback_table
    ft = build_feedback_table(t_grid, [float(pop.metallicity_Z)], quantities,
                              realization_metadata(pop, spec, include_supernovae),
                              population=population_record(pop))
    log = {h: ft.log_nodes(q)[0] for h, q in _SF_NAMES.items()}
    return GenerationTable(time_grid_myr=np.asarray(t_grid, float), Z=float(pop.metallicity_Z), log=log,
                           linear_components=linear_components or {}, drawn_mass_msun=pop.drawn_mass_msun,
                           ages_included=pop.ages_included, population_md5=pop.arrays_md5(),
                           feedback_table=ft)


def _snap_sum(ei, masses, births, t_grid, chunk=4000):
    """Linear sum over stars of the nearest-track feedback at age t - birth, zero before
    birth. Same rule as EEPFeedbackInterpolator._snap per star (nearest grid mass by
    argmin |dm|, np.interp in age of the linear quantity), grouped by track."""
    out = {q: np.zeros(len(t_grid)) for q in _FEEDBACK_QUANTITIES}
    if len(masses) == 0:
        return out
    idx = np.array([int(np.argmin(np.abs(ei._db_masses - m))) for m in masses])
    for j in np.unique(idx):
        sel = idx == j
        # linear quantity, interpolated linearly in age (same convention as the spectra)
        lin = {q: np.power(10.0, ei._fb[q][j]) for q in _FEEDBACK_QUANTITIES}
        b = births[sel]
        if np.all(b == b[0]):
            ages = t_grid - b[0]
            formed = ages >= 0.0
            n = float(np.sum(sel))
            for q in _FEEDBACK_QUANTITIES:
                v = np.zeros(len(t_grid))
                v[formed] = np.interp(ages[formed], ei.age_grid, lin[q])
                out[q] += n * v
            continue
        for s in range(0, len(b), chunk):
            bb = b[s:s + chunk]
            ages = t_grid[None, :] - bb[:, None]
            formed = ages >= 0.0
            for q in _FEEDBACK_QUANTITIES:
                v = np.interp(ages, ei.age_grid, lin[q])
                out[q] += np.sum(np.where(formed, v, 0.0), axis=0)
    return out


def _eep_sum(ei, masses, births, t_grid):
    out = {q: np.zeros(len(t_grid)) for q in _FEEDBACK_QUANTITIES}
    for m, b in zip(masses, births):
        ages = t_grid - b
        formed = ages >= 0.0
        if not np.any(formed):
            continue
        fb = ei.query(float(m), ages[formed])
        for q in _FEEDBACK_QUANTITIES:
            out[q][formed] += np.power(10.0, fb[q])
    return out


def generation_time_grid(spec, t_max_myr=DEFAULT_T_MAX_MYR):
    n = int(round(t_max_myr / spec.time_step_myr))
    return np.arange(n + 1) * spec.time_step_myr


_TABLE_CACHE = {}


def build_generation_table(pop, spec, t_max_myr=DEFAULT_T_MAX_MYR, include_supernovae=True):
    """Feedback table of one generation (cached per process on the population and spec).

    Stars below the track minimum give no feedback (they are not passed to the lookup,
    which would raise them to the lowest track). Normalization: per 1e6 Msun of the
    full drawn mass. Built by toddlers.feedback_tables.build_feedback_table (the one
    writer rule: 1e-50 floor per 1e6 Msun, time-grid checks, IMF record check)."""
    t_grid = generation_time_grid(spec, t_max_myr)
    key = (pop.arrays_md5(), pop.database_md5, float(pop.metallicity_Z), spec.interpolation_mode,
           float(spec.time_step_myr), float(t_max_myr), bool(include_supernovae))
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]
    label = metallicity_label(pop.metallicity_Z)
    ei = _interpolator(spec.database_path, label)
    lit = pop.masses_msun >= pop.track_min_mass_msun
    m, b = pop.masses_msun[lit], pop.birth_times_myr[lit]
    lin = _snap_sum(ei, m, b, t_grid) if spec.interpolation_mode == 'snap' else _eep_sum(ei, m, b, t_grid)
    sn_power = np.zeros(len(t_grid))
    sn_momentum = np.zeros(len(t_grid))
    if include_supernovae and len(m):
        from .interpolants import supernova_feedback
        sn = supernova_feedback(m, b, t_grid, spec.database_path, label)
        sn_power, sn_momentum = sn['sn_power'], sn['sn_momentum']
    # pySB99 bands are nested (Q_HI counts every photon below 912 A, Q_HeI below 504 A,
    # Q_HeII below 228 A; pysb99_core.py), so the hydrogen-ionizing rate is Q_HI and the
    # ionizing luminosity is the < 912 A integral L_ion_HI.
    drawn = pop.drawn_mass_msun
    per = 1e6 / drawn
    q = {'L_bol': lin['L_bol'] * per, 'L_ion': lin['L_ion_HI'] * per, 'Q_HI': lin['Q_HI'] * per,
         'Q_HeI': lin['Q_HeI'] * per, 'Q_HeII': lin['Q_HeII'] * per, 'L_LyW': lin['L_LyW'] * per,
         'L_mech_wind': lin['wind_power'] * per, 'L_mech_SN': sn_power * per,
         'F_ram_wind': lin['wind_momentum'] * per, 'F_ram_SN': sn_momentum * per}
    q = {k: np.asarray(v, dtype=float)[None, :] for k, v in q.items()}     # (n_Z = 1, n_t)
    comps = {'wind_power': lin['wind_power'], 'wind_momentum': lin['wind_momentum'],
             'sn_power': sn_power, 'sn_momentum': sn_momentum, 'L_bol': lin['L_bol'],
             'Q_HI': lin['Q_HI'], 'Q_ion': lin['Q_HI'], 'L_ion_HI': lin['L_ion_HI']}
    table = generation_table_from_quantities(pop, t_grid, q, spec, include_supernovae, comps)
    _TABLE_CACHE[key] = table
    return table


# ---------------------------------------------------------------------------
# Star-list storage next to an evolution output
# ---------------------------------------------------------------------------
def write_star_lists(populations, path):
    """Write every generation's masses and birth times to a compressed .npz; return its md5."""
    arrays = {}
    for p in populations:
        arrays[f'masses_msun_gen{p.generation_index}'] = p.masses_msun
        arrays[f'birth_times_myr_gen{p.generation_index}'] = p.birth_times_myr
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return file_md5(path)


def population_records(populations, sidecar_path=None, sidecar_md5=None):
    """JSON-ready per-generation records (scalars; arrays inline below INLINE_STAR_LIMIT)."""
    recs = []
    for p in populations:
        r = p.scalars()
        r['population_md5'] = p.arrays_md5()
        if sidecar_path is not None:
            r['star_list_file'] = os.path.basename(sidecar_path)
            r['star_list_file_md5'] = sidecar_md5
        if len(p.masses_msun) < INLINE_STAR_LIMIT:
            r['masses_msun'] = p.masses_msun.tolist()
            r['birth_times_myr'] = p.birth_times_myr.tolist()
        recs.append(r)
    return recs


def populations_from_records(records, dat_path=None):
    """Rebuild GenerationPopulation objects from the records of an evolution .dat (inline
    arrays, or the .npz sidecar next to ``dat_path``, checked against its md5)."""
    pops, npz = [], None
    for r in records:
        if 'masses_msun' in r:
            m, b = r['masses_msun'], r['birth_times_myr']
        else:
            if npz is None:
                path = os.path.join(os.path.dirname(dat_path), r['star_list_file'])
                if file_md5(path) != r['star_list_file_md5']:
                    raise ValueError(f"{path}: md5 differs from the one recorded in the evolution output")
                npz = np.load(path)
            i = r['generation_index']
            m, b = npz[f'masses_msun_gen{i}'], npz[f'birth_times_myr_gen{i}']
        p = GenerationPopulation.from_record(r, m, b)
        if p.arrays_md5() != r['population_md5']:
            raise ValueError(f"generation {r['generation_index']}: star list md5 mismatch")
        pops.append(p)
    return pops


def write_run_generation_tables(populations, tables, output_path, star_list_path=None, star_list_md5=None):
    """Write every generation's table of an evolution run next to its output:
    ``<output stem>_gen<i>_feedback.h5``; a star list of INLINE_STAR_LIMIT stars or more refers
    to the run's ``_stars.npz`` (star_list_path, with its md5). Returns the paths."""
    from ...feedback_tables import write_table_file
    stem = output_path[:-len('.dat')] if output_path.endswith('.dat') else output_path
    paths = []
    for pop, tab in zip(populations, tables):
        p = f"{stem}_gen{pop.generation_index}_feedback.h5"
        side = None
        if len(pop.masses_msun) >= INLINE_STAR_LIMIT:
            if star_list_path is None:
                raise ValueError("a generation of 1e4 stars or more needs the run's star-list file")
            side = (os.path.basename(star_list_path), star_list_md5)
        write_table_file(tab.feedback_table, p, star_list_sidecar=side)
        paths.append(p)
    return paths


def write_missing_run_generation_tables(dat_path, database_path=None):
    """Write the generation tables an evolution run does not have yet (a run made with
    save_generation_tables=False, reused from the cache with True). Each generation is taken
    from toddlers.feedback_tables.load_run_generations: a saved table that is current is kept;
    otherwise the table is rebuilt from the recorded star list (md5 and redraw checked) and
    written to ``<output stem>_gen<i>_feedback.h5``, after an outdated file of that name is
    moved to ``stale/<UTC time>/`` next to the output. Returns the paths written."""
    import json
    from datetime import datetime, timezone
    from ...feedback_tables import load_run_generations, write_table_file
    with open(dat_path) as f:
        recs = {r['generation_index']: r for r in
                json.load(f)['parameters']['stochastic']['generations']}
    stem = dat_path[:-len('.dat')] if dat_path.endswith('.dat') else dat_path
    written = []
    for g in load_run_generations(dat_path, database_path):
        if g['source'] == 'file':
            continue
        i = g['population'].generation_index
        path = f"{stem}_gen{i}_feedback.h5"
        if os.path.exists(path):
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            dest = os.path.join(os.path.dirname(path), 'stale', stamp)
            os.makedirs(dest, exist_ok=True)
            os.replace(path, os.path.join(dest, os.path.basename(path)))
        side = None
        if len(g['masses_msun']) >= INLINE_STAR_LIMIT:
            side = (recs[i]['star_list_file'], recs[i]['star_list_file_md5'])
        write_table_file(g['table'], path, star_list_sidecar=side)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# Cloudy star table of one generation
# ---------------------------------------------------------------------------
STAR_TABLE_AGES_MYR = np.logspace(np.log10(0.01), np.log10(50.0), 50)


def star_table_name(pop, spec):
    """File name of a generation's Cloudy star table; the key covers the star list, the
    database, Z and the age grid, so a different generation never shares a table."""
    h = hashlib.md5()
    h.update(repr((pop.arrays_md5(), pop.database_md5, float(pop.metallicity_Z),
                   STAR_TABLE_AGES_MYR.tolist())).encode())
    kind = 'agesIncl' if pop.ages_included else 'burst'
    return (f"pySB99{spec.label}_seed{pop.base_seed}_gen{pop.generation_index}"
            f"_{h.hexdigest()[:8]}_{kind}.ascii")


def _refresh_pairing(path, content_md5, verbose=False):
    """Set ``paired_feedback_content_md5`` in the sidecar of a reused star table."""
    import json
    from ...cloudy_stellar_spectra_generator import _write_json_atomic
    meta_path = path + '.meta.json'
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return  # no readable sidecar: nothing recorded to refresh
    if meta.get('paired_feedback_content_md5') == content_md5:
        return
    if verbose:
        print(f"* '{os.path.basename(path)}': paired feedback md5 "
              f"{meta.get('paired_feedback_content_md5')} -> {content_md5}")
    meta['paired_feedback_content_md5'] = content_md5
    _write_json_atomic(meta_path, meta)


def ensure_star_table(pop, spec, output_dir=None, verbose=False, paired_feedback_table=None):
    """Write (once) the Cloudy star table of one generation from its star list and return
    its file name. Snap mode only: no EEP spectra exist, so an EEP-mode generation would
    pair EEP luminosities with snapped spectral shapes.

    The table is written in place by write_cloudy_star_table (temporary file renamed, so a
    partial table never carries the final name) with its sidecar ``<name>.meta.json``; the
    sidecar records ``paired_feedback_content_md5``, the content md5 of the generation's
    feedback table (``paired_feedback_table``, default: built from the population).

    An existing file is reused only if its sidecar matches this request's content key
    (star_table_is_current, the rule of every Cloudy star table); otherwise it is moved
    to ``stale_tables/`` and the table is written again. The key covers the star list, not
    the feedback table, so on reuse with ``paired_feedback_table`` given the sidecar's
    ``paired_feedback_content_md5`` is set to that table's content md5 (the star table is
    not rewritten). Without it, a reused table keeps its recorded pairing."""
    if spec.interpolation_mode != 'snap':
        raise ValueError("Cloudy star tables exist for interpolation_mode='snap' only "
                         "(no EEP spectra); run the Cloudy stage on a snap-mode evolution.")
    from .interpolants import (generate_cloudy_spectral_table, _stochastic_table_key,
                               _reuse_stochastic_table)
    from .database import METALLICITY_MAPPING
    if output_dir is None:
        from ...constants import CLOUDY_DATA_DIR
        output_dir = CLOUDY_DATA_DIR
    name = star_table_name(pop, spec)
    path = os.path.join(output_dir, name)
    lit = pop.masses_msun >= pop.track_min_mass_msun
    masses, birth_times = pop.masses_msun[lit], pop.birth_times_myr[lit]
    label = metallicity_label(pop.metallicity_Z)
    # the key generate_cloudy_spectral_table writes into the sidecar for this request
    key = _stochastic_table_key(masses, birth_times, spec.database_path, [label], False,
                                STAR_TABLE_AGES_MYR)
    from ..._fileutils import DirLock
    os.makedirs(output_dir, exist_ok=True)
    # reuse check and write under one lock: a second worker cannot move a fresh table aside
    # before its sidecar lands
    with DirLock(path + '.lock', what='star-table lock'):
        if _reuse_stochastic_table(path, output_dir, key, STAR_TABLE_AGES_MYR * 1e6,
                                   [METALLICITY_MAPPING[label]], verbose):
            if paired_feedback_table is not None:
                _refresh_pairing(path, paired_feedback_table.content_md5(), verbose)
            return name
        ft = paired_feedback_table
        if ft is None:
            ft = build_generation_table(pop, spec).feedback_table
        generate_cloudy_spectral_table(masses, birth_times, spec.database_path, label,
                                       output_filename=name, time_grid_myr=STAR_TABLE_AGES_MYR,
                                       n_processes=spec.n_processes, verbose=verbose,
                                       output_dir=output_dir,
                                       paired_feedback_content_md5=ft.content_md5())
    return name
