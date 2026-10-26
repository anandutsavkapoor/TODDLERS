"""The unified stellar feedback table: schema, writer and reader (schema version 1).

One HDF5 file per table, the same quantity set for every template:

``/axes/t``
    age since the burst [Myr], strictly ascending (attribute ``units`` = 'Myr').
``/axes/Z``
    metal mass fraction (``units`` = '', dimensionless), strictly ascending; one node is
    allowed (a single-metallicity table, e.g. a stochastic realization).
``/quantities/<name>``
    (n_Z, n_t) float64, LINEAR, per :data:`PER_MASS_MSUN` Msun of the reference population
    (root attribute ``normalization``). Attributes ``units`` (an astropy unit string),
    ``definition`` and ``floored_nodes``. The names and units are fixed by :data:`QUANTITIES`;
    further mechanical channels are named ``L_mech_<channel>`` (erg s-1) and
    ``F_ram_<channel>`` (dyn). A quantity the source cannot provide has no dataset and is
    listed with its reason in the group attribute ``missing`` (JSON); never zeros.
``/population`` (realization tables only)
    the draw record of a stochastic population (attributes) and its star list (datasets
    ``masses_msun``, ``birth_times_myr`` below :data:`INLINE_STAR_LIMIT` stars, else a
    ``.npz`` sidecar named in the attribute ``star_list_file`` with its md5).

Root attributes: ``schema``, ``schema_version``, ``writer_version``, ``template``, ``imf``,
``star_type`` ('' if the template has none), ``name``, ``legacy``, and as JSON strings
``normalization``, ``imf_definition``, ``source``, ``conventions``, ``time_grid_checks``,
``floor``, ``derived_on_load``. No timestamps, so the same content written twice on one
machine has the same md5 (the md5 is the evolution track-cache key).

Not stored: totals, time derivatives, spectral hardness, mean ionizing photon energy. On load
the totals are the sums of the channels, and :data:`DERIVED` quantities are computed from the
stored ones with one definition each.

Values below :data:`FLOOR` (in each quantity's units) are set to it by the writer and counted;
negative or non-finite input is refused.
"""
import hashlib
import json
import os

import numpy as np

from .._fileutils import unique_tmp_path

SCHEMA = 'toddlers.feedback_table'
SCHEMA_VERSION = 1
#: Bump when the bytes written for unchanged inputs change.
WRITER_VERSION = 2
#: Floor in each quantity's linear units.
FLOOR = 1e-50
#: Tables are per this mass of the reference population.
PER_MASS_MSUN = 1e6
#: Star lists with fewer stars are stored inside the table; longer ones in a .npz sidecar.
INLINE_STAR_LIMIT = 10000
#: First non-zero node of StellarFeedback's precalculation grid; queries at t = 0 are held at
#: the first table node.
FIRST_QUERY_MYR = 0.025

#: name -> (astropy unit string, definition). Every table carries each of these as a dataset
#: or lists it as missing.
QUANTITIES = {
    'L_bol': ('erg s-1', 'total stellar luminosity'),
    'L_ion': ('erg s-1', 'stellar luminosity below 912 Angstrom (one band; includes the He-ionizing part)'),
    'Q_HI': ('s-1', 'photon rate below 912 Angstrom'),
    'Q_HeI': ('s-1', 'photon rate below 504 Angstrom (nested: includes Q_HeII)'),
    'Q_HeII': ('s-1', 'photon rate below 228 Angstrom'),
    'L_LyW': ('erg s-1 Angstrom-1', 'mean L_lambda over the source wavelength nodes in [995, 1005] Angstrom'),
    'L_mech_wind': ('erg s-1', 'mechanical luminosity of stellar winds'),
    'L_mech_SN': ('erg s-1', 'mechanical luminosity of supernovae'),
    'F_ram_wind': ('dyn', 'momentum injection rate of stellar winds'),
    'F_ram_SN': ('dyn', 'momentum injection rate of supernovae'),
}
CHANNEL_UNITS = {'L_mech': 'erg s-1', 'F_ram': 'dyn'}
#: Totals, summed on load over every stored channel L_mech_* / F_ram_*.
TOTALS = {'L_mech': ('erg s-1', 'total mechanical luminosity (sum of the L_mech_* channels)'),
          'F_ram': ('dyn', 'total momentum injection rate (sum of the F_ram_* channels)')}
EV_ERG = 1.602176634e-12
#: Derived on load: name -> (units, definition, numerator, denominator, factor).
DERIVED = {
    'hardness_HeI': ('', 'Q_HeI / Q_HI (photon ratio)', 'Q_HeI', 'Q_HI', 1.0),
    'hardness_HeII': ('', 'Q_HeII / Q_HI (photon ratio)', 'Q_HeII', 'Q_HI', 1.0),
    'mean_ionizing_photon_energy': ('erg', 'L_ion / Q_HI', 'L_ion', 'Q_HI', 1.0),
    'mean_ionizing_photon_energy_eV': ('eV', 'L_ion / Q_HI in eV', 'L_ion', 'Q_HI', 1.0 / EV_ERG),
}
AXES = {'t': ('Myr', 'age since the burst'), 'Z': ('', 'metal mass fraction')}
# schema quantities a table may not list as missing (StellarFeedback reads them)
MANDATORY = ('L_bol', 'L_ion', 'Q_HI', 'L_LyW')
REQUIRED_METADATA = ('template', 'imf', 'normalization', 'imf_definition', 'source', 'conventions')
JSON_ATTRS = ('normalization', 'imf_definition', 'source', 'conventions', 'time_grid_checks', 'floor',
              'derived_on_load')


def table_filename(template, imf, star_type=None):
    """``<template>_<imf>[_<star_type>]_feedback.h5`` (star_type only where the template
    distinguishes it, e.g. BPASS single/binary)."""
    return f"{template}_{imf}_{star_type}_feedback.h5" if star_type else f"{template}_{imf}_feedback.h5"


def file_md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))


def quantity_units(name):
    """Schema units of a stored, total or derived quantity (KeyError if it has none)."""
    if name in QUANTITIES:
        return QUANTITIES[name][0]
    if name in TOTALS:
        return TOTALS[name][0]
    if name in DERIVED:
        return DERIVED[name][0]
    for prefix, u in CHANNEL_UNITS.items():
        if name.startswith(prefix + '_'):
            return u
    raise KeyError(f"'{name}' is not a quantity of {SCHEMA}")


def _is_channel(name):
    return any(name.startswith(p + '_') for p in CHANNEL_UNITS)


def units_equal(stored, expected):
    """Parsed-unit equality (astropy), so 'erg/s' and 'erg s-1' are the same unit."""
    import astropy.units as u
    try:
        return u.Unit(stored) == u.Unit(expected)
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# IMF definitions and the normalization check
# ---------------------------------------------------------------------------
def broken_power_law(exponents, mass_limits, total_mass_msun=PER_MASS_MSUN):
    """IMF definition dN/dm = A_i m^-alpha_i on [m_i, m_{i+1}], continuous at the breaks and
    normalized so that the integral of m dN/dm over [m_low, m_up] is ``total_mass_msun``
    (the rule of pySB99's calc_Nostars and of Starburst99)."""
    a = [float(x) for x in exponents]
    lim = [float(x) for x in mass_limits]
    if len(lim) != len(a) + 1 or np.any(np.diff(lim) <= 0) or lim[0] <= 0:
        raise ValueError(f"IMF mass limits {lim} must be ascending and one more than the exponents {a}")
    rel = [1.0]
    for i in range(1, len(a)):
        rel.append(rel[i - 1] * lim[i] ** (a[i] - a[i - 1]))
    k = []
    for i in range(len(a)):
        lo, hi = lim[i], lim[i + 1]
        k.append(np.log(hi / lo) if a[i] == 2.0 else (hi ** (2.0 - a[i]) - lo ** (2.0 - a[i])) / (2.0 - a[i]))
    S = total_mass_msun / float(np.dot(rel, k))
    segs = [{'m_low_msun': lim[i], 'm_up_msun': lim[i + 1], 'slope': a[i], 'coefficient': rel[i] * S}
            for i in range(len(a))]
    return {'kind': 'broken_power_law', 'form': 'dN/dm = coefficient * m^-slope per segment',
            'segments': segs, 'm_low_msun': lim[0], 'm_up_msun': lim[-1],
            'exponents': a, 'mass_limits_msun': lim}


def imf_mass_integral(defn):
    """Integral of m dN/dm of a broken-power-law definition over its segments [Msun] (closed form)."""
    tot = 0.0
    for s in defn['segments']:
        a, lo, hi, c = s['slope'], s['m_low_msun'], s['m_up_msun'], s['coefficient']
        tot += c * (np.log(hi / lo) if a == 2.0 else (hi ** (2.0 - a) - lo ** (2.0 - a)) / (2.0 - a))
    return float(tot)


def check_imf_record(imf_definition, normalization, population=None):
    """IMF record check of a table; raises ValueError on a violation.

    * broken power law: segments contiguous over [m_low, m_up]; the mass integral equals the
      normalization's population mass within rel 1e-6;
    * custom population: the sum of n m over the recorded stars equals the population mass
      within rel 1e-6;
    * stochastic draw: needs the population record; its drawn mass equals the sum of its star
      list (rel 1e-12), its m_low / m_up are the IMF's, every star lies inside them;
    * read from source (e.g. BPASS): source file names and md5 must be recorded.
    """
    kind = imf_definition.get('kind')
    pm = normalization.get('population_mass_msun')
    if kind == 'broken_power_law':
        segs = imf_definition['segments']
        if segs[0]['m_low_msun'] != imf_definition['m_low_msun'] or segs[-1]['m_up_msun'] != imf_definition['m_up_msun']:
            raise ValueError("IMF segments do not span [m_low, m_up]")
        for s0, s1 in zip(segs[:-1], segs[1:]):
            if s0['m_up_msun'] != s1['m_low_msun']:
                raise ValueError("IMF segments are not contiguous")
        if pm is None:
            raise ValueError("normalization records no population_mass_msun")
        I = imf_mass_integral(imf_definition)
        if not abs(I / pm - 1.0) <= 1e-6:
            raise ValueError(f"IMF definition integrates to {I:.8e} Msun over [{imf_definition['m_low_msun']}, "
                             f"{imf_definition['m_up_msun']}] Msun, normalization says {pm:.8e} Msun")
        if normalization.get('m_low_msun') is not None and normalization['m_low_msun'] != imf_definition['m_low_msun']:
            raise ValueError(f"normalization m_low {normalization['m_low_msun']} differs from the IMF's "
                             f"{imf_definition['m_low_msun']}")
    elif kind == 'custom_population':
        if 'star_numbers_msun_per_Z' in imf_definition:
            per = normalization.get('population_mass_msun_per_Z') or {}
            groups = [(k, v, per.get(k)) for k, v in imf_definition['star_numbers_msun_per_Z'].items()]
        else:
            groups = [('', imf_definition['star_numbers_msun'], pm)]
        for label, stars, mass in groups:
            M = float(sum(float(m) * float(n) for m, n in stars.items()))
            if mass is None or not abs(M / mass - 1.0) <= 1e-6:
                raise ValueError(f"custom population {label} holds {M:.8e} Msun, normalization says {mass}")
    elif kind == 'stochastic_draw':
        if population is None:
            raise ValueError("a stochastic-draw table needs its /population record")
        sc = population['scalars']
        m = np.asarray(population['masses_msun'], dtype=float)
        drawn = float(sc['drawn_mass_msun'])
        if not abs(float(np.sum(m)) / drawn - 1.0) <= 1e-12:
            raise ValueError(f"recorded drawn mass {drawn:.12e} Msun differs from the star list sum "
                             f"{float(np.sum(m)):.12e} Msun")
        if pm is None or not abs(pm / drawn - 1.0) <= 1e-12:
            raise ValueError(f"normalization population mass {pm} differs from the drawn mass {drawn}")
        if (float(sc['imf_m_low_msun']) != float(imf_definition['m_low_msun'])
                or float(sc['imf_m_up_msun']) != float(imf_definition['m_up_msun'])):
            raise ValueError("population IMF range differs from the table's IMF definition")
        if normalization.get('m_low_msun') is not None and float(normalization['m_low_msun']) != float(sc['imf_m_low_msun']):
            raise ValueError(f"normalization m_low {normalization['m_low_msun']} is not the IMF lower limit "
                             f"{sc['imf_m_low_msun']} (a track-grid minimum is not an IMF limit)")
        if m.size and (m.min() < float(sc['imf_m_low_msun']) or m.max() > float(sc['imf_m_up_msun'])):
            raise ValueError("star list has stars outside the recorded IMF range")
    elif kind == 'read_from_source':
        if not imf_definition.get('source_files_md5'):
            raise ValueError("an IMF definition read from the source must record the source files and md5")
    else:
        raise ValueError(f"unknown imf_definition kind {kind!r}")


# ---------------------------------------------------------------------------
# Time grid checks
# ---------------------------------------------------------------------------
def time_grid_checks(t_myr):
    """The time-grid checks of a feedback table (dict of booleans and the numbers behind them)."""
    t = np.asarray(t_myr, dtype=float)
    win = (t >= 2.0) & (t <= 6.0)
    idx = np.where(win)[0]
    if idx.size:
        lo, hi = max(idx[0] - 1, 0), min(idx[-1] + 1, t.size - 1)
        max_dt = float(np.max(np.diff(t[lo:hi + 1])))
    else:
        i = np.searchsorted(t, 2.0)
        max_dt = float(t[min(i, t.size - 1)] - t[max(i - 1, 0)])
    return {
        'first_node_myr': float(t[0]), 'last_node_myr': float(t[-1]),
        'max_spacing_2_6_myr': max_dt,
        'first_node_at_or_before_first_query': bool(t[0] <= FIRST_QUERY_MYR),
        'last_node_beyond_30_myr': bool(t[-1] > 30.0),
        'spacing_2_6_myr_at_most_0p1': bool(max_dt <= 0.1 + 1e-9),
    }


def _check_axis(a, label, min_size=2):
    a = np.asarray(a, dtype=float)
    if a.ndim != 1 or a.size < min_size:
        raise ValueError(f"{label} must be a 1-D array with at least {min_size} node(s)")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{label} has non-finite nodes")
    if np.any(np.diff(a) <= 0):
        raise ValueError(f"{label} must be strictly ascending (no duplicates)")
    return a


def _check_values(name, v, shape):
    v = np.asarray(v, dtype=float)
    if v.shape != shape:
        raise ValueError(f"{name} has shape {v.shape}; expected (n_Z, n_t) = {shape}")
    bad = ~np.isfinite(v)
    if bad.any():
        idx = tuple(int(i) for i in np.argwhere(bad)[0])
        raise ValueError(f"{name} has {int(bad.sum())} non-finite value(s), first at (Z index, t index) {idx}")
    if np.any(v < 0):
        idx = tuple(int(i) for i in np.argwhere(v < 0)[0])
        raise ValueError(f"{name} has {int((v < 0).sum())} negative value(s), first at (Z index, t index) {idx}")
    return v


# ---------------------------------------------------------------------------
# The in-memory table
# ---------------------------------------------------------------------------
class _RowTable:
    """log10 values on a time grid at one metallicity node, called as f(t_myr, Z): linear in
    t, held at the grid ends, refuses another Z (the run-time stochastic table rule)."""

    def __init__(self, t_myr, log_values, Z):
        self._x = np.asarray(t_myr, dtype=float)
        self._z = np.asarray(log_values, dtype=float)
        self._y = np.array([float(Z)])

    def __call__(self, t_myr, Z):
        if not np.isclose(Z, self._y[0], rtol=1e-6, atol=0.0):
            raise ValueError(f"table built at Z={self._y[0]}, queried at Z={Z}")
        return np.interp(t_myr, self._x, self._z)


#: Totals whose supernova channel holds point events: the wind channel is read in
#: log10 as every other quantity, the SN channel linearly in t, so each event delivers its
#: bin's energy and momentum exactly.
POINT_EVENT_CHANNELS = {'L_mech': ('L_mech_wind', 'L_mech_SN'), 'F_ram': ('F_ram_wind', 'F_ram_SN')}


class _PointEventTotal:
    """log10 of (wind channel interpolated in log10 + SN channel interpolated linearly in t),
    called as f(t_myr, Z) at a metallicity node; held at the grid ends."""

    def __init__(self, table, name):
        wind, sn = POINT_EVENT_CHANNELS[name]
        self._x = np.asarray(table.t, dtype=float)
        self._y = np.asarray(table.Z, dtype=float)
        self._log_wind = table.log_nodes(wind)
        self._sn = table.linear(sn)

    def __call__(self, t_myr, Z):
        m = np.isclose(self._y, Z, rtol=1e-6, atol=0.0)
        if not m.any():
            raise ValueError(f"Z={Z} is not a node of this table {list(self._y)}")
        i = int(np.argmax(m))
        wind = 10.0 ** np.interp(t_myr, self._x, self._log_wind[i])
        return np.log10(np.maximum(wind + np.interp(t_myr, self._x, self._sn[i]), FLOOR))


class FeedbackTable:
    """A validated feedback table in memory (from :func:`build_feedback_table` or a file).

    Attributes:
        t (ndarray): time nodes [Myr].
        Z (ndarray): metallicity nodes (mass fractions).
        values (dict): stored quantity -> (n_Z, n_t) linear values (floored).
        missing (dict): quantity -> reason, for schema quantities the table does not hold.
        metadata (dict): root attributes (JSON ones decoded).
        population (dict or None): realization record {'scalars', 'masses_msun', 'birth_times_myr'}.
        path, md5: the file and its md5 (None for a table not read from or written to a file).
    """

    def __init__(self, t, Z, values, missing, metadata, population=None, path=None, md5=None):
        self.t = t
        self.Z = Z
        self.values = values
        self.missing = dict(missing)
        self.metadata = metadata
        self.population = population
        self.path = path
        self.md5 = md5
        self._log_cache = {}

    # -- names ---------------------------------------------------------------
    def channels(self, total):
        """Stored channels of a total ('L_mech' or 'F_ram'), sorted by name."""
        return sorted(k for k in self.values if k.startswith(total + '_'))

    @property
    def quantities(self):
        """Every name :meth:`get` answers: stored, totals and derived (missing ones excluded)."""
        names = list(self.values)
        names += [k for k in TOTALS if self.channels(k)]
        names += [k for k, d in DERIVED.items() if d[2] in self.values and d[3] in self.values]
        return names

    @property
    def imf(self):
        """IMF name, definition and normalization of the table (the checked IMF record)."""
        n = self.metadata['normalization']
        return {'name': self.metadata['imf'], 'definition': self.metadata['imf_definition'],
                'normalization': n, 'normalization_mass_msun': n.get('population_mass_msun'),
                'per_mass_msun': n.get('per_mass_msun')}

    def units(self, name):
        return quantity_units(name)

    # -- values --------------------------------------------------------------
    def linear(self, name):
        """(n_Z, n_t) linear node values of a stored quantity or total (floored)."""
        if name in self.values:
            return self.values[name]
        if name in TOTALS:
            ch = self.channels(name)
            if not ch:
                raise KeyError(f"{name}: no {name}_* channel stored")
            return np.maximum(np.sum([self.values[k] for k in ch], axis=0), FLOOR)
        self._raise_unknown(name)

    def log_nodes(self, name):
        """(n_Z, n_t) log10 node values: log10(max(value, FLOOR)); totals summed first."""
        if name not in self._log_cache:
            self._log_cache[name] = np.log10(np.maximum(self.linear(name), FLOOR))
        return self._log_cache[name]

    def log_interpolant(self, name):
        """Callable f(t_myr, Z) of log10 of a stored quantity or total: bilinear on (t, Z) for
        several Z nodes (Interp2DLinear, edges held), linear in t at one node otherwise."""
        logs = self.log_nodes(name)
        if self.Z.size == 1:
            return _RowTable(self.t, logs[0], self.Z[0])
        from .._interp_compat import Interp2DLinear
        return Interp2DLinear(self.t, self.Z, logs)

    @property
    def sn_point_events(self):
        """True if the SN channels hold point events (custom star lists, stochastic draws):
        the table's ``conventions['sn_time_deposition_mode']`` is 'point'."""
        return self.metadata.get('conventions', {}).get('sn_time_deposition_mode') == 'point'

    def feedback_interpolant(self, name):
        """The interpolant StellarFeedback uses for a quantity: :meth:`log_interpolant`, except
        L_mech and F_ram of a point-event table, whose SN channel is linear in t."""
        if name in POINT_EVENT_CHANNELS and self.sn_point_events:
            return _PointEventTotal(self, name)
        return self.log_interpolant(name)

    def _raise_unknown(self, name):
        if name in self.missing:
            raise KeyError(f"{name} is missing from this table: {self.missing[name]}")
        raise KeyError(f"{name} is not a quantity of this table; available: {self.quantities}")

    def z_index(self, Z):
        if Z is None:
            if self.Z.size == 1:
                return 0
            raise ValueError(f"Z is required; table nodes {list(self.Z)}")
        m = np.isclose(self.Z, Z, rtol=1e-6, atol=0.0)
        if not m.any():
            raise ValueError(f"Z={Z} is not a node of this table {list(self.Z)}; tables are not "
                             f"interpolated in Z")
        return int(np.argmax(m))

    def get(self, name, t_myr, Z=None):
        """Value of any quantity at age(s) t_myr [Myr] and metallicity node Z, as an astropy
        Quantity in the schema units, per ``per_mass_msun`` of the reference population.

        Stored quantities and totals: log10 interpolated linearly in t (held at the grid ends),
        the rule StellarFeedback uses (for L_mech and F_ram of a point-event table, see
        :meth:`feedback_interpolant`). Derived quantities: the ratio of the interpolated stored
        values (one definition, :data:`DERIVED`)."""
        import astropy.units as u
        i = self.z_index(Z)
        if name in DERIVED:
            units, _, num, den, fac = DERIVED[name]
            for k in (num, den):
                if k not in self.values:
                    self._raise_unknown(k)
            v = self._interp_linear(num, i, t_myr) / self._interp_linear(den, i, t_myr) * fac
        else:
            units = quantity_units(name) if (name in self.values or name in TOTALS) else None
            if units is None:
                self._raise_unknown(name)
            if name in POINT_EVENT_CHANNELS and self.sn_point_events:
                v = 10.0 ** self.feedback_interpolant(name)(np.asarray(t_myr, dtype=float), self.Z[i])
            else:
                v = self._interp_linear(name, i, t_myr)
        return v * u.Unit(units)

    def _interp_linear(self, name, i, t_myr):
        return 10.0 ** np.interp(np.asarray(t_myr, dtype=float), self.t, self.log_nodes(name)[i])

    def content_md5(self):
        """md5 of the table content (axes, stored values, missing list, metadata, population),
        independent of the file bytes; recorded in every file as the root attribute
        ``content_md5`` and used to pair an in-memory (run-time) table with a Cloudy star table."""
        h = hashlib.md5()
        h.update(np.ascontiguousarray(self.t, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(self.Z, dtype=np.float64).tobytes())
        for k in sorted(self.values):
            h.update(k.encode())
            h.update(np.ascontiguousarray(self.values[k], dtype=np.float64).tobytes())
        h.update(json.dumps(self.missing, sort_keys=True).encode())
        meta = {k: v for k, v in self.metadata.items() if k != 'content_md5'}
        h.update(json.dumps(meta, sort_keys=True).encode())
        if self.population is not None:
            h.update(json.dumps(_jsonable(self.population['scalars']), sort_keys=True).encode())
            for k in ('masses_msun', 'birth_times_myr'):
                h.update(np.ascontiguousarray(self.population[k], dtype=np.float64).tobytes())
        return h.hexdigest()

    def __repr__(self):
        return (f"FeedbackTable({self.metadata.get('template')}, {self.metadata.get('imf')}, "
                f"star_type={self.metadata.get('star_type') or None}, Z={list(self.Z)}, "
                f"{self.t.size} ages, missing={sorted(self.missing)})")


def build_feedback_table(t_myr, Z, quantities, metadata, *, missing=None, legacy=False, population=None):
    """
    Validate a feedback table and return it in memory (:class:`FeedbackTable`).

    Args:
        t_myr (array): time nodes [Myr], strictly ascending.
        Z (array): metallicity nodes (mass fractions), strictly ascending; one node allowed.
        quantities (dict): name -> linear (n_Z, n_t) array in the schema units
            (:data:`QUANTITIES`, plus any ``L_mech_<channel>`` / ``F_ram_<channel>``), per
            ``PER_MASS_MSUN`` Msun of the reference population. Finite and non-negative.
        metadata (dict): ``template``, ``imf``, ``star_type`` (None if the template has none),
            ``normalization`` (dict: 'rule', 'per_mass_msun', 'population_mass_msun',
            'm_low_msun', 'm_up_msun'), ``imf_definition`` (see :func:`check_imf_record`),
            ``source``, ``conventions``; optional ``name``.
        missing (dict, optional): schema quantity -> reason, for each one not given.
        legacy (bool): a released table being reproduced: failed time-grid checks are recorded
            instead of raising.
        population (dict, optional): realization record {'scalars': dict, 'masses_msun',
            'birth_times_myr'} (stochastic tables).

    Returns:
        FeedbackTable
    """
    t = _check_axis(t_myr, 't_myr')
    Zn = _check_axis(Z, 'Z', min_size=1)
    if np.any(Zn < 0):
        raise ValueError("Z nodes must be non-negative mass fractions")
    shape = (Zn.size, t.size)
    missing = dict(missing or {})
    unknown = [k for k in quantities if k not in QUANTITIES and not _is_channel(k)]
    if unknown:
        raise ValueError(f"unknown quantities {unknown}; schema: {list(QUANTITIES)} and channels "
                         f"L_mech_<channel> / F_ram_<channel>")
    both = [k for k in missing if k in quantities]
    if both:
        raise ValueError(f"{both} given and listed as missing")
    bad_missing = [k for k in missing if k not in QUANTITIES]
    if bad_missing:
        raise ValueError(f"only schema quantities can be missing; got {bad_missing}")
    absent = [k for k in QUANTITIES if k not in quantities and k not in missing]
    if absent:
        raise ValueError(f"quantities {absent} are neither given nor listed as missing (with a reason)")
    for total in TOTALS:
        if not any(k.startswith(total + '_') for k in quantities):
            raise ValueError(f"no {total}_* channel given")
    for k in MANDATORY:
        if k in missing:
            raise ValueError(f"{k} cannot be missing: StellarFeedback reads it")
    meta = dict(metadata)
    for k in REQUIRED_METADATA:
        if k not in meta or meta[k] is None:
            raise ValueError(f"metadata needs '{k}'")
    norm = dict(meta['normalization'])
    norm.setdefault('per_mass_msun', PER_MASS_MSUN)
    if norm['per_mass_msun'] != PER_MASS_MSUN:
        raise ValueError(f"tables are per {PER_MASS_MSUN:g} Msun; normalization says {norm['per_mass_msun']}")
    meta['normalization'] = norm
    check_imf_record(meta['imf_definition'], norm, population)

    vals = {k: _check_values(k, v, shape) for k, v in quantities.items()}
    floored = {k: int(np.sum(v < FLOOR)) for k, v in vals.items()}
    vals = {k: np.maximum(v, FLOOR) for k, v in vals.items()}

    checks = time_grid_checks(t)
    failed = [k for k, v in checks.items() if v is False]
    if failed and not legacy:
        msgs = {'first_node_at_or_before_first_query': f"first node {t[0]} Myr is after the first "
                                                       f"evolution query ({FIRST_QUERY_MYR} Myr)",
                'last_node_beyond_30_myr': f"last node {t[-1]} Myr is not beyond 30 Myr",
                'spacing_2_6_myr_at_most_0p1': f"spacing {checks['max_spacing_2_6_myr']:.4g} Myr "
                                               f"between 2 and 6 Myr exceeds 0.1 Myr"}
        raise ValueError("time grid: " + "; ".join(msgs[k] for k in failed)
                         + " (pass legacy=True only to reproduce a released table)")
    if population is not None:
        population = {'scalars': dict(population['scalars']),
                      'masses_msun': np.asarray(population['masses_msun'], dtype=float),
                      'birth_times_myr': np.asarray(population['birth_times_myr'], dtype=float)}
        if population['masses_msun'].shape != population['birth_times_myr'].shape:
            raise ValueError("population masses and birth times differ in length")

    star_type = meta.get('star_type') or ''
    out_meta = {
        'schema': SCHEMA, 'schema_version': SCHEMA_VERSION, 'writer_version': WRITER_VERSION,
        'template': str(meta['template']), 'imf': str(meta['imf']), 'star_type': str(star_type),
        'name': str(meta.get('name') or table_filename(meta['template'], meta['imf'], star_type or None)[:-3]),
        'legacy': bool(legacy),
        'normalization': norm, 'imf_definition': meta['imf_definition'], 'source': meta['source'],
        'conventions': meta['conventions'], 'time_grid_checks': checks,
        'floor': {'value': FLOOR, 'rule': 'max(x, floor) in linear units, applied by the writer; '
                                          'totals summed, floored again, then log10 on load',
                  'floored_nodes': floored},
        'derived_on_load': {k: d[1] for k, d in DERIVED.items()},
    }
    out_meta = _jsonable(out_meta)
    return FeedbackTable(t, Zn, vals, missing, out_meta, population=population)


# ---------------------------------------------------------------------------
# HDF5 writer and reader
# ---------------------------------------------------------------------------
def _ds(group, name, data, units, definition=None, floored=None):
    d = group.create_dataset(name, data=np.asarray(data, dtype=np.float64), compression='gzip',
                             compression_opts=4, shuffle=True, track_times=False)
    d.attrs['units'] = units
    if definition is not None:
        d.attrs['definition'] = definition
    if floored is not None:
        d.attrs['floored_nodes'] = int(floored)
    return d


def write_table_file(table, path, star_list_sidecar=None):
    """Write a :class:`FeedbackTable` to ``path`` (HDF5; atomic: temporary file renamed).

    Args:
        star_list_sidecar (tuple, optional): (file name, md5) of the .npz holding a realization's
            star list when it has :data:`INLINE_STAR_LIMIT` stars or more; the file is written
            by the caller next to ``path``.
    Returns:
        str: md5 of the file written (also set as ``table.md5``; ``table.path`` = path).
    """
    import h5py
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = unique_tmp_path(path)
    meta = table.metadata
    with h5py.File(tmp, 'w') as f:
        for k in ('schema', 'schema_version', 'writer_version', 'template', 'imf', 'star_type', 'name', 'legacy'):
            f.attrs[k] = meta[k]
        for k in JSON_ATTRS:
            f.attrs[k] = json.dumps(meta[k], sort_keys=True)
        f.attrs['content_md5'] = table.content_md5()
        ax = f.create_group('axes')
        _ds(ax, 't', table.t, AXES['t'][0], AXES['t'][1])
        _ds(ax, 'Z', table.Z, AXES['Z'][0], AXES['Z'][1])
        q = f.create_group('quantities')
        q.attrs['missing'] = json.dumps(table.missing, sort_keys=True)
        fl = meta['floor']['floored_nodes']
        for name in list(QUANTITIES) + sorted(k for k in table.values if k not in QUANTITIES):
            if name in table.values:
                definition = QUANTITIES[name][1] if name in QUANTITIES else 'further mechanical channel'
                _ds(q, name, table.values[name], quantity_units(name), definition, fl.get(name, 0))
        if table.population is not None:
            p = f.create_group('population')
            p.attrs['record'] = json.dumps(_jsonable(table.population['scalars']), sort_keys=True)
            n = int(table.population['masses_msun'].size)
            if star_list_sidecar is None:
                if n >= INLINE_STAR_LIMIT:
                    raise ValueError(f"{n} stars: pass star_list_sidecar=(name, md5) for lists of "
                                     f"{INLINE_STAR_LIMIT} stars or more")
                _ds(p, 'masses_msun', table.population['masses_msun'], 'Msun', 'initial stellar masses')
                _ds(p, 'birth_times_myr', table.population['birth_times_myr'], 'Myr',
                    'birth times since the generation started')
            else:
                p.attrs['star_list_file'] = star_list_sidecar[0]
                p.attrs['star_list_file_md5'] = star_list_sidecar[1]
    os.replace(tmp, path)
    table.path = path
    table.md5 = file_md5(path)
    _CACHE.pop(os.path.realpath(path), None)
    return table.md5


def write_feedback_table(t_myr, Z, quantities, metadata, out_dir, *, filename=None, missing=None,
                         legacy=False, population=None, star_list_sidecar=None):
    """Build (:func:`build_feedback_table`) and write a table to ``out_dir/filename``
    (default :func:`table_filename`). Returns the :class:`FeedbackTable` (with path and md5)."""
    table = build_feedback_table(t_myr, Z, quantities, metadata, missing=missing, legacy=legacy,
                                 population=population)
    name = filename or table_filename(metadata['template'], metadata['imf'], metadata.get('star_type') or None)
    write_table_file(table, os.path.join(out_dir, name), star_list_sidecar=star_list_sidecar)
    return table


_CACHE = {}


def read_table_file(path):
    """Read and check a table file (cached per process on path, size and mtime).

    Refused (ValueError): another schema or an unsupported schema_version; a dataset or axis
    without units, or with units that are not the schema's (parsed-unit equality); a schema
    quantity neither stored nor listed as missing; one of :data:`MANDATORY` listed missing,
    or no ``L_mech_*`` / ``F_ram_*`` channel stored (the writer's rules); non-finite or
    negative values; axes not strictly ascending; a realization whose star list fails its
    md5 or the IMF record check.
    """
    import h5py
    rp = os.path.realpath(path)
    st = os.stat(rp)
    key = (st.st_size, st.st_mtime_ns)
    hit = _CACHE.get(rp)
    if hit is not None and hit[0] == key:
        return hit[1]
    with h5py.File(rp, 'r') as f:
        if f.attrs.get('schema') != SCHEMA:
            raise ValueError(f"{path}: not a {SCHEMA} file (schema {f.attrs.get('schema')!r})")
        ver = int(f.attrs.get('schema_version', -1))
        if ver != SCHEMA_VERSION:
            raise ValueError(f"{path}: schema_version {ver} is not supported (this code reads {SCHEMA_VERSION})")
        meta = {}
        for k in ('schema', 'schema_version', 'writer_version', 'template', 'imf', 'star_type', 'name', 'legacy'):
            v = f.attrs[k]
            meta[k] = v.item() if hasattr(v, 'item') else v
        meta['legacy'] = bool(meta['legacy'])
        for k in JSON_ATTRS:
            meta[k] = json.loads(f.attrs[k])
        axes = {}
        for a, (u, _) in AXES.items():
            ds = f['axes'][a]
            if 'units' not in ds.attrs or not units_equal(ds.attrs['units'], u):
                raise ValueError(f"{path}: axis {a} units {ds.attrs.get('units')!r}, expected {u!r}")
            axes[a] = np.asarray(ds[()], dtype=float)
        t = _check_axis(axes['t'], f"{path}: axis t")
        Zn = _check_axis(axes['Z'], f"{path}: axis Z", min_size=1)
        q = f['quantities']
        missing = json.loads(q.attrs.get('missing', '{}'))
        vals = {}
        for name in q:
            ds = q[name]
            try:
                expect = quantity_units(name)
            except KeyError:
                raise ValueError(f"{path}: unknown dataset quantities/{name}")
            if name in TOTALS or name in DERIVED:
                raise ValueError(f"{path}: {name} must not be stored (summed or derived on load)")
            if 'units' not in ds.attrs:
                raise ValueError(f"{path}: quantities/{name} has no units attribute")
            if not units_equal(ds.attrs['units'], expect):
                raise ValueError(f"{path}: quantities/{name} has units {ds.attrs['units']!r}; expected "
                                 f"{expect!r} (unit mismatch)")
            vals[name] = _check_values(f"{path}: {name}", ds[()], (Zn.size, t.size))
        absent = [k for k in QUANTITIES if k not in vals and k not in missing]
        if absent:
            raise ValueError(f"{path}: quantities {absent} are neither stored nor listed as missing")
        # the writer's mandatory rules (build_feedback_table), applied on read too
        for total in TOTALS:
            if not any(k.startswith(total + '_') for k in vals):
                raise ValueError(f"{path}: no {total}_* channel stored")
        for k in MANDATORY:
            if k in missing:
                raise ValueError(f"{path}: {k} is listed missing; StellarFeedback reads it")
        population = None
        if 'population' in f:
            p = f['population']
            scalars = json.loads(p.attrs['record'])
            if 'masses_msun' in p:
                for k, u in (('masses_msun', 'Msun'), ('birth_times_myr', 'Myr')):
                    if not units_equal(p[k].attrs.get('units', None) or 'x', u):
                        raise ValueError(f"{path}: population/{k} units must be {u}")
                m, b = np.asarray(p['masses_msun'][()]), np.asarray(p['birth_times_myr'][()])
            else:
                side = os.path.join(os.path.dirname(rp), str(p.attrs['star_list_file']))
                if file_md5(side) != p.attrs['star_list_file_md5']:
                    raise ValueError(f"{side}: md5 differs from the one recorded in {path}")
                gi = scalars['generation_index']
                with np.load(side) as z:
                    m, b = z[f'masses_msun_gen{gi}'], z[f'birth_times_myr_gen{gi}']
            population = {'scalars': scalars, 'masses_msun': np.asarray(m, float),
                          'birth_times_myr': np.asarray(b, float)}
        stored_content = f.attrs.get('content_md5')
    check_imf_record(meta['imf_definition'], meta['normalization'], population)
    table = FeedbackTable(t, Zn, vals, missing, meta, population=population, path=rp, md5=file_md5(rp))
    if stored_content is not None and table.content_md5() != stored_content:
        raise ValueError(f"{path}: content md5 {table.content_md5()} differs from the recorded "
                         f"{stored_content}; the file changed after it was written")
    _CACHE[rp] = (key, table)
    return table


def find_table(template, imf, star_type=None, data_dir=None):
    """Path of the table file for (template, imf, star_type) in the data dir, or None."""
    from .._paths import get_data_dir
    d = data_dir or get_data_dir()
    p = os.path.join(d, table_filename(template, imf, star_type))
    return p if os.path.exists(p) else None


def load_table(template_or_path, imf=None, star_type=None):
    """
    Open a feedback table: ``load_table('SB99', 'kroupa100')``, ``load_table('BPASS',
    'chab100', 'bin')``, ``load_table('pySB99', 'kroupa_100')`` or ``load_table(path)`` (any
    table file, including a stochastic generation table written next to an evolution run).

    By names, the file's recorded template, imf and star type must equal the request
    (:func:`check_table_names`). Returns a :class:`FeedbackTable`: ``get(name, t_myr, Z)`` (astropy Quantity),
    ``.quantities``, ``.missing``, ``.metadata``, ``.imf``, ``.population``. No cloud
    parameters are needed. Built-in names take their default IMF and star type when omitted
    (SB99 kroupa100; BPASS chab100, bin).
    """
    if imf is None and (os.sep in str(template_or_path) or str(template_or_path).endswith('.h5')):
        return read_table_file(template_or_path)
    template = template_or_path
    family, imf, star_type = resolve_names(template, imf, star_type)
    p = find_table(family, imf, star_type)
    if p is None:
        from .._paths import get_data_dir
        raise FileNotFoundError(f"no feedback table {table_filename(family, imf, star_type)} in "
                                f"{get_data_dir()} (2.0 .obj tables: see toddlers.feedback_tables.legacy)")
    table = read_table_file(p)
    check_table_names(table, family, imf, star_type)
    return table


def check_table_names(table, family, imf, star_type):
    """Raise ValueError unless the table's recorded template, imf and star_type are the
    requested (file-name) ones: a file copied or renamed to another name is refused. Used by
    :func:`load_table` and StellarFeedback."""
    want = {'template': family, 'imf': imf, 'star_type': star_type or ''}
    got = {k: table.metadata[k] for k in want}
    if got != want:
        raise ValueError(f"{table.path}: the table records {got}, the request is {want}")


def resolve_names(template, imf, star_type):
    """(family, imf, star_type) used in file names: BPASS* -> 'BPASS' (default chab100, bin),
    SB99* -> 'SB99' (default kroupa100, no star type), pySB99 (no star type); any other
    template as given (imf required)."""
    if template.startswith('BPASS'):
        return 'BPASS', imf or 'chab100', star_type or 'bin'
    if template.startswith('SB99'):
        if star_type == 'bin':
            raise ValueError("SB99 does not support binary stars")
        return 'SB99', imf or 'kroupa100', None
    if template == 'pySB99':
        if star_type == 'bin':
            raise ValueError("pySB99 does not support binary stars")
        if imf is None:
            raise ValueError("pySB99 needs imf= (the name the table was generated under)")
        return 'pySB99', imf, None
    if imf is None:
        raise ValueError(f"template {template!r} needs imf= (the name its table was written under)")
    return template, imf, star_type
