"""
EEP-aligned mass interpolation for pySTARBURST99 single-star tracks.

Motivation
----------
A stochastically sampled star almost never lands exactly on a track-grid mass, so
its feedback must be assigned somehow. There are two ends of the spectrum:

  - Snapping (the production default): round the star to the nearest grid mass and
    read feedback directly from that real track. No interpolation is needed, and
    every star stays on a physically self-consistent evolutionary history.
  - EEP interpolation (this module): interpolate feedback in mass between the two
    bracketing grid tracks. This is only physically sound if done along Equivalent
    Evolutionary Points (EEPs), i.e. by aligning evolutionary *phases* across tracks
    before interpolating, rather than at fixed clock age (which would blend, say, a
    main-sequence star with a Wolf-Rayet one).

Primary markers (the phase anchors; full table in MARKER_INFO). Each is grounded in
a pySTARBURST99 spectral-classifier boundary (P, see pysb99_core.py), a MIST
primary-EEP criterion (M, Dotter 2016), or a feedback Teff threshold (F)::

    ZAMS      track start                                          (M: ZAMS)
    MS_TO     turn-off: logTeff drops 0.05 dex below its run-max   (M: TAMS / turn-off)
    T_DOWN    logTeff < 4.45, ionizing-decline onset               (F; near P WR cut 4.4)
    TEFF_MIN  coolest point, argmin logTeff                        (M: RGBTip; P cool <3.65)
    WN_ON     surface X(H) < 0.4, envelope stripping               (P: WR gate, :1316)
    WC_ON     surface X(C) > X(N) after WN                         (P: WN/WC split, :1337)
    T_UP      logTeff > 4.30 after the minimum, blue return        (P: OB cut 20000 K, :1400)
    END       track end                                            (terminal / pre-SN)

MS_MID (a mid-MS proxy for MIST IAMS) is detected but dropped by default: there is
no central H to define it and it had no measurable effect in validation.

Method (secondary-EEP placement follows MIST, Dotter 2016, Eq. 1)::

    1. Detect the primary markers above on each track (`detect_eeps_comprehensive`).
    2. For a pair of bracketing grid masses, take the markers they share (longest
       common subsequence) as phase-aligned segment boundaries.
    3. Within each segment, place secondary EEPs at equal increments of the MIST HRD
       metric (`_segment_ages_metric`): Euclidean path length in (logTeff, logL),
       each normalised to span [0, 1] over the track so the two contribute equally
       (Dotter 2016, Eq. 1, with weights w = 1/range^2).
    4. Interpolate feedback in log at fixed EEP, with log-mass as the independent
       variable (`interpolate_feedback_pairwise`).

Validation lives in eep_tests/; the leave-one-out study there shows the
reconstructed feedback history is insensitive to the within-segment sampler, so
the structurally principled MIST metric is used.

Raw track column layout (in ``pySB99_files/*_tracks.npy``), confirmed by inspection::

    col 0  : within-track index (not used)
    col 1  : age                       [yr]
    col 2  : current (actual) mass      [Msun]
    col 3  : logL                       [log Lsun]
    col 4  : logTeff                    [log K]
    col 5  : surface X(H)               [mass fraction]
    col 6  : surface X(He)              [mass fraction]
    col 7  : surface X(C12)             [mass fraction]
    col 8  : surface X(N14)             [mass fraction]
    col 9  : surface X(O16)             [mass fraction]
    col 10 : (labelled core temperature -- UNRELIABLE, ignored)
    col 11 : log mass-loss rate         [log Msun/yr]

Each track is exactly 400 rows with no trailing padding; age is in years.
"""
import os
import sys

import numpy as np

# make the pysb99 package importable when run as a script
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_THIS_DIR)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from toddlers.pysb99.pysb99_core import StellarPopulationConfig, StellarDataLoader

# raw-track column indices
COL_AGE = 1
COL_MASS = 2
COL_LOGL = 3
COL_LOGTEFF = 4
COL_XH = 5
COL_XHE = 6
COL_XC = 7
COL_XN = 8
COL_XO = 9
COL_LOGMDOT = 11

# fields exposed on a clean track
FIELDS = {
    "age": COL_AGE,
    "mass": COL_MASS,
    "logL": COL_LOGL,
    "logTeff": COL_LOGTEFF,
    "X_H": COL_XH,
    "X_He": COL_XHE,
    "X_C": COL_XC,
    "X_N": COL_XN,
    "X_O": COL_XO,
    "log_mdot": COL_LOGMDOT,
}


# highest upper IMF mass limit accepted by StellarDataLoader._get_mass_grid per Z
# (the grid logic raises if imf_mass_limits[-1] exceeds the track coverage)
_MAX_UPPER = {
    "MW": 500.0, "MWC": 300.0, "LMC": 300.0,
    "SMC": 120.0, "IZw18": 120.0, "Z0": 300.0,
}


def _make_loader(metallicity, rotation):
    """Build a StellarDataLoader configured to expose the full (VMS) mass grid."""
    upper = _MAX_UPPER.get(metallicity, 120.0)
    if rotation and metallicity in ("MW", "Z0"):
        upper = min(upper, 120.0 if metallicity == "Z0" else 300.0)
    cfg = StellarPopulationConfig(
        metallicity=metallicity,
        rotation=rotation,
        imf_mass_limits=(0.1, 0.5, upper),
    )
    return StellarDataLoader(cfg)


def load_all_tracks(metallicity="MW", rotation=False):
    """
    Load every single-star track for a metallicity as a list of clean dicts,
    sorted by ascending initial mass.

    Returns
    -------
    masses : np.ndarray
        Initial masses [Msun], ascending.
    tracks : list[dict]
        One dict per mass with keys from FIELDS, each a 1D array along the track,
        plus 'initial_mass'. Same order as `masses`.
    """
    loader = _make_loader(metallicity, rotation)
    grid = np.asarray(loader.mass_grid, dtype=float)         # as ordered in the file
    raw = loader.load_evolution_tracks()                     # (n_mass*n_t, ncol)
    blocks = np.array_split(raw, len(grid))                  # matches production parsing

    tracks = []
    init_masses = []
    for m_init, blk in zip(grid, blocks):
        d = {name: np.asarray(blk[:, col], dtype=float) for name, col in FIELDS.items()}
        d["initial_mass"] = float(m_init)
        tracks.append(d)
        init_masses.append(float(m_init))

    order = np.argsort(init_masses)
    masses = np.asarray(init_masses)[order]
    tracks = [tracks[i] for i in order]
    return masses, tracks


def load_clean_track(metallicity, mass, rotation=False, tol=0.05):
    """Return the clean track dict whose initial mass matches `mass` (within tol, relative)."""
    masses, tracks = load_all_tracks(metallicity, rotation)
    i = int(np.argmin(np.abs(masses - mass)))
    if abs(masses[i] - mass) > tol * mass:
        raise ValueError(
            f"No track within {tol:.0%} of {mass} Msun for {metallicity}; "
            f"nearest is {masses[i]:.3g} Msun."
        )
    return tracks[i]


# ---------------------------------------------------------------------------
# Primary EEP markers (MIST-driven)
# ---------------------------------------------------------------------------
# Feedback-relevant thresholds in effective temperature. The ionizing output
# Q(H) is controlled by Teff and its decline sets in at logTeff ~ 4.45
# (data-driven; the -0.5 dex Q(H) point clusters there across mass), so T_DOWN
# is anchored at T_CLIFF and the blue return at T_RETURN (hysteresis gap below).
T_CLIFF = 4.45   # logTeff onset of the ionizing decline (cool-excursion entry)
T_RETURN = 4.30  # logTeff for counting a recovery (hysteresis gap below T_CLIFF)
WR_XH_THRESHOLD = 0.4   # surface H marking a Wolf-Rayet (WN onset)


# Each marker carries its detectable definition (`defn`) and its provenance
# (`source`). We cannot use MIST's primary EEPs verbatim because those key off
# CENTRAL abundances/temperature (Dotter 2016), which pySTARBURST99 does not
# expose; instead each marker is grounded in one of three places:
#   - "pysb99": a threshold the pySTARBURST99 spectral classifier itself uses to
#     switch spectral libraries (pysb99_core.py), so the marker coincides with a
#     real change of stellar type in the production code;
#   - "MIST": a MIST primary-EEP criterion that IS expressible in our observables
#     (ZAMS; RGBTip = Teff minimum; TAMS ~ the observational turn-off);
#   - "feedback": an effective-temperature threshold marking the ionizing-output
#     transition, validated empirically (dropping it degrades Q(H) alignment; see
#     eep_curvature_anchor_test.py).
# The two pysb99 hot boundaries (WR cut at logTeff 4.4, OB cut at Teff 20000 K =
# logTeff 4.301) sit right at our T_DOWN/T_UP, so those are not arbitrary.
#
# Markers are listed in nominal evolutionary order. A track carries only those it
# reaches ("not reached" -> absent). Pairwise interpolation uses the markers two
# adjacent masses share.
MARKER_INFO = {
    "ZAMS":     dict(defn="track start",                         source="MIST ZAMS", reliable=True),
    "MS_MID":   dict(defn="mid-MS, halfway in age to MS_TO",     source="MIST IAMS (weak: no central H)", reliable=False),
    "MS_TO":    dict(defn="turn-off: Teff drops 0.05 dex below its running max", source="MIST TAMS / standard turn-off (no pysb99 analog)", reliable=True),
    "T_DOWN":   dict(defn="logTeff < 4.45 (ionizing decline onset)", source="feedback; ~ pysb99 WR hot cut logTeff 4.4 (pysb99_core.py:1316)", reliable=True),
    "TEFF_MIN": dict(defn="coolest point (argmin logTeff)",      source="MIST RGBTip (Teff-min); pysb99 cool-star regime logTeff<3.65 (pysb99_core.py:1393)", reliable=True),
    "WN_ON":    dict(defn="surface X(H) < 0.4 (envelope stripping)", source="pysb99 WR gate (pysb99_core.py:1316)", reliable=True),
    "WC_ON":    dict(defn="surface X(C) > X(N) after WN",        source="pysb99 WN->WC C/N split (pysb99_core.py:1337,1351)", reliable=True),
    "T_UP":     dict(defn="logTeff > 4.30 after the minimum (blue return)", source="pysb99 OB cut Teff>20000K=logTeff 4.301 (pysb99_core.py:1400)", reliable=True),
    "END":      dict(defn="track end",                           source="terminal / pre-SN", reliable=True),
}
MS_TO_DROP = 0.05   # dex below the running-max logTeff that marks leaving the MS band


def detect_eeps_comprehensive(track, t_cliff=T_CLIFF, t_return=T_RETURN):
    """
    Detect the full MIST-driven marker set on a track.

    Returns
    -------
    eeps : dict[str, int]
        Only the markers that are actually reached, mapped to their track index,
        returned in ascending index order.
    """
    Te = track["logTeff"]
    XH = track["X_H"]
    XC = track["X_C"]
    XN = track["X_N"]
    n = len(Te)
    eeps = {"ZAMS": 0}

    # MS turn-off: first point where Teff has dropped MS_TO_DROP dex below its
    # running maximum (i.e. the star leaves the hot main-sequence band).
    runmax = np.maximum.accumulate(Te)
    left = np.where(Te < runmax - MS_TO_DROP)[0]
    i_to = int(left[0]) if len(left) else None
    if i_to is not None:
        eeps["MS_TO"] = i_to
        # mid-MS proxy (IAMS): halfway in age between ZAMS and MS turn-off
        age = track["age"]
        t_half = 0.5 * (age[0] + age[i_to])
        eeps["MS_MID"] = int(np.argmin(np.abs(age[:i_to + 1] - t_half)))

    below = np.where(Te < t_cliff)[0]
    if len(below):
        eeps["T_DOWN"] = int(below[0])
        i_min = int(np.argmin(Te))
        eeps["TEFF_MIN"] = i_min
        up = np.where(Te[i_min:] > t_return)[0]
        if len(up):
            eeps["T_UP"] = i_min + int(up[0])

    wn = np.where(XH < WR_XH_THRESHOLD)[0]
    if len(wn):
        eeps["WN_ON"] = int(wn[0])
        wc = np.where((XC[wn[0]:] > XN[wn[0]:]))[0]
        if len(wc):
            eeps["WC_ON"] = wn[0] + int(wc[0])

    eeps["END"] = n - 1
    # return ordered by index
    return dict(sorted(eeps.items(), key=lambda kv: kv[1]))


# ---------------------------------------------------------------------------
# Feedback database
# ---------------------------------------------------------------------------
# feedback database (single_star_tracks.h5) metallicity group names
_DB_ZGRP = {
    "MWC": "Z_0.0200", "MW": "Z_0.0140", "LMC": "Z_0.0060", "SMC": "Z_0.0020",
}
from toddlers._paths import get_database_path
_DB_PATH = get_database_path()
FEEDBACK_QUANTITIES = ["Q_HI", "Q_HeI", "Q_HeII", "L_bol", "wind_power", "wind_momentum"]


def load_feedback_db(metallicity, db_path=_DB_PATH, quantities=None):
    """
    Load the feedback database for a metallicity.

    Parameters
    ----------
    quantities : list[str] or None
        Which feedback quantities to read. Defaults to FEEDBACK_QUANTITIES; the
        database also stores L_LyW and the L_ion_* channels, which the stochastic
        feedback path requests, so any subset present in the file may be passed.

    Returns
    -------
    masses : np.ndarray   ascending initial masses [Msun]
    age : np.ndarray      common time grid [Myr]
    fb : dict[str, np.ndarray]   each (n_mass, n_age), in log10 (DB native units)
    """
    import h5py
    if quantities is None:
        quantities = FEEDBACK_QUANTITIES
    zgrp = _DB_ZGRP[metallicity]
    with h5py.File(db_path, "r") as f:
        age = np.array(f["metadata"]["time_grid"])
        g = f[zgrp]
        keys = [k for k in g if k.startswith("mass_") and isinstance(g[k], h5py.Group)]
        ms = np.array([g[k].attrs["mass_Msun"] for k in keys])
        order = np.argsort(ms)
        keys = [keys[i] for i in order]
        masses = ms[order]
        fb = {q: np.array([np.array(g[k][q]) for k in keys]) for q in quantities}
    return masses, age, fb


# ---------------------------------------------------------------------------
# EEP interpolation: pairwise shared-marker basis + MIST secondary sampling
# ---------------------------------------------------------------------------
def _lcs(a, b):
    """Longest common subsequence of two ordered name lists (order-preserving)."""
    na, nb = len(a), len(b)
    dp = [[0] * (nb + 1) for _ in range(na + 1)]
    for i in range(na - 1, -1, -1):
        for j in range(nb - 1, -1, -1):
            dp[i][j] = (1 + dp[i + 1][j + 1]) if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    out, i, j = [], 0, 0
    while i < na and j < nb:
        if a[i] == b[j]:
            out.append(a[i]); i += 1; j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def _segment_ages_metric(track, lo, hi, n_sec):
    """`n_sec` ages within [lo, hi] placed at equal increments of the MIST HRD
    metric distance (Dotter 2016, Eq. 1): the cumulative Euclidean path length in
    (logTeff, logL), weighted so each spans [0, 1] over the full track and hence
    contributes to the distance in roughly equal amounts. The metric is
    monotonically non-decreasing by construction, so the returned ages increase.
    More EEPs land where the track moves fast in the HRD (the transitions)."""
    Te, L = track["logTeff"], track["logL"]
    wT = 1.0 / max(Te.max() - Te.min(), 1e-6) ** 2
    wL = 1.0 / max(L.max() - L.min(), 1e-6) ** 2
    step = 1 if lo < hi else -1
    idx = np.arange(lo, hi + step, step)
    age_ev = track["age"][idx] / 1e6
    d = np.sqrt(wT * np.diff(Te[idx]) ** 2 + wL * np.diff(L[idx]) ** 2)
    D = np.concatenate([[0.0], np.cumsum(d)])
    if D[-1] <= 0:
        return np.full(n_sec, age_ev[-1])
    targets = np.linspace(0.0, D[-1], n_sec + 1)[1:]
    j = np.clip(np.searchsorted(D, targets), 1, len(D) - 1)
    D0, D1 = D[j - 1], D[j]
    frac = np.where(D1 > D0, (targets - D0) / (D1 - D0), 0.0)
    return age_ev[j - 1] + frac * (age_ev[j] - age_ev[j - 1])


def _segment_ages_var(track, lo, hi, var, n_sec):
    """`n_sec` ages within [lo, hi] sampled uniformly in `var`
    ('age'|'logTeff'|'logL'|'metric') using a monotonic envelope so the returned
    ages are monotonically increasing. 'metric' uses the MIST HRD path length."""
    if hi == lo:
        return np.full(n_sec, track["age"][lo] / 1e6)
    if var == "metric":
        return _segment_ages_metric(track, lo, hi, n_sec)
    step = 1 if lo < hi else -1
    idx = np.arange(lo, hi + step, step)
    age_ev = track["age"][idx] / 1e6
    if var in ("age", "time"):
        return np.linspace(age_ev[0], age_ev[-1], n_sec + 1)[1:]
    V = track[var][idx]
    V0, V1 = V[0], V[-1]
    targets = np.linspace(V0, V1, n_sec + 1)[1:]
    if V1 < V0:
        env = np.minimum.accumulate(V)
        return np.array([age_ev[int(np.argmax(env <= t))] for t in targets])
    env = np.maximum.accumulate(V)
    return np.array([age_ev[int(np.argmax(env >= t))] for t in targets])


# Per-segment sampler: chooses the coordinate along which secondary EEPs are
# placed within each (marker_a -> marker_b) segment. `sampler_mist` (the MIST HRD
# metric) is the production choice; the others exist only for the validation
# comparison in eep_tests/ (a leave-one-out study showed the reconstructed
# feedback HISTORY is insensitive to this choice, so we pick the structurally
# principled MIST metric -- see eep_shape_validation.py).
_TEFF_CHAIN = {"MS_TO", "T_DOWN", "TEFF_MIN", "T_UP"}


def sampler_mist(a, b):
    """Place secondary EEPs by the MIST HRD metric (Dotter 2016) in every segment."""
    return "metric"


def sampler_time(a, b):
    return "time"


def sampler_teff(a, b):
    return "logTeff" if (a in _TEFF_CHAIN and b in _TEFF_CHAIN) else "time"


def sampler_phys(a, b):
    return "logTeff" if (a in _TEFF_CHAIN and b in _TEFF_CHAIN) else "logL"


def pairwise_basis_ages(track, eeps, shared, n_sec, sampler):
    """Ages at the EEP points for `track`, using `shared` (ordered marker names)
    as segment boundaries and `sampler(a,b)` to pick each segment's variable."""
    ages = [track["age"][eeps[shared[0]]] / 1e6]
    for a, b in zip(shared[:-1], shared[1:]):
        ages.extend(_segment_ages_var(track, eeps[a], eeps[b], sampler(a, b), n_sec).tolist())
    return np.asarray(ages)


def interpolate_feedback_pairwise(metallicity, m_target, m_lo, m_hi, n_sec=20,
                                  sampler=sampler_mist, use_ms_mid=False,
                                  rotation=False, db=None):
    """
    EEP-interpolate feedback to `m_target` from bracketing grid masses, using the
    comprehensive primary markers shared (longest common subsequence) by the two
    bracketing tracks as phase-aligned segment boundaries, and the MIST HRD metric
    (`sampler_mist`) to place secondary EEPs within each segment. Feedback is
    interpolated in log at fixed EEP, with log-mass as the independent variable.

    The unreliable `MS_MID` marker is dropped by default (`use_ms_mid=False`); it
    had no measurable effect in validation. `sampler` and `use_ms_mid` are exposed
    only so the validation scripts can compare alternatives.

    Returns (ages, fb_dict, ok). ok=False if the two tracks share no usable
    ZAMS..END marker sequence or their bases differ in length.
    """
    if db is None:
        db = load_feedback_db(metallicity)
    t_lo = load_clean_track(metallicity, m_lo, rotation)
    t_hi = load_clean_track(metallicity, m_hi, rotation)
    e_lo = detect_eeps_comprehensive(t_lo)
    e_hi = detect_eeps_comprehensive(t_hi)
    if not use_ms_mid:
        e_lo.pop("MS_MID", None); e_hi.pop("MS_MID", None)
    shared = _lcs(list(e_lo.keys()), list(e_hi.keys()))
    if len(shared) < 2 or shared[0] != "ZAMS" or shared[-1] != "END":
        return None, None, False
    a_lo = pairwise_basis_ages(t_lo, e_lo, shared, n_sec, sampler)
    a_hi = pairwise_basis_ages(t_hi, e_hi, shared, n_sec, sampler)
    if len(a_lo) != len(a_hi):
        return None, None, False

    masses_db, age_grid, fb = db
    i_lo = int(np.argmin(np.abs(masses_db - t_lo["initial_mass"])))
    i_hi = int(np.argmin(np.abs(masses_db - t_hi["initial_mass"])))
    w = (np.log10(m_target) - np.log10(m_lo)) / (np.log10(m_hi) - np.log10(m_lo))
    ages = (1 - w) * a_lo + w * a_hi
    out = {}
    for q in FEEDBACK_QUANTITIES:
        lo = np.interp(a_lo, age_grid, fb[q][i_lo])
        hi = np.interp(a_hi, age_grid, fb[q][i_hi])
        out[q] = np.power(10.0, (1 - w) * lo + w * hi)
    return ages, out, True


# ---------------------------------------------------------------------------
# Production driver: cached, multi-quantity EEP feedback for the stochastic path
# ---------------------------------------------------------------------------
class EEPFeedbackInterpolator:
    """
    Efficient EEP feedback evaluator for one metallicity, built once and queried
    per star. It loads the tracks and the feedback database a single time and
    precomputes, for every adjacent grid bracket, the shared markers and the
    bracketing feedback on their EEP bases. `query(mass, ages)` then costs only a
    log-mass blend plus an interpolation onto the requested ages.

    Output convention matches stochastic.database.query_database: a dict mapping
    each requested quantity to a log10 array aligned with `ages_myr`. (That was
    checked: query_database returns the database values, which are stored in log10,
    and the caller converts with 10**.) A mass at or beyond the grid ends, or in a
    bracket whose two tracks share no usable marker sequence, falls back to
    nearest-track snapping (nearest grid mass, age-interpolated), the same
    assignment the production default makes there.
    """

    def __init__(self, metallicity, database_path=None, quantities=None,
                 rotation=False, n_sec=20):
        self.metallicity = metallicity
        self.quantities = list(quantities) if quantities else list(FEEDBACK_QUANTITIES)
        self.n_sec = n_sec
        self.masses, self.tracks = load_all_tracks(metallicity, rotation)
        db_path = database_path or _DB_PATH
        self._db_masses, self.age_grid, self._fb = load_feedback_db(
            metallicity, db_path, quantities=self.quantities)
        self._db_idx = [int(np.argmin(np.abs(self._db_masses - m))) for m in self.masses]
        self._brackets = [self._build_bracket(k) for k in range(len(self.masses) - 1)]

    def _build_bracket(self, k):
        """Precompute one bracket, or None if its tracks share no usable sequence."""
        t_lo, t_hi = self.tracks[k], self.tracks[k + 1]
        e_lo = detect_eeps_comprehensive(t_lo); e_lo.pop("MS_MID", None)
        e_hi = detect_eeps_comprehensive(t_hi); e_hi.pop("MS_MID", None)
        shared = _lcs(list(e_lo.keys()), list(e_hi.keys()))
        if len(shared) < 2 or shared[0] != "ZAMS" or shared[-1] != "END":
            return None
        a_lo = pairwise_basis_ages(t_lo, e_lo, shared, self.n_sec, sampler_mist)
        a_hi = pairwise_basis_ages(t_hi, e_hi, shared, self.n_sec, sampler_mist)
        if len(a_lo) != len(a_hi):
            return None
        i_lo, i_hi = self._db_idx[k], self._db_idx[k + 1]
        lo = {q: np.interp(a_lo, self.age_grid, self._fb[q][i_lo]) for q in self.quantities}
        hi = {q: np.interp(a_hi, self.age_grid, self._fb[q][i_hi]) for q in self.quantities}
        return dict(a_lo=a_lo, a_hi=a_hi, lo=lo, hi=hi)

    def _snap(self, mass, ages):
        i = int(np.argmin(np.abs(self._db_masses - mass)))
        return {q: np.interp(ages, self.age_grid, self._fb[q][i]) for q in self.quantities}

    def query(self, mass, ages_myr):
        """log10 feedback for a (possibly off-grid) mass at the given ages [Myr]."""
        ages_myr = np.asarray(ages_myr, dtype=float)
        if mass <= self.masses[0] or mass >= self.masses[-1]:
            return self._snap(mass, ages_myr)
        k = int(np.searchsorted(self.masses, mass)) - 1
        br = self._brackets[k]
        if br is None:
            return self._snap(mass, ages_myr)
        m_lo, m_hi = self.masses[k], self.masses[k + 1]
        w = (np.log10(mass) - np.log10(m_lo)) / (np.log10(m_hi) - np.log10(m_lo))
        ages_eep = (1 - w) * br["a_lo"] + w * br["a_hi"]    # monotonic by construction
        out = {}
        for q in self.quantities:
            blended = (1 - w) * br["lo"][q] + w * br["hi"][q]   # log10, at ages_eep
            out[q] = np.interp(ages_myr, ages_eep, blended)
        return out


if __name__ == "__main__":
    # quick smoke test
    for Z in ["MW", "LMC", "MWC", "SMC", "IZw18"]:
        masses, tracks = load_all_tracks(Z)
        lifetimes = [t["age"].max() / 1e6 for t in tracks]
        print(f"{Z:6s}: {len(masses):2d} masses "
              f"[{masses.min():.2g}-{masses.max():.4g} Msun], "
              f"npts/track={len(tracks[0]['age'])}, "
              f"lifetime(Mmax)={lifetimes[-1]:.2f} Myr, "
              f"lifetime(Mmin)={lifetimes[0]/1e3:.2f} Gyr")
