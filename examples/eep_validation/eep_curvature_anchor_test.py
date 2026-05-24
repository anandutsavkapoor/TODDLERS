"""
Head-to-head: geometry-only (high-change) anchors vs the semantic marker anchors.

Motivation. The production anchors mix robust turning points (MS turn-off, RSG tip)
with two hand-tuned Teff threshold crossings (T_DOWN at logTeff=4.45, T_UP at 4.30).
This asks whether the thresholds can be dropped in favour of purely geometric
"points of high change" in the HR diagram.

Geometry-only anchors. On the normalised HRD path (logTeff, logL each scaled to
[0,1] over the track) we run Ramer-Douglas-Peucker: it repeatedly inserts the point
of maximum perpendicular deviation from the current polyline, so the first K points
are the K most significant corners, ordered along the track. Endpoints (ZAMS, END)
are included by construction. Two bracketing tracks get the same K, matched by order;
secondary EEPs are filled between anchors with the MIST metric (as in production).

We compare, by leave-one-out reconstruction of held-out grid masses, the feedback
HISTORY error (median |dex| over the alive lifetime -- the fidelity-relevant metric)
and the time-integrated budget error, for the semantic markers vs RDP-K anchors.
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from toddlers.pysb99.eep_interpolation import (
    load_all_tracks, load_clean_track, load_feedback_db,
    interpolate_feedback_pairwise, sampler_mist,
    _segment_ages_metric, FEEDBACK_QUANTITIES)

QOI = ["Q_HI", "wind_power", "wind_momentum"]
MIN_MASS = 15.0
FLOOR_MARGIN = 1.0
N_SEC = 20
K_LIST = [4, 6, 8]


# --- geometry-only anchors via Ramer-Douglas-Peucker on the normalised HRD -----
def _norm_hrd(track):
    Te, L = track["logTeff"], track["logL"]
    tn = (Te - Te.min()) / max(Te.max() - Te.min(), 1e-9)
    ln = (L - L.min()) / max(L.max() - L.min(), 1e-9)
    return np.column_stack([tn, ln])


def _max_dev(P, a, b):
    """Index in (a, b) of max perpendicular distance from the chord P[a]-P[b]."""
    p0, p1 = P[a], P[b]
    seg = p1 - p0
    n = np.hypot(*seg)
    pts = P[a + 1:b]
    if len(pts) == 0:
        return None, -1.0
    if n < 1e-12:
        d = np.hypot(*(pts - p0).T)
    else:
        d = np.abs(seg[0] * (p0[1] - pts[:, 1]) - (p0[0] - pts[:, 0]) * seg[1]) / n
    j = int(np.argmax(d))
    return a + 1 + j, float(d[j])


def rdp_anchors(track, K):
    """Indices of the K most significant HRD corners (endpoints included), sorted."""
    P = _norm_hrd(track)
    idx = [0, len(P) - 1]
    while len(idx) < K:
        s = sorted(idx)
        best_i, best_d = None, -1.0
        for a, b in zip(s[:-1], s[1:]):
            if b - a < 2:
                continue
            i, d = _max_dev(P, a, b)
            if i is not None and d > best_d:
                best_i, best_d = i, d
        if best_i is None:
            break
        idx.append(best_i)
    return sorted(idx)


def _basis_from_idx(track, idx, n_sec):
    ages = [track["age"][idx[0]] / 1e6]
    for a, b in zip(idx[:-1], idx[1:]):
        ages.extend(_segment_ages_metric(track, a, b, n_sec).tolist())
    return np.asarray(ages)


def interp_rdp(Z, m_t, m_lo, m_hi, K, db, n_sec=N_SEC):
    t_lo, t_hi = load_clean_track(Z, m_lo), load_clean_track(Z, m_hi)
    i_lo_idx, i_hi_idx = rdp_anchors(t_lo, K), rdp_anchors(t_hi, K)
    if len(i_lo_idx) != len(i_hi_idx):
        return None, None, False
    a_lo, a_hi = _basis_from_idx(t_lo, i_lo_idx, n_sec), _basis_from_idx(t_hi, i_hi_idx, n_sec)
    if len(a_lo) != len(a_hi):
        return None, None, False
    masses_db, age_grid, fb = db
    j_lo = int(np.argmin(np.abs(masses_db - t_lo["initial_mass"])))
    j_hi = int(np.argmin(np.abs(masses_db - t_hi["initial_mass"])))
    w = (np.log10(m_t) - np.log10(m_lo)) / (np.log10(m_hi) - np.log10(m_lo))
    ages = (1 - w) * a_lo + w * a_hi
    out = {q: np.power(10.0, (1 - w) * np.interp(a_lo, age_grid, fb[q][j_lo])
                       + w * np.interp(a_hi, age_grid, fb[q][j_hi]))
           for q in FEEDBACK_QUANTITIES}
    return ages, out, True


# --- validation ---------------------------------------------------------------
def shape_and_integ(ages, fbp, k, db, life, floor):
    age_grid = db[1]
    o = np.argsort(ages); ages = ages[o]
    res = {}
    for q in QOI:
        true_log = db[2][q][k]
        alive = (true_log > floor[q] + FLOOR_MARGIN) & (age_grid <= life) \
                & (age_grid >= ages[0]) & (age_grid <= ages[-1])
        sh = (np.median(np.abs(np.interp(age_grid[alive], ages, np.log10(fbp[q][o]))
                               - true_log[alive])) if alive.sum() >= 3 else np.nan)
        full = age_grid <= life
        t = np.trapz(np.power(10.0, true_log[full]), age_grid[full])
        rec = np.power(10.0, np.interp(age_grid[full], ages, np.log10(fbp[q][o])))
        ie = abs(100 * (np.trapz(rec, age_grid[full]) - t) / t) if t > 0 else np.nan
        res[q] = (sh, ie)
    return res


def run(Z):
    db = load_feedback_db(Z)
    grid, age_grid, fb = db
    masses, tracks = load_all_tracks(Z)
    floor = {q: fb[q].min() for q in QOI}
    methods = ["semantic"] + [f"rdp{K}" for K in K_LIST]
    sh = {m: {q: [] for q in QOI} for m in methods}
    ie = {m: {q: [] for q in QOI} for m in methods}
    cover = {m: 0 for m in methods}

    for k in range(1, len(grid) - 1):
        m_t, m_lo, m_hi = grid[k], grid[k - 1], grid[k + 1]
        if m_t < MIN_MASS:
            continue
        life = tracks[int(np.argmin(np.abs(masses - m_t)))]["age"].max() / 1e6
        runs = {"semantic": interpolate_feedback_pairwise(
            Z, m_t, m_lo, m_hi, sampler=sampler_mist, use_ms_mid=False, db=db)}
        for K in K_LIST:
            runs[f"rdp{K}"] = interp_rdp(Z, m_t, m_lo, m_hi, K, db)
        for meth, (ages, fbp, ok) in runs.items():
            if not ok:
                continue
            cover[meth] += 1
            r = shape_and_integ(ages, fbp, k, db, life, floor)
            for q in QOI:
                if np.isfinite(r[q][0]):
                    sh[meth][q].append(r[q][0])
                if np.isfinite(r[q][1]):
                    ie[meth][q].append(r[q][1])

    print(f"=== {Z}: semantic vs geometry-only (RDP-K) anchors ===")
    print("  shape = median |dex| of feedback history;  int = |%| integrated budget\n")
    hdr = f"{'anchors':>9} {'cov':>4} | " + " | ".join(q.center(18) for q in QOI)
    print(hdr)
    print(f"{'':>9} {'':>4} | " + " | ".join(f"{'shape':>8} {'int':>7}" for _ in QOI))
    print("-" * len(hdr))
    for meth in methods:
        row = f"{meth:>9} {cover[meth]:>4} | "
        cells = []
        for q in QOI:
            s = np.median(sh[meth][q]) if sh[meth][q] else np.nan
            i = np.median(ie[meth][q]) if ie[meth][q] else np.nan
            cells.append(f"{s:>7.3f}d {i:>6.1f}%")
        print(row + " | ".join(cells))
    print()


if __name__ == "__main__":
    for z in (sys.argv[1:] or ["MW", "LMC", "SMC", "MWC"]):
        run(z)
