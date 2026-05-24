"""
Decide the pairwise-EEP configuration empirically.

Compares, by leave-one-out reconstruction of held-out grid masses, the choices:
  - per-segment sampler: time / teff-valley / physical (Teff valley + L elsewhere)
  - MS_MID marker: included or not

For each held-out interior mass m_t we reconstruct it from its grid neighbours
(m_lo, m_hi) with `interpolate_feedback_pairwise`, integrate each feedback channel
over [0, lifetime], and compare to the true (database) integral. We report the
median |relative error| per channel for each config. (The reference is the same
for every config, so this is a fair *relative* comparison even though the database
is coarse at the transitions.)
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from toddlers.pysb99.eep_interpolation import (load_feedback_db, load_all_tracks,
                                      interpolate_feedback_pairwise,
                                      sampler_time, sampler_teff, sampler_phys,
                                      sampler_mist)

QOI = ["Q_HI", "wind_power", "wind_momentum"]
MIN_MASS = 15.0
SAMPLERS = {"time": sampler_time, "teff": sampler_teff,
            "phys": sampler_phys, "mist": sampler_mist}


def lifetime(masses, tracks, m):
    return tracks[int(np.argmin(np.abs(masses - m)))]["age"].max() / 1e6


def integ(age, val, t_max):
    m = age <= t_max
    return np.trapz(val[m], age[m]) if m.sum() >= 2 else np.nan


def run(Z="MW"):
    db = load_feedback_db(Z)
    grid, age_grid, fb = db
    masses, tracks = load_all_tracks(Z)

    configs = [(s, mm) for s in SAMPLERS for mm in (True, False)]
    err = {c: {q: [] for q in QOI} for c in configs}
    cover = {c: 0 for c in configs}

    for k in range(1, len(grid) - 1):
        m_t, m_lo, m_hi = grid[k], grid[k - 1], grid[k + 1]
        if m_t < MIN_MASS:
            continue
        life = lifetime(masses, tracks, m_t)
        true = {q: integ(age_grid, np.power(10.0, fb[q][k]), life) for q in QOI}
        for (sname, mm) in configs:
            ages, fbp, ok = interpolate_feedback_pairwise(
                Z, m_t, m_lo, m_hi, sampler=SAMPLERS[sname], use_ms_mid=mm, db=db)
            if not ok:
                continue
            o = np.argsort(ages); ages = ages[o]
            cover[(sname, mm)] += 1
            for q in QOI:
                t = true[q]
                if np.isfinite(t) and t > 0:
                    val = integ(ages, fbp[q][o], life)
                    if np.isfinite(val):
                        err[(sname, mm)][q].append(100 * (val - t) / t)

    print(f"=== {Z}: leave-one-out median |error| per channel (lower=better) ===")
    print(f"{'sampler':>8} {'MS_MID':>7} {'cover':>6} " + "".join(q.rjust(15) for q in QOI))
    for (sname, mm) in configs:
        row = f"{sname:>8} {str(mm):>7} {cover[(sname, mm)]:>6} "
        for q in QOI:
            a = np.abs(err[(sname, mm)][q])
            row += (f"{np.median(a):.1f}%".rjust(15) if len(a) else "n/a".rjust(15))
        print(row)


if __name__ == "__main__":
    zs = sys.argv[1:] or ["MW", "LMC", "SMC", "MWC"]
    for z in zs:
        run(z)
        print()
