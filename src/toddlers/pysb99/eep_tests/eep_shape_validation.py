"""
Validate EEP interpolation on the feedback *history*, not just its time integral.

Why a separate test. The integral test (`eep_config_validation.py`) rewards good
time-quadrature: uniform-in-age sampling is the optimal quadrature of int Q dt, so
it scores well there for reasons that have nothing to do with phase alignment. But
an EEP basis exists to align physical *state* across masses, and the shell evolution
responds to the feedback history Q(t) instant by instant (it sets fragmentation
timing), not only to the lifetime budget. So we score the reconstructed history
directly: median |Δ log Q| over the star's alive lifetime, where uniform-age has no
built-in advantage. The integral error is reported alongside, for contrast.

Method (leave-one-out). For each interior grid mass m_t (>= MIN_MASS) we reconstruct
it from its neighbours (m_lo, m_hi) with `interpolate_feedback_pairwise`, map the
reconstructed feedback onto the database age grid, and compare to the true track over
the alive portion (true log feedback more than FLOOR_MARGIN dex above the floor).
The reference is identical for every sampler, so this is a fair *relative* ranking
even though the database is coarse at the transitions.
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
                                      sampler_time, sampler_teff,
                                      sampler_phys, sampler_mist)

QOI = ["Q_HI", "wind_power", "wind_momentum"]
MIN_MASS = 15.0
FLOOR_MARGIN = 1.0   # dex above the per-channel floor counted as "alive"
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
    floor = {q: fb[q].min() for q in QOI}

    shape = {s: {q: [] for q in QOI} for s in SAMPLERS}    # median |dlogQ| over life
    integ_sp = {s: {q: [] for q in QOI} for s in SAMPLERS}  # integral over sparse EEPs
    integ_fn = {s: {q: [] for q in QOI} for s in SAMPLERS}  # integral after fine interp
    cover = {s: 0 for s in SAMPLERS}

    for k in range(1, len(grid) - 1):
        m_t, m_lo, m_hi = grid[k], grid[k - 1], grid[k + 1]
        if m_t < MIN_MASS:
            continue
        life = lifetime(masses, tracks, m_t)
        full = age_grid <= life
        for sname, sfn in SAMPLERS.items():
            ages, fbp, ok = interpolate_feedback_pairwise(
                Z, m_t, m_lo, m_hi, sampler=sfn, use_ms_mid=False, db=db)
            if not ok:
                continue
            o = np.argsort(ages); ages = ages[o]
            cover[sname] += 1
            for q in QOI:
                true_log = fb[q][k]
                alive = (true_log > floor[q] + FLOOR_MARGIN) & full \
                        & (age_grid >= ages[0]) & (age_grid <= ages[-1])
                if alive.sum() >= 3:
                    rec_log = np.interp(age_grid[alive], ages, np.log10(fbp[q][o]))
                    shape[sname][q].append(np.median(np.abs(rec_log - true_log[alive])))
                t = integ(age_grid, np.power(10.0, true_log), life)
                # (a) integral over the sparse EEP points directly (quadrature-sensitive)
                v_sp = integ(ages, fbp[q][o], life)
                if np.isfinite(t) and t > 0 and np.isfinite(v_sp):
                    integ_sp[sname][q].append(abs(100 * (v_sp - t) / t))
                # (b) integral the way the evolution consumes it: interpolate the EEP
                #     curve onto the fine grid first, then integrate
                rec_fine = np.power(10.0, np.interp(age_grid[full], ages,
                                                    np.log10(fbp[q][o])))
                v_fn = np.trapz(rec_fine, age_grid[full])
                if np.isfinite(t) and t > 0 and np.isfinite(v_fn):
                    integ_fn[sname][q].append(abs(100 * (v_fn - t) / t))

    print(f"=== {Z}: leave-one-out, {cover[list(SAMPLERS)[0]]} interior masses >= {MIN_MASS:.0f} Msun ===")
    print("  shape   : median |dex| error of the feedback HISTORY over the alive lifetime")
    print("  int(sp) : |error| of budget integrated over the SPARSE EEP points (quadrature-sensitive)")
    print("  int(fn) : |error| of budget after interpolating to the fine grid (how evolution uses it)\n")
    hdr = f"{'sampler':>8} | " + " | ".join(q.center(26) for q in QOI)
    print(hdr)
    print(f"{'':>8} | " + " | ".join(f"{'shape':>8} {'int(sp)':>8} {'int(fn)':>7}" for _ in QOI))
    print("-" * len(hdr))
    for sname in SAMPLERS:
        cells = []
        for q in QOI:
            sh = np.median(shape[sname][q]) if shape[sname][q] else np.nan
            isp = np.median(integ_sp[sname][q]) if integ_sp[sname][q] else np.nan
            ifn = np.median(integ_fn[sname][q]) if integ_fn[sname][q] else np.nan
            cells.append(f"{sh:>7.3f}d {isp:>7.1f}% {ifn:>6.1f}%")
        print(f"{sname:>8} | " + " | ".join(cells))


if __name__ == "__main__":
    zs = sys.argv[1:] or ["MW", "LMC", "SMC", "MWC"]
    for z in zs:
        run(z)
        print()
