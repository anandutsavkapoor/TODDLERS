"""
Which structural variable best aligns each feedback channel?

Rather than assume a single anchor, we test several candidate variables and let
the data rank them, per feedback quantity.

Method. For each star, normalise the feedback to its main-sequence value, so we
compare the *transition* (the drop/rise relative to the MS), not the absolute
level (which EEP interpolation handles by interpolating in mass). For a candidate
variable V, a good aligner makes log(Q/Q_MS) a tight, single-valued function of V
across masses: at fixed V every star is at the same point in its own feedback
evolution. We bin in V and report the mean cross-mass scatter of log(Q/Q_MS).
Lower scatter = better aligner for that channel.

Candidate variables: logTeff, surface X(H), log mass-loss rate, logL, surface X(He).
Channels: Q(H) (ionizing), wind power, wind momentum.
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from toddlers.pysb99.eep_interpolation import load_all_tracks, load_feedback_db

Z = "MW"
MIN_MASS = 20.0          # feedback-relevant
FLOOR_MARGIN = 1.0       # exclude points within this many dex of the floor (dead star)
N_BINS = 25

CANDIDATES = {
    "logTeff": "logTeff",
    "X_H": "X_H",
    "log_mdot": "log_mdot",
    "logL": "logL",
    "X_He": "X_He",
}
CHANNELS = ["Q_HI", "wind_power", "wind_momentum"]


def main():
    masses, tracks = load_all_tracks(Z)
    mdb, age, fb = load_feedback_db(Z)
    use = [m for m in masses if m >= MIN_MASS]

    # collect, per channel and candidate, pooled (V, dlogQ) with a mass label
    scores = {}
    for ch in CHANNELS:
        floor = fb[ch].min()  # log floor value used for dead stars
        for vname, vkey in CANDIDATES.items():
            Vs, Ds, Ms = [], [], []
            for m in use:
                i = int(np.argmin(np.abs(masses - m)))
                idb = int(np.argmin(np.abs(mdb - m)))
                t = tracks[i]
                V_on_age = np.interp(age, t["age"] / 1e6, t[vkey])
                q = fb[ch][idb]
                alive = q > floor + FLOOR_MARGIN
                if alive.sum() < 5:
                    continue
                Vs.append(V_on_age[alive])
                Ds.append(q[alive] - q[alive][0])   # log(Q/Q_MS) ~ q - q at first alive pt
                Ms.append(np.full(alive.sum(), m))
            V = np.concatenate(Vs); D = np.concatenate(Ds); M = np.concatenate(Ms)
            # bin in V; per bin compute scatter of D across DIFFERENT masses
            edges = np.linspace(np.percentile(V, 1), np.percentile(V, 99), N_BINS + 1)
            which = np.digitize(V, edges)
            per_bin = []
            for b in range(1, N_BINS + 1):
                sel = which == b
                if np.unique(M[sel]).size >= 3:     # need >=3 masses in the bin
                    per_bin.append(np.std(D[sel]))
            scores[(ch, vname)] = np.mean(per_bin) if per_bin else np.nan

    # report
    print(f"Z={Z}, masses>= {MIN_MASS}  (mean cross-mass scatter of log(Q/Q_MS) at fixed V; lower=better)\n")
    header = "channel".ljust(15) + "".join(v.rjust(11) for v in CANDIDATES)
    print(header)
    for ch in CHANNELS:
        row = ch.ljust(15)
        vals = {v: scores[(ch, v)] for v in CANDIDATES}
        best = min(vals, key=lambda k: (np.inf if np.isnan(vals[k]) else vals[k]))
        for v in CANDIDATES:
            s = vals[v]
            cell = f"{s:.3f}" if np.isfinite(s) else "  nan"
            row += (("*" + cell).rjust(11) if v == best else cell.rjust(11))
        print(row)
    print("\n(* = best aligner for that channel)")


if __name__ == "__main__":
    main()
