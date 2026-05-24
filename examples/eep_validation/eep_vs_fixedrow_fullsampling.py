"""
Why not use EEP for the fully-sampled (deterministic) feedback too?

The deterministic pySTARBURST99 interpolates feedback in mass by a fixed-track-row
scheme (interpolate_param in pysb99_core.py: linear in mass at fixed row index,
relying on the track set's native row correspondence). EEP instead aligns by
detected evolutionary markers. This script asks how much the choice matters once
the IMF is fully sampled, by integrating each method's feedback over the IMF and
comparing the time-integrated budgets.

Three fully-sampled (continuous-IMF) integrals per channel:
  snap     : nearest grid track (no interpolation)
  fixedrow : interpolate feedback at fixed track-row index (deterministic-style)
  eep      : phase-aligned (MIST-marker) interpolation

The fixed-row interpolator is a feedback-space proxy for the deterministic scheme:
it maps the database feedback onto each track's rows, interpolates linearly in mass
at fixed row (blending the two tracks' ages and feedback per row), then resamples to
the time grid. It reproduces the fixed-row phase alignment (and its misalignment at
topology transitions) without re-running the spectral synthesis.
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from toddlers.pysb99.eep_interpolation import EEPFeedbackInterpolator

DB = os.path.join(ROOT, "src", "database", "single_star_tracks.h5")
QOI = ["Q_HI", "wind_power", "wind_momentum"]
MIN_MASS = 8.0
TGRID = np.arange(0.0, 30.01, 0.1)


class FixedRowInterp:
    """Deterministic-style fixed-row mass interpolation of the feedback."""

    def __init__(self, ei):
        self.ei = ei
        self.row_age = [t["age"] / 1e6 for t in ei.tracks]      # ages per track row [Myr]
        self.row_fb = []
        for k in range(len(ei.tracks)):
            i = ei._db_idx[k]
            self.row_fb.append({q: np.interp(self.row_age[k], ei.age_grid, ei._fb[q][i])
                                for q in ei.quantities})

    def query(self, mass, ages):
        ms = self.ei.masses
        if mass <= ms[0] or mass >= ms[-1]:
            return self.ei._snap(mass, ages)
        k = int(np.searchsorted(ms, mass)) - 1
        w = (mass - ms[k]) / (ms[k + 1] - ms[k])               # linear in mass (as pySB99)
        age_i = (1 - w) * self.row_age[k] + w * self.row_age[k + 1]
        out = {}
        for q in self.ei.quantities:
            logfb_i = (1 - w) * self.row_fb[k][q] + w * self.row_fb[k + 1][q]
            order = np.argsort(age_i)
            out[q] = np.interp(ages, age_i[order], logfb_i[order])
        return out


def _kroupa(m):
    return np.where(m < 0.5, m ** -1.3, 0.5 * m ** -2.3)


def full_imf_integral(query_fn, masses_grid, m_cl=1e6):
    """Continuous-IMF integrated budget per channel for a feedback query function."""
    m_full = np.geomspace(0.1, 100.0, 6000)
    xi = _kroupa(m_full)
    dm = np.gradient(m_full)
    norm_mass = np.trapz(m_full * xi, m_full)
    sel = m_full >= MIN_MASS
    mm, dN = m_full[sel], (xi[sel] * dm[sel]) * m_cl / norm_mass
    out = {q: 0.0 for q in QOI}
    # accumulate dN(m) * Q(m,t), integrate over t
    acc = {q: np.zeros(len(TGRID)) for q in QOI}
    for m, n in zip(mm, dN):
        fb = query_fn(m, TGRID)
        for q in QOI:
            acc[q] += n * np.power(10.0, fb[q])
    return {q: np.trapz(acc[q], TGRID) for q in QOI}


def main(Z="MW"):
    ei = EEPFeedbackInterpolator(Z, DB, quantities=QOI)
    fr = FixedRowInterp(ei)
    print(f"{Z}: fully-sampled (continuous-IMF) integrated budget, three interpolation schemes\n")
    I_snap = full_imf_integral(lambda m, a: ei._snap(m, a), ei.masses)
    I_fr = full_imf_integral(lambda m, a: fr.query(m, a), ei.masses)
    I_eep = full_imf_integral(lambda m, a: ei.query(m, a), ei.masses)
    print(f"{'channel':>14} {'snap/eep':>10} {'fixedrow/eep':>14}")
    for q in QOI:
        print(f"{q:>14} {I_snap[q]/I_eep[q]:>9.3f}  {I_fr[q]/I_eep[q]:>13.3f}")
    print("\n(ratios to EEP; fixedrow≈1 means the deterministic scheme already agrees with EEP)")


if __name__ == "__main__":
    for z in (sys.argv[1:] or ["MW", "LMC", "SMC", "MWC"]):
        main(z)
        print()
