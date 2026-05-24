"""
Population feedback: snapping vs EEP interpolation, with the fully-sampled limit.

Answers the referee's two questions in one figure:
  (1) Does interpolating each star's feedback in mass (EEP) instead of snapping it
      to the nearest grid track change the POPULATION feedback?
  (2) What happens when the IMF is fully sampled?

For a burst of cluster mass M_cl we draw the IMF (continuous masses), then assign
each massive star's feedback two ways with the SAME realization:
  - snap : nearest grid track (production default; no interpolation)
  - eep  : phase-aligned mass interpolation between bracketing grid tracks
and sum into a population time series. We repeat over realizations and report the
time-integrated budget per channel: mean and scatter for each method.

The fully-sampled limit is the IMF-weighted integral over the track GRID (each grid
mass weighted by the IMF count in its nearest-mass bin). This is what the
deterministic population synthesis computes (it bins the IMF onto grid masses), and
it is the large-N limit that snapping converges to. EEP interpolates between grid
masses, so it converges to a slightly different value (the inter-grid refinement).

Speed note: only stars above MIN_MASS contribute appreciably to Q(H)/winds, so we
sum those. Feedback is read from the in-memory EEP interpolator: `.query` (eep) and
`._snap` (snap, identical to the production nearest-track lookup).
"""
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from toddlers.pysb99.eep_interpolation import EEPFeedbackInterpolator
from toddlers.pysb99.stochastic.sampling import sample_imf

STYLE = os.path.join(ROOT, "paper", "paper_style.mplstyle")
if os.path.exists(STYLE):
    plt.style.use(STYLE)
DB = os.path.join(ROOT, "src", "database", "single_star_tracks.h5")

Z = "MW"
QOI = [("Q_HI", r"$\int Q(\mathrm{H})\,dt$"),
       ("wind_power", r"$\int \dot E_{\rm wind}\,dt$")]
MIN_MASS = 8.0                       # feedback-relevant stars
CLUSTER_MASSES = [3e2, 1e3, 3e3, 1e4, 3e4, 1e5]
N_REAL = 30
TGRID = np.arange(0.0, 30.01, 0.1)   # Myr


def _kroupa(m):
    return np.where(m < 0.5, m ** -1.3, 0.5 * m ** -2.3)


def population_integral(ei, masses, mode):
    """Time-integrated feedback for one population realization (linear-space sum)."""
    tot = {q: np.zeros(len(TGRID)) for q, _ in QOI}
    for m in masses:
        fb = ei.query(m, TGRID) if mode == "eep" else ei._snap(m, TGRID)
        for q, _ in QOI:
            tot[q] += np.power(10.0, fb[q])
    return {q: np.trapz(tot[q], TGRID) for q, _ in QOI}


def full_imf_interp_per_msun(ei):
    """Fully-sampled budget per Msun using EEP (phase-aligned) interpolation, i.e.
    the continuous-IMF integral of the interpolated feedback. This is the limit the
    deterministic path corresponds to: pySTARBURST99 interpolates tracks in mass
    (fixed evolutionary-row index), which agrees with EEP to ~3% (see
    eep_vs_fixedrow_fullsampling.py). Linear in cluster mass, so computed once."""
    m_full = np.geomspace(0.1, 100.0, 6000)
    xi = _kroupa(m_full)
    dm = np.gradient(m_full)
    norm_mass = np.trapz(m_full * xi, m_full)
    sel = m_full >= MIN_MASS
    mm, dN = m_full[sel], xi[sel] * dm[sel] / norm_mass     # per Msun of cluster
    out = {}
    for q, _ in QOI:
        acc = np.zeros(len(TGRID))
        for m, n in zip(mm, dN):
            acc += n * np.power(10.0, ei.query(m, TGRID)[q])
        out[q] = float(np.trapz(acc, TGRID))
    return out


def main():
    ei = EEPFeedbackInterpolator(Z, DB, quantities=[q for q, _ in QOI])
    print(f"{Z}: interpolator built ({len(ei.masses)} masses)")
    rng = np.random.default_rng(0)
    full_per_msun = full_imf_interp_per_msun(ei)    # interpolated (deterministic-equiv) limit

    res = {q: {k: [] for k in ("snap_m", "snap_s", "eep_m", "eep_s", "full")} for q, _ in QOI}
    for m_cl in CLUSTER_MASSES:
        snap_I = {q: [] for q, _ in QOI}
        eep_I = {q: [] for q, _ in QOI}
        for _ in range(N_REAL):
            seed = int(rng.integers(1 << 31))
            masses = sample_imf(total_mass=m_cl, imf_name="kroupa",
                                m_min=0.1, m_max=100.0, seed=seed)
            masses = masses[masses >= MIN_MASS]
            Is = population_integral(ei, masses, "snap")
            Ie = population_integral(ei, masses, "eep")
            for q, _ in QOI:
                snap_I[q].append(Is[q]); eep_I[q].append(Ie[q])
        for q, _ in QOI:
            res[q]["snap_m"].append(np.mean(snap_I[q]))
            res[q]["snap_s"].append(np.std(snap_I[q]))
            res[q]["eep_m"].append(np.mean(eep_I[q]))
            res[q]["eep_s"].append(np.std(eep_I[q]))
            res[q]["full"].append(full_per_msun[q] * m_cl)
        line = f"  M_cl={m_cl:.0e}: "
        for q, _ in QOI:
            fl = full_per_msun[q] * m_cl
            sm, em = np.mean(snap_I[q]), np.mean(eep_I[q])
            line += (f"{q}: snap/full={sm/fl:.3f} eep/full={em/fl:.3f} "
                     f"(scatter {100*np.std(snap_I[q])/sm:.0f}%)  ")
        print(line)

    Mc = np.array(CLUSTER_MASSES)
    fig, axes = plt.subplots(1, len(QOI), figsize=(3.6 * len(QOI), 3.2), squeeze=False)
    for j, (q, qlab) in enumerate(QOI):
        ax = axes[0][j]
        R = res[q]
        full = np.array(R["full"])
        ax.errorbar(Mc, np.array(R["snap_m"]) / full, yerr=np.array(R["snap_s"]) / full,
                    fmt="o-", color="#4683DE", ms=4, capsize=2, label="snap", zorder=2)
        ax.errorbar(Mc * 1.08, np.array(R["eep_m"]) / full, yerr=np.array(R["eep_s"]) / full,
                    fmt="s-", color="#CB4035", ms=4, capsize=2, label="eep", zorder=3)
        ax.axhline(1.0, color="0.4", lw=0.9, ls="--",
                   label="full IMF sampling (interpolated)", zorder=1)
        ax.set_xscale("log")
        ax.set_xlabel(r"$M_{\rm cl}$  [$M_\odot$]")
        ax.set_ylabel(qlab + r"  / full-IMF")
        ax.set_title(q, fontsize=8)
        if j == 0:
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"{Z}: snap vs EEP population feedback, and the fully-sampled limit",
                 fontsize=9)
    out = os.path.join(HERE, "eep_population_snap_vs_eep.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
