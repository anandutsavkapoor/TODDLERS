"""
Visual confirmation that feedback aligns on the EEP index.

For every track we build its own EEP basis (comprehensive primary markers +
MIST-metric secondary EEPs, the production scheme) and read the feedback onto it.
Masses that share the same primary-marker sequence share an index meaning, so for
each such group we plot a feedback channel two ways:

  - vs age   : the transition (e.g. the ionizing cliff) lands at a different clock
               age for each mass -> smeared. This is why fixed-age interpolation
               blends unlike phases.
  - vs EEP index : the same transitions snap onto the SAME index across masses ->
               the phase alignment works, and interpolating at fixed index is
               meaningful. Primary markers are drawn as vertical lines.

Run:  python3 eep_index_alignment.py [Z]   (default MW; lists the marker groups).
Writes eep_index_alignment_<Z>.pdf next to the script.
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
ROOT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "src")  # editable-install src/
sys.path.insert(0, ROOT)
from toddlers.pysb99.eep_interpolation import (
    load_all_tracks, load_clean_track, load_feedback_db,
    detect_eeps_comprehensive, pairwise_basis_ages, sampler_mist)

STYLE = os.path.join(ROOT, "paper", "paper_style.mplstyle")
if os.path.exists(STYLE):
    plt.style.use(STYLE)

N_SEC = 20
MIN_MASS = 15.0
QOI = [("Q_HI", r"$\log\,Q(\mathrm{H})$"),
       ("wind_power", r"$\log\,\dot E_{\rm wind}$")]
# robust radiation skeleton shared by nearly all feedback-relevant tracks; used as
# the common index basis here (the WR markers WN_ON/WC_ON still drive the real
# pairwise interpolation, they are just not the grouping key for this figure).
SKELETON = ["ZAMS", "MS_TO", "T_DOWN", "TEFF_MIN", "END"]


def track_eep(Z, m, db, skeleton=SKELETON, n_sec=N_SEC):
    """Per-track EEP basis on the fixed skeleton (+ MIST sampling) and feedback on it.
    Returns None if the track lacks a skeleton marker (e.g. never cools)."""
    t = load_clean_track(Z, m)
    e = detect_eeps_comprehensive(t)
    if any(k not in e for k in skeleton):
        return None
    ages = pairwise_basis_ages(t, e, skeleton, n_sec, sampler_mist)
    masses_db, age_grid, fb = db
    i = int(np.argmin(np.abs(masses_db - t["initial_mass"])))
    fbe = {q: np.interp(ages, age_grid, fb[q][i]) for q, _ in QOI}   # log10
    return ages, fbe


def main(Z="MW"):
    db = load_feedback_db(Z)
    masses, _ = load_all_tracks(Z)
    use = [m for m in masses if m >= MIN_MASS]

    cache, excluded = {}, []
    for m in use:
        res = track_eep(Z, m, db)
        if res is None:
            excluded.append(m)
        else:
            cache[m] = res
    grp = sorted(cache)
    seq = SKELETON
    print(f"{Z}: {len(grp)} masses on the {'-'.join(SKELETON)} skeleton "
          f"[{min(grp):.0f}-{max(grp):.0f} Msun]"
          + (f"; excluded (missing a skeleton marker): "
             f"{[round(m) for m in excluded]}" if excluded else ""))

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(np.log10(min(grp)), np.log10(max(grp)))
    fig, axes = plt.subplots(len(QOI), 2, figsize=(7.2, 2.7 * len(QOI)),
                             squeeze=False)
    # dex below the plateau to show: the cliff's *departure* from the plateau is
    # the alignment signal, so clip the dead-star floor (would span ~60 dex).
    span = {"Q_HI": 14.0, "wind_power": 10.0}
    for row, (q, qlab) in enumerate(QOI):
        ax_age, ax_idx = axes[row]
        plateau = max(np.max(fbe[q]) for _, fbe in cache.values())
        for m in grp:
            ages, fbe = cache[m]
            c = cmap(norm(np.log10(m)))
            ax_age.plot(ages, fbe[q], "-", color=c, lw=0.9)
            ax_idx.plot(np.arange(len(ages)), fbe[q], "-", color=c, lw=0.9)
        for ax in (ax_age, ax_idx):
            ax.set_ylim(plateau - span[q], plateau + 0.5)
        # primary markers sit at index j*N_SEC in the basis
        for j, name in enumerate(seq):
            x = j * N_SEC
            ax_idx.axvline(x, color="0.6", lw=0.6, ls=":", zorder=0)
            if row == 0:
                ax_idx.text(x, ax_idx.get_ylim()[1], name, rotation=90,
                            va="top", ha="right", fontsize=5.5, color="0.4")
        ax_age.set_ylabel(qlab)
        if row == len(QOI) - 1:
            ax_age.set_xlabel("age  [Myr]")
            ax_idx.set_xlabel("EEP index")
        if row == 0:
            ax_age.set_title("vs age (smeared)", fontsize=8)
            ax_idx.set_title("vs EEP index (aligned)", fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes, fraction=0.04, pad=0.02)
    cb.set_label(r"$\log\,M_{\rm init}$  [$M_\odot$]", fontsize=8)
    fig.suptitle(f"{Z}: feedback aligns on the EEP index "
                 f"({len(grp)} masses, {min(grp):.0f}-{max(grp):.0f} "
                 rf"$M_\odot$)", fontsize=9)
    fig.savefig(os.path.join(HERE, f"eep_index_alignment_{Z}.pdf"))
    plt.close(fig)
    print(f"wrote eep_index_alignment_{Z}.pdf")


if __name__ == "__main__":
    for z in (sys.argv[1:] or ["MW", "LMC", "SMC", "MWC"]):
        main(z)
