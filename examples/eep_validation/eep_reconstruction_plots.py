"""
Visualise the EEP interpolation: (1) how the MIST metric distributes secondary
EEPs along the HRD, and (2) leave-one-out reconstruction of a held-out grid mass
against the true track and against naive fixed-age interpolation.

Naive fixed-age interpolation (what "interpolate the feedback in mass" means if you
ignore evolutionary phase) blends the two bracketing tracks at equal clock age. EEP
interpolation instead aligns evolutionary phase (shared primary markers) and places
secondary EEPs by the MIST HRD metric before interpolating. The plots show why the
phase alignment matters and that the reconstruction tracks the truth.

Run:  python3 eep_reconstruction_plots.py [Z]
Outputs eep_hrd_placement_<Z>.pdf and eep_reconstruction_<Z>.pdf next to the script.
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
    detect_eeps_comprehensive, pairwise_basis_ages,
    interpolate_feedback_pairwise, sampler_mist)

STYLE = os.path.join(ROOT, "paper", "paper_style.mplstyle")
if os.path.exists(STYLE):
    plt.style.use(STYLE)

N_SEC = 20
CHAN = [("Q_HI", r"$\log\,Q(\mathrm{H})$  [s$^{-1}$]"),
        ("wind_power", r"$\log\,\dot E_{\rm wind}$  [erg s$^{-1}$]")]
# dex below the alive plateau to show (clips the dead-star floor / naive collapse,
# which otherwise span tens of decades and crush the interesting range)
SPAN = {"Q_HI": 8.0, "wind_power": 6.0}


def alive_ylim(true_log, life_mask, q, pad=0.4):
    """y-limits focused on the alive feedback: [plateau - SPAN, plateau + pad]."""
    plateau = float(np.max(true_log[life_mask]))
    return plateau - SPAN[q], plateau + pad


def eep_hrd_points(Z, m, n_sec=N_SEC):
    """Return (logTeff, logL) of all MIST-placed EEPs and of the primary markers."""
    t = load_clean_track(Z, m)
    e = detect_eeps_comprehensive(t)
    e.pop("MS_MID", None)
    shared = list(e.keys())
    ages = pairwise_basis_ages(t, e, shared, n_sec, sampler_mist)
    age_t = t["age"] / 1e6
    Te = np.interp(ages, age_t, t["logTeff"])
    L = np.interp(ages, age_t, t["logL"])
    pm_Te = np.array([t["logTeff"][e[k]] for k in shared])
    pm_L = np.array([t["logL"][e[k]] for k in shared])
    return t, (Te, L), (pm_Te, pm_L), shared


def naive_fixed_age(Z, m_t, m_lo, m_hi, db):
    """Blend the two bracketing tracks at equal clock age (no phase alignment)."""
    masses_db, age, fb = db
    i_lo = int(np.argmin(np.abs(masses_db - m_lo)))
    i_hi = int(np.argmin(np.abs(masses_db - m_hi)))
    w = (np.log10(m_t) - np.log10(m_lo)) / (np.log10(m_hi) - np.log10(m_lo))
    out = {q: (1 - w) * fb[q][i_lo] + w * fb[q][i_hi] for q in (c[0] for c in CHAN)}
    return age, out


def pick_masses(Z, db, n_hrd=3, n_recon=3, m_min=20.0):
    """Choose, data-driven, the HRD masses and the leave-one-out triples to plot.

    HRD masses: up to `n_hrd` feedback-relevant masses spread across the grid.
    Triples: interior masses whose two neighbours share a usable marker sequence
    (so the EEP reconstruction is defined), spread across the grid."""
    grid, _ = load_all_tracks(Z)
    feedback = [m for m in grid if m >= m_min]
    hrd = [feedback[int(round(i))] for i in
           np.linspace(0, len(feedback) - 1, min(n_hrd, len(feedback)))]
    valid = []
    for k in range(1, len(grid) - 1):
        if grid[k] < m_min:
            continue
        _, _, ok = interpolate_feedback_pairwise(
            Z, grid[k], grid[k - 1], grid[k + 1], sampler=sampler_mist,
            use_ms_mid=False, db=db)
        if ok:
            valid.append((grid[k - 1], grid[k], grid[k + 1]))
    if valid:
        idx = np.linspace(0, len(valid) - 1, min(n_recon, len(valid)))
        triples = [valid[int(round(i))] for i in idx]
    else:
        triples = []
    return hrd, triples


def plot_hrd(Z, masses):
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    colors = ["#4683DE", "#CB4035", "#4BC55A"]
    for m, c in zip(masses, colors):
        t, (Te, L), (pTe, pL), _ = eep_hrd_points(Z, m)
        ax.plot(t["logTeff"], t["logL"], "-", color=c, lw=0.8, alpha=0.5, zorder=1)
        ax.plot(Te, L, ".", color=c, ms=3, zorder=2)
        ax.plot(pTe, pL, "o", color=c, ms=5, mfc="white", mew=1.0, zorder=3,
                label=rf"${m:.0f}\,M_\odot$")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\log\,T_{\rm eff}$  [K]")
    ax.set_ylabel(r"$\log\,L$  [$L_\odot$]")
    ax.set_title(f"{Z}: MIST EEP placement", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.savefig(os.path.join(HERE, f"eep_hrd_placement_{Z}.pdf"))
    plt.close(fig)
    print(f"wrote eep_hrd_placement_{Z}.pdf  (open dots = primary markers, small = MIST secondary EEPs)")


def plot_reconstruction(Z, triples, db):
    masses, tracks = load_all_tracks(Z)
    n = len(triples)
    fig, axes = plt.subplots(len(CHAN), n, figsize=(3.4 * n, 5.4), squeeze=False)
    for col, (m_lo, m_t, m_hi) in enumerate(triples):
        life = tracks[int(np.argmin(np.abs(masses - m_t)))]["age"].max() / 1e6
        k = int(np.argmin(np.abs(db[0] - m_t)))
        age_db = db[1]
        ages, fbp, ok = interpolate_feedback_pairwise(
            Z, m_t, m_lo, m_hi, sampler=sampler_mist, use_ms_mid=False, db=db)
        o = np.argsort(ages); ages = ages[o]
        age_nv, nv = naive_fixed_age(Z, m_t, m_lo, m_hi, db)
        for row, (q, qlab) in enumerate(CHAN):
            ax = axes[row][col]
            true_log = db[2][q][k]
            sel = age_db <= life
            ax.plot(age_db[sel], true_log[sel], "-", color="black", lw=1.4,
                    label=rf"true ${m_t:.0f}\,M_\odot$", zorder=3)
            nsel = age_nv <= life
            ax.plot(age_nv[nsel], nv[q][nsel], "--", color="#999999", lw=1.0,
                    label="naive fixed-age", zorder=1)
            ax.plot(ages, np.log10(fbp[q][o]), ".-", color="#CB4035", lw=0.9, ms=3,
                    label="EEP (MIST)", zorder=2)
            ax.set_xlim(0, life)
            ax.set_ylim(*alive_ylim(true_log, sel, q))
            if row == len(CHAN) - 1:
                ax.set_xlabel("age  [Myr]")
            if col == 0:
                ax.set_ylabel(qlab)
            if row == 0:
                ax.set_title(rf"hold out ${m_t:.0f}$ from ${m_lo:.0f},{m_hi:.0f}\,M_\odot$",
                             fontsize=8)
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.suptitle(f"{Z}: leave-one-out feedback reconstruction", fontsize=9)
    fig.savefig(os.path.join(HERE, f"eep_reconstruction_{Z}.pdf"))
    plt.close(fig)
    print(f"wrote eep_reconstruction_{Z}.pdf")


def all_valid_triples(Z, db, m_min=15.0):
    """Every interior grid mass (>= m_min) whose neighbours give a defined EEP
    reconstruction, as (m_lo, m_t, m_hi)."""
    grid, _ = load_all_tracks(Z)
    out = []
    for k in range(1, len(grid) - 1):
        if grid[k] < m_min:
            continue
        _, _, ok = interpolate_feedback_pairwise(
            Z, grid[k], grid[k - 1], grid[k + 1], sampler=sampler_mist,
            use_ms_mid=False, db=db)
        if ok:
            out.append((grid[k - 1], grid[k], grid[k + 1]))
    return out


def plot_grid(Z, db, q, qlab, ncols=4):
    """Full-coverage leave-one-out: one panel per valid interior mass, for a single
    channel (true vs EEP vs naive fixed-age)."""
    masses, tracks = load_all_tracks(Z)
    triples = all_valid_triples(Z, db)
    n = len(triples)
    if n == 0:
        print(f"{Z}/{q}: no valid masses")
        return
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 2.2 * nrows),
                             squeeze=False)
    age_db = db[1]
    for ax in axes.flat:
        ax.set_visible(False)
    for p, (m_lo, m_t, m_hi) in enumerate(triples):
        ax = axes[p // ncols][p % ncols]
        ax.set_visible(True)
        life = tracks[int(np.argmin(np.abs(masses - m_t)))]["age"].max() / 1e6
        k = int(np.argmin(np.abs(db[0] - m_t)))
        ages, fbp, _ = interpolate_feedback_pairwise(
            Z, m_t, m_lo, m_hi, sampler=sampler_mist, use_ms_mid=False, db=db)
        o = np.argsort(ages); ages = ages[o]
        age_nv, nv = naive_fixed_age(Z, m_t, m_lo, m_hi, db)
        sel = age_db <= life
        true_log = db[2][q][k]
        ax.plot(age_db[sel], true_log[sel], "-", color="black", lw=1.3,
                label="true", zorder=3)
        nsel = age_nv <= life
        ax.plot(age_nv[nsel], nv[q][nsel], "--", color="#999999", lw=0.9,
                label="naive fixed-age", zorder=1)
        ax.plot(ages, np.log10(fbp[q][o]), ".-", color="#CB4035", lw=0.8, ms=2.5,
                label="EEP (MIST)", zorder=2)
        ax.set_xlim(0, life)
        ax.set_ylim(*alive_ylim(true_log, sel, q))
        ax.set_title(rf"${m_t:.0f}\,M_\odot$  (from ${m_lo:.0f},{m_hi:.0f}$)",
                     fontsize=7)
        ax.tick_params(labelsize=6)
        if p == 0:
            ax.legend(frameon=False, fontsize=6, loc="lower left")
    fig.supxlabel("age  [Myr]", fontsize=8)
    fig.supylabel(qlab, fontsize=8)
    fig.suptitle(rf"{Z}: leave-one-out {qlab} reconstruction, all {n} interior masses",
                 fontsize=9)
    tag = q.replace("_", "")
    fig.savefig(os.path.join(HERE, f"eep_recon_grid_{tag}_{Z}.pdf"))
    plt.close(fig)
    print(f"wrote eep_recon_grid_{tag}_{Z}.pdf  ({n} masses)")


if __name__ == "__main__":
    zs = sys.argv[1:] or ["MW", "LMC", "SMC", "MWC"]
    for Z in zs:
        db = load_feedback_db(Z)
        hrd, triples = pick_masses(Z, db)
        plot_hrd(Z, hrd)
        if triples:
            plot_reconstruction(Z, triples, db)   # the curated 3-mass, 2-channel example
        for q, qlab in CHAN:                       # full-coverage grids, per channel
            plot_grid(Z, db, q, qlab)
