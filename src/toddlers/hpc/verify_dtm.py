#!/usr/bin/env python3
"""Verify a DTM's derived interpolants are sound BEFORE deleting its (truth) cloudy_output.

This is the safety gate for the bounded per-DTM run: a DTM's bulky ``cloudy_output`` (~0.5 TiB
for the full grid) may only be deleted once its derived interpolant ``.pkl`` products are
proven good. Deleting is recoverable (the evolution ``.dat`` makes Cloudy reproducible) but
expensive, so the gate guards against the costly failure mode of deleting then discovering the
derived data was bad.

Checks (all must pass; exit 0 = safe to delete, nonzero = HALT and preserve):

  1. integrity      - the four per-DTM SED interpolants exist, load, have the right rank and
                      are mostly finite (not a sentinel-filled shell).
  2. completeness   - every (Z, SFE, n_cl, M_cl) cloud carries finite SED data over (lambda, t)
                      -- no empty cloud slice.
  3. physical       - dust bump present + lines resolved at the FIR-emission-PEAK time (the most
                      embedded / dust-reprocessing epoch), and Halpha present and age-dependent
                      (NOT required to decline monotonically -- f_esc evolution and recollapse make
                      emergent Halpha non-monotonic / multi-peak).
  4. recollapse-correlation - (optional, needs --evolution-dir) the intrinsic Halpha must track the
                      evolution's absorbed ionizing rate Q_abs = Q_ion*(1-f_esc) through the full
                      history. Q_ion + f_esc are read from the .dat (per-generation, so recollapse
                      generations are included); a log-log time cross-correlation (interpolated onto
                      the Cloudy grid) should be high. Catches a Cloudy SED that failed to track a
                      (recollapse) generation. Reads the .dat -- correct at gate time, no recollapse_data.h5.
  5. non-degenerate - vs the previously-built DTM: the SED is not byte-identical, and the FIR
                      bump moves in the right direction with f_dust (the degenerate-axis check).
  6. cloudy-truth   - (optional) a sample of this DTM's raw Cloudy .cont files are valid
                      (non-empty, no NUL corruption, finite), i.e. the source was sound.

Usage::

    python -m toddlers.hpc.verify_dtm --interp-dir <interp_tables> --prefix BPASS_chab100_bin \
        --dtm 0.10 [--prev-dtm 0.02] [--evolution-dir <dir>] [--cloudy-output <dir>]
"""
import argparse
import glob
import hashlib
import os
import pickle
import warnings

import numpy as np

SENTINEL = -90.0      # interpolants store log10(L); <= -90 (or non-finite) marks "absent"
SEC_PER_MYR = 3.1557e13


def _suffix(dtm):
    return "" if abs(float(dtm) - 1.0) < 1e-9 else f"_dtm{float(dtm):g}"   # mirrors utils.dtm_label


def _paths(interp_dir, prefix, dtm):
    s = _suffix(dtm)
    return {
        "totSED_lr (Dust lr)":          f"{interp_dir}/TODDLERS_totSED_lr_{prefix}{s}.pkl",
        "inciSED_lr (noDust lr)":       f"{interp_dir}/TODDLERS_inciSED_lr_{prefix}{s}.pkl",
        "tot_hr emergent (Dust hr)":    f"{interp_dir}/TODDLERS_tot_hr_{prefix}_lines_emergent=True{s}.pkl",
        "tot_hr intrinsic (noDust hr)": f"{interp_dir}/TODDLERS_tot_hr_{prefix}_lines_emergent=False{s}.pkl",
    }


def _load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _finite(v):
    return np.isfinite(v) & (v > SENTINEL)


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rep_cloud(shape):
    """Representative cloud node: highest Z/SFE/n/M (densest, dustiest) -> strongest signal."""
    return tuple(s - 1 for s in shape[2:])   # (iZ, iS, iN, iM)


# ----------------------------------------------------------------------------- checks
def check_integrity(paths):
    msgs, ok = [], True
    for name, p in paths.items():
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            ok = False; msgs.append(f"FAIL  {name}: missing or empty"); continue
        try:
            v = np.asarray(_load(p).values)
        except Exception as e:                                   # noqa: BLE001
            ok = False; msgs.append(f"FAIL  {name}: unloadable ({e})"); continue
        if v.ndim != 6:
            ok = False; msgs.append(f"FAIL  {name}: rank {v.ndim} != 6"); continue
        ff = float(np.mean(_finite(v)))
        if ff < 0.5:
            ok = False; msgs.append(f"FAIL  {name}: only {ff:.0%} finite (sentinel shell?)")
        else:
            msgs.append(f"ok    {name}: shape {v.shape}, {ff:.0%} finite")
    return ok, msgs


def check_completeness(paths):
    o = _load(paths["totSED_lr (Dust lr)"])
    v = np.asarray(o.values)                                     # (lam, t, Z, SFE, n, M)
    cloud_has_data = _finite(v).any(axis=(0, 1))                 # (Z, SFE, n, M)
    n = cloud_has_data.size
    bad = int((~cloud_has_data).sum())
    ok = bad == 0
    msg = f"{n - bad}/{n} clouds carry SED data"
    if not ok:
        msg += f"   ** {bad} EMPTY cloud slices **"
    return ok, [("ok    " if ok else "FAIL  ") + msg]


def _band_integral(wl, sed_1d, lo, hi):
    m = (wl >= lo) & (wl <= hi) & np.isfinite(sed_1d)
    if not m.any():
        return 0.0
    return float(np.trapz((wl[m] * sed_1d[m]) / wl[m], wl[m]))   # integral of lambda*L over ln-lambda


def check_physical(paths, interp_dir):
    msgs, ok = [], True
    dust = _load(paths["totSED_lr (Dust lr)"])
    nod = _load(paths["inciSED_lr (noDust lr)"])
    hr = _load(paths["tot_hr emergent (Dust hr)"])
    wl_lr = 10.0 ** np.asarray(dust.grid[0])                     # micron
    t = np.asarray(dust.grid[1])                                 # Myr
    c = _rep_cloud(np.asarray(dust.values).shape)
    dust_v = np.asarray(dust.values)

    # Evaluate dust/line morphology in the EMBEDDED phase (before the cloud disperses), using the
    # dissolution time from the evolution runs (dissolution_time_interp.pkl: DTM-independent, built
    # with the interpolants from the phase2->dissolution transition, so it is correct at gate time
    # -- unlike recollapse_data.h5, which is regenerated only at the final STAB step). We do NOT
    # assert anything post-dissolution: recollapse-prone clouds (low SFE / high n / high M)
    # re-brighten after the first dissolution, so only the first embedded phase is a guaranteed
    # dust+line epoch. Fall back to the brightest timepoint if dissolution is unavailable.
    k_emb, where = None, ""
    diss_path = os.path.join(interp_dir, "dissolution_time_interp.pkl")
    if os.path.exists(diss_path):
        try:
            t_diss = float(np.asarray(_load(diss_path).values)[c]) / SEC_PER_MYR
            if np.isfinite(t_diss) and t_diss > t.min():
                k_emb = int(np.argmin(np.abs(t - 0.5 * t_diss)))
                where = f"embedded t={t[k_emb]:.2f} Myr (dissolution {t_diss:.2f} Myr)"
        except Exception:                                       # noqa: BLE001
            k_emb = None
    if k_emb is None:
        k_emb = int(np.nanargmax((10.0 ** dust_v[(slice(None), slice(None)) + c]).sum(axis=0)))
        where = f"brightest t={t[k_emb]:.2f} Myr (dissolution time unavailable)"
    dv = 10.0 ** dust_v[(slice(None), k_emb) + c]
    nv = 10.0 ** np.asarray(nod.values)[(slice(None), k_emb) + c]

    # (a) dust bump: in the embedded phase the Dust FIR emission exceeds the dust-free SED
    fir_dust = _band_integral(wl_lr, dv, 30, 300)
    fir_nod = _band_integral(wl_lr, nv, 30, 300)
    if fir_dust > 1.5 * fir_nod:
        msgs.append(f"ok    dust FIR bump present @ {where} (Dust/noDust FIR = {fir_dust/max(fir_nod,1e-300):.1f})")
    else:
        ok = False; msgs.append(f"FAIL  no dust FIR bump @ {where} (Dust/noDust FIR = {fir_dust/max(fir_nod,1e-300):.2f})")

    # (b) Halpha present and age-dependent. The emergent Halpha is NOT monotonic in age:
    # the escape fraction f_esc evolves as the shell thins and cloud recollapse injects new
    # ionizing generations, so a valid cell can dip then rebound. We therefore check that
    # Halpha is PRESENT and VARIES with age -- not that it strictly declines (which would
    # false-fail good data).
    wl_hr = 10.0 ** np.asarray(hr.grid[0])
    ha = (wl_hr >= 0.6520) & (wl_hr <= 0.6610)
    if ha.any():
        hv = np.asarray(hr.values)
        ha_t = np.array([np.max(10.0 ** hv[(ha, k) + c]) for k in range(len(t))])
        finite_pos = ha_t[np.isfinite(ha_t) & (ha_t > 0)]
        if finite_pos.size and finite_pos.max() > 0:
            msgs.append(f"ok    Halpha present (peak {finite_pos.max():.2e} at "
                        f"t={t[int(np.nanargmax(ha_t))]:.1f} Myr)")
        else:
            ok = False; msgs.append("FAIL  Halpha absent/zero at all ages")
        if finite_pos.size and finite_pos.max() / finite_pos.min() > 2:
            msgs.append(f"ok    Halpha varies with age (max/min = {finite_pos.max()/finite_pos.min():.1f}; "
                        f"non-monotonic is allowed)")
        else:
            ok = False; msgs.append("FAIL  Halpha ~constant in age (degenerate time axis?)")
        # (c) lines resolved: Halpha peak well above the local continuum (at its peak age)
        kpk = int(np.nanargmax(ha_t))
        cont = (wl_hr >= 0.60) & (wl_hr <= 0.64)
        peak = float(np.max(10.0 ** hv[(ha, kpk) + c])) / max(
            float(np.median(10.0 ** hv[(cont, kpk) + c])), 1e-300)
        if peak > 3:
            msgs.append(f"ok    emission lines resolved (Halpha peak/continuum = {peak:.0f})")
        else:
            ok = False; msgs.append(f"FAIL  lines not resolved (Halpha peak/continuum = {peak:.1f})")
    else:
        ok = False; msgs.append("FAIL  Halpha wavelength not in hr grid")
    return ok, msgs


def check_nondegenerate(interp_dir, prefix, dtm, prev_dtm):
    this_p = _paths(interp_dir, prefix, dtm)["totSED_lr (Dust lr)"]
    prev_p = _paths(interp_dir, prefix, prev_dtm)["totSED_lr (Dust lr)"]
    if not os.path.exists(prev_p):
        return True, [f"skip  non-degenerate: previous DTM {prev_dtm} product not present"]
    if _md5(this_p) == _md5(prev_p):
        return False, [f"FAIL  totSED is BYTE-IDENTICAL to DTM {prev_dtm} (degenerate axis)"]
    a = _load(this_p); b = _load(prev_p)
    wl = 10.0 ** np.asarray(a.grid[0]); t = np.asarray(a.grid[1])
    c = _rep_cloud(np.asarray(a.values).shape)
    k = int(np.argmin(np.abs(t - 1.0)))
    fa = _band_integral(wl, 10.0 ** np.asarray(a.values)[(slice(None), k) + c], 30, 300)
    fb = _band_integral(wl, 10.0 ** np.asarray(b.values)[(slice(None), k) + c], 30, 300)
    # more f_dust -> bigger FIR bump
    up = (float(dtm) > float(prev_dtm))
    good = (fa > fb) if up else (fa < fb)
    rel = abs(fa - fb) / max(fb, 1e-300)
    if rel < 1e-3:
        return False, [f"FAIL  FIR bump essentially identical to DTM {prev_dtm} (rel {rel:.1e})"]
    if not good:
        return False, [f"FAIL  FIR bump moves the WRONG way vs f_dust "
                       f"(DTM {dtm}: {fa:.2e}, DTM {prev_dtm}: {fb:.2e})"]
    return True, [f"ok    distinct + monotonic vs DTM {prev_dtm} "
                  f"(FIR {fb:.2e} -> {fa:.2e}, md5 differ)"]


def check_cloudy_truth(cloudy_output, dtm, sample):
    s = _suffix(dtm)
    # model dirs for this DTM (1.0 = unsuffixed -> match dirs WITHOUT any _dtm token)
    alldirs = [d for d in glob.glob(f"{cloudy_output}/**/", recursive=True)]
    if s:
        dirs = [d for d in alldirs if s.strip("_") in d]
    else:
        dirs = [d for d in alldirs if "_dtm" not in d]
    conts = []
    for d in dirs:
        conts += glob.glob(f"{d}/*.cont")
    if not conts:
        return False, [f"FAIL  no .cont files found for DTM {dtm} under {cloudy_output}"]
    bad = []
    pick = conts[:: max(1, len(conts) // max(sample, 1))][:sample]
    for f in pick:
        try:
            if os.path.getsize(f) == 0:
                bad.append(f"{os.path.basename(f)}: empty"); continue
            with open(f, "rb") as fh:
                head = fh.read(4096)
            if b"\x00" in head:
                bad.append(f"{os.path.basename(f)}: NUL corruption"); continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                arr = np.loadtxt(f, comments="#", usecols=(0,), max_rows=50)
            if not np.all(np.isfinite(arr)) or arr.size == 0:
                bad.append(f"{os.path.basename(f)}: non-finite/empty data")
        except Exception as e:                                   # noqa: BLE001
            bad.append(f"{os.path.basename(f)}: {e}")
    ok = not bad
    msg = f"sampled {len(pick)} of {len(conts)} .cont files"
    return ok, [("ok    " if ok else "FAIL  ") + msg + ("" if ok else "  ** " + "; ".join(bad) + " **")]


# ---------------------------------------------------------- recollapse / case-B correlation
def evolution_qabs(dat_path):
    """Absorbed ionizing-photon-rate history Q_abs(t)=Q_i*(1-f_esc_i) from an evolution .dat.

    Concatenates all generations (so recollapse generations are included -- the whole point:
    a single coeval population's Q would miss them). Returns (t_Myr, Q_abs) sorted by time,
    and the number of generations. Reads via the package loader (track_simulation)."""
    from ..track_simulation import load_output_file
    _, results = load_output_file(dat_path)
    ts, qs = [], []
    for gen in results:
        t = np.asarray(gen["time"], float)                       # seconds
        Q = np.asarray(gen["stellar_feedback"]["Q_i"], float)    # ionizing photon rate
        fe = np.asarray(gen["shell_properties"]["f_esc_i"], float)
        # feedback is stored with a trailing component axis (N, k) -> total ionizing rate;
        # shell f_esc is a single fraction per step.
        if Q.ndim > 1:
            Q = Q.sum(axis=tuple(range(1, Q.ndim)))
        if fe.ndim > 1:
            fe = fe[(slice(None),) + (0,) * (fe.ndim - 1)]
        m = min(t.shape[0], Q.shape[0], fe.shape[0])
        if m == 0:
            continue
        ts.append(t[:m] / SEC_PER_MYR)
        qs.append(Q[:m] * np.clip(1.0 - fe[:m], 0.0, 1.0))
    if not ts:
        return None, None, 0
    t = np.concatenate(ts); q = np.concatenate(qs)
    o = np.argsort(t)
    return t[o], q[o], len(results)


def _index_dats(evolution_dir):
    """List (Z, eta_sf, n_cl, logM, path) for every sim_*.dat in evolution_dir."""
    from ..track_simulation import extract_params_from_path
    out = []
    for p in glob.glob(os.path.join(evolution_dir, "sim_*.dat")):
        try:
            pr = extract_params_from_path(p)
            out.append((pr["Z"], pr["eta_sf"], pr["n_cl"], np.log10(pr["M_cl_init"]), p))
        except Exception:                                        # noqa: BLE001
            continue
    return out


def _nearest_dat(dat_list, Z, eta, n, logM):
    """.dat whose (Z, eta, n, logM) is closest to the requested cell (relative match)."""
    best, bd = None, 1e9
    for dZ, de, dn, dM, p in dat_list:
        d = (abs(dZ - Z) / max(Z, 1e-9) + abs(de - eta) / max(eta, 1e-9)
             + abs(dn - n) / max(n, 1e-9) + abs(dM - logM) / max(abs(logM), 1e-9))
        if d < bd:
            best, bd = p, d
    return best if bd < 0.05 else None                           # require a real match


def cell_logr(t_cl, ha_t, t_ev, qabs, min_pts=5):
    """log-log Pearson r between intrinsic Halpha(t_cl) and Q_abs interpolated onto t_cl.

    Returns None if fewer than ``min_pts`` overlapping valid points; 0.0 if either series is
    flat over the overlap (constant -> not tracking, which should not pass)."""
    m = (t_cl >= t_ev.min()) & (t_cl <= t_ev.max())
    if int(m.sum()) < min_pts:
        return None
    q_cl = np.interp(t_cl[m], t_ev, qabs)
    h = ha_t[m]
    good = np.isfinite(h) & (h > 0) & np.isfinite(q_cl) & (q_cl > 0)
    if int(good.sum()) < min_pts:
        return None
    lh, lq = np.log10(h[good]), np.log10(q_cl[good])
    if np.ptp(lh) == 0 or np.ptp(lq) == 0:
        return 0.0
    return float(np.corrcoef(lh, lq)[0, 1])


def check_recollapse_correlation(paths, evolution_dir, sample=12, strict=False, r_pass=0.85):
    """Intrinsic Halpha must track the evolution's absorbed ionizing rate through the full
    history (recollapse included). For a sample of cells (recollapse-prone first), correlate
    log intrinsic Halpha (noDust hr) with log Q_abs from the .dat, interpolated onto the
    Cloudy time grid. High correlation = the Cloudy SED faithfully follows the ionizing output
    (including the recollapse jumps); a low value flags a missed/mismatched generation."""
    if not evolution_dir or not os.path.isdir(evolution_dir):
        return True, ["skip  no --evolution-dir given (recollapse-correlation check skipped)"]
    dat_list = _index_dats(evolution_dir)
    if not dat_list:
        return True, [f"skip  no sim_*.dat under {evolution_dir}"]
    hr = _load(paths["tot_hr intrinsic (noDust hr)"])
    wl = 10.0 ** np.asarray(hr.grid[0]); t_cl = np.asarray(hr.grid[1])
    g = [np.asarray(x) for x in hr.grid]; hv = np.asarray(hr.values)
    ha = (wl >= 0.6520) & (wl <= 0.6610)
    nZ, nS, nN, nM = hv.shape[2:]
    # recollapse-prone first: low SFE, high density, high mass
    cells = sorted([(iZ, iS, iN, iM) for iZ in range(nZ) for iS in range(nS)
                    for iN in range(nN) for iM in range(nM)],
                   key=lambda c: (c[1], -c[2], -c[3]))
    rows = []
    for (iZ, iS, iN, iM) in cells[:sample]:
        Z, eta, n, logM = 10**g[2][iZ], g[3][iS], 10**g[4][iN], g[5][iM]
        dp = _nearest_dat(dat_list, Z, eta, n, logM)
        if dp is None:
            continue
        t_ev, qabs, ngen = evolution_qabs(dp)
        if t_ev is None:
            continue
        ha_t = np.array([np.max(10.0 ** hv[(ha, k, iZ, iS, iN, iM)]) for k in range(t_cl.size)])
        r = cell_logr(t_cl, ha_t, t_ev, qabs)
        if r is None:
            continue
        rows.append((r, ngen, Z, eta, n, logM))
    if not rows:
        return True, ["skip  no cells with enough ionizing history to correlate"]
    rv = np.array([x[0] for x in rows])
    med, worst = float(np.median(rv)), float(np.min(rv))
    ok = med >= r_pass and (worst >= 0.5 or not strict)
    msgs = [("ok    " if ok else "FAIL  ")
            + f"intrinsic-Halpha vs evolution Q_abs: median r={med:.3f}, worst r={worst:.3f} "
              f"over {len(rows)} cells ({sum(1 for x in rows if x[1] > 1)} multi-generation)"]
    for r, ngen, Z, eta, n, logM in sorted(rows)[:3]:
        msgs.append(f"    r={r:.3f}  {ngen}-gen  Z={Z:g} eta={eta:g} n={n:g} logM={logM:g}")
    return ok, msgs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interp-dir", required=True)
    ap.add_argument("--prefix", default="BPASS_chab100_bin")
    ap.add_argument("--dtm", type=float, required=True)
    ap.add_argument("--prev-dtm", type=float, default=None,
                    help="previously-built DTM for the non-degenerate check (omit for the first DTM)")
    ap.add_argument("--cloudy-output", default=None,
                    help="cloudy_output dir for the source spot-check (recommended before deleting it)")
    ap.add_argument("--evolution-dir", default=None,
                    help="evolution .dat dir; enables the intrinsic-Halpha vs ionizing-flux "
                         "cross-correlation check (recollapse-aware)")
    ap.add_argument("--corr-sample", type=int, default=12,
                    help="number of cells (recollapse-prone first) for the correlation check")
    ap.add_argument("--strict", action="store_true",
                    help="fail the correlation check on any low-r cell, not just a low median")
    ap.add_argument("--sample", type=int, default=8)
    args = ap.parse_args()

    paths = _paths(args.interp_dir, args.prefix, args.dtm)
    results = []
    results.append(("integrity", *check_integrity(paths)))
    if results[-1][1]:   # only continue heavy checks if files load
        results.append(("completeness", *check_completeness(paths)))
        results.append(("physical", *check_physical(paths, args.interp_dir)))
        if args.evolution_dir:
            results.append(("recollapse-correlation", *check_recollapse_correlation(
                paths, args.evolution_dir, args.corr_sample, args.strict)))
    if args.prev_dtm is not None:
        results.append(("non-degenerate", *check_nondegenerate(
            args.interp_dir, args.prefix, args.dtm, args.prev_dtm)))
    if args.cloudy_output:
        results.append(("cloudy-truth", *check_cloudy_truth(
            args.cloudy_output, args.dtm, args.sample)))

    print(f"\n=== verify DTM={args.dtm} ({args.prefix}) ===")
    all_ok = True
    for name, ok, msgs in results:
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        for m in msgs:
            print(f"        {m}")
    verdict = "SAFE TO DELETE cloudy_output" if all_ok else "DO NOT DELETE -- preserve and investigate"
    print(f"\n>>> DTM={args.dtm}: {'ALL PASS' if all_ok else 'FAILED'} -> {verdict}\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
