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
  3. physical       - dust bump present (Dust FIR > noDust FIR), Halpha present and age-dependent
                      (NOT required to decline monotonically -- f_esc evolution and cloud
                      recollapse make emergent Halpha non-monotonic), lines resolved in the hr SED.
  4. non-degenerate - vs the previously-built DTM: the SED is not byte-identical, and the FIR
                      bump moves in the right direction with f_dust (the degenerate-axis check).
  5. cloudy-truth   - (optional) a sample of this DTM's raw Cloudy .cont files are valid
                      (non-empty, no NUL corruption, finite), i.e. the source was sound.

Usage::

    python -m toddlers.hpc.verify_dtm --interp-dir <interp_tables> --prefix BPASS_chab100_bin \
        --dtm 0.10 [--prev-dtm 0.02] [--cloudy-output <dir>] [--sample 8]
"""
import argparse
import glob
import hashlib
import os
import pickle
import warnings

import numpy as np

SENTINEL = -90.0   # interpolants store log10(L); <= -90 (or non-finite) marks "absent"


def _suffix(dtm):
    return "" if abs(float(dtm) - 1.0) < 1e-9 else f"_dtm{float(dtm):.2f}"


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


def check_physical(paths):
    msgs, ok = [], True
    dust = _load(paths["totSED_lr (Dust lr)"])
    nod = _load(paths["inciSED_lr (noDust lr)"])
    hr = _load(paths["tot_hr emergent (Dust hr)"])
    wl_lr = 10.0 ** np.asarray(dust.grid[0])                     # micron
    t = np.asarray(dust.grid[1])                                 # Myr
    c = _rep_cloud(np.asarray(dust.values).shape)
    # pick an early, bright timepoint (embedded phase)
    # Evaluate morphology at the BRIGHTEST timepoint, not a fixed early one: low-SFE /
    # high-density / high-mass clouds recollapse and form multiple generations, so the
    # emission can peak late or in several bursts -- a hardcoded ~1 Myr could land on a
    # weak/empty time for those systems.
    dust_v = np.asarray(dust.values)
    bol = (10.0 ** dust_v[(slice(None), slice(None)) + c]).sum(axis=0)   # total flux vs t
    k_sig = int(np.nanargmax(bol))
    dv = 10.0 ** dust_v[(slice(None), k_sig) + c]
    nv = 10.0 ** np.asarray(nod.values)[(slice(None), k_sig) + c]

    # (a) dust bump: Dust FIR emission exceeds the dust-free SED there
    fir_dust = _band_integral(wl_lr, dv, 30, 300)
    fir_nod = _band_integral(wl_lr, nv, 30, 300)
    if fir_dust > 1.5 * fir_nod:
        msgs.append(f"ok    dust FIR bump present (Dust/noDust FIR = {fir_dust/max(fir_nod,1e-300):.1f})")
    else:
        ok = False; msgs.append(f"FAIL  no dust FIR bump (Dust/noDust FIR = {fir_dust/max(fir_nod,1e-300):.2f})")

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
    ap.add_argument("--sample", type=int, default=8)
    args = ap.parse_args()

    paths = _paths(args.interp_dir, args.prefix, args.dtm)
    results = []
    results.append(("integrity", *check_integrity(paths)))
    if results[-1][1]:   # only continue heavy checks if files load
        results.append(("completeness", *check_completeness(paths)))
        results.append(("physical", *check_physical(paths)))
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
