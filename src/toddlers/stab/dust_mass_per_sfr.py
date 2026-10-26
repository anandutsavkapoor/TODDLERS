"""Dust mass per unit SFR for the SFR-normalized TODDLERS templates.

This mirrors the ``sed_per_sfr`` estimator in :mod:`toddlers.stab.sfr_scaling`,
replacing the per-cloud SED with a per-cloud dust mass. It lets one recover the
dust content carried by the SFR-normalized (constant-SFR) template family without
re-running any Cloudy or evolution model, using only:

* the recollapse-aware sampled population (a pickled
  :class:`toddlers.stab.recollapse.StarFormationSimulation`, the ``.sim`` file),
  which fixes the cloud-mass-function weights ``w_ki``, the per-cloud stellar and
  cloud masses, and the uniform ages that stand in for a constant SFR, and
* the stored dissolution-time grid (``dissolution_time_interp.pkl``), which sets
  when each cloud disperses.

Physical model (matching the evolution pipeline):

* A cloud of initial mass ``M_cloud`` forms ``eps * M_cloud`` in stars per
  star-formation event and retains the rest as gas. After ``g`` star-formation events
  (the first at formation plus one per recollapse), the total stellar mass formed is
  ``M_cloud * [1 - (1-eps)**g]``, so the gas left is ``M_cloud * (1-eps)**g``. The
  evolution model implements this by multiplying the cloud gas by ``(1-eps)`` at each
  recollapse.
* The dust in that gas is fixed at
  ``m_dust(t) = f_dust * xi * Z * X(Z) * (1-eps)**g(t) * M_cloud``
  from formation until the cloud dissolves at ``t_diss``, and is zero afterwards, where
  ``g(t)`` is the number of star-formation events by age ``t`` (``1 +`` the number of
  recollapse times before ``t``). Here ``xi = m_dust / (m_H * Z) = 0.643`` is the
  (metallicity-independent) dust mass per hydrogen mass per metallicity from the
  TODDLERS 2.0 appendix, and ``X(Z)`` is the hydrogen mass fraction, so
  ``xi * X(Z) = DTM(Z)`` (the dust-to-metal ratio).
* ``f_dust`` is the grain-abundance scaling of the v2-DTM library (1.0 = full dust).
* If no recollapse-time table is supplied, ``g(t) = 1`` for all ages (single event):
  exact for non-recollapsing clouds, an over-estimate of the gas for recollapsed ones.

The estimator is then, exactly analogous to ``sed_per_sfr``::

    M_dust / SFR = sum_k  w_k * m_dust(k) * [age_k < t_diss(k)]
                   -------------------------------------------------
                        sum_k  M_star_k / (AGE_LIMIT - AGE_START)

with the sum over sampled cloud particles ``k``. Because the ages are uniform over the
averaging window, clouds with ``t_diss`` longer than the window are naturally clipped at
the window edge, so the 10 Myr and 30 Myr windows give genuinely different results rather
than a rescaling.

Run as a script to build the full (Z, eta, n_cl) grid from a directory of ``.sim`` files
for the current window (set ``TODDLERS_STAB_AGE_LIMIT``)::

    python -m toddlers.stab.dust_mass_per_sfr \
        --sim-dir  <dir with recollapse_sim_*.sim> \
        --dissolution-interp <path to dissolution_time_interp.pkl> \
        --f-dust 1.0 --output dustMasses.h5
"""

import os
import glob
import pickle
import argparse
import warnings

import numpy as np

# Unpickling both the .sim populations and the dissolution interpolator can hit a
# scipy private-module rename across versions; the same shim used in sfr_scaling.py.
import scipy.interpolate
import sys
sys.modules.setdefault('scipy.interpolate._interpolate', scipy.interpolate)

from .config import AGE_LIMIT, AGE_START, METALLICITIES, \
    STAR_FORMATION_EFFICIENCIES, CLOUD_DENSITIES

# Dust mass per hydrogen mass per unit metallicity, m_dust / (m_H * Z). Constant across
# metallicity by construction of the v2 dust model (TODDLERS 2.0 appendix, "Variable
# dust-to-metal ratio templates"): DTM = xi / mu_gas = xi * X(Z), giving DTM = 0.456 at
# Z = 0.014.
XI_DUST_PER_H_PER_Z = 0.643

# Helium abundance of the GASS10 set used for the templates (Dopita et al. 2002;
# TODDLERS I, Eq. 34; cloudy_abundances_generator._calc_He): He/H = 0.0737 +
# 0.024 Z/Z_sun by number, with Z_sun = 0.014. Sets the hydrogen mass fraction
# X = (1 - Z) / (1 + (m_He/m_H) He/H) and hence mu_gas = 1/X.
_HE_H_ZERO = 0.0737
_HE_H_SLOPE = 0.024
_Z_SUN_GASS10 = 0.014
_M_HE_OVER_M_H = 4.0026 / 1.00794

SECONDS_PER_MYR = 3.1556952e13
SECONDS_PER_YR = 3.1556952e7


def hydrogen_mass_fraction(Z):
    """Hydrogen mass fraction X(Z) of the template gas (GASS10, Dopita He/H)."""
    Z = np.asarray(Z, dtype=float)
    he_h = _HE_H_ZERO + _HE_H_SLOPE * Z / _Z_SUN_GASS10
    return (1.0 - Z) / (1.0 + _M_HE_OVER_M_H * he_h)


def dust_to_metal_ratio(Z):
    """Dust-to-metal ratio DTM(Z) = xi * X(Z) (= 0.456 at Z = 0.014)."""
    return XI_DUST_PER_H_PER_Z * hydrogen_mass_fraction(Z)


def _load_pickle(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def load_recollapse_times(recollapse_data_path):
    """Index the recollapse-event times keyed on (Z, eta, n_cl, logM).

    Returns a dict mapping (round(Z,5), round(eta,5), round(n_cl,1), round(logM,2)) to a
    numpy array of recollapse times in Myr, read from the ``recollapse_data_*.hdf5`` file
    (group ``recollapse_data/simulations/<name>`` with attrs Z, eta_sf, logM, n_cl and a
    ``recollapse_times_myr`` dataset).
    """
    import h5py
    out = {}
    with h5py.File(recollapse_data_path, 'r') as fh:
        sims = fh['recollapse_data']['simulations']
        for name in sims:
            g = sims[name]
            a = g.attrs
            key = (round(float(a['Z']), 5), round(float(a['eta_sf']), 5),
                   round(float(a['n_cl']), 1), round(float(a['logM']), 2))
            t = np.asarray(g['recollapse_times_myr'][()], dtype=float).ravel() \
                if 'recollapse_times_myr' in g else np.array([])
            out[key] = t
    return out


def _events_by_age(ages, recollapse_times_myr):
    """Number of star-formation events by each age: 1 + count(recollapse_times < age)."""
    if recollapse_times_myr is None or len(recollapse_times_myr) == 0:
        return np.ones_like(ages)
    rt = np.sort(np.asarray(recollapse_times_myr, dtype=float))
    return 1.0 + np.searchsorted(rt, ages, side='right')


def dust_mass_per_sfr(sim, dissolution_interp, f_dust=1.0,
                      age_limit=None, age_start=None, recollapse_times_lookup=None):
    """Dust mass per unit SFR [yr] for one sampled population.

    Args:
        sim: a :class:`~toddlers.stab.recollapse.StarFormationSimulation` (loaded
            ``.sim``), carrying ``ages`` [Myr], ``w_ki``, ``M_star_particles`` [Msun],
            ``M_cloud_particles`` [Msun], ``epsilon``, ``Z`` and ``n_cl`` per particle.
        dissolution_interp: RegularGridInterpolator over
            ``(log10 Z, eta, log10 n_cl, log10 M_cloud)`` returning the dissolution
            time in seconds (the stored ``dissolution_time_interp.pkl``).
        f_dust: grain-abundance scaling (1.0 = full v2 dust).
        age_limit, age_start: averaging window edges [Myr]; default from config.

    Returns:
        M_dust / SFR in yr.
    """
    if age_limit is None:
        age_limit = AGE_LIMIT
    if age_start is None:
        age_start = AGE_START

    ages = np.asarray(sim.ages, dtype=float)          # Myr
    mask = ages <= age_limit                          # safety, matches sed_per_sfr
    ages = ages[mask]
    w_ki = np.asarray(sim.w_ki, dtype=float)[mask]
    M_cloud = np.asarray(sim.M_cloud_particles, dtype=float)[mask]
    M_star = np.asarray(sim.M_star_particles, dtype=float)
    eps = np.asarray(sim.epsilon, dtype=float)[mask]
    Z = np.asarray(sim.Z, dtype=float)[mask]
    n_cl = np.asarray(sim.n_cl, dtype=float)[mask]

    # dissolution time per cloud [Myr], from the stored grid
    pts = np.column_stack((np.log10(Z), eps, np.log10(n_cl), np.log10(M_cloud)))
    t_diss_myr = np.asarray(dissolution_interp(pts), dtype=float) / SECONDS_PER_MYR
    alive = ages < t_diss_myr

    # number of star-formation events by each age (1 for non-recollapsing clouds)
    g = np.ones_like(ages)
    if recollapse_times_lookup is not None:
        logM = np.log10(M_cloud)
        for b in np.unique(M_cloud):
            sel = M_cloud == b
            key = (round(float(Z[sel][0]), 5), round(float(eps[sel][0]), 5),
                   round(float(n_cl[sel][0]), 1), round(float(logM[sel][0]), 2))
            g[sel] = _events_by_age(ages[sel], recollapse_times_lookup.get(key))

    # gas left after g star-formation events = (1-eps)**g * M_cloud; dust rides that gas
    m_gas = (1.0 - eps) ** g * M_cloud
    m_dust = f_dust * XI_DUST_PER_H_PER_Z * Z * hydrogen_mass_fraction(Z) * m_gas

    M_dust_total = np.sum(w_ki * m_dust * alive)                       # Msun

    # SFR = sum M_star / (age_limit - age_start); span in yr -> SFR in Msun/yr
    span_yr = (age_limit - age_start) * 1.0e6
    sfr = np.sum(M_star) / span_yr                                    # Msun/yr

    return M_dust_total / sfr                                         # yr


def _sim_params(sim):
    """(Z, eta, n_cl) of a sampled population (constant across its particles)."""
    return (float(np.asarray(sim.Z)[0]),
            float(np.asarray(sim.epsilon)[0]),
            float(np.asarray(sim.n_cl)[0]))


def build_grid(sim_dir, dissolution_interp_path, f_dust=1.0,
               age_limit=None, age_start=None, pattern='recollapse_sim_*.sim',
               recollapse_data_path=None):
    """Compute M_dust/SFR for every ``.sim`` in ``sim_dir``.

    If ``recollapse_data_path`` is given, the per-generation gas depletion
    ``(1-eps)**g(t)`` is applied (correct for clouds with multiple star-formation
    events); otherwise a single event is assumed.

    Returns a dict mapping (Z, eta, n_cl) -> M_dust/SFR [yr], and the loaded
    dissolution interpolator.
    """
    dissolution_interp = _load_pickle(dissolution_interp_path)
    lookup = load_recollapse_times(recollapse_data_path) if recollapse_data_path else None
    out = {}
    for path in sorted(glob.glob(os.path.join(sim_dir, pattern))):
        sim = _load_pickle(path)
        key = _sim_params(sim)
        out[key] = dust_mass_per_sfr(sim, dissolution_interp, f_dust=f_dust,
                                     age_limit=age_limit, age_start=age_start,
                                     recollapse_times_lookup=lookup)
    return out, dissolution_interp


def grid_to_cube(grid, Z_axis=None, eta_axis=None, n_axis=None):
    """Pack a {(Z,eta,n): value} dict into (Z, eta, n_cl) cubes, matching the config grid.

    Returns Z, eta, nCloud, mdustPerSFR arrays each of shape (nZ, neta, nn); missing
    cells are NaN.
    """
    Z_axis = np.asarray(METALLICITIES if Z_axis is None else Z_axis, dtype=float)
    eta_axis = np.asarray(STAR_FORMATION_EFFICIENCIES if eta_axis is None else eta_axis, dtype=float)
    n_axis = np.asarray(CLOUD_DENSITIES if n_axis is None else n_axis, dtype=float)
    shape = (Z_axis.size, eta_axis.size, n_axis.size)
    Zc = np.full(shape, np.nan); etac = np.full(shape, np.nan)
    nc = np.full(shape, np.nan); md = np.full(shape, np.nan)
    for i, Zv in enumerate(Z_axis):
        for j, ev in enumerate(eta_axis):
            for k, nv in enumerate(n_axis):
                Zc[i, j, k] = Zv; etac[i, j, k] = ev; nc[i, j, k] = nv
                for (Zk, ek, nk), val in grid.items():
                    if np.isclose(Zk, Zv) and np.isclose(ek, ev) and np.isclose(nk, nv):
                        md[i, j, k] = val
                        break
    return Zc, etac, nc, md


def build_interpolator(grid, Z_axis=None, eta_axis=None, n_axis=None):
    """A RegularGridInterpolator for M_dust/SFR over (log10 Z, eta, log10 n_cl).

    The interpolator stores log10(M_dust/SFR) at ``f_dust = 1`` (it is exactly linear in
    f_dust, so a caller scales the result). Axes and ordering match the config grid and the
    other pipeline interpolants (dissolution_time_interp, radius_interp).

    Returns the RegularGridInterpolator. Evaluate it via :func:`evaluate_interpolator`.
    """
    from scipy.interpolate import RegularGridInterpolator
    Z_axis = np.asarray(METALLICITIES if Z_axis is None else Z_axis, dtype=float)
    eta_axis = np.asarray(STAR_FORMATION_EFFICIENCIES if eta_axis is None else eta_axis, dtype=float)
    n_axis = np.asarray(CLOUD_DENSITIES if n_axis is None else n_axis, dtype=float)
    cube = np.full((Z_axis.size, eta_axis.size, n_axis.size), np.nan)
    for i, Zv in enumerate(Z_axis):
        for j, ev in enumerate(eta_axis):
            for k, nv in enumerate(n_axis):
                for (Zk, ek, nk), val in grid.items():
                    if np.isclose(Zk, Zv) and np.isclose(ek, ev) and np.isclose(nk, nv):
                        cube[i, j, k] = val
                        break
    if not np.all(np.isfinite(cube)):
        raise ValueError("grid is missing cells; cannot build a full interpolator")
    return RegularGridInterpolator(
        (np.log10(Z_axis), eta_axis, np.log10(n_axis)), np.log10(cube),
        bounds_error=False, fill_value=None)  # fill_value=None -> linear extrapolation


def evaluate_interpolator(interp, Z, eta, n_cl, f_dust=1.0, out_of_grid='warn'):
    """M_dust/SFR [yr] at (Z, eta, n_cl, f_dust) from a :func:`build_interpolator` object.

    Linear in f_dust; the caller is responsible for capping f_dust at 1 if desired.

    The interpolator itself extrapolates linearly outside its (Z, eta, n_cl) box.
    ``out_of_grid`` sets what happens to such points:
    'warn' (default) returns the extrapolated values with one warning giving the count
    and the box, 'raise' raises ValueError, 'clip' clamps the points to the box. The box
    is read from ``interp.grid``, so a pickled interpolator works as it is.
    """
    if out_of_grid not in ('warn', 'raise', 'clip'):
        raise ValueError(f"out_of_grid must be 'warn', 'raise' or 'clip', got {out_of_grid!r}")
    Z = np.atleast_1d(np.asarray(Z, dtype=float))
    eta = np.broadcast_to(np.asarray(eta, dtype=float), Z.shape)
    n_cl = np.broadcast_to(np.asarray(n_cl, dtype=float), Z.shape)
    pts = np.column_stack((np.log10(Z), eta, np.log10(n_cl)))
    lo = np.array([ax[0] for ax in interp.grid])
    hi = np.array([ax[-1] for ax in interp.grid])
    outside = np.any((pts < lo) | (pts > hi), axis=1)
    if outside.any():
        box = (f"Z [{10 ** lo[0]:.3g}, {10 ** hi[0]:.3g}], eta [{lo[1]:.3g}, {hi[1]:.3g}], "
               f"n_cl [{10 ** lo[2]:.3g}, {10 ** hi[2]:.3g}] cm^-3")
        msg = f"{int(outside.sum())} of {outside.size} points outside the M_dust/SFR grid ({box})"
        if out_of_grid == 'raise':
            raise ValueError(msg)
        if out_of_grid == 'clip':
            pts = np.clip(pts, lo, hi)
        else:
            warnings.warn(msg + "; values are linearly extrapolated", UserWarning, stacklevel=2)
    out = f_dust * 10.0 ** np.asarray(interp(pts), dtype=float)
    return out if out.size > 1 else float(out[0])


def _save_h5(path, Zc, etac, nc, md, f_dust, age_limit):
    import h5py
    with h5py.File(path, 'w') as fh:
        fh.create_dataset('Z', data=Zc)
        fh.create_dataset('eta', data=etac)
        fh.create_dataset('nCloud', data=nc)
        fh.create_dataset('mdustPerSFR', data=md)
        fh.attrs['f_dust'] = f_dust
        fh.attrs['age_limit_myr'] = age_limit
        fh.attrs['xi_dust_per_h_per_z'] = XI_DUST_PER_H_PER_Z
        fh.attrs['units_mdustPerSFR'] = 'yr'


def main():
    p = argparse.ArgumentParser(description='Build M_dust/SFR grid from sampled populations.')
    p.add_argument('--sim-dir', required=True, help='directory of recollapse_sim_*.sim files')
    p.add_argument('--dissolution-interp', required=True, help='path to dissolution_time_interp.pkl')
    p.add_argument('--recollapse-data', default=None,
                   help='path to recollapse_data_*.hdf5; enables (1-eps)**g per-generation '
                        'gas depletion for clouds with multiple star-formation events')
    p.add_argument('--f-dust', type=float, default=1.0)
    p.add_argument('--output', default='dustMasses.h5')
    p.add_argument('--interp-out', default=None,
                   help='also pickle a RegularGridInterpolator of M_dust/SFR (log10, at '
                        'f_dust=1) over (log10 Z, eta, log10 n_cl) to this path')
    p.add_argument('--age-limit', type=float, default=None,
                   help='override AGE_LIMIT [Myr]; default from config/env')
    args = p.parse_args()

    age_limit = args.age_limit if args.age_limit is not None else AGE_LIMIT
    grid, _ = build_grid(args.sim_dir, args.dissolution_interp, f_dust=args.f_dust,
                         age_limit=age_limit, recollapse_data_path=args.recollapse_data)
    Zc, etac, nc, md = grid_to_cube(grid)
    _save_h5(args.output, Zc, etac, nc, md, args.f_dust, age_limit)
    if args.interp_out is not None:
        if args.f_dust != 1.0:
            raise SystemExit("build the interpolator at --f-dust 1.0 (it is linear in f_dust)")
        interp = build_interpolator(grid)
        with open(args.interp_out, 'wb') as fh:
            pickle.dump(interp, fh)
        print(f"saved interpolator {args.interp_out}")
    finite = np.isfinite(md)
    print(f"cells: {finite.sum()}/{md.size}  window={age_limit} Myr  f_dust={args.f_dust}")
    print(f"M_dust/SFR [yr]: min={np.nanmin(md):.3e} max={np.nanmax(md):.3e} "
          f"median={np.nanmedian(md):.3e}")
    print(f"saved {args.output}")


if __name__ == '__main__':
    main()
