"""pySB99 spectral assignment: large, cool stars keep their light.

Regression test for the cool-supergiant branch of
``StellarPopulationSynthesis._assign_spectra_to_grid_WR``. Stars with
log Teff < 3.65 above the WR minimum mass used to receive no spectrum, so
the list of assigned spectra was shorter than the list of stars and every
later star was paired with the wrong N(M) and radius. They now take a
Lejeune (low-mass library) spectrum, as Starburst99 does.

Check, for a Z = 0.014 Kroupa (0.1-100 Msun) population at 5.5 Myr, where
large, cool stars are present:
- every star gets a spectrum (assigned count equals star count);
- the integrated population luminosity equals sum_i N_i L_i to within the
  constant offset of -0.0063 dex seen at ages without cool supergiants
  (pySB99's radius formula implies L_sun = 3.784e33 erg/s, against
  3.839e33 used here). Tolerance 0.0005 dex. Before the fix this age gives
  -0.0647 dex.
"""
import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.data

L_SUN_CGS = 3.839e33
CLEAN_AGE_OFFSET_DEX = -0.0063
TOLERANCE_DEX = 5e-4


@pytest.fixture(scope="module")
def kroupa_mw():
    from toddlers.pysb99.pysb99_core import StellarPopulationConfig, StellarPopulationSynthesis

    cfg = StellarPopulationConfig(total_mass=1e6, metallicity='MW', spectral_library='FW',
                                  imf_exponents=[1.3, 2.3], imf_mass_limits=(0.1, 0.5, 100.0),
                                  auto_extend_imf=False, run_speed_mode='DEFAULT')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        syn = StellarPopulationSynthesis(cfg)
    ind = syn._get_timestep_0_ind(syn.times_steps_SB99[0])
    p0 = syn._get_timestep_params(syn.times_fine[0], ind)
    initial_masses = np.array(p0[3])
    No_stars, *_ = syn.calc_Nostars(p0[3], cfg.imf_exponents, cfg.imf_mass_limits)
    syn.No_stars = No_stars
    return syn, ind, initial_masses, No_stars


def _population_vs_star_sum(syn, ind, initial_masses, No_stars, age_myr):
    k = int(np.argmin(abs(syn.times_fine - age_myr * 1e6)))
    p = syn._get_timestep_params(syn.times_fine[k], ind)
    temps, lums, masses, H = p[1], p[2], p[3], p[4]
    _, loggs, radii, cotests, bbfluxes = syn._get_specsyn_params(temps, masses, lums)
    grid = syn.ob_spectral_grid
    assigned = syn._assign_spectra_to_grid_WR(temps, grid['params'], grid['teffs'], grid['logls'],
                                              lums, H, grid['loggs'], masses, initial_masses,
                                              cotests, loggs, p[12], p[13])
    lums = np.array(lums)
    n_cool_large = int(np.sum((np.array(temps) < 3.65) & (lums > -19)
                              & (initial_masses > syn.loader.minimum_wr_mass)))
    n_assigned = len(assigned[2])
    if n_assigned != len(lums):
        return n_assigned, len(lums), n_cool_large, None
    flux, _ = syn._specsyn(assigned[0], bbfluxes, assigned[2], radii, No_stars)
    w = syn.wavelength_grid
    L_pop = abs(np.trapz(flux[::-1], w[::-1])) * 1e20
    alive = lums > -19
    L_sum = float(np.sum(No_stars[alive] * 10**lums[alive]) * L_SUN_CGS)
    return n_assigned, len(lums), n_cool_large, np.log10(L_pop / L_sum)


def test_cool_supergiants_keep_spectrum_and_light(kroupa_mw):
    syn, ind, initial_masses, No_stars = kroupa_mw
    n_assigned, n_stars, n_cool_large, diff = _population_vs_star_sum(
        syn, ind, initial_masses, No_stars, 5.5)
    assert n_cool_large > 0, "test age must contain large, cool stars"
    assert n_assigned == n_stars
    assert abs(diff - CLEAN_AGE_OFFSET_DEX) <= TOLERANCE_DEX


def test_clean_age_offset_unchanged(kroupa_mw):
    syn, ind, initial_masses, No_stars = kroupa_mw
    n_assigned, n_stars, n_cool_large, diff = _population_vs_star_sum(
        syn, ind, initial_masses, No_stars, 3.0)
    assert n_cool_large == 0
    assert n_assigned == n_stars
    assert abs(diff - CLEAN_AGE_OFFSET_DEX) <= TOLERANCE_DEX

