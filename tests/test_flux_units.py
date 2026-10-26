"""The flux unit strings say what the stored numbers are.

pySB99 stores flux_spectra = L_lambda / 1e20 (pysb99_core: 4 pi r^2 F / 1e20), so a stored
value v means v x 1e20 erg/s/A: astropy unit '1e20 erg s-1 Angstrom-1'. The pySB99 Cloudy
table is written as flux_spectra x 1e-20, so its unit is '1e40 erg s-1 Angstrom-1'. Bars:
each string converts a known value the way the code itself converts it
(L_LyW = log10(mean flux_spectra) + 20 in the database).
"""
import types

import astropy.units as u
import numpy as np
import pytest

from toddlers.pysb99.stochastic import database as db
from toddlers.cloudy_stellar_spectra_generator import SpectralTableGenerator

CGS = u.Unit('erg s-1 Angstrom-1')


def test_single_star_flux_unit_converts_a_stored_value():
    assert (2.5 * u.Unit(db.SINGLE_STAR_UNITS['flux_spectra'])).to(CGS).value == pytest.approx(2.5e20, rel=1e-12)


def test_database_LyW_agrees_with_the_flux_unit():
    t = np.array([0.1e6, 1e6, 2e6])
    wl = np.array([900.0, 995.0, 1000.0, 1005.0, 1200.0])
    flux = np.tile([1.0, 3.0, 3.0, 3.0, 2.0], (3, 1))
    logs = np.zeros(3)
    res = types.SimpleNamespace(times=t, wavelength_grid=wl, flux_spectra=flux,
                                bolometric_luminosity=logs, wind_momentum=logs, wind_power=logs,
                                hi_ionizing_flux=logs, hei_ionizing_flux=logs, heii_ionizing_flux=logs,
                                hi_ionizing_luminosity=logs, hei_ionizing_luminosity=logs,
                                heii_ionizing_luminosity=logs)
    fb = db._extract_feedback(res, t / 1e6)
    via_unit = (3.0 * u.Unit(db.SINGLE_STAR_UNITS['flux_spectra'])).to(CGS).value
    assert 10 ** fb['L_LyW'] == pytest.approx(np.full(3, via_unit), rel=1e-9)


def test_pysb99_cloudy_table_unit():
    g = SpectralTableGenerator.__new__(SpectralTableGenerator)
    g.template = 'pySB99'
    side = g._sidecar_units()
    # the table holds flux_spectra x 1e-20: a written value w is w x 1e20 x 1e20 erg/s/A
    assert (1.0 * u.Unit(side['F_lambda'])).to(CGS).value == pytest.approx(1e40, rel=1e-12)
