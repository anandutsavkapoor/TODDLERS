"""EEP feedback interpolator consistency (requires the stellar database).

On a grid mass, the phase-aligned EEP interpolator must reproduce the database
feedback it was built from (no interpolation happens at a grid point).
"""
import numpy as np
import pytest

from toddlers.pysb99.eep_interpolation import EEPFeedbackInterpolator
from toddlers.pysb99.stochastic.database import query_database
from toddlers._paths import get_database_path

pytestmark = pytest.mark.data

Z = "MW"
QUANTS = ["Q_HI", "wind_power"]


def test_on_grid_query_matches_database():
    ei = EEPFeedbackInterpolator(Z, get_database_path(), quantities=QUANTS)
    mass = float(ei.masses[len(ei.masses) // 2])      # an exact grid mass
    ages = np.array([0.5, 1.0, 3.0, 5.0])

    eep = ei.query(mass, ages)
    db = query_database(database_path=get_database_path(), metallicity=Z,
                        mass=mass, ages_myr=ages, quantities=QUANTS)
    for q in QUANTS:
        # both are log10 quantities; agree to well under 0.05 dex
        assert np.allclose(eep[q], db[q], atol=0.05)
