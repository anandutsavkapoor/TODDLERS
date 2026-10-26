"""The in-memory nearest-track lookup used by the run-time stochastic build
(EEPFeedbackInterpolator._snap) must equal the per-call database query
(stochastic.database.query_database) EXACTLY, star by star, since the build uses it in place of the query.
50 continuous masses (Kroupa, 0.1-100 Msun, so
sub-track masses are included), ages 0-30 Myr at 0.025 Myr, every quantity the
feedback build reads. Requires the single-star database."""
import numpy as np
import pytest

from toddlers._paths import get_database_path
from toddlers.pysb99.stochastic.database import query_database
from toddlers.pysb99.stochastic.sampling import sample_imf

pytestmark = pytest.mark.data

QUANTITIES = ['wind_power', 'wind_momentum', 'L_bol', 'Q_HI', 'Q_HeI', 'Q_HeII', 'L_LyW',
              'L_ion_HI', 'L_ion_HeI', 'L_ion_HeII']


@pytest.mark.parametrize("Z", ["MW", "MWC"])
def test_snap_equals_query_database(Z):
    from toddlers.pysb99.eep_interpolation import EEPFeedbackInterpolator
    db = get_database_path()
    ei = EEPFeedbackInterpolator(Z, db, quantities=QUANTITIES)
    masses = sample_imf(total_mass=60.0, imf_name='kroupa', m_min=0.1, m_max=100.0, seed=3)
    rng = np.random.default_rng(5)
    masses = np.concatenate([masses[:30], rng.uniform(0.8, 100.0, 20)])[:50]
    assert len(masses) == 50
    ages = np.arange(0.0, 30.0 + 1e-9, 0.025)
    for m in masses:
        a = ei._snap(float(m), ages)
        b = query_database(database_path=db, metallicity=Z, mass=float(m), ages_myr=ages,
                           quantities=QUANTITIES)
        for q in QUANTITIES:
            assert np.array_equal(a[q], b[q]), (Z, m, q, np.max(np.abs(a[q] - b[q])))
