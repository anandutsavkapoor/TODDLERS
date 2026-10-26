"""chabrier_imf must be continuous at the 1 Msun junction between the log-normal and the
power law (Chabrier 2003): the two pieces agree there to rounding."""
import numpy as np

from toddlers.pysb99.stochastic.sampling import chabrier_imf


def test_chabrier_continuous_at_one_msun():
    below, above = chabrier_imf(np.array([1.0 - 1e-9, 1.0]))
    assert abs(below / above - 1.0) < 1e-6
