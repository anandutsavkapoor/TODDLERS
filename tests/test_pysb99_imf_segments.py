"""pySB99 calc_Nostars against the analytic piecewise power-law IMF.

Each mass bin is integrated over every IMF segment it overlaps, each segment with its own
slope and normalization; this matters whenever a break lies above the lowest tracked mass.
"""
import io
import contextlib

import numpy as np
import pytest

from toddlers.pysb99.pysb99_core import StellarPopulationSynthesis


def _calc(masses, exps, lims, m_total=1e6):
    sps = object.__new__(StellarPopulationSynthesis)
    sps.M_total = m_total
    with contextlib.redirect_stdout(io.StringIO()):
        return sps.calc_Nostars(np.asarray(masses, dtype=float), exps, lims)


def _analytic(masses, exps, lims, m_total=1e6):
    """N per bin for xi(m) = A_i m^-alpha_i, continuous at the breaks, with
    int m xi dm over [lims[0], lims[-1]] = m_total; bins as in calc_Nostars."""
    masses = np.asarray(masses, dtype=float)
    A = [1.0]
    for i in range(1, len(exps)):
        A.append(A[-1] * lims[i] ** (exps[i] - exps[i - 1]))

    def seg_int(a, b, alpha, p):  # int_a^b m^(p - alpha) dm
        q = p - alpha + 1
        return np.log(b / a) if q == 0 else (b ** q - a ** q) / q

    norm = sum(A[i] * seg_int(lims[i], lims[i + 1], exps[i], 1) for i in range(len(exps)))
    A = np.array(A) * m_total / norm
    mid = 0.5 * (masses[1:] + masses[:-1])
    lo = np.concatenate(([masses[0]], mid))
    hi = np.concatenate((mid, [masses[-1]]))
    N = np.zeros(masses.size)
    for k in range(masses.size):
        for i in range(len(exps)):
            a, b = max(lo[k], lims[i]), min(hi[k], lims[i + 1])
            if b > a:
                N[k] += A[i] * seg_int(a, b, exps[i], 0)
    return N


MASSES = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 9.0, 15.0, 25.0, 40.0, 60.0, 85.0, 100.0])


@pytest.mark.parametrize("exps,lims", [
    ([1.3, 2.3], (0.1, 1.0, 100.0)),       # break inside the tracked range
    ([1.3, 2.3], (0.1, 0.5, 100.0)),       # Kroupa: break below the tracked range
    ([1.3, 2.0], (0.1, 0.8, 100.0)),       # break on a bin edge region
    ([2.35], (0.6, 100.0)),                # single segment
    ([1.3, 2.3], (0.1, 0.5, 40.0)),        # upper IMF limit inside the tracked range: bins clipped
    ([2.35], (1.1, 100.0)),                # lower IMF limit inside the tracked range
])
def test_calc_nostars_matches_piecewise_integral(exps, lims):
    N = np.asarray(_calc(MASSES, exps, lims)[0], dtype=float)
    np.testing.assert_allclose(N, _analytic(MASSES, exps, lims), rtol=1e-10)
