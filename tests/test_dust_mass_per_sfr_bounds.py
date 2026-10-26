"""evaluate_interpolator must not extrapolate silently outside the (Z, eta, n_cl) grid."""
import warnings

import numpy as np
import pytest

from toddlers.stab.dust_mass_per_sfr import build_interpolator, evaluate_interpolator

Z_AX = [0.001, 0.008, 0.04]
ETA_AX = [0.01, 0.05, 0.1]
N_AX = [40.0, 160.0, 640.0]


@pytest.fixture(scope="module")
def interp():
    grid = {(Z, e, n): 1e6 * Z * (1 - e) / e * (n / 160.0) ** 0.3
            for Z in Z_AX for e in ETA_AX for n in N_AX}
    return build_interpolator(grid, Z_AX, ETA_AX, N_AX)


def test_in_grid_is_silent_and_unchanged(interp):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        v = evaluate_interpolator(interp, 0.008, 0.05, 160.0)
    assert v == pytest.approx(1e6 * 0.008 * 0.95 / 0.05)


@pytest.mark.parametrize("Z,n", [(1e-4, 160.0), (0.008, 2000.0)])
def test_out_of_grid_warns_by_default(interp, Z, n):
    with pytest.warns(UserWarning, match="outside"):
        v = evaluate_interpolator(interp, Z, 0.05, n)
    pts = np.array([[np.log10(Z), 0.05, np.log10(n)]])
    assert v == pytest.approx(10 ** float(interp(pts)[0]))  # the interpolator's own (extrapolated) value


def test_out_of_grid_raise(interp):
    with pytest.raises(ValueError):
        evaluate_interpolator(interp, 1e-4, 0.05, 160.0, out_of_grid='raise')


def test_out_of_grid_clip(interp):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        v = evaluate_interpolator(interp, 1e-4, 0.05, 2000.0, out_of_grid='clip')
    assert v == pytest.approx(evaluate_interpolator(interp, 0.001, 0.05, 640.0))
