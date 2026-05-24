"""Consistency and bounds of the tanh interpolation helpers."""
import numpy as np

from toddlers.utils import interpolate_with_tanh


# Reference hand-written forms that interpolate_with_tanh should reproduce.
def _orig_velocity(V):
    b, a = 20.0, 100.0
    return 0.5 * np.tanh((V - a) / b) + 0.5


def _new_velocity(V):
    b, a = 20.0, 100.0
    return interpolate_with_tanh(a - b, 0, a + b, 1, V)


def _orig_fesc(f):
    b, a = 2e-2, 5e-2
    return 0.5 * np.tanh((f - a) / b) + 0.5


def _new_fesc(f):
    b, a = 2e-2, 5e-2
    return interpolate_with_tanh(a - b, 0, a + b, 1, f)


def test_tanh_velocity_matches_reference():
    V = np.linspace(0.0, 300.0, 500)
    assert np.allclose(_orig_velocity(V), _new_velocity(V), atol=1e-8)


def test_tanh_fesc_matches_reference():
    f = np.linspace(0.0, 0.2, 500)
    assert np.allclose(_orig_fesc(f), _new_fesc(f), atol=1e-8)


def test_tanh_is_bounded_0_1():
    V = np.linspace(-200.0, 600.0, 1000)
    y = _new_velocity(V)
    assert np.all(y >= -1e-9)
    assert np.all(y <= 1.0 + 1e-9)


def test_tanh_is_monotonic():
    V = np.linspace(0.0, 300.0, 500)
    y = _new_velocity(V)
    assert np.all(np.diff(y) >= -1e-12)
