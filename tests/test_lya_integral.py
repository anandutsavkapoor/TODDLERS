"""Dusty Lyman-alpha luminosity: the recombination integral over the ionized layer.

Bars: the 3-point Gauss-Legendre integral on the solve's steps agrees with a fine-grid integral of the dense output to
1e-6; it moves by less than 1e-6 when the shell radius moves by 1e-10 (smooth in the state).
"""
import numpy as np

from toddlers.lyman_alpha import LymanAlpha
from toddlers.shell_structure import ShellStructure


class _Feedback:
    def get_ionizing_photon_rate(self, age):
        return 2.31e51

    def get_ionizing_luminosity(self, age):
        return 7.37e40

    def get_bolometric_luminosity(self, age):
        return 7.37e40 + 1.19e41


def _solved(R):
    ss = ShellStructure(0.014, 171.6, 1.117e39, 8000.0, _Feedback(), 0.0, 1.0)
    ss.solve_shell_structure(R)
    return ss.shell_strctre_i


def _fine(sol, k=2000):
    total, tau_end = 0.0, sol.y[2][-1]
    for a, b in zip(sol.t[:-1], sol.t[1:]):
        r = np.linspace(a, b, k)
        n, tau = sol.sol(r)[[0, 2]]
        total += np.trapz(n**2 * np.exp(-(tau_end - tau)) * 4 * np.pi * r**2, r)
    return total


def test_recombination_integral_matches_a_fine_grid():
    sol = _solved(9.22e19)
    assert abs(LymanAlpha.recombination_integral(sol) / _fine(sol) - 1) < 1e-6


def test_recombination_integral_is_smooth_in_the_shell_radius():
    R = 9.22e19
    a = LymanAlpha.recombination_integral(_solved(R))
    for dR in (1e-10, -1e-10, 2.2e-13):
        assert abs(LymanAlpha.recombination_integral(_solved(R * (1 + dR))) / a - 1) < 1e-6
