"""Shell ionization front and dissolution test.

Bars: the neutral layer starts at the pressure of the ionized gas at the front (to 1e-12);
dissolution_measure returns the selected quantity; an unknown criterion is rejected.
"""
import contextlib
import io

import numpy as np
import pytest

from toddlers.constants import MU_N, MU_P, K_BOLTZMANN, T_NEUTRAL, M_SUN
from toddlers.evolution import Evolution, dissolution_measure
from toddlers.shell_structure import ShellStructure


class _Feedback:
    def get_ionizing_photon_rate(self, age):
        return 2.31e51

    def get_ionizing_luminosity(self, age):
        return 7.37e40

    def get_bolometric_luminosity(self, age):
        return 7.37e40 + 1.19e41


def test_neutral_layer_starts_in_pressure_equilibrium():
    T_ion = 8000.0
    ss = ShellStructure(0.014, 171.6, 1.117e39, T_ion, _Feedback(), 0.0, 1.0)
    ss.solve_shell_structure(9.22e19)
    assert ss.shell_strctre_n is not None
    n_front = ss.shell_strctre_i.y[0][-1]
    n_neutral = ss.shell_strctre_n.y[0][0]
    P_ion = n_front * (MU_N / MU_P) * K_BOLTZMANN * T_ion
    P_neutral = n_neutral * K_BOLTZMANN * T_NEUTRAL
    assert P_neutral == pytest.approx(P_ion, rel=1e-12)


def test_dissolution_measure():
    assert dissolution_measure('density', 3.0, 0.5, 1.0, 4.0) == 3.0
    assert dissolution_measure('push', 3.0, 0.5, 1.0, 4.0) == pytest.approx(0.625)
    assert dissolution_measure('push', 3.0, 0.5, 0.0, 4.0) == 0.5
    assert dissolution_measure('push', 3.0, 0.5, 1.0, 0.0) == np.inf
    with pytest.raises(ValueError):
        dissolution_measure('peak', 3.0, 0.5, 1.0, 4.0)


def test_unknown_dissolution_criterion_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("TODDLERS_OUTPUT_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="dissolution_criterion"), contextlib.redirect_stdout(io.StringIO()):
        Evolution(Z=0.008, eta_sf=0.05, n_cl=320.0, M_cl_init=1e5 * M_SUN, template='SB99',
                  dissolution_criterion='peak')
