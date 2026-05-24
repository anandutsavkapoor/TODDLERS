"""Analytic regression: the energy-driven shell must reproduce the Weaver (1977)
self-similar wind-bubble solution.

Driving the integrator with a constant mechanical luminosity in a uniform medium,
with gravity / external pressure / Lyman-alpha and the radiation force switched off,
the energy-conserving (Phase 1) shell obeys
    R(t) = (250 / 308 pi)^(1/5) (L_w / rho_0)^(1/5) t^(3/5),  constant = 0.763.
This guards the dynamics core: if the integration is silently broken, the recovered
self-similar constant drifts off 0.763.
"""
import os

import numpy as np
import pytest

from toddlers.evolution import Evolution
from toddlers.constants import M_SUN, MYR_TO_SEC, MU_N

pytestmark = [pytest.mark.data, pytest.mark.slow]

WEAVER_C = (250.0 / (308.0 * np.pi)) ** 0.2     # ~= 0.763


def test_energy_driven_self_similar_constant():
    L_w, n0 = 3.0e38, 1000.0
    kw = dict(Z=0.02, eta_sf=0.05, n_cl=n0, M_cl_init=1e6 * M_SUN,
              template="SB99", star_type="sin", profile_type="uniform",
              dynamic_cloud_density=False,
              include_gravity=False, include_lyman_alpha=False,
              include_external_pressure=False)
    ev = Evolution(**kw)
    if os.path.exists(ev.output_path):           # clear stale cached .dat
        os.remove(ev.output_path)
        ev = Evolution(**kw)
    ev.stellar_feedback.get_mechanical_luminosity = lambda t: L_w
    ev.get_radiation_force = lambda: 0.0
    ev.y0 = ev.get_initial_conditions(ev.t0, dt=ev.t0)   # re-seed with constant L_w

    g = ev.run_simulation()[0]
    t, R = g["time"], g["radius"]
    trans = g.get("phase_transitions", [])
    t_end = trans[0][1] if trans else t[-1]
    keep = t < t_end                                     # Phase 1 only
    assert keep.sum() > 10

    rho0 = n0 * MU_N
    ss = R[keep] / ((L_w / rho0) ** 0.2 * t[keep] ** 0.6)
    late = (t[keep] / MYR_TO_SEC) > 0.3                  # past the IC transient
    const = np.median(ss[late])
    assert np.isclose(const, WEAVER_C, rtol=0.03), \
        f"self-similar constant {const:.3f} drifted from Weaver {WEAVER_C:.3f}"
