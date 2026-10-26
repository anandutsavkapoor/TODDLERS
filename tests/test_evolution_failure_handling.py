"""Failures in the evolution path must be loud.

E1 a non-finite feedback node is refused at load, and the interpolator's scalar path
   never lets a zero-weight neighbour turn a node value into NaN;
E2 a run whose attempts all fail raises instead of returning None;
E3 an interpolant error propagates instead of becoming zero feedback;
E4 a cached track is reused only for the feedback tables it was computed with.
"""
import io
import json
import os
import pickle
import shutil
import contextlib

import numpy as np
import pytest

from toddlers._interp_compat import Interp2DLinear
from toddlers._paths import get_data_dir
from toddlers.constants import M_SUN
from toddlers.stellar_feedback import StellarFeedback


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# ---------------------------------------------------------------- E1
def test_scalar_path_ignores_zero_weight_neighbours():
    t = np.array([1.0, 2.0, 3.0])
    Z = np.array([0.004, 0.008, 0.02])
    z = np.array([[-np.inf, -np.inf, -np.inf],         # z is (len(Z), len(t)); the lower
                  [40.0, 40.0, 40.0],                   # neighbour of the Z = 0.008 node
                  [40.0, 40.0, 40.0]])                  # is -inf
    f = Interp2DLinear(t, Z, z)
    assert f(2.0, 0.008)[0] == 40.0                     # exact node
    assert f(2.5, 0.008)[0] == 40.0                     # between time nodes, exact Z node


@pytest.mark.data
def test_non_finite_feedback_node_is_refused(tmp_path, monkeypatch):
    import h5py
    src = get_data_dir()
    shutil.copy(os.path.join(src, 'SB99_kroupa100_feedback.h5'), tmp_path)
    p = tmp_path / 'SB99_kroupa100_feedback.h5'
    with h5py.File(p, 'a') as f:                         # one L_mech_wind node
        d = f['quantities/L_mech_wind'][()]
        d[2, 5] = np.inf
        f['quantities/L_mech_wind'][...] = d
    monkeypatch.setenv('TODDLERS_DATA', str(tmp_path))
    with pytest.raises(ValueError, match='non-finite'):
        _quiet(StellarFeedback, template='SB99', Z=0.02, M_cl_init=2e7 * M_SUN, eta_sf=0.05,
               t_list_collapse=[], imf='kroupa100')


# ---------------------------------------------------------------- E3
@pytest.mark.data
def test_interpolant_error_propagates(monkeypatch):
    orig = StellarFeedback.load_interpolants

    def load_then_break(self):
        orig(self)
        good = self.Q_ion_interpolant

        def q(t, Z):
            if t > 2.0:
                raise RuntimeError("probe: Q interpolant fails after 2 Myr")
            return good(t, Z)
        self.Q_ion_interpolant = q

    monkeypatch.setattr(StellarFeedback, 'load_interpolants', load_then_break)
    with pytest.raises(RuntimeError, match='probe'):
        sf = _quiet(StellarFeedback, template='SB99', Z=0.02, M_cl_init=2e7 * M_SUN, eta_sf=0.05,
                    t_list_collapse=[], imf='kroupa100')
        sf.get_ionizing_photon_rate(3.0 * 3.15576e13)


# ---------------------------------------------------------------- E2, E4
def _evolution(tmp_path, monkeypatch):
    from toddlers.evolution import Evolution
    out = str(tmp_path / 'sim.dat')
    monkeypatch.setattr(Evolution, 'get_output_paths', lambda self: (out, str(tmp_path / 'sim.log')))
    return Evolution, out


@pytest.mark.data
def test_failed_run_raises(tmp_path, monkeypatch):
    Evolution, _ = _evolution(tmp_path, monkeypatch)

    def boom(self, solvers, rtol):
        raise ValueError("probe: forced failure")
    monkeypatch.setattr(Evolution, 'simulation_attempt', boom)
    monkeypatch.setattr(Evolution, 'cleanup_after_attempt', lambda self: None)
    ev = _quiet(Evolution, Z=0.02, eta_sf=0.05, n_cl=100.0, M_cl_init=1e5 * M_SUN,
                template='SB99', imf='kroupa100')
    with pytest.raises(RuntimeError) as err:
        _quiet(ev.run_simulation)
    assert 'probe: forced failure' in repr(err.value.__cause__)


def _write_dat(path, params):
    with open(path, 'w') as f:
        json.dump({'parameters': params, 'generations': []}, f)


@pytest.mark.data
@pytest.mark.parametrize("stored,reused", [("same", True), ("other", False), (None, True)])
def test_cached_track_matches_its_feedback(tmp_path, monkeypatch, stored, reused):
    from toddlers.evolution import TRACK_CODE_VERSION
    from toddlers.stellar_feedback import feedback_table_md5
    Evolution, out = _evolution(tmp_path, monkeypatch)
    kw = dict(Z=0.02, eta_sf=0.05, n_cl=100.0, M_cl_init=1e5 * M_SUN, template='SB99', imf='kroupa100')
    params = {'Z': 0.02, 'template': 'SB99'}
    if stored is not None:
        params['feedback_md5'] = feedback_table_md5('SB99', 'kroupa100', None) if stored == 'same' else 'x' * 32
        params['track_code_version'] = TRACK_CODE_VERSION
        params['inputs'] = _quiet(Evolution, **kw)._inputs_record()     # no .dat yet: a fresh run
    _write_dat(out, params)
    ev = _quiet(Evolution, **kw)
    assert bool(getattr(ev, '_output_exists', False)) == reused
    if not reused:
        moved = list((tmp_path / 'stale').rglob('sim.dat'))
        assert len(moved) == 1 and not os.path.exists(out)
