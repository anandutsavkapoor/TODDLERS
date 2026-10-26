"""Legacy .obj feedback tables (the format of earlier releases): the deprecated read-only
loader and the converter.

Pass rules, on a synthetic .obj table:
(a) StellarFeedback reads it with a deprecation warning (a FutureWarning) when no unified table exists, and
    feedback_table_md5 is the md5 over the two .obj files (cached tracks computed from them stay valid);
(b) convert_legacy_obj writes a unified table whose log10 nodes of L_mech, F_ram, L_bol,
    L_ion, Q_HI and L_LyW equal the .obj entries 0-4 and the LyW file to rel 1e-12, with the
    mechanical totals as the one channel '*_unsplit' and Q_HeI, Q_HeII listed as missing;
(c) once the unified table exists, StellarFeedback reads it (no warning) and returns the
    same getters as from the .obj to rel 1e-12; the fingerprint becomes the HDF5 md5;
(d) the loader never writes: the data dir holds the same files before and after a read.
"""
import contextlib
import hashlib
import io
import os
import pickle
import warnings

import numpy as np
import pytest

from toddlers._interp_compat import Interp2DLinear
from toddlers.constants import M_SUN, MYR_TO_SEC
from toddlers.feedback_tables import broken_power_law, read_table_file
from toddlers.feedback_tables.legacy import convert_legacy_obj, legacy_nodes
from toddlers.stellar_feedback import StellarFeedback, feedback_table_md5

T = np.concatenate([np.linspace(0.0, 1.9, 20), np.arange(2.0, 6.01, 0.1), np.linspace(7.0, 40.0, 34)])
ZN = np.array([0.004, 0.02])


def _legacy(d):
    t = T[None, :]
    z = (ZN / 0.02)[:, None]
    logs = [40 - t / 20 + 0 * z, 31 - t / 25 + 0 * z, 42 - t / 30 + 0.1 * z, 41.5 - t / 10 + 0 * z,
            52 - t / 9 - 0.1 * z]
    main = [Interp2DLinear(T, ZN, v) for v in logs] + [Interp2DLinear(T, ZN, 0 * logs[0])] * 2
    with open(os.path.join(d, 'SB99interpolation_oldimf.obj'), 'wb') as f:
        pickle.dump(main, f)
    with open(os.path.join(d, 'SB99interpolation_oldimf_LumLymanWerner.obj'), 'wb') as f:
        pickle.dump(Interp2DLinear(T, ZN, 38 - t / 40 + 0 * z), f)
    return logs


def _sf(**kw):
    a = dict(template='SB99', imf='oldimf', Z=0.02, M_cl_init=1e6 / 0.05 * M_SUN, eta_sf=0.05, t_list_collapse=[])
    a.update(kw)
    with contextlib.redirect_stdout(io.StringIO()):
        return StellarFeedback(**a)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv('TODDLERS_DATA', str(tmp_path))
    return tmp_path


def test_legacy_read_convert_and_switch(data_dir):
    logs = _legacy(str(data_dir))
    before = sorted(os.listdir(data_dir))
    with pytest.warns(FutureWarning, match="deprecated"):
        old = _sf()
    assert sorted(os.listdir(data_dir)) == before                      # (d)
    h = hashlib.md5()
    for n in ('SB99interpolation_oldimf.obj', 'SB99interpolation_oldimf_LumLymanWerner.obj'):
        h.update(open(os.path.join(data_dir, n), 'rb').read())
    assert feedback_table_md5('SB99', 'oldimf') == h.hexdigest()       # (a)
    meta = {'imf_definition': broken_power_law([1.3, 2.3], [0.1, 0.5, 100.0]),
            'normalization': {'rule': 'per 1e6 Msun of the full IMF', 'population_mass_msun': 1e6,
                              'm_low_msun': 0.1, 'm_up_msun': 100.0}}
    tab = convert_legacy_obj('SB99', 'oldimf', None, str(data_dir), metadata=meta)
    tab = read_table_file(tab.path)
    t, Z, nodes = legacy_nodes('SB99', 'oldimf')
    for k, q in (('L_mech', 'L_mech'), ('F_ram', 'F_ram'), ('L_bolo', 'L_bol'), ('L_ion', 'L_ion'),
                 ('Q_ion', 'Q_HI'), ('L_LyW', 'L_LyW')):
        np.testing.assert_allclose(tab.log_nodes(q), nodes[k], rtol=1e-12, atol=0)   # (b)
    assert tab.channels('L_mech') == ['L_mech_unsplit'] and tab.channels('F_ram') == ['F_ram_unsplit']
    assert set(tab.missing) == {'Q_HeI', 'Q_HeII', 'L_mech_wind', 'L_mech_SN', 'F_ram_wind', 'F_ram_SN'}
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        new = _sf()                                                      # (c)
    assert new.feedback_table is not None and feedback_table_md5('SB99', 'oldimf') == tab.md5
    for tm in (1.0, 3.0, 10.0):
        for g in ('get_bolometric_luminosity', 'get_ionizing_photon_rate', 'get_mechanical_luminosity',
                  'get_ram_force', 'get_ionizing_luminosity', 'get_lyman_werner_specific_luminosity'):
            a, b = getattr(old, g)(tm * MYR_TO_SEC), getattr(new, g)(tm * MYR_TO_SEC)
            assert b == pytest.approx(a, rel=1e-12), (g, tm)


def test_legacy_warning_shown_under_default_filters(data_dir):
    """A user script reading a legacy .obj table under Python's default warning
    filters sees the deprecation (a DeprecationWarning raised from inside the package is
    hidden outside __main__). Run in a fresh interpreter, no pytest filters, -W unset.
    Bar: stderr contains 'deprecated'."""
    import subprocess
    import sys
    _legacy(str(data_dir))
    code = ("from toddlers.constants import M_SUN\n"
            "from toddlers.stellar_feedback import StellarFeedback\n"
            "StellarFeedback(template='SB99', imf='oldimf', Z=0.02, M_cl_init=1e6 / 0.05 * M_SUN,"
            " eta_sf=0.05, t_list_collapse=[])\n")
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONWARNINGS'}
    r = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stderr
    assert 'TODDLERS 2.0 .obj feedback tables are deprecated' in r.stderr
