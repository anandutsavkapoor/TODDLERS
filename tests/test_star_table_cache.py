"""Star-table cache keyed on content, not on the filename.

A table is reused only when its sidecar (<table>.meta.json) matches the request and its
own age/logZ header matches the requested grid. A table without a sidecar is reused only
if its md5 is a released table's; anything else is moved to stale_tables/ (never deleted)
and regenerated.
"""
import hashlib
import json
import os

import numpy as np
import pytest

import toddlers.cloudy_stellar_spectra_generator as csg


class FakeGen(csg.SpectralTableGenerator):
    """A small synthetic SB99-like generator: no source files are read."""

    def __init__(self, src='v1', scale=1.0):
        self.template, self.imf, self.star_type = 'SB99', 'test', 'sin'
        self.wavelength_resolution_factor, self.resample = 1, 'decimate'
        self.max_age = 1e7
        self.src = src
        lam = np.array([100.0, 500.0, 912.0, 5000.0])
        ages = np.array([0.0, 1e6, 1e7])
        Z = np.array([0.004, 0.02])
        flux = scale * np.ones((Z.size, ages.size, lam.size))
        self.burst_data = self.data = {'ages': ages, 'metallicities': Z, 'wavelengths': lam, 'fluxes': flux}
        self._setup_interpolator()
        self.convolution_dt = 0.025 * csg.MYR_TO_SEC

    def _source_fingerprint(self):
        return {'fake_source': self.src}


AGES = np.array([1e6, 5e6])
ZS = [0.004, 0.02]
NAME = 'SB99_test_sin_burst.ascii'


@pytest.fixture
def cdir(tmp_path, monkeypatch):
    monkeypatch.setattr(csg, 'CLOUDY_DATA_DIR', str(tmp_path))
    return tmp_path


def _md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def _stale(cdir):
    d = cdir / 'stale_tables'
    return sorted(str(p.relative_to(d)) for p in d.rglob('*') if p.is_file()) if d.exists() else []


def test_first_write_has_sidecar_and_is_reused(cdir):
    FakeGen().generate_spectral_table(AGES, ZS)
    table, meta = cdir / NAME, cdir / (NAME + '.meta.json')
    assert table.exists() and meta.exists()
    assert json.loads(meta.read_text())['key']['template'] == 'SB99'
    mtime = os.path.getmtime(table)
    FakeGen().generate_spectral_table(AGES, ZS)
    assert os.path.getmtime(table) == mtime and _stale(cdir) == []


@pytest.mark.parametrize("change", ["source", "grid"])
def test_changed_request_moves_aside_and_regenerates(cdir, change):
    FakeGen().generate_spectral_table(AGES, ZS)
    old = _md5(cdir / NAME)
    if change == "source":
        FakeGen(src='v2', scale=2.0).generate_spectral_table(AGES, ZS)
    else:
        FakeGen().generate_spectral_table(AGES, [0.004])
    assert _md5(cdir / NAME) != old
    moved = [p for p in (cdir / 'stale_tables').rglob(NAME)]
    assert len(moved) == 1 and _md5(moved[0]) == old          # nothing deleted, content intact


def test_registered_legacy_table_is_accepted(cdir, monkeypatch):
    FakeGen().generate_spectral_table(AGES, ZS)
    os.remove(cdir / (NAME + '.meta.json'))
    monkeypatch.setitem(csg.RELEASED_TABLE_MD5, NAME, _md5(cdir / NAME))
    mtime = os.path.getmtime(cdir / NAME)
    FakeGen(src='other').generate_spectral_table(AGES, ZS)
    assert os.path.getmtime(cdir / NAME) == mtime and _stale(cdir) == []
    assert json.loads((cdir / (NAME + '.meta.json')).read_text())['origin'] == 'released-registry'


def test_unregistered_legacy_table_is_moved_aside(cdir):
    (cdir / NAME).write_text("stale table from an older code\n")
    old = _md5(cdir / NAME)
    FakeGen().generate_spectral_table(AGES, ZS)
    assert (cdir / (NAME + '.meta.json')).exists()
    moved = [p for p in (cdir / 'stale_tables').rglob(NAME)]
    assert len(moved) == 1 and _md5(moved[0]) == old


def test_registry_lists_the_released_bundle():
    assert csg.RELEASED_TABLE_MD5['BPASS_chab100_bin_burst_resFac10.ascii'] == '9275d20d0f72ceb0e155d47783339c5a'
    assert csg.RELEASED_TABLE_MD5['SB99_kroupa100_sin_burst.ascii'] == '7f51036e8c5d5db1fbafdee05916217e'
    assert len(csg.RELEASED_TABLE_MD5) == 9


def test_unreadable_sidecar_means_not_current(cdir):
    FakeGen().generate_spectral_table(AGES, ZS)
    (cdir / (NAME + '.meta.json')).write_text('{"origin": "gener')       # truncated
    old = _md5(cdir / NAME)
    FakeGen().generate_spectral_table(AGES, ZS)
    moved = [p for p in (cdir / 'stale_tables').rglob(NAME)]
    assert len(moved) == 1 and _md5(moved[0]) == old
    json.loads((cdir / (NAME + '.meta.json')).read_text())


def test_numpy_scalars_in_the_key(cdir):
    """A numpy formation timescale must not break the sidecar (and so every later call)."""
    g = FakeGen()
    g.generate_spectral_table(AGES, ZS, formation_timescale=np.float32(2e6))
    name = g._generate_output_filename(np.float32(2e6))
    json.loads((cdir / (name + '.meta.json')).read_text())
    mtime = os.path.getmtime(cdir / name)
    FakeGen().generate_spectral_table(AGES, ZS, formation_timescale=np.float64(2e6))
    assert os.path.getmtime(cdir / name) == mtime


def test_hand_edited_table_is_not_current(cdir):
    FakeGen().generate_spectral_table(AGES, ZS)
    with open(cdir / NAME, 'a') as f:
        f.write('1.0\n')
    old = _md5(cdir / NAME)
    FakeGen().generate_spectral_table(AGES, ZS)
    moved = [p for p in (cdir / 'stale_tables').rglob(NAME)]
    assert len(moved) == 1 and _md5(moved[0]) == old
