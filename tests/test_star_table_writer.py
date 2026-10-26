"""The Cloudy star-table writer shared by SpectralTableGenerator and the stochastic tables.

The stochastic tables (generate_cloudy_spectral_table, generate_cloudy_spectral_table_2d)
go through the same writer as SpectralTableGenerator: the reference bytes, a sidecar
with the content key and the paired feedback-table md5, grid checks, reuse only on
matching content. The spectra are synthetic (compute_time_delayed_spectra is replaced),
so no database is needed.
"""
import hashlib
import json
import os

import numpy as np
import pytest

import toddlers.constants
from toddlers.pysb99.stochastic import interpolants as stoch

# reference md5 of the tables for the synthetic inputs below (the writer must reproduce
# them byte for byte).
GOLDEN_MD5_1D = "8198a1dc7d75783ac16226936c0f9a3a"
GOLDEN_MD5_2D = "eac1f5e4fb23f0e6b4d44b7f0e2d26be"

WAVELENGTHS = np.linspace(100.0, 1.0e4, 23)
METS = {'SMC': 1.0, 'LMC': 2.0, 'MW': 3.0}


def fake_spectra(masses, initial_ages, time_grid_myr, database_path, metallicity,
                 rotation=False, n_processes=None, verbose=True):
    """Deterministic stand-in for compute_time_delayed_spectra."""
    t = np.asarray(time_grid_myr, dtype=float)
    scale = METS.get(metallicity, 1.0) * float(np.sum(masses))
    spectra = scale * np.outer(1.0 + t, WAVELENGTHS ** -1.5) * 1e30
    return WAVELENGTHS.copy(), spectra


def _md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


@pytest.fixture
def cloudy_dir(tmp_path, monkeypatch):
    d = tmp_path / 'cloudy_data'
    d.mkdir()
    monkeypatch.setattr(toddlers.constants, 'CLOUDY_DATA_DIR', str(d))
    import toddlers.cloudy_stellar_spectra_generator as csg
    monkeypatch.setattr(csg, 'CLOUDY_DATA_DIR', str(d), raising=False)
    monkeypatch.setattr(stoch, 'compute_time_delayed_spectra', fake_spectra)
    return d


def _inputs():
    masses = np.array([1.0, 10.0, 40.0])
    ages = np.zeros(3)
    tg = np.array([0.01, 0.5, 1.0, 3.0, 10.0])
    db = os.path.join(os.path.dirname(__file__), 'conftest.py')  # any existing file stands in for the h5
    return masses, ages, tg, db


def write_1d(name='stoch_1d.ascii', masses=None, **kw):
    m, a, tg, db = _inputs()
    return stoch.generate_cloudy_spectral_table(m if masses is None else masses, a, db, 'MW', name,
                                                time_grid_myr=tg, verbose=False, **kw)


def write_2d(name='stoch_2d.ascii', **kw):
    m, a, tg, db = _inputs()
    return stoch.generate_cloudy_spectral_table_2d(m, a, db, metallicities=list(METS),
                                                   output_filename=name, time_grid_myr=tg,
                                                   verbose=False, **kw)


def test_stochastic_tables_bytes_unchanged(cloudy_dir):
    assert GOLDEN_MD5_1D is not None and GOLDEN_MD5_2D is not None
    assert _md5(write_1d()) == GOLDEN_MD5_1D
    assert _md5(write_2d()) == GOLDEN_MD5_2D


def test_stochastic_table_sidecar(cloudy_dir, tmp_path):
    fb = tmp_path / 'fb.obj'
    fb.write_bytes(b'feedback')
    lw = tmp_path / 'fb_LumLymanWerner.obj'
    lw.write_bytes(b'lyw')
    path = write_1d(paired_feedback_files=[str(fb), str(lw)])
    with open(path + '.meta.json') as f:
        meta = json.load(f)
    assert meta['md5'] == _md5(path)
    assert meta['origin'] == 'generated'
    assert meta['paired_feedback_md5'] == hashlib.md5(b'feedback' + b'lyw').hexdigest()
    assert meta['key']['kind'] == 'pySB99_stochastic'


def test_stochastic_table_reuse_and_move_aside(cloudy_dir):
    path = write_1d()
    mtime = os.stat(path).st_mtime_ns
    write_1d()
    assert os.stat(path).st_mtime_ns == mtime          # same content: reused, not rewritten
    write_1d(masses=np.array([1.0, 10.0, 50.0]))       # different population: regenerated
    stale = cloudy_dir / 'stale_tables'
    assert stale.is_dir() and any(stale.rglob('stoch_1d.ascii'))


def test_writer_refuses_bad_grid(cloudy_dir):
    from toddlers.cloudy_stellar_spectra_generator import write_cloudy_star_table
    good = np.ones((1, 2, WAVELENGTHS.size))
    p = str(cloudy_dir / 'bad.ascii')
    with pytest.raises(ValueError, match="finite"):
        bad = good.copy()
        bad[0, 1, 3] = np.nan
        write_cloudy_star_table(p, [0.0, 1e6], [0.014], WAVELENGTHS, bad, sidecar={})
    with pytest.raises(ValueError, match="ascending"):
        write_cloudy_star_table(p, [1e6, 0.0], [0.014], WAVELENGTHS, good, sidecar={})
    with pytest.raises(ValueError, match="metallicit"):
        write_cloudy_star_table(p, [0.0, 1e6], [0.0], WAVELENGTHS, good, sidecar={})
    with pytest.raises(ValueError, match="shape"):
        write_cloudy_star_table(p, [0.0, 1e6, 2e6], [0.014], WAVELENGTHS, good, sidecar={})
    assert not os.path.exists(p)


def test_spectral_table_generator_records_paired_feedback_md5(cloudy_dir):
    """SpectralTableGenerator's sidecar names the feedback table it pairs with."""
    from toddlers.cloudy_stellar_spectra_generator import SpectralTableGenerator
    from toddlers.stellar_feedback import feedback_table_md5
    g = SpectralTableGenerator.__new__(SpectralTableGenerator)
    g.template, g.imf, g.star_type = 'SB99', 'kroupa100', 'sin'
    g.wavelength_resolution_factor, g.resample, g.max_age = 1, 'decimate', 3.5e7
    g.burst_data = {'wavelengths': WAVELENGTHS,
                    'ages': np.array([0.0, 1e6, 2e6]), 'metallicities': np.array([0.02])}
    g._source_fingerprint = lambda: {'synthetic': True}
    g._get_burst_spectrum = lambda t, Z: WAVELENGTHS ** -1.5 * (1 + t / 1e6)
    g.generate_spectral_table(np.array([0.0, 1e6]), [0.02])
    path = os.path.join(str(cloudy_dir), g._generate_output_filename())
    with open(path + '.meta.json') as f:
        meta = json.load(f)
    assert meta['md5'] == _md5(path)
    assert meta['paired_feedback_md5'] == feedback_table_md5('SB99', 'kroupa100', 'sin')
