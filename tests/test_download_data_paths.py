"""scripts/download_data.py puts the full-resolution BPASS source where the generator
reads it, and checks what it downloaded.

Bars: the downloaded BPASSv2.2.1_*.ascii table lands at the path
SpectralTableGenerator._get_burst_table_path returns (src/others/Cloudy Spectra/), before
the generator runs; a download whose md5 differs from the pinned value is refused (returns
False, nothing left at the source path). Network and the generator are stubbed; the
repository tree is mapped into tmp_path so nothing is written into the checkout.
"""
import hashlib
import importlib.util
import lzma
import os
import types
from pathlib import Path

import pytest

pytest.importorskip('requests')  # scripts/download_data.py imports it at module level

ROOT = str(Path(__file__).resolve().parents[1])


def _load(tmp_path, monkeypatch, payload):
    real_abspath = os.path.abspath

    def mapped(p):
        a = real_abspath(p)
        return str(tmp_path / 'repo') + a[len(ROOT):] if a.startswith(ROOT + os.sep) else a
    spec = importlib.util.spec_from_file_location('download_data_under_test',
                                                  os.path.join(ROOT, 'scripts', 'download_data.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(os.path, 'abspath', mapped)

    blob = lzma.compress(payload)

    class Resp:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield blob
    monkeypatch.setattr(mod.requests, 'get', lambda url, stream=True: Resp())

    import toddlers.cloudy_stellar_spectra_generator as csg
    seen = {}

    class StubGen:
        def __init__(self, template, imf, star_type, wavelength_resolution_factor=1, **kw):
            fake = types.SimpleNamespace(template=template, imf=imf, star_type=star_type)
            seen['read_path'] = csg.SpectralTableGenerator._get_burst_table_path(fake)
            seen['present'] = os.path.exists(seen['read_path'])
            self.max_age = 1e8

        def generate_spectral_table(self, ages, Zs):
            seen['generated'] = True
    monkeypatch.setattr(mod, 'SpectralTableGenerator', StubGen)
    return mod, seen, hashlib.md5(blob).hexdigest(), hashlib.md5(payload).hexdigest()


def test_bpass_source_lands_where_the_generator_reads(tmp_path, monkeypatch):
    payload = b'fake BPASS table\n'
    mod, seen, xz_md5, ascii_md5 = _load(tmp_path, monkeypatch, payload)
    name = 'BPASSv2.2.1_imf_chab100_burst_binary.ascii'
    monkeypatch.setattr(mod, 'BPASS_SOURCE_MD5', {name: {'xz': xz_md5, 'ascii': ascii_md5}},
                        raising=False)
    cdir = tmp_path / 'cloudy_data'
    cdir.mkdir()
    assert mod.download_bpass_table('chab100', 'bin', str(cdir)) is True
    assert seen.get('present') is True, f"generator reads {seen.get('read_path')}, file not there"
    assert seen.get('generated') is True


def test_md5_mismatch_is_refused(tmp_path, monkeypatch):
    mod, seen, xz_md5, ascii_md5 = _load(tmp_path, monkeypatch, b'not the real table\n')
    cdir = tmp_path / 'cloudy_data'
    cdir.mkdir()
    assert mod.download_bpass_table('chab100', 'bin', str(cdir)) is False
    assert 'generated' not in seen
    left = [p for p in (tmp_path / 'repo').rglob('BPASSv2.2.1_*')] if (tmp_path / 'repo').exists() else []
    assert left == []
