"""download_data.py follows the versioned resources file.

Data on Drive is versioned and never replaced in place; data_resources.json names each
resource's version, Drive id and md5. Bars: the file is well formed; a present database
with the pinned md5 is kept; one with another md5 is moved to stale_data/ (never deleted)
and replaced; a download with the wrong md5 is refused.
"""
import hashlib
import importlib.util
import os
import re
import sys

import pytest

from toddlers._paths import data_resources

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytest.importorskip('requests')


def _load():
    spec = importlib.util.spec_from_file_location('download_data_s3', os.path.join(ROOT, 'scripts', 'download_data.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resources_file_is_well_formed():
    res = data_resources()
    assert {'cloudy_resources', 'single_star_tracks'} <= set(res)
    for name, r in res.items():
        assert re.fullmatch(r'[0-9a-f]{32}', r['md5']), name
        assert r['version'] and r['kind'] in ('file', 'zip') and r['install_to']
        assert r['drive_id'] is None or isinstance(r['drive_id'], str)


@pytest.fixture
def dd(tmp_path, monkeypatch):
    monkeypatch.setenv('TODDLERS_DATA', str(tmp_path))
    mod = _load()
    good = b'the 2.1.0 database' * 1000
    monkeypatch.setattr(mod, 'SINGLE_STAR_TRACKS_MD5', hashlib.md5(good).hexdigest())
    monkeypatch.setattr(mod, 'SINGLE_STAR_TRACKS_DRIVE_ID', 'drive-id')
    monkeypatch.setattr(mod, '_TRACKS_MIN_MB', 0)
    calls = []

    def fake_download(url, dest, quiet=False, payload=good):
        calls.append(url)
        open(dest, 'wb').write(payload)
        return dest
    gdown = type(sys)('gdown')
    gdown.download = fake_download
    monkeypatch.setitem(sys.modules, 'gdown', gdown)
    return mod, tmp_path, good, calls, gdown


def test_current_database_is_kept(dd):
    mod, d, good, calls, _ = dd
    (d / 'single_star_tracks.h5').write_bytes(good)
    assert mod.download_single_star_tracks()
    assert calls == []


def test_old_database_is_moved_aside_and_replaced(dd):
    mod, d, good, calls, _ = dd
    (d / 'single_star_tracks.h5').write_bytes(b'a database of another version')
    assert mod.download_single_star_tracks()
    assert len(calls) == 1 and (d / 'single_star_tracks.h5').read_bytes() == good
    kept = list((d / 'stale_data').rglob('single_star_tracks.h5'))
    assert len(kept) == 1 and kept[0].read_bytes() == b'a database of another version'


def test_corrupt_download_is_refused(dd):
    mod, d, good, calls, gdown = dd
    gdown.download = lambda url, dest, quiet=False: (open(dest, 'wb').write(b'html error page'), dest)[1]
    assert not mod.download_single_star_tracks()
    assert not (d / 'single_star_tracks.h5').exists()


def test_version_not_on_drive_yet_keeps_the_present_file(dd, monkeypatch):
    mod, d, good, calls, _ = dd
    monkeypatch.setattr(mod, 'SINGLE_STAR_TRACKS_DRIVE_ID', None)
    (d / 'single_star_tracks.h5').write_bytes(b'a database of another version')
    assert not mod.download_single_star_tracks()
    assert (d / 'single_star_tracks.h5').read_bytes() == b'a database of another version' and calls == []
