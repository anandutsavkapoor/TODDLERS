"""CloudyOutputHandler with check_file_stability (the positional flag at
cloudy_output_handler.py:45, and the stability check per model type).

(a) The flag turns on the stability check only; it does not reach print_traceback, the
    first positional parameter of check_cloudy_success. Bar: with stability on, a pending
    model (no 'Cloudy exited OK') prints nothing, and the stability check still runs on a
    finished one.
(b) The stability check asks for the files each phase writes; a dissolved model writes only
    .cont. Bar: dissolved checks .cont only; shell, unified and dig check .cont, .phy, .rad.
is_file_stable is replaced by a recorder (it sleeps for real, up to 120 s per missing file);
parse_data=False.
"""
import pytest

from toddlers import cloudy_output_handler as H
from toddlers.constants import MYR_TO_SEC

T = 1.5 * MYR_TO_SEC


def _write(tmp_path, prefix, exts, ok=True):
    base = tmp_path / f"{prefix}_1.50"
    (tmp_path / f"{prefix}_1.50.out").write_text(" Cloudy exited OK\n" if ok else " still running\n")
    for ext in exts:
        (tmp_path / f"{prefix}_1.50.{ext}").write_text("#header\n1 2 3\n")
    return base


@pytest.fixture
def stable_calls(monkeypatch):
    calls = []

    def fake(path, *a, **k):
        calls.append(path.rsplit('.', 1)[-1])
        import os
        return os.path.exists(path)
    monkeypatch.setattr(H, 'is_file_stable', fake)
    return calls


def test_pending_model_prints_nothing(tmp_path, monkeypatch, capsys, stable_calls):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, 'shell', ['cont', 'phy', 'rad'], ok=False)
    h = H.CloudyOutputHandler('shell', T, check_file_stability=True, parse_data=False)
    assert capsys.readouterr().out == ''
    assert not h.check_cloudy_success() and stable_calls == []


@pytest.mark.parametrize('prefix, exts, checked', [
    ('dissolved', ['cont'], ['cont']),
    ('shell', ['cont', 'ovr', 'phy', 'rad', 'cum'], ['cont', 'phy', 'rad']),
    ('unified', ['cont', 'ovr', 'phy', 'rad', 'cum'], ['cont', 'phy', 'rad']),
    ('dig', ['cont', 'ovr', 'phy', 'rad', 'cum'], ['cont', 'phy', 'rad']),
], ids=['dissolved', 'shell', 'unified', 'dig'])
def test_stability_files_per_phase(tmp_path, monkeypatch, stable_calls, prefix, exts, checked):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, prefix, exts)
    # an input without the .cumEmer save (no grains), so .cumEmer is not required
    (tmp_path / f"{prefix}_1.50.in").write_text('save last continuum ".cont"\n')
    h = H.CloudyOutputHandler(prefix, T, check_file_stability=True, parse_data=False)
    stable_calls.clear()
    assert h.check_cloudy_success()
    assert stable_calls == checked
