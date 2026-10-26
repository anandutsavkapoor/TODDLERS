"""The Cloudy line-list reader fails loudly.

_read_line_list feeds every line-luminosity reader (CloudyOutputHandler, the STAB line
maps), so a missing, unreadable or malformed list must be an error, not an empty or
shortened line set downstream. Bar: a missing file raises FileNotFoundError naming the
path; a malformed line raises ValueError naming the file and line number. The shipped v2
list parses with no line dropped except the documented skip (mg 7 9.03094 micron):
148 entries, 147 returned.
"""
import io

import pytest

import toddlers.cloudy_output_handler as coh


def _handler():
    h = coh.CloudyOutputHandler.__new__(coh.CloudyOutputHandler)
    h.skip_lines = [('mg', 7, 9.03094)]
    return h


def _fake_open(text=None):
    real = open

    def fake(path, *a, **k):
        if str(path).endswith('cloudy_lines_TODDLERS_v2.dat'):
            if text is None:
                raise FileNotFoundError(2, 'No such file or directory', str(path))
            return io.StringIO(text)
        return real(path, *a, **k)
    return fake


def test_missing_list_raises(monkeypatch):
    monkeypatch.setattr(coh, 'open', _fake_open(None), raising=False)
    with pytest.raises(FileNotFoundError, match='cloudy_lines_TODDLERS_v2.dat'):
        _handler()._read_line_list()


def test_malformed_line_raises(monkeypatch):
    monkeypatch.setattr(coh, 'open', _fake_open('H  1  6562.81A\nO  3  not-a-wavelength\n'),
                        raising=False)
    with pytest.raises(ValueError, match='line 2'):
        _handler()._read_line_list()


def test_shipped_list_parses_whole():
    lines = _handler()._read_line_list()
    assert len(lines) == 147
