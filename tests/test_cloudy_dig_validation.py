"""The Cloudy success check requires .cumEmer only from models whose input asked for it
(a DIG model runs without grains and so writes no .cumEmer).

Pass rules:

V1  A DIG model that exited OK with every save file but no .cumEmer, its input requesting none,
    passes check_cloudy_success.
V2  A shell model whose input requests .cumEmer fails when the .cumEmer is missing, and passes
    when it is present.
V3  The DIG model of V1 with a 0-byte .cont fails (the truncation guard applies).
V4  A dissolved model needs only .cont.
"""
import os

import pytest

from toddlers.cloudy_output_handler import CloudyOutputHandler
from toddlers.constants import MYR_TO_SEC

T = 1.47 * MYR_TO_SEC
HEADERS = {'cont': '#Cont  nu\tincident\ttrans\n1 2 3\n', 'ovr': '#depth\tTe\n1 2\n', 'phy': '#PhyC depth\tTe\n1 2\n',
           'rad': '#NZONE\tradius\n1 2\n', 'cum': '#depth\tH  1 6562.80A\n1 2\n', 'cumEmer': '#depth\tH  1 6562.80A\n1 2\n'}
EMER = 'save last lines zone cumulative emergent ".cumEmer"\n'


def _model(d, prefix, with_cumemer_file, request_cumemer, skip=()):
    base = os.path.join(d, f"{prefix}_{T / MYR_TO_SEC:.2f}")
    with open(base + '.in', 'w') as f:
        f.write('title test\nsave last continuum ".cont"\nsave last lines zone cumulative intrinsic ".cum"\n'
                + (EMER if request_cumemer else ''))
    with open(base + '.out', 'w') as f:
        f.write(" Cloudy ends: 226 zones, 3 iterations.\n [Stop in cdMain at maincl.cpp:590, Cloudy exited OK]\n")
    for ext, text in HEADERS.items():
        if ext in skip or (ext == 'cumEmer' and not with_cumemer_file):
            continue
        with open(f"{base}.{ext}", 'w') as f:
            f.write(text)
    return base


def _ok(d, prefix):
    cwd = os.getcwd()
    try:
        return CloudyOutputHandler(prefix, T, absolute_path=d, parse_data=False).check_cloudy_success()
    finally:
        os.chdir(cwd)


def test_V1_dig_without_cumemer_passes(tmp_path):
    _model(str(tmp_path), 'dig', with_cumemer_file=False, request_cumemer=False)
    assert _ok(str(tmp_path), 'dig')


def test_V2_requested_cumemer_still_required(tmp_path):
    _model(str(tmp_path), 'shell', with_cumemer_file=False, request_cumemer=True)
    assert not _ok(str(tmp_path), 'shell')
    with open(os.path.join(str(tmp_path), f"shell_{T / MYR_TO_SEC:.2f}.cumEmer"), 'w') as f:
        f.write(HEADERS['cumEmer'])
    assert _ok(str(tmp_path), 'shell')


def test_V3_truncated_cont_still_fails(tmp_path):
    base = _model(str(tmp_path), 'dig', with_cumemer_file=False, request_cumemer=False)
    open(base + '.cont', 'w').close()
    assert not _ok(str(tmp_path), 'dig')


def test_V4_dissolved_needs_only_cont(tmp_path):
    _model(str(tmp_path), 'dissolved', with_cumemer_file=False, request_cumemer=False,
           skip=('ovr', 'phy', 'rad', 'cum'))
    assert _ok(str(tmp_path), 'dissolved')
