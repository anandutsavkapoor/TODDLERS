#!/usr/bin/env python3
"""The dust-to-metal filename label must round-trip and must NOT collapse small values.

The library labels Cloudy outputs / interpolants / SEDs with a ``_dtm<val>`` suffix and parses
the f_dust axis back from those names. A fixed 2-decimal format silently turned 1e-3 into
``_fdust0.00`` (i.e. 0.0), mislabeling the model and breaking the log-scaled f_dust axis. These
tests pin the shared ``f_dust_label`` / ``parse_f_dust`` helpers (single source of truth).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toddlers.utils import f_dust_label, parse_f_dust


@pytest.mark.parametrize("dtm,suffix", [
    (1.0, ""), (0.001, "_fdust0.001"), (0.02, "_fdust0.02"), (0.1, "_fdust0.1"),
    (0.4, "_fdust0.4"), (0.8, "_fdust0.8"),
])
def test_label(dtm, suffix):
    assert f_dust_label(dtm) == suffix


@pytest.mark.parametrize("dtm", [0.001, 0.02, 0.1, 0.4, 0.6, 0.8])
def test_round_trip(dtm):
    name = f"TODDLERS_totSED_lr_BPASS_chab100_bin{f_dust_label(dtm)}.pkl"
    assert abs(parse_f_dust(name) - dtm) < 1e-12


def test_no_suffix_parses_to_one():
    assert parse_f_dust("TODDLERS_totSED_lr_BPASS_chab100_bin.pkl") == 1.0
    assert f_dust_label(1.0) == ""


def test_1e3_not_collapsed_to_zero():
    # the whole reason for the change: 1e-3 must stay 0.001, never 0.00
    assert f_dust_label(0.001) == "_fdust0.001"
    assert f_dust_label(0.001) != "_fdust0.00"
    assert parse_f_dust(f"x{f_dust_label(0.001)}.pkl") == 0.001


def test_label_never_emits_legacy_dtm_token():
    # we WRITE only the new _fdust form now; the misnamed _dtm token must never be produced
    for fd in (0.001, 0.02, 0.1, 0.4, 0.8):
        assert "_dtm" not in f_dust_label(fd)


@pytest.mark.parametrize("fd", [0.001, 0.02, 0.1, 0.4, 0.8])
def test_parse_reads_legacy_dtm_suffix(fd):
    # back-compat: pre-rename artifacts used a "_dtm<val>" suffix for the same f_dust value;
    # parse_f_dust must still read those (we only stopped WRITING them).
    assert abs(parse_f_dust(f"TODDLERS_totSED_lr_BPASS_chab100_bin_dtm{fd:g}.pkl") - fd) < 1e-12


def test_parse_prefers_fdust_over_legacy_when_both_present():
    # an _fdust token wins over a stray legacy _dtm token in the same name
    assert parse_f_dust("x_dtm0.4_fdust0.02.pkl") == 0.02


def test_orchestration_suffix_matches_label():
    # verify_dtm / dtm_sweep keep an inline _suffix (to stay matplotlib-free) -- it must
    # produce exactly the same labels as the central f_dust_label, or the wrapper and the
    # pipeline would disagree on filenames.
    from toddlers.hpc import verify_dtm, dtm_sweep
    for dtm in [0.001, 0.02, 0.1, 0.4, 0.8, 1.0]:
        assert verify_dtm._suffix(dtm) == f_dust_label(dtm)
        assert dtm_sweep._suffix(dtm) == f_dust_label(dtm)
