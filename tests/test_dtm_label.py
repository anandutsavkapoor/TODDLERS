#!/usr/bin/env python3
"""The dust-to-metal filename label must round-trip and must NOT collapse small values.

The library labels Cloudy outputs / interpolants / SEDs with a ``_dtm<val>`` suffix and parses
the f_dust axis back from those names. A fixed 2-decimal format silently turned 1e-3 into
``_dtm0.00`` (i.e. 0.0), mislabeling the model and breaking the log-scaled f_dust axis. These
tests pin the shared ``dtm_label`` / ``parse_dtm`` helpers (single source of truth).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toddlers.utils import dtm_label, parse_dtm


@pytest.mark.parametrize("dtm,suffix", [
    (1.0, ""), (0.001, "_dtm0.001"), (0.02, "_dtm0.02"), (0.1, "_dtm0.1"),
    (0.4, "_dtm0.4"), (0.8, "_dtm0.8"),
])
def test_label(dtm, suffix):
    assert dtm_label(dtm) == suffix


@pytest.mark.parametrize("dtm", [0.001, 0.02, 0.1, 0.4, 0.6, 0.8])
def test_round_trip(dtm):
    name = f"TODDLERS_totSED_lr_BPASS_chab100_bin{dtm_label(dtm)}.pkl"
    assert abs(parse_dtm(name) - dtm) < 1e-12


def test_no_suffix_parses_to_one():
    assert parse_dtm("TODDLERS_totSED_lr_BPASS_chab100_bin.pkl") == 1.0
    assert dtm_label(1.0) == ""


def test_1e3_not_collapsed_to_zero():
    # the whole reason for the change: 1e-3 must stay 0.001, never 0.00
    assert dtm_label(0.001) == "_dtm0.001"
    assert dtm_label(0.001) != "_dtm0.00"
    assert parse_dtm(f"x{dtm_label(0.001)}.pkl") == 0.001


def test_orchestration_suffix_matches_label():
    # verify_dtm / dtm_sweep keep an inline _suffix (to stay matplotlib-free) -- it must
    # produce exactly the same labels as the central dtm_label, or the wrapper and the
    # pipeline would disagree on filenames.
    from toddlers.hpc import verify_dtm, dtm_sweep
    for dtm in [0.001, 0.02, 0.1, 0.4, 0.8, 1.0]:
        assert verify_dtm._suffix(dtm) == dtm_label(dtm)
        assert dtm_sweep._suffix(dtm) == dtm_label(dtm)
