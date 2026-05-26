"""Unit tests for LogStallWatchdog: escalate on a *stalled log*, not a flat timeout.

These use small (second-scale) thresholds and SIGALRM, so they run in the main thread
only (pytest does). They verify the three behaviours the evolution retry ladder relies
on: a growing log is never interrupted, a frozen log escalates, and the absolute
hard-cap still fires when there is no log to watch.
"""
import os
import time

import pytest

from toddlers.timeout_manager import LogStallWatchdog


def test_growing_log_is_not_interrupted(tmp_path):
    # The log keeps growing for ~3 s with a 2 s stall threshold -> must NOT raise,
    # however "slow" the operation is.
    log = tmp_path / "run.log"
    log.write_text("start\n")
    reached_end = False
    with LogStallWatchdog(str(log), stall_timeout=2, poll=1):
        t0 = time.time()
        with open(log, "a") as f:
            while time.time() - t0 < 3.0:
                f.write("still working\n")
                f.flush()
                os.fsync(f.fileno())
                time.sleep(0.4)
        reached_end = True
    assert reached_end  # exited the block without a TimeoutError


def test_frozen_log_escalates(tmp_path):
    # The log never grows -> the watchdog must raise within ~stall_timeout, well before
    # the (bounded) busy-loop would otherwise end.
    log = tmp_path / "run.log"
    log.write_text("start\n")
    with pytest.raises(TimeoutError):
        with LogStallWatchdog(str(log), stall_timeout=2, poll=1):
            t0 = time.time()
            while time.time() - t0 < 8.0:  # safety bound; watchdog should fire at ~2 s
                pass


def test_hard_cap_fires_without_a_log(tmp_path):
    # No log file to watch -> stall detection is inert, but the absolute backstop must
    # still interrupt a runaway.
    with pytest.raises(TimeoutError):
        with LogStallWatchdog(None, stall_timeout=10, poll=1, hard_cap=2):
            t0 = time.time()
            while time.time() - t0 < 8.0:
                pass


def test_handler_is_restored(tmp_path):
    # The context manager must leave SIGALRM as it found it.
    import signal
    log = tmp_path / "run.log"
    log.write_text("x\n")
    before = signal.getsignal(signal.SIGALRM)
    with LogStallWatchdog(str(log), stall_timeout=5, poll=1):
        pass
    assert signal.getsignal(signal.SIGALRM) == before
