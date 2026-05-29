#!/usr/bin/env python3
"""Enumeration must not open a per-cloud log file for every evolution file.

CloudyLogger opens a logging.FileHandler per CloudySimulationManager (unique logger name, never
closed). Building one manager per .dat across a full grid (thousands of clouds) therefore leaks
file descriptors and dies with [Errno 24] Too many open files -- which aborted task generation on
the first real run. Enumeration only reads parameters / time points, so it constructs managers
with add_logger=False (the print-wrapper path, no FileHandler). This test pins that contract.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_enumeration_builds_managers_without_file_logger(monkeypatch):
    import toddlers.cloudy_simulation_manager as csm
    from toddlers.hpc import enumerate_cloudy as ec

    seen_kwargs = []

    class StubManager:
        def __init__(self, sim_file, **kw):
            seen_kwargs.append(kw)
            self.all_results = []                       # no dissolution transition
            self.simulation_params = {"dust_to_metal": 1.0}

        def get_time_points(self):
            return []                                   # no time points -> no task rows needed

    # enumerate imports the class at call time from the source module, so patch it there.
    monkeypatch.setattr(csm, "CloudySimulationManager", StubManager)

    # enumerate_cloudy_tasks is a generator; consume it so the managers are actually built.
    tasks = list(ec.enumerate_cloudy_tasks(["/fake/sim_a.dat", "/fake/sim_b.dat"], verbose=False))

    assert tasks == []                                  # stub yields no work
    assert len(seen_kwargs) == 2                        # one manager per file
    # the leak fix: every manager built during enumeration must disable the file logger
    assert all(kw.get("add_logger") is False for kw in seen_kwargs)
