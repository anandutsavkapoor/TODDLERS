"""Shared pytest fixtures for the TODDLERS test suite."""
import pytest


@pytest.fixture
def output_dir(tmp_path):
    """A temporary, per-test output directory (replaces the old missing fixture)."""
    return str(tmp_path)
