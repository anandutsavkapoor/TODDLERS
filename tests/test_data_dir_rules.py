"""Where the package finds its data: an empty $TODDLERS_DATA counts as unset (the shipped
feedback tables stay visible), and the campaign's default data directory is
<toddlers-src>/toddlers/database for any source path."""
import argparse
import os

from toddlers import _paths


def test_empty_toddlers_data_is_unset(monkeypatch):
    monkeypatch.setenv("TODDLERS_DATA", "")
    assert _paths.get_data_dir() == os.path.join(_paths._PKG_DIR, "database")


def test_set_toddlers_data_is_used(monkeypatch, tmp_path):
    monkeypatch.setenv("TODDLERS_DATA", str(tmp_path))
    assert _paths.get_data_dir() == str(tmp_path)


def test_campaign_default_data_dir():
    from toddlers.hpc.campaign import _common_replacements
    args = argparse.Namespace(account="a", partition="p", ntasks=1, walltime="1:00:00",
                              python_module="", activate_env="", output_root="",
                              toddlers_src="/home/u/src/TODDLERS/src", toddlers_data=None)
    assert _common_replacements(args)["@TODDLERS_DATA@"] == "/home/u/src/TODDLERS/src/toddlers/database"
