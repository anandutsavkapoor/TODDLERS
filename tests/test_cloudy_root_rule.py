"""One rule for where Cloudy output lives: next to the evolution_output/ that holds the .dat.

The Cloudy stage (CloudySimulationManager), the STAB build (TODDLERSInterpolantGenerator) and
the campaign's archive step must agree, also when $TODDLERS_OUTPUT_ROOT points elsewhere
(post-processing existing runs while the variable is set for scratch).
"""
import argparse
import os

import pytest

from toddlers.utils import output_root_of

LEAF = os.path.join("template_SB99", "imf_kroupa100", "star_type_sin", "cluster_mode_burst",
                    "profile_type_uniform")


def test_output_root_of(tmp_path):
    dat = tmp_path / "run" / "evolution_output" / LEAF / "sim.dat"
    assert output_root_of(str(dat)) == str(tmp_path / "run")
    assert output_root_of(str(dat.parent)) == str(tmp_path / "run")
    nested = tmp_path / "evolution_output" / "copy" / "evolution_output" / LEAF
    assert output_root_of(str(nested)) == str(tmp_path / "evolution_output" / "copy")   # the last one
    assert output_root_of(str(tmp_path / "elsewhere" / "sim.dat")) is None


class _Stop(Exception):
    pass


def test_stab_reader_follows_the_writer_rule(tmp_path, monkeypatch):
    from toddlers.stab import interpolants as I
    run = tmp_path / "run"
    evo = run / "evolution_output" / LEAF
    (run / "cloudy_output" / LEAF).mkdir(parents=True)
    evo.mkdir(parents=True)
    monkeypatch.setenv("TODDLERS_OUTPUT_ROOT", str(tmp_path / "scratch"))

    def stop(self):
        raise _Stop

    monkeypatch.setattr(I.TODDLERSInterpolantGenerator, "_extract_template_config", stop)
    gen = I.TODDLERSInterpolantGenerator.__new__(I.TODDLERSInterpolantGenerator)
    with pytest.raises(_Stop):
        gen.__init__(str(evo))
    assert gen.root_dir == run
    assert gen.cloudy_dir == run / "cloudy_output" / LEAF


def test_campaign_archive_root_follows_the_evolution_dir(tmp_path):
    from toddlers.hpc.campaign import _cloudy_root
    evo = tmp_path / "run" / "evolution_output" / LEAF
    args = argparse.Namespace(evolution_dir=str(evo), output_root=str(tmp_path / "scratch"),
                              toddlers_src=str(tmp_path / "src"))
    assert _cloudy_root(args) == str(tmp_path / "run")
    fresh = argparse.Namespace(evolution_dir=None, output_root=str(tmp_path / "scratch"),
                               toddlers_src=str(tmp_path / "src"))
    assert _cloudy_root(fresh) == str(tmp_path / "scratch")


def test_stab_reader_with_a_symlinked_directory(tmp_path, monkeypatch):
    """A directory below evolution_output/ that is a symlink to elsewhere keeps its place."""
    from toddlers.stab import interpolants as I
    run = tmp_path / "run"
    target = tmp_path / "elsewhere" / "template_SB99"
    (target / "imf_kroupa100").mkdir(parents=True)
    (run / "evolution_output").mkdir(parents=True)
    (run / "evolution_output" / "template_SB99").symlink_to(target)
    evo = run / "evolution_output" / "template_SB99" / "imf_kroupa100"
    (run / "cloudy_output" / "template_SB99" / "imf_kroupa100").mkdir(parents=True)

    def stop(self):
        raise _Stop

    monkeypatch.setattr(I.TODDLERSInterpolantGenerator, "_extract_template_config", stop)
    gen = I.TODDLERSInterpolantGenerator.__new__(I.TODDLERSInterpolantGenerator)
    with pytest.raises(_Stop):
        gen.__init__(str(evo))
    assert gen.cloudy_dir == run / "cloudy_output" / "template_SB99" / "imf_kroupa100"
