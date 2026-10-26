"""A stochastic run without the single-star database stops with a ConfigurationError that names
the download command, before any file is written (not a bare FileNotFoundError)."""
import os

import pytest

from toddlers.constants import M_SUN
from toddlers.exceptions import ConfigurationError
from toddlers.pysb99.stochastic.population import IMFSpec, StochasticSpec


def test_spec_without_database(tmp_path):
    spec = StochasticSpec(IMFSpec("kroupa", 0.1, 100.0), base_seed=1,
                          database_path=str(tmp_path / "missing.h5"))
    with pytest.raises(ConfigurationError, match="download_data.py --stochastic-tracks"):
        spec.database_md5()


def test_evolution_without_database(tmp_path):
    from toddlers.evolution import Evolution
    spec = StochasticSpec(IMFSpec("kroupa", 0.1, 100.0), base_seed=1,
                          database_path=str(tmp_path / "missing.h5"))
    out = tmp_path / "out"
    with pytest.raises(ConfigurationError, match="download_data.py --stochastic-tracks"):
        Evolution(Z=0.014, eta_sf=0.01, n_cl=2560.0, M_cl_init=1e6 * M_SUN, template="pySB99",
                  imf=None, stochastic=spec, profile_type="uniform", output_root=str(out))
    assert not out.exists() or not any(out.rglob("*.dat"))
