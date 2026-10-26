"""A solver retry must keep every constructor argument.

cleanup_after_attempt re-runs __init__ between attempts. Bar: after it, every constructor
argument (f_dust, use_Tion_model, include_gravity, include_lyman_alpha,
include_external_pressure and the others set here to non-default values) keeps its value,
and the run keeps its output path.
"""
import io
import contextlib

import pytest

from toddlers.constants import M_SUN, MYR_TO_SEC
import toddlers.evolution as evolution_module
from toddlers.evolution import Evolution
from toddlers.pysb99.stochastic.population import IMFSpec, StochasticSpec

NON_DEFAULT = dict(
    f_dust=0.3, use_Tion_model=True, include_gravity=False, include_lyman_alpha=False,
    include_external_pressure=False, dynamic_cloud_density=True, dark_matter_fraction=0.1,
    add_cover_frac=True, post_sweep_covering_fraction=0.7, init_with_fragmentation=True,
    cluster_formation_mode='constant_sfr', formation_timescale=3 * MYR_TO_SEC,
)


@pytest.fixture
def evolution(tmp_path, monkeypatch):
    monkeypatch.setenv("TODDLERS_OUTPUT_ROOT", str(tmp_path))
    with contextlib.redirect_stdout(io.StringIO()):
        return Evolution(Z=0.008, eta_sf=0.05, n_cl=320.0, M_cl_init=1e5 * M_SUN,
                         template='SB99', **NON_DEFAULT)


def test_retry_keeps_constructor_arguments(evolution):
    before = {k: getattr(evolution, k) for k in NON_DEFAULT}
    path_before = evolution.output_path
    with contextlib.redirect_stdout(io.StringIO()):
        evolution.cleanup_after_attempt()
    after = {k: getattr(evolution, k) for k in NON_DEFAULT}
    assert after == before
    assert evolution.output_path == path_before


# The same check for a stochastic run (template pySB99, imf=None): the stochastic spec is a
# constructor argument and must survive the retry with everything else. No single-star
# database is read: StellarFeedback is replaced by a stand-in that records the arguments it
# was given and serves the deterministic pySB99 kroupa_100 table (the initial conditions
# need a working feedback object), and the spec's database is a small placeholder file
# (its md5 enters the output fingerprint only).
_RealStellarFeedback = evolution_module.StellarFeedback


class _FeedbackStub(_RealStellarFeedback):
    def __init__(self, *args, **kwargs):
        self.kwargs = dict(kwargs)
        kwargs.update(stochastic=None, imf='kroupa_100')
        super().__init__(*args, **kwargs)


@pytest.fixture
def stochastic_evolution(tmp_path, monkeypatch):
    monkeypatch.setenv("TODDLERS_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(evolution_module, "StellarFeedback", _FeedbackStub)
    db = tmp_path / "placeholder_single_star_tracks.h5"
    db.write_bytes(b"placeholder")
    spec = StochasticSpec(imf=IMFSpec('kroupa', 0.1, 100.0), base_seed=17,
                          database_path=str(db), interpolation_mode='eep')
    kwargs = dict(NON_DEFAULT, stochastic=spec)
    with contextlib.redirect_stdout(io.StringIO()):
        evo = Evolution(Z=0.014, eta_sf=0.05, n_cl=320.0, M_cl_init=1e5 * M_SUN,
                        template='pySB99', imf=None, **kwargs)
    return evo, kwargs


def test_retry_keeps_constructor_arguments_stochastic(stochastic_evolution):
    evolution, kwargs = stochastic_evolution
    before = {k: getattr(evolution, k) for k in kwargs}
    path_before = evolution.output_path
    md5_before = evolution.feedback_md5
    with contextlib.redirect_stdout(io.StringIO()):
        evolution.cleanup_after_attempt()
    after = {k: getattr(evolution, k) for k in kwargs}
    assert after == before
    assert evolution.stochastic is not None and evolution.stochastic == kwargs['stochastic']
    assert evolution.stellar_feedback.kwargs['stochastic'] == kwargs['stochastic']
    assert evolution.feedback_md5 == md5_before
    assert evolution.output_path == path_before
