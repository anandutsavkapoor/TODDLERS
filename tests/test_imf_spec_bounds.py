"""IMF ranges the stochastic sampler can represent.

The kroupa form here has two segments (slope 1.3 from 0.08 to 0.5 Msun) and no upper cut.
Below 0.08 Msun Kroupa (2001) has a third segment (slope 0.3) that this form does not
carry, and a star above the database's top track would be given that track's feedback
(nearest-track snapping). Bars: IMFSpec('kroupa', m_low < 0.08, ...) raises ValueError; a
draw whose m_up exceeds the top track mass of the single-star database raises
ConfigurationError naming both masses, in every generation (0 and a recollapse generation
1); m_up equal to the top track and the standard 0.1-100 range draw.
Data-free: the database is a stand-in file and its track masses a stub.
"""
import types

import numpy as np
import pytest

from toddlers.exceptions import ConfigurationError
from toddlers.pysb99.stochastic import population as P

TRACKS = np.array([0.8, 1.0, 5.0, 20.0, 120.0, 498.5])


@pytest.fixture
def db(tmp_path, monkeypatch):
    p = tmp_path / 'single_star_tracks.h5'
    p.write_bytes(b'stand-in')
    monkeypatch.setattr(P, '_interpolator', lambda path, label: types.SimpleNamespace(_db_masses=TRACKS))
    return str(p)


def test_kroupa_below_008_refused():
    with pytest.raises(ValueError, match="0.08"):
        P.IMFSpec('kroupa', 0.05, 100.0)
    P.IMFSpec('kroupa', 0.08, 100.0)
    P.IMFSpec('top_heavy', 0.05, 100.0)   # not Kroupa: its own definition, accepted


@pytest.mark.parametrize('generation', [0, 1])
def test_m_up_above_top_track_refused(db, generation):
    spec = P.StochasticSpec(P.IMFSpec('kroupa', 0.1, 1000.0), base_seed=3, database_path=db)
    with pytest.raises(ConfigurationError, match="498.5"):
        P.draw_generation(spec, generation, 1e4, 'burst', 0.014)


@pytest.mark.parametrize('m_up', [100.0, 498.5])
def test_ranges_inside_the_tracks_draw(db, m_up):
    spec = P.StochasticSpec(P.IMFSpec('kroupa', 0.1, m_up), base_seed=3, database_path=db)
    for g in (0, 1):
        pop = P.draw_generation(spec, g, 1e4, 'burst', 0.014)
        assert pop.masses_msun.max() <= m_up and pop.drawn_mass_msun >= 1e4
