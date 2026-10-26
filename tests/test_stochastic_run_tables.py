"""Stochastic generation tables as unified feedback tables
Data tests (single-star database).

Pre-drawn tables are not read back as input (stochastic runs draw at run time
only).

Pass rule:
R20    An evolution run with save_generation_tables=True and one with False (same spec, seed,
       cloud) give identical load_run_generations(...)['table'].get values for every quantity
       of the table at 1, 3, 10 Myr; the first reads the tables from file, the second rebuilds
       them; the star lists are identical.
"""
import os

import numpy as np
import pytest

from toddlers.constants import M_SUN
from toddlers.pysb99.stochastic import population as P

pytestmark = pytest.mark.data
KROUPA = P.IMFSpec('kroupa', 0.1, 100.0)
Z = 0.014


@pytest.mark.slow
def test_R20_saved_and_rebuilt_generation_tables_agree(tmp_path):
    """Both runs write to the package output path (Evolution has no output-root option), so each
    run's files are moved to tmp_path before the next run, which then cannot reuse them."""
    import glob, shutil
    from toddlers.evolution import Evolution
    from toddlers.feedback_tables import load_run_generations
    spec = P.StochasticSpec(KROUPA, 23)
    runs = {}
    for on in (True, False):
        kw = dict(Z=Z, eta_sf=0.05, n_cl=160.0, M_cl_init=2e5 * M_SUN, template='pySB99', imf=None,
                  stochastic=spec, save_generation_tables=on)
        ev = Evolution(**kw)
        stem = ev.output_path[:-len('.dat')]
        for f in glob.glob(stem + '*'):           # a cached track of an earlier run
            os.remove(f)
        ev = Evolution(**kw)
        ev.run_simulation()
        dest = tmp_path / f'on_{on}'
        dest.mkdir()
        for f in glob.glob(stem + '*'):
            shutil.move(f, dest / os.path.basename(f))
        dat = str(dest / os.path.basename(ev.output_path))
        assert os.path.exists(dat[:-len('.dat')] + '_gen0_feedback.h5') == on
        runs[on] = load_run_generations(dat)
    a, b = runs[True], runs[False]
    assert len(a) == len(b) >= 1
    for ga, gb in zip(a, b):
        assert ga['source'] == 'file' and gb['source'] == 'rebuilt'
        assert np.array_equal(ga['masses_msun'], gb['masses_msun'])
        for q in ga['table'].quantities:
            va = ga['table'].get(q, [1.0, 3.0, 10.0]).value
            vb = gb['table'].get(q, [1.0, 3.0, 10.0]).value
            assert np.array_equal(va, vb), q
