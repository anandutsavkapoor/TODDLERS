"""Multi-generation stochastic runs at the Cloudy input stage (data test: single-star database).

Pass rule:

R3  A two-generation stochastic run (collapse at 3 Myr; burst, and constant SFR with
    tau 1 Myr), rebuilt by the Cloudy input generator from its saved star-list records,
    gives at 6 Myr (both generations shining) two `table star` sources with DIFFERENT
    star-table names; each table's sidecar paired_feedback_content_md5 equals the content
    md5 of its OWN generation's feedback table and differs from the other's; each
    generation's star list equals the draw of generation_rng(base_seed, i).
"""
import json
import logging
import os
import re

import numpy as np
import pytest

from toddlers.constants import M_SUN, MYR_TO_SEC
from toddlers.pysb99.stochastic import population as P
from toddlers.stellar_feedback import StellarFeedback

pytestmark = [pytest.mark.data, pytest.mark.slow]

KROUPA = P.IMFSpec('kroupa', 0.1, 100.0)
Z, ETA, M_CL, SEED = 0.014, 0.05, 4e4, 11
T_COLLAPSE = [3 * MYR_TO_SEC]


@pytest.mark.parametrize("mode,tau_myr", [("burst", None), ("constant_sfr", 1.0)])
def test_R3_each_generation_gets_its_own_star_table(tmp_path, monkeypatch, mode, tau_myr):
    import toddlers.constants as C
    from toddlers.cloudy_base_input_generator import BaseCloudyInputGenerator
    monkeypatch.setattr(C, 'CLOUDY_DATA_DIR', str(tmp_path))
    spec = P.StochasticSpec(KROUPA, SEED)
    tau = None if tau_myr is None else tau_myr * MYR_TO_SEC
    run = StellarFeedback('pySB99', Z, M_CL * M_SUN, ETA, T_COLLAPSE, mode=mode,
                          formation_timescale=tau, stochastic=spec)
    rec = dict(spec.to_dict(), generations=P.population_records(run.populations))
    params = {'Z': Z, 'eta_sf': ETA, 'n_cl': 160.0, 'M_cl_init': M_CL, 'template': 'pySB99',
              'imf': None, 'star_type': 'sin', 'cluster_formation_mode': mode,
              'formation_timescale': tau, 'stochastic': rec}

    class Gen(BaseCloudyInputGenerator):
        def get_density_options(self, t): pass
        def get_geometry_options(self, t): pass
        def get_stopping_criteria(self): pass
        def write_input_file(self, t, filename): pass

    gen = Gen(params, T_COLLAPSE, None, (None,) * 7, str(tmp_path), logger=logging.getLogger('R3'))
    sf = gen.stellar_feedback
    assert len(sf.populations) == 2
    for i, p in enumerate(sf.populations):
        assert np.array_equal(p.masses_msun,
                              P.draw_stop_after(KROUPA, p.target_mass_msun, P.generation_rng(SEED, i)))
    opts = gen.get_primary_source_options(6 * MYR_TO_SEC)
    names = [re.match(r'table star "([^"]+)"', o).group(1) for o in opts if o.startswith('table star')]
    assert len(names) == 2 and names[0] != names[1], names
    md5 = [sf.generation_tables[i].feedback_table.content_md5() for i in range(2)]
    assert md5[0] != md5[1]
    for i, name in enumerate(names):
        with open(os.path.join(str(tmp_path), name + '.meta.json')) as f:
            side = json.load(f)
        assert side['paired_feedback_content_md5'] == md5[i], (i, name)
