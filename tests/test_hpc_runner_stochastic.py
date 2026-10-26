"""The HPC evolution runner carries a stochastic spec.

A task is one JSON line, so a stochastic spec travels as a nested object:
``"stochastic": {"imf": {"name": "kroupa", "m_low": 0.1, "m_up": 100.0}, "base_seed": 23}``
(optional keys: database_path, interpolation_mode, n_processes, time_step_myr; the database
defaults to the worker machine's own). In a grid file the axis is a list of such objects.
Bars: the spec reaches Evolution unchanged through a grid -> task line -> runner round trip
(run_evolution_task passes the key through); a spec key Evolution does not know (a typo) is
refused, naming the key.

The row is the recollapsing stochastic cloud of test_output_root_evolution (Z 0.014,
eta_sf 0.01, n 2560, 1e6 Msun, seed 23: two or more generations). In the default suite
Evolution is replaced by a recorder; the real multi-generation run (needs the 2.1.0
single-star database) is the data/slow test at the end.
"""
import glob
import json
import os
import types

import pytest

from toddlers.hpc import generate_tasks, runner

SPEC = {"imf": {"name": "kroupa", "m_low": 0.1, "m_up": 100.0}, "base_seed": 23}
GRID = {"Z": [0.014], "eta_sf": [0.01], "n_cl": [2560.0], "M_cl_init": [1e6], "template": ["pySB99"],
        "stochastic": [SPEC]}


def _task_rows(tmp_path):
    grid = tmp_path / 'grid.json'
    grid.write_text(json.dumps(GRID))
    generate_tasks.cmd_evolution(types.SimpleNamespace(grid=str(grid), outdir=str(tmp_path / 'tasks')))
    with open(tmp_path / 'tasks' / 'evolution.tasks') as f:
        return [json.loads(line) for line in f if line.strip()]


class _Recorder:
    calls = []

    def __init__(self, **kw):
        _Recorder.calls.append(kw)

    def run_simulation(self):
        return []


def test_stochastic_spec_reaches_evolution(tmp_path, monkeypatch):
    import toddlers.evolution
    _Recorder.calls = []
    monkeypatch.setattr(toddlers.evolution, 'Evolution', _Recorder)
    rows = _task_rows(tmp_path)
    assert len(rows) == 1
    runner.run_evolution_task(rows[0])
    kw = _Recorder.calls[0]
    assert kw['stochastic'] == SPEC
    from toddlers.pysb99.stochastic.population import StochasticSpec
    spec = StochasticSpec.coerce(kw['stochastic'])
    assert (spec.imf.name, spec.imf.m_low, spec.imf.m_up, spec.base_seed) == ('kroupa', 0.1, 100.0, 23)


def test_unknown_spec_key_refused(tmp_path, monkeypatch):
    import toddlers.evolution
    monkeypatch.setattr(toddlers.evolution, 'Evolution', _Recorder)
    row = _task_rows(tmp_path)[0]
    row['stochastic'] = dict(SPEC, interp_mode='eep')
    with pytest.raises(ValueError, match="interp_mode"):
        runner.run_evolution_task(row)


@pytest.mark.data
@pytest.mark.slow
def test_recollapsing_stochastic_row_runs(tmp_path, monkeypatch):
    from toddlers._paths import get_database_path
    from toddlers.pysb99.stochastic.population import file_md5
    db = get_database_path()
    from toddlers._paths import data_resources
    want = data_resources()['single_star_tracks']['md5']
    if not os.path.exists(db) or file_md5(db) != want:
        pytest.skip(f"needs the single-star database of this version (md5 {want})")
    monkeypatch.setenv('TODDLERS_OUTPUT_ROOT', str(tmp_path / 'root'))
    runner.run_evolution_task(_task_rows(tmp_path)[0])
    dats = glob.glob(str(tmp_path / 'root' / 'evolution_output' / '**' / '*.dat'), recursive=True)
    assert len(dats) == 1
    params = json.load(open(dats[0]))['parameters']
    assert params['stochastic']['base_seed'] == 23
    assert len(params['stochastic']['generations']) >= 2
    assert len(glob.glob(dats[0][:-len('.dat')] + '_gen*_feedback.h5')) >= 2
