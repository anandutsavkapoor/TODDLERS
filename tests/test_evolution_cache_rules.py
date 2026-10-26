"""Which cached evolution outputs are reused (S4, S5, S6, S12 below).

S4: the .dat is written atomically; an unreadable (truncated) .dat is stale, not an error
that blocks every resume. S5: a pySB99 track without a feedback-table fingerprint is stale;
SB99/BPASS tracks without one are reused as legacy. S6: the .dat ``inputs`` record keys the
cache, so runs differing only in dark_matter_fraction, the force switches, use_Tion_model or
init_with_fragmentation do not share a track. S12: a saved generation table is used only
with the run's database, the current table rule and a recorded SN deposition mode; moving
an output aside takes its side files along.
"""
import contextlib
import io
import json
import os
import types

import numpy as np
import pytest

import toddlers.evolution as E
from toddlers.constants import M_SUN
from toddlers.feedback_tables.runs import _saved_table_is_current
from toddlers.pysb99.stochastic.population import TABLE_RULE_VERSION
from toddlers.track_simulation import TrackSimulation

MD5 = 'a' * 32
INPUTS = {'Z': 0.02, 'eta_sf': 0.05, 'dark_matter_fraction': 0.0, 'include_gravity': True,
          'profile_params': {'r_core': 1.0}, 'init_with_fragmentation': False}


def _dat(path, **params):
    path.write_text(json.dumps({'parameters': params, 'generations': []}))
    return str(path)


def _current(**over):
    p = {'template': 'SB99', 'feedback_md5': MD5, 'track_code_version': E.TRACK_CODE_VERSION,
         'inputs': dict(INPUTS)}
    p.update(over)
    return p


def test_truncated_dat_is_stale(tmp_path):
    p = tmp_path / 'sim.dat'
    p.write_text('{"parameters": {"feedback_md5": "' + MD5)
    assert E.cached_output_status(str(p), MD5, INPUTS) == 'stale'


def test_dat_write_is_atomic(tmp_path, monkeypatch):
    out = tmp_path / 'sim.dat'
    out.write_text('previous complete file')

    def boom(*a, **k):
        raise KeyboardInterrupt('killed while writing')
    monkeypatch.setattr(json, 'dump', boom)
    with pytest.raises(KeyboardInterrupt):
        TrackSimulation.write_output_file(None, str(out), {}, [], None)
    assert out.read_text() == 'previous complete file'
    assert [f.name for f in tmp_path.iterdir()] == ['sim.dat']


@pytest.mark.parametrize('template,status', [('pySB99', 'stale'), ('SB99', 'legacy'), ('BPASS', 'legacy')])
def test_unfingerprinted_tracks(tmp_path, template, status):
    p = _dat(tmp_path / 'sim.dat', template=template, Z=0.02)
    assert E.cached_output_status(p, MD5, INPUTS) == status


@pytest.mark.parametrize('key,value', [('dark_matter_fraction', 0.1), ('include_gravity', False),
                                       ('init_with_fragmentation', True), ('profile_params', {'r_core': 2.0})])
def test_inputs_key_the_cache(tmp_path, key, value):
    p = _dat(tmp_path / 'sim.dat', **_current())
    assert E.cached_output_status(p, MD5, INPUTS) == 'current'
    assert E.cached_output_status(p, MD5, dict(INPUTS, **{key: value})) == 'stale'


def test_dat_without_inputs_record_is_stale(tmp_path):
    p = _dat(tmp_path / 'sim.dat', **{k: v for k, v in _current().items() if k != 'inputs'})
    assert E.cached_output_status(p, MD5, INPUTS) == 'stale'


def test_older_track_code_is_stale(tmp_path):
    p = _dat(tmp_path / 'sim.dat', **_current(track_code_version=1))
    assert E.TRACK_CODE_VERSION >= 2
    assert E.cached_output_status(p, MD5, INPUTS) == 'stale'


def test_move_aside_takes_the_side_files(tmp_path):
    names = ['sim.dat', 'sim.summary', 'sim_stars.npz', 'sim_gen0_feedback.h5', 'sim_gen1_feedback.h5']
    for n in names:
        (tmp_path / n).write_text('x')
    dest = E.move_output_aside(str(tmp_path / 'sim.dat'))
    assert sorted(os.listdir(dest)) == sorted(names)
    assert not any((tmp_path / n).exists() for n in names)


def _table(md5='p', db='d', rule=TABLE_RULE_VERSION, mode=True):
    conv = {'sn_time_deposition_mode': 'point'} if mode else {}
    return types.SimpleNamespace(population={'scalars': {'population_md5': md5}},
                                 metadata={'source': {'single_star_database': {'database_md5': db},
                                                      'table_rule_version': rule},
                                           'conventions': conv})


@pytest.mark.parametrize('table,ok', [(_table(), True), (_table(md5='q'), False), (_table(db='e'), False),
                                      (_table(rule=TABLE_RULE_VERSION - 1), False), (_table(mode=False), False)])
def test_saved_generation_table_rules(table, ok):
    pop = types.SimpleNamespace(arrays_md5=lambda: 'p')
    assert _saved_table_is_current(table, pop, {'database_md5': 'd'}) is ok


@pytest.mark.data
def test_truncated_dat_does_not_block_a_resume(tmp_path, monkeypatch):
    out = str(tmp_path / 'sim.dat')
    monkeypatch.setattr(E.Evolution, 'get_output_paths', lambda self: (out, str(tmp_path / 'sim.log')))
    with open(out, 'w') as f:
        f.write('{"parameters": {"feedback_md5": "trunc')
    with contextlib.redirect_stdout(io.StringIO()):
        ev = E.Evolution(Z=0.02, eta_sf=0.05, n_cl=100.0, M_cl_init=1e5 * M_SUN, template='SB99', imf='kroupa100')
    assert not getattr(ev, '_output_exists', False)
    assert len(list((tmp_path / 'stale').rglob('sim.dat'))) == 1
