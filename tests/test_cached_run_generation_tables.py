"""A cached stochastic run writes its generation tables when asked.

Evolution reuses a current .dat without running, and save_generation_tables is not part of
the cache key. Bar: constructing a cached run (first made with save_generation_tables=False)
with save_generation_tables=True writes every missing generation table (rebuilt from the
recorded star lists, as load_run_generations does), after which load_run_generations reads
every generation from file; with False nothing is written. A table already current is left
as it is.

Multi-generation: the recorded run has two generations (a recollapse), and both tables
must appear, each with its own generation's population.
Data-free seam: the single-star database is a stand-in file, the cache status is forced
to 'current', and the per-generation feedback (build_generation_table, check_redraw) is
replaced by a synthetic table built with the real table builder.
"""
import json
import os

import numpy as np
import pytest

from toddlers import evolution as E
from toddlers.feedback_tables import load_run_generations, read_table_file
from toddlers.pysb99.stochastic import population as P


def _pop(gen, db_md5):
    rng = np.random.default_rng(gen)
    m = np.sort(rng.uniform(1.0, 60.0, 20))
    return P.GenerationPopulation(
        masses_msun=m, birth_times_myr=np.zeros(m.size), imf_name='kroupa', imf_m_low_msun=0.1,
        imf_m_up_msun=100.0, base_seed=17, generation_index=gen, target_mass_msun=float(m.sum()),
        sf_mode='burst', tau_sf_myr=None, metallicity_Z=0.014, track_min_mass_msun=0.8,
        interpolation_mode='snap', database_md5=db_md5, seed_spawn_key=[gen])


def _synthetic_table(pop, spec, *a, **k):
    t = P.generation_time_grid(spec)
    base = np.exp(-t / 10.0)[None, :] * (1 + pop.generation_index)
    q = {'L_bol': 1e40 * base, 'L_ion': 4e39 * base, 'Q_HI': 1e51 * base, 'Q_HeI': 1e50 * base,
         'Q_HeII': 1e48 * base, 'L_LyW': 1e37 * base, 'L_mech_wind': 1e38 * base,
         'L_mech_SN': 0 * base, 'F_ram_wind': 1e30 * base, 'F_ram_SN': 0 * base}
    return P.generation_table_from_quantities(pop, t, q, spec)


def _write_dat(path, spec, pops):
    rec = spec.to_dict()
    rec['generations'] = P.population_records(pops)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'parameters': {'stochastic': rec}}, f)


def _output_path(tmp_path, spec, monkeypatch):
    """The .dat path Evolution uses for these arguments (read from get_output_paths, the
    construction stopped right after it)."""
    calls = {}
    real = E.Evolution.get_output_paths

    def spy(self):
        out = real(self)
        calls['path'] = out[0]
        raise _Stop
    monkeypatch.setattr(E.Evolution, 'get_output_paths', spy)
    with pytest.raises(_Stop):
        E.Evolution(0.014, 0.05, 160.0, 1e5, 'pySB99', stochastic=spec,
                    output_root=str(tmp_path / 'root'), skip_logger_init=True)
    monkeypatch.setattr(E.Evolution, 'get_output_paths', real)
    return calls['path']


class _Stop(Exception):
    pass


@pytest.mark.parametrize('save', [True, False])
def test_cached_run_writes_missing_tables(tmp_path, monkeypatch, save):
    db = tmp_path / 'single_star_tracks.h5'
    db.write_bytes(b'stand-in single-star database')
    spec = P.StochasticSpec(imf=P.IMFSpec('kroupa', 0.1, 100.0), base_seed=17, database_path=str(db))
    monkeypatch.setattr(P, 'build_generation_table', _synthetic_table)
    monkeypatch.setattr(P, 'check_redraw', lambda pop, spec: None)
    monkeypatch.setattr(E, 'cached_output_status', lambda *a, **k: 'current')
    pops = [_pop(g, spec.database_md5()) for g in (0, 1)]
    dat = _output_path(tmp_path, spec, monkeypatch)
    _write_dat(dat, spec, pops)
    stem = dat[:-len('.dat')]

    evo = E.Evolution(0.014, 0.05, 160.0, 1e5, 'pySB99', stochastic=spec,
                      save_generation_tables=save, output_root=str(tmp_path / 'root'),
                      skip_logger_init=True)
    assert evo._output_exists
    files = [f"{stem}_gen{g}_feedback.h5" for g in (0, 1)]
    if not save:
        assert not any(os.path.exists(f) for f in files)
        return
    assert all(os.path.exists(f) for f in files)
    gens = load_run_generations(dat)
    assert [g['source'] for g in gens] == ['file', 'file']
    for g, p in zip(gens, pops):
        assert g['table'].population['scalars']['population_md5'] == p.arrays_md5()
        assert g['table'].population['scalars']['generation_index'] == p.generation_index
    # a second construction leaves the current tables untouched
    mtimes = [os.stat(f).st_mtime_ns for f in files]
    E.Evolution(0.014, 0.05, 160.0, 1e5, 'pySB99', stochastic=spec, output_root=str(tmp_path / 'root'),
                skip_logger_init=True)
    assert [os.stat(f).st_mtime_ns for f in files] == mtimes


def test_outdated_table_moved_aside_not_overwritten(tmp_path, monkeypatch):
    db = tmp_path / 'single_star_tracks.h5'
    db.write_bytes(b'stand-in single-star database')
    spec = P.StochasticSpec(imf=P.IMFSpec('kroupa', 0.1, 100.0), base_seed=17, database_path=str(db))
    monkeypatch.setattr(P, 'build_generation_table', _synthetic_table)
    monkeypatch.setattr(P, 'check_redraw', lambda pop, spec: None)
    monkeypatch.setattr(E, 'cached_output_status', lambda *a, **k: 'current')
    pops = [_pop(g, spec.database_md5()) for g in (0, 1)]
    dat = _output_path(tmp_path, spec, monkeypatch)
    _write_dat(dat, spec, pops)
    stem = dat[:-len('.dat')]
    other = _pop(5, spec.database_md5())          # another star list at the gen0 name
    other.generation_index = 0
    from toddlers.feedback_tables import write_table_file
    write_table_file(_synthetic_table(other, spec).feedback_table, f"{stem}_gen0_feedback.h5")
    old_md5 = read_table_file(f"{stem}_gen0_feedback.h5").md5
    E.Evolution(0.014, 0.05, 160.0, 1e5, 'pySB99', stochastic=spec, output_root=str(tmp_path / 'root'),
                skip_logger_init=True)
    moved = [p for p in (tmp_path / 'root').rglob('stale/*/*_gen0_feedback.h5')]
    assert len(moved) == 1 and read_table_file(str(moved[0])).md5 == old_md5
    assert [g['source'] for g in load_run_generations(dat)] == ['file', 'file']
