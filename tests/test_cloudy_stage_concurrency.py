"""Shared data-dir files under parallel workers, and runs moved between machines
(S7-S11 below).

S7: temporary names carry pid, thread and a random part (the local parallel runner uses
threads, so the pid alone does not make a name unique). S8: the patch check waits for a
shell or unified model (a dissolved model writes no .diffContUnatt, so it cannot decide the
check). S9: the stochastic star-table reuse check and write run under one lock. S10: a
lock left by a dead process on this host is broken at once, without waiting for the lock
timeout. S11: a recorded single-star database path missing on this machine falls back to
the local database when its md5 is the recorded one.
"""
import json
import os
import subprocess
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from toddlers import _fileutils
from toddlers._fileutils import DirLock, unique_tmp_path
from toddlers.cloudy_grains_generator import _write_text_if_changed
import toddlers.hpc.runner as runner
import toddlers.pysb99.stochastic.population as P
import toddlers.pysb99.stochastic.interpolants as I


def test_tmp_names_differ_per_thread_and_call(tmp_path):
    p = str(tmp_path / 'x.ascii')
    names = set()
    with ThreadPoolExecutor(8) as ex:
        for n in ex.map(lambda _: unique_tmp_path(p), range(200)):
            names.add(n)
    assert len(names) == 200


def test_threaded_writers_do_not_collide(tmp_path):
    p = str(tmp_path / 'grains.szd')
    texts = [f"content {i}\n" * 50 for i in range(240)]
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda t: _write_text_if_changed(p, t), texts))
    assert open(p).read() in texts
    assert [f.name for f in tmp_path.iterdir()] == ['grains.szd']


def _dead_pid():
    proc = subprocess.Popen([sys.executable, '-c', 'pass'])
    proc.wait()
    return proc.pid


def test_lock_of_a_dead_process_is_broken(tmp_path):
    lock = tmp_path / 'x.lock'
    lock.mkdir()
    import socket
    (lock / 'owner.json').write_text(json.dumps({'host': socket.gethostname(), 'pid': _dead_pid(), 'time': 0}))
    t0 = time.monotonic()
    with DirLock(str(lock), timeout=30):
        assert json.loads((lock / 'owner.json').read_text())['pid'] == os.getpid()
    assert time.monotonic() - t0 < 5 and not lock.exists()


def test_old_unowned_lock_is_broken(tmp_path):
    lock = tmp_path / 'x.lock'
    lock.mkdir()
    old = time.time() - 2 * _fileutils.UNOWNED_LOCK_AGE_S
    os.utime(lock, (old, old))
    with DirLock(str(lock), timeout=5):
        pass


def test_lock_held_from_another_host_is_kept(tmp_path):
    lock = tmp_path / 'x.lock'
    lock.mkdir()
    (lock / 'owner.json').write_text(json.dumps({'host': 'another-host', 'pid': 1, 'time': 0}))
    with pytest.raises(RuntimeError, match='another-host'):
        with DirLock(str(lock), timeout=0.5, poll=0.05):
            pass
    assert lock.exists()


def test_lock_excludes_threads_of_one_process(tmp_path):
    lock, inside, overlap = str(tmp_path / 'x.lock'), [0], []

    def work(_):
        with DirLock(lock, timeout=30, poll=0.01):
            inside[0] += 1
            overlap.append(inside[0])
            time.sleep(0.01)
            inside[0] -= 1
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(work, range(40)))
    assert max(overlap) == 1


def test_star_table_reuse_and_write_are_one_critical_section(tmp_path, monkeypatch):
    state = {'inside': 0, 'max': 0, 'written': False}

    def enter():
        state['inside'] += 1
        state['max'] = max(state['max'], state['inside'])
        time.sleep(0.02)
        state['inside'] -= 1

    def reuse(*a, **k):
        enter()
        return state['written']

    def generate(*a, **k):
        enter()
        state['written'] = True
    monkeypatch.setattr(I, '_reuse_stochastic_table', reuse)
    monkeypatch.setattr(I, 'generate_cloudy_spectral_table', generate)
    monkeypatch.setattr(I, '_stochastic_table_key', lambda *a, **k: 'k')
    monkeypatch.setattr(P, 'star_table_name', lambda pop, spec: 'gen0_burst.ascii')
    pop = types.SimpleNamespace(masses_msun=np.array([20.0]), birth_times_myr=np.array([0.0]),
                                track_min_mass_msun=0.8, metallicity_Z=0.014)
    spec = types.SimpleNamespace(interpolation_mode='snap', database_path='db.h5', n_processes=1)
    ft = types.SimpleNamespace(content_md5=lambda: 'c')
    with ThreadPoolExecutor(4) as ex:
        names = list(ex.map(lambda _: P.ensure_star_table(pop, spec, str(tmp_path), paired_feedback_table=ft),
                            range(4)))
    assert names == ['gen0_burst.ascii'] * 4
    assert state['max'] == 1


@pytest.mark.parametrize('order', [['dissolved', 'dissolved', 'shell'], ['unified']])
def test_patch_check_waits_for_a_shell_or_unified_model(tmp_path, monkeypatch, capsys, order):
    monkeypatch.setattr(runner, '_PATCH_CHECK_LOGGED', False)
    stock = tmp_path / 'm.diffContUnatt'
    ran = []
    for phase in order:
        path = str(stock) if phase != 'dissolved' else str(tmp_path / 'none.diffContUnatt')
        if phase != 'dissolved':
            stock.write_text('#energy/um\tConEmitLocal\n')
        ran.append(runner._patch_check(phase, path, 'cloudy.exe'))
    assert ran[-1] is True and not any(ran[:-1])
    assert '[patch-check]' in capsys.readouterr().out


def _rec(tmp_path, md5):
    return {'imf': {'name': 'kroupa', 'm_low': 0.1, 'm_up': 100.0}, 'base_seed': 17,
            'database_path': str(tmp_path / 'elsewhere' / 'single_star_tracks.h5'), 'database_md5': md5}


def test_relocated_database_found_by_md5(tmp_path, monkeypatch):
    local = tmp_path / 'local.h5'
    local.write_bytes(b'single-star database')
    monkeypatch.setattr('toddlers._paths.get_database_path', lambda: str(local))
    spec = P.StochasticSpec.from_run_record(_rec(tmp_path, P.file_md5(str(local))))
    assert spec.database_path == str(local)


def test_relocated_database_with_another_md5_is_refused(tmp_path, monkeypatch):
    local = tmp_path / 'local.h5'
    local.write_bytes(b'another database')
    monkeypatch.setattr('toddlers._paths.get_database_path', lambda: str(local))
    with pytest.raises(FileNotFoundError, match='does not exist here'):
        P.StochasticSpec.from_run_record(_rec(tmp_path, '0' * 32))
