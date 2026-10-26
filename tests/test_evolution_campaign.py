"""Evolution and campaign contracts.

Pass rules:

F1  numpy-typed constructor arguments (np.bool_, np.float32, np.int64, numpy values in
      profile_params) give a JSON-serialisable ``inputs`` record; the .dat is written; a
      second Evolution with the same numpy-typed inputs finds it current; a .dat written from
      plain-typed inputs is found current by the numpy-typed twin (no TypeError).
"""
import json
import time
import os

import numpy as np
import pytest

import toddlers.evolution as E
from toddlers.constants import M_SUN
from toddlers.evolution_logging import NullEvolutionLogger
from toddlers.track_simulation import TrackSimulation

BASE = dict(Z=0.02, eta_sf=0.05, n_cl=160.0, M_cl_init=1e5 * M_SUN, template='SB99', imf='kroupa100')


def _np_kwargs():
    return dict(BASE, n_cl=np.int64(160), add_cover_frac=np.bool_(True),
                post_sweep_covering_fraction=np.float32(0.5), dark_matter_fraction=np.float32(0.0),
                include_gravity=np.bool_(True))


def _plain_kwargs():
    return dict(BASE, add_cover_frac=True, post_sweep_covering_fraction=0.5)


def _write_dat(ev, monkeypatch):
    """Write ev's .dat as run_simulation does (parameters incl. inputs; no generations)."""
    monkeypatch.setattr(TrackSimulation, 'write_summary', lambda *a, **k: None)
    params = {'template': ev.template, 'feedback_md5': ev.feedback_md5,
              'track_code_version': E.TRACK_CODE_VERSION, 'inputs': ev._inputs_record()}
    TrackSimulation().write_output_file(ev.output_path, params, [], NullEvolutionLogger())


# ---------------------------------------------------------------- F1
def test_F1_numpy_inputs_record_is_plain(tmp_path):
    ev = E.Evolution(**_np_kwargs(), skip_logger_init=True, output_root=str(tmp_path))
    ev.profile_params = {'r_core': np.float32(1.5), 'grid': np.arange(3), 'flag': np.bool_(False)}
    rec = ev._inputs_record()
    text = json.dumps(rec)
    assert json.loads(text)['profile_params'] == {'r_core': 1.5, 'grid': [0, 1, 2], 'flag': False}
    assert type(rec['add_cover_frac']) is bool and type(rec['n_cl']) is int


def test_F1_numpy_inputs_fresh_write_then_cached(tmp_path, monkeypatch):
    kw = dict(_np_kwargs(), skip_logger_init=True, output_root=str(tmp_path))
    ev = E.Evolution(**kw)
    assert not getattr(ev, '_output_exists', False)
    _write_dat(ev, monkeypatch)
    assert json.load(open(ev.output_path))['parameters']['inputs']['add_cover_frac'] is True
    again = E.Evolution(**kw)
    assert getattr(again, '_output_exists', False)


def test_F1_numpy_twin_of_a_plain_cached_dat(tmp_path, monkeypatch):
    plain = E.Evolution(**_plain_kwargs(), skip_logger_init=True, output_root=str(tmp_path))
    _write_dat(plain, monkeypatch)
    twin = E.Evolution(**_np_kwargs(), skip_logger_init=True, output_root=str(tmp_path))
    assert twin.output_path == plain.output_path
    assert getattr(twin, '_output_exists', False)


def _gen_record(**over):
    """A generation's results as run_simulation hands them to write_output_file."""
    arr = np.linspace(1.0, 2.0, 3)
    g = {'status': 'complete', 'phase_transitions': [('phase2', np.float64(1e13), 'swept')],
         'cluster_formation_mode': 'burst', 'formation_timescale': None, 'star type': 'sin',
         'generation': np.int64(0), 'cloud_mass': np.float64(1e38), 'cloud_radius': np.float64(1e19),
         'post_sweep_covering_fraction': np.float32(0.5), 'time': arr, 'radius': arr, 'velocity': arr,
         'mass': arr, 'n_cloud_avg': arr, 'shell_properties': {'n_shell_in': arr},
         'cloud_properties': {'M': arr}, 'stellar_feedback': {'L': arr}, 'energetics': {'E': arr},
         'forces': {'F': arr}}
    g.update(over)
    return g


def test_F1_numpy_values_in_generation_records_written(tmp_path, monkeypatch):
    """Multi-generation: numpy-typed arguments also reach the generation records
    (post_sweep_covering_fraction np.float32): the write must take them, for every generation."""
    monkeypatch.setattr(TrackSimulation, 'write_summary', lambda *a, **k: None)
    out = tmp_path / 'sim.dat'
    gens = [_gen_record(), _gen_record(generation=np.int64(1)), _gen_record(generation=np.int64(2))]
    TrackSimulation().write_output_file(str(out), {'inputs': {'Z': np.float32(0.5)}}, gens,
                                        NullEvolutionLogger())
    d = json.load(open(out))
    assert [g['generation'] for g in d['generations']] == [0, 1, 2]
    assert d['generations'][2]['post_sweep_covering_fraction'] == 0.5


# ---------------------------------------------------------------- F2
# Pass rule: with $TODDLERS_OUTPUT_ROOT set and no --output-root, every campaign stage uses
# that root: the from-scratch evolution leaf, the evolution and stage-2 job exports, and in
# from-existing mode the Cloudy job export, the build job export and the archiver path.
def _campaign_argv(tmp_path, mode):
    common = ["--work-dir", str(tmp_path / "work"), "--stab-dir", str(tmp_path / "stab"),
              "--account", "acc", "--partition", "doduo", "--toddlers-src", "/src",
              "--cloudy-exe", "/cl.exe", "--cloudy-data", "/d", "--dry-run"]
    if mode == "grid":
        g = tmp_path / "grid.json"
        g.write_text(json.dumps({'template': 'SB99', 'imf': 'kroupa100', 'star_type': 'sin'}))
        return ["--grid", str(g)] + common
    evo = tmp_path / "evo"
    evo.mkdir()
    (evo / "sim_Z0.020_eta0.050_n160.0_logM6.00.dat").write_text("{}")
    return ["--evolution-dir", str(evo), "--f-dust", "1.0", "--stab", "none"] + common


def test_F2_campaign_from_scratch_follows_the_variable(tmp_path, monkeypatch, capsys):
    from toddlers.hpc import campaign
    root = str(tmp_path / "scratch_root")
    monkeypatch.setenv("TODDLERS_OUTPUT_ROOT", root)
    campaign.main(_campaign_argv(tmp_path, "grid"))
    leaf = os.path.join(root, 'evolution_output', 'template_SB99', 'imf_kroupa100', 'star_type_sin',
                        'cluster_mode_burst', 'profile_type_uniform')
    assert f"Cloudy chain + STAB on {leaf}." in capsys.readouterr().out
    work = tmp_path / "work"
    assert f"export TODDLERS_OUTPUT_ROOT={root}\n" in (work / "campaign_evolution.sh").read_text()
    assert f"export TODDLERS_OUTPUT_ROOT={root}\n" in (work / "campaign_stage2.sh").read_text()


def test_F2_campaign_from_existing_follows_the_variable(tmp_path, monkeypatch):
    from toddlers.hpc import campaign
    root = str(tmp_path / "scratch_root")
    monkeypatch.setenv("TODDLERS_OUTPUT_ROOT", root)
    campaign.main(_campaign_argv(tmp_path, "existing"))
    work = tmp_path / "work"
    stab = (work / "campaign_stab.sh").read_text()
    assert f"export TODDLERS_OUTPUT_ROOT={root}\n" in (work / "campaign_cloudy.sh").read_text()
    assert f"export TODDLERS_OUTPUT_ROOT={root}\n" in stab
    assert f"archive_cloudy_output {root}/cloudy_output" in stab


# ---------------------------------------------------------------- F3
# Pass rule: the .dat parameters record the
# resolved output root as `output_root` (absolute; not in `inputs`); CloudySimulationManager
# puts cloudy_output/ under: its own output_root argument, else the root the .dat lives under
# (parent of its evolution_output/; a moved run follows its new place), else the recorded root,
# else $TODDLERS_OUTPUT_ROOT, else the default.
def _dat_from_evolution(ev, monkeypatch):
    monkeypatch.setattr(TrackSimulation, 'write_summary', lambda *a, **k: None)
    TrackSimulation().write_output_file(ev.output_path, ev._dat_parameters(), [], NullEvolutionLogger())
    return ev.output_path


def _results_dir(dat, **kw):
    from toddlers.cloudy_simulation_manager import CloudySimulationManager
    return CloudySimulationManager(dat, complete_init=False, add_logger=False, **kw).results_dir


def test_F3_root_recorded_and_followed_by_cloudy_stage(tmp_path, monkeypatch):
    x = str(tmp_path / 'X')
    monkeypatch.setenv('TODDLERS_OUTPUT_ROOT', str(tmp_path / 'Y'))
    ev = E.Evolution(**BASE, skip_logger_init=True, output_root=x)
    dat = _dat_from_evolution(ev, monkeypatch)
    assert _results_dir(dat) == os.path.join(x, 'cloudy_output')                 # recorded > variable
    params = json.load(open(dat))['parameters']
    assert params['output_root'] == x and 'output_root' not in params['inputs']
    z = str(tmp_path / 'Z')
    assert _results_dir(dat, output_root=z) == os.path.join(z, 'cloudy_output')  # argument first


def test_F3_variable_root_is_recorded(tmp_path, monkeypatch):
    y = str(tmp_path / 'Y')
    monkeypatch.setenv('TODDLERS_OUTPUT_ROOT', y)
    ev = E.Evolution(**BASE, skip_logger_init=True)
    dat = _dat_from_evolution(ev, monkeypatch)
    assert json.load(open(dat))['parameters']['output_root'] == y
    monkeypatch.delenv('TODDLERS_OUTPUT_ROOT')
    assert _results_dir(dat) == os.path.join(y, 'cloudy_output')


@pytest.mark.parametrize('via', ['argument', 'variable'])
def test_F3_root_recorded_with_custom_output_paths(tmp_path, monkeypatch, via):
    """With get_output_paths replaced (as the run-log tests do), the .dat still records the root."""
    root = str(tmp_path / 'R')
    if via == 'variable':
        monkeypatch.setenv('TODDLERS_OUTPUT_ROOT', root)
    monkeypatch.setattr(E.Evolution, 'get_output_paths',
                        lambda self: (str(tmp_path / 'sim.dat'), str(tmp_path / 'sim.log')))
    ev = E.Evolution(**BASE, skip_logger_init=True, output_root=root if via == 'argument' else None)
    assert ev._dat_parameters()['output_root'] == root


def _flat_copy(dat, d, drop_field=False):
    """A copy of the .dat outside the evolution_output/ layout."""
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, 'sim.dat')
    data = json.load(open(dat))
    if drop_field:
        data['parameters'].pop('output_root', None)
    json.dump(data, open(out, 'w'))
    return out


def test_F3_moved_run_follows_its_new_place(tmp_path, monkeypatch):
    x = str(tmp_path / 'X')
    ev = E.Evolution(**BASE, skip_logger_init=True, output_root=x)
    dat = _dat_from_evolution(ev, monkeypatch)
    m = tmp_path / 'M'
    moved = m / os.path.relpath(dat, x)
    os.makedirs(moved.parent)
    os.replace(dat, moved)
    monkeypatch.setenv('TODDLERS_OUTPUT_ROOT', str(tmp_path / 'Y'))
    assert _results_dir(str(moved)) == os.path.join(str(m), 'cloudy_output')      # location > recorded X


def test_F3_dat_outside_the_layout_falls_back(tmp_path, monkeypatch):
    import toddlers.cloudy_simulation_manager as csm
    x = str(tmp_path / 'X')
    ev = E.Evolution(**BASE, skip_logger_init=True, output_root=x)
    dat = _dat_from_evolution(ev, monkeypatch)
    flat = _flat_copy(dat, str(tmp_path / 'flat'))
    y = str(tmp_path / 'Y')
    monkeypatch.setenv('TODDLERS_OUTPUT_ROOT', y)
    assert _results_dir(flat) == os.path.join(x, 'cloudy_output')                # recorded > variable
    bare = _flat_copy(dat, str(tmp_path / 'bare'), drop_field=True)
    assert _results_dir(bare) == os.path.join(y, 'cloudy_output')                # then the variable
    monkeypatch.delenv('TODDLERS_OUTPUT_ROOT')
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(csm.__file__)))
    monkeypatch.setattr(csm.os, 'makedirs', lambda *a, **k: None)   # no cloudy_output/ in the package tree
    assert _results_dir(bare) == os.path.join(pkg_parent, 'cloudy_output')       # then the default


# ---------------------------------------------------------------- F8
# Pass rule: a relative output_root is made absolute at construction (against the working
# directory at that moment); the stored root and both paths are absolute.
def test_F8_relative_output_root_made_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ev = E.Evolution(**BASE, skip_logger_init=True, output_root='rel_root')
    want = os.path.join(str(tmp_path.resolve()), 'rel_root')
    assert os.path.realpath(ev.output_root) == want and os.path.isabs(ev.output_root)
    assert os.path.isabs(ev.output_path) and os.path.isabs(ev.log_path)
    assert os.path.realpath(ev.output_path).startswith(want + os.sep)


# ---------------------------------------------------------------- F4
# Pass rule: a legacy pre-drawn stochastic name (stochastic_seed<N>_M<mass>) is refused with a ConfigurationError naming StochasticSpec, by Evolution before
# any file access (no directory, no log file, a cached .dat at its path not reused) and by
# StellarFeedback before reading a table.
PRE_DRAWN = dict(Z=0.014, eta_sf=0.05, n_cl=160.0, M_cl_init=1e5 * M_SUN, template='pySB99')


@pytest.mark.parametrize('name', ['stochastic_seed17_M1000', 'stochastic_seed31_M1000', 'stochastic_seed5_M100000'])
def test_F4_evolution_refuses_pre_drawn_name_up_front(tmp_path, name):
    from toddlers.exceptions import ConfigurationError
    dat_dir = (tmp_path / 'evolution_output' / 'template_pySB99' / f'imf_{name}' / 'star_type_sin'
               / 'cluster_mode_burst' / 'profile_type_uniform')
    dat_dir.mkdir(parents=True)
    dat = dat_dir / 'sim_Z0.014_eta0.05_n160.0_logM5.00.dat'
    dat.write_text(json.dumps({'parameters': {'template': 'pySB99', 'imf': name}, 'generations': []}))
    before = sorted(str(p) for p in tmp_path.rglob('*'))
    with pytest.raises(ConfigurationError, match='StochasticSpec'):
        E.Evolution(**PRE_DRAWN, imf=name, output_root=str(tmp_path))
    assert sorted(str(p) for p in tmp_path.rglob('*')) == before      # no log, no new dir


def test_F4_stellar_feedback_refuses_pre_drawn_name():
    from toddlers.exceptions import ConfigurationError
    from toddlers.stellar_feedback import StellarFeedback
    with pytest.raises(ConfigurationError, match='StochasticSpec'):
        StellarFeedback('pySB99', 0.014, 1e5 * M_SUN, 0.05, [], imf='stochastic_seed17_M1000')


# ---------------------------------------------------------------- F9
# Pass rule: the platform line of the RUN RECORD names the solvers of attempt 1 and of
# attempt 2 (the alternative set) as the constants define them; on macOS attempt 2 uses Radau
# in phase 2.
def test_F9_solver_note_matches_the_attempt_plan():
    import platform
    from toddlers import constants as C
    from toddlers.evolution_logging import EvolutionLogger
    line = " ".join(EvolutionLogger.solver_lines())
    assert f"phase 2 {C.SOLVER_PH2}" in line and f"attempt 2: phase 2 {C.ALT_SOLVER_PH2}" in line
    if platform.system().lower() == 'darwin':
        assert "Radau" in line


# ---------------------------------------------------------------- F6
# Pass rule: a Cloudy .in is written atomically by all four model generators: an error while
# the options are assembled (here in get_cr_options, after the first lines are written)
# leaves the previous .in unchanged (or no .in) and no temporary file; a complete write
# gives the full file.
class _StubBase:
    dissolution_time = 0.0
    boom = False

    def get_effective_covering_fraction(self, t):
        return 1.0

    def get_cr_options(self, *a, **k):
        if self.boom:
            raise RuntimeError('failed while assembling the options')
        return ['cosmic rays background']

    def __getattr__(self, name):
        return lambda *a, **k: [f'{name} line']


def _generator(kind):
    import toddlers.cloudy_dig_model_generator as dig
    import toddlers.cloudy_dissolved_model_generator as dis
    import toddlers.cloudy_shell_model_generator as sh
    import toddlers.cloudy_unified_model_generator as un
    cls = {'shell': sh.ShellModelGenerator, 'unified': un.UnifiedModelGenerator,
           'dissolved': dis.DissolvedModelGenerator, 'dig': dig.DIGModelGenerator}[kind]
    g = object.__new__(cls)
    g.base, g.model_prefix, g.logger = _StubBase(), kind, NullEvolutionLogger()
    g.add_old_stellar_radiation, g.include_grains, g.logU_background = False, True, -4.0
    for m in ('get_density_options', 'get_geometry_options', 'get_stopping_criteria', 'get_source_options',
              'get_output_options', 'get_density_and_geometry_options', 'get_turbulence_options',
              'get_stopping_criterion'):
        setattr(g, m, lambda *a, _m=m, **k: [f'{_m} line'])
    return g


def _write(g, kind, t):
    return g.write_input_file(t, 'shell') if kind == 'dig' else g.write_input_file(t)


@pytest.mark.parametrize('kind', ['shell', 'unified', 'dissolved', 'dig'])
def test_F6_in_file_written_atomically(tmp_path, monkeypatch, kind):
    from toddlers.constants import MYR_TO_SEC
    monkeypatch.chdir(tmp_path)
    g, t = _generator(kind), 1.0 * MYR_TO_SEC
    name = f'{kind}_1.00.in'
    g.base.boom = True
    with pytest.raises(RuntimeError):
        _write(g, kind, t)
    assert sorted(os.listdir(tmp_path)) == []                    # no partial .in, no temp file
    g.base.boom = False
    _write(g, kind, t)
    full = (tmp_path / name).read_text()
    assert 'cosmic rays background' in full and full.rstrip().endswith('line')
    g.base.boom = True
    with pytest.raises(RuntimeError):
        _write(g, kind, t)
    assert sorted(os.listdir(tmp_path)) == [name] and (tmp_path / name).read_text() == full


# ---------------------------------------------------------------- L1
# Pass rule: the atomic .dat gets the mode a plain open() would give (0o666 & ~umask), for
# two umasks.
@pytest.mark.parametrize('umask', [0o022, 0o002])
def test_L1_dat_mode_follows_umask(tmp_path, monkeypatch, umask):
    import stat
    monkeypatch.setattr(TrackSimulation, 'write_summary', lambda *a, **k: None)
    old = os.umask(umask)
    try:
        out = tmp_path / 'sim.dat'
        TrackSimulation().write_output_file(str(out), {}, [], NullEvolutionLogger())
    finally:
        os.umask(old)
    assert stat.S_IMODE(os.stat(out).st_mode) == 0o666 & ~umask


# ---------------------------------------------------------------- L3
# Pass rules: (a) DirLock.__exit__ removes the lock directory only if its owner.json names
# this holder (host, pid and this lock's own start time): a lock that was broken and taken
# by another holder (here another holder of the same process, which only the start time
# tells apart) survives the first holder's exit. (b) A waiter that judged a lock abandoned
# and then renames it away checks that the moved directory is the one it judged (owner and
# directory identity); if a live lock took its place in between, that lock is put back and
# the waiter keeps waiting (times out), instead of removing it and taking a second hold.
def _dead_pid():
    import subprocess
    import sys
    proc = subprocess.Popen([sys.executable, '-c', 'pass'])
    proc.wait()
    return proc.pid


def _owner_file(lock, **owner):
    (lock / 'owner.json').write_text(json.dumps(owner))


def test_L3_exit_keeps_a_lock_it_no_longer_owns(tmp_path):
    import shutil
    import socket
    from toddlers._fileutils import DirLock
    lock = tmp_path / 'x.lock'
    a = DirLock(str(lock), timeout=5)
    a.__enter__()
    shutil.rmtree(lock)                                   # broken by a waiter ...
    lock.mkdir()                                          # ... and taken by holder C
    c_owner = {'host': socket.gethostname(), 'pid': os.getpid(), 'time': time.time() + 1.0}
    _owner_file(lock, **c_owner)
    a.__exit__(None, None, None)
    assert lock.exists() and json.loads((lock / 'owner.json').read_text()) == c_owner


def test_L3_exit_removes_its_own_lock(tmp_path):
    from toddlers._fileutils import DirLock
    lock = tmp_path / 'x.lock'
    with DirLock(str(lock), timeout=5):
        assert lock.exists()
    assert not lock.exists()


def test_L3_break_does_not_remove_a_live_lock_that_replaced_the_dead_one(tmp_path, monkeypatch):
    import shutil
    import socket
    from toddlers import _fileutils
    from toddlers._fileutils import DirLock
    lock = tmp_path / 'x.lock'
    lock.mkdir()
    _owner_file(lock, host=socket.gethostname(), pid=_dead_pid(), time=0)
    live = {'host': socket.gethostname(), 'pid': os.getpid(), 'time': time.time()}
    original, swapped = DirLock._abandoned, []

    def judged_then_replaced(self):
        verdict = original(self)
        if verdict and not swapped:                       # another waiter breaks it, C takes it
            shutil.rmtree(self.path)
            os.mkdir(self.path)
            _owner_file(lock, **live)
            swapped.append(1)
        return verdict
    monkeypatch.setattr(DirLock, '_abandoned', judged_then_replaced)
    with pytest.raises(RuntimeError, match='Timed out'):
        with DirLock(str(lock), timeout=0.6, poll=0.05):
            pass
    assert swapped and lock.exists() and json.loads((lock / 'owner.json').read_text()) == live
    assert [p.name for p in tmp_path.iterdir()] == ['x.lock']


def test_L3_dead_lock_still_broken(tmp_path):
    import socket
    from toddlers._fileutils import DirLock
    lock = tmp_path / 'x.lock'
    lock.mkdir()
    _owner_file(lock, host=socket.gethostname(), pid=_dead_pid(), time=0)
    with DirLock(str(lock), timeout=5):
        assert json.loads((lock / 'owner.json').read_text())['pid'] == os.getpid()
    assert [p.name for p in tmp_path.iterdir()] == []


# ---------------------------------------------------------------- I3
# Pass rule: with include_supernovae True and no run carrying supernova_events, the error
# says that the supernova event record is missing and for which metallicities (not "set()").
def test_I3_missing_sn_metadata_message():
    import types
    from toddlers.pysb99.generate_pysb99_interpolants import PySB99InterpolantGenerator
    g = object.__new__(PySB99InterpolantGenerator)
    g.include_supernovae = True
    g.results_dict = {0.014: types.SimpleNamespace(), 0.02: types.SimpleNamespace(supernova_events=None)}
    with pytest.raises(ValueError) as e:
        g.sn_time_deposition_mode()
    msg = str(e.value)
    assert 'set()' not in msg and 'supernova_events' in msg and '0.014' in msg and '0.02' in msg
