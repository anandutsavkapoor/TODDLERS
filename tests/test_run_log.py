"""Run log and per-run record (2.1.0).

Pass rules (a GENERATION line is also written for generation 0 at the start, so L2
counts len(t_list_collapse) + 1 lines; the .dat record is one key, parameters['run_record']):

L3  Watchdog: (a) with a progress counter that stays put while other lines keep the log
    growing, LogStallWatchdog raises TimeoutError once stall_timeout passes; (b) with the
    counter advancing it does not; (c) without a
    counter (file-size mode, kept for TimeoutManager) a growing file is progress.
T   Tables: every state-table row is as wide as its border; integers print as integers,
    booleans as True/False.
L2  Multi-generation run (Z 0.02, eta_sf 0.01, n_cl 2560, 1e6 Msun, SB99; 9 generations):
    one `GENERATION index i starts at t = X Myr` line per generation (indices 0..n-1, n =
    the .dat generation count = len(t_list_collapse) + 1); for i >= 1, X equals generation
    i-1's recollapse time to 0.01 Myr; the stellar mass formed equals
    eta_sf (1 - eta_sf)^i M_cl_init and the cloud gas left (1 - eta_sf)^(i+1) M_cl_init, rel 1e-3.
L5  The run record block names every constructor argument (`arg <name> =`), the feedback md5
    equal to the .dat feedback_md5, and the running Python version; the heartbeat (interval
    set to 5 s here) writes `heartbeat:` lines.
S   The run summary names the attempt that succeeded and its solvers, the generation count
    and every recollapse time; parameters['run_record'] holds attempt_used, attempts,
    solvers_used, rtol_used, rhs_calls (> 0), wall_time_s, versions.
L4  With a RuntimeError injected at the first recollapse of attempt 1 only, attempt 2
    succeeds and the summary and run_record say so, with attempt 1's error class.
F   With a ConfigurationError at the first recollapse, the summary is still written and
    says the run failed, not retried.
"""
import glob
import json
import os
import re
import sys
import threading
import time

import pytest

from toddlers.constants import M_SUN, MYR_TO_SEC


# ---------------------------------------------------------------- L3
def _writer(path, stop, progress=None):
    while not stop.is_set():
        with open(path, 'a') as f:
            f.write("heartbeat: still here\n")
        if progress is not None:
            progress[0] += 1
        time.sleep(0.2)


def _watch(tmp_path, progress_fn, advance):
    from toddlers.timeout_manager import LogStallWatchdog
    path = str(tmp_path / 'run.log')
    open(path, 'w').close()
    counter = [0]
    stop = threading.Event()
    th = threading.Thread(target=_writer, args=(path, stop, counter if advance else None))
    th.start()
    try:
        fn = None if progress_fn is None else (lambda: counter[0])
        with LogStallWatchdog(path, stall_timeout=2, poll=1, progress=fn):
            t0 = time.time()
            while time.time() - t0 < 5:
                time.sleep(0.05)
        return False
    except TimeoutError:
        return True
    finally:
        stop.set()
        th.join()


def test_L3a_frozen_progress_escalates_while_log_grows(tmp_path):
    assert _watch(tmp_path, progress_fn=True, advance=False)


def test_L3b_advancing_progress_does_not_escalate(tmp_path):
    assert not _watch(tmp_path, progress_fn=True, advance=True)


def test_L3c_file_size_mode_counts_growth(tmp_path):
    assert not _watch(tmp_path, progress_fn=None, advance=False)


# ---------------------------------------------------------------- T
def test_T_rows_as_wide_as_border():
    from toddlers.utils import format_row, TABLE_BORDER
    # scalars and short strings (state tables pass no lists; a list value cannot fit a
    # fixed column, so it is not part of this width check)
    for v in (1.0, 12345.0, 3, True, "x", "1.458726e-08"):
        assert len(format_row("Direct rad. force (UV+IR)", v, "dyne/cm^2")) == len(TABLE_BORDER)
    assert format_row("No. of generations", 9, "").split("|")[2].strip() == "9"
    assert format_row("Shell accelerating", True, "").split("|")[2].strip() == "True"


# ---------------------------------------------------------------- runs
MULTIGEN = dict(Z=0.02, eta_sf=0.01, n_cl=2560.0, M_cl_init=1e6 * M_SUN, template='SB99', imf='kroupa100')
GEN_RE = re.compile(r"GENERATION index (\d+) starts at t = ([\d.]+) Myr.*stellar mass formed ([\d.e+]+) M_sun.*"
                    r"cloud gas left ([\d.e+]+) M_sun")


def _run(tmp_path, monkeypatch, inject=None, heartbeat_s=None):
    """Run the multi-generation cloud into tmp_path; return (log text, .dat dict or None, error)."""
    import toddlers.evolution as evmod
    from toddlers.evolution import Evolution
    monkeypatch.setattr(Evolution, 'get_output_paths',
                        lambda self: (str(tmp_path / 'sim.dat'), str(tmp_path / 'sim.log')))
    if heartbeat_s is not None:
        import toddlers.evolution_logging as lg
        monkeypatch.setattr(lg, 'HEARTBEAT_S', heartbeat_s)
    if inject is not None:
        exc, attempts = inject
        orig = Evolution.add_new_generation
        seen = {'attempt': 0}
        orig_attempt = Evolution.simulation_attempt

        def counted(self, solvers, rtol):
            seen['attempt'] += 1
            seen['first'] = True
            return orig_attempt(self, solvers, rtol)

        def boom(self):
            if seen['attempt'] in attempts and seen.pop('first', False):
                raise exc("probe: injected at the first recollapse")
            return orig(self)
        monkeypatch.setattr(Evolution, 'simulation_attempt', counted)
        monkeypatch.setattr(Evolution, 'add_new_generation', boom)
    err = None
    ev = Evolution(**MULTIGEN)
    try:
        ev.run_simulation()
    except Exception as e:
        err = e
    logs = sorted(glob.glob(str(tmp_path / 'sim_uid*.log')))
    assert len(logs) == 1, logs
    text = open(logs[0]).read()
    dat = json.load(open(tmp_path / 'sim.dat')) if os.path.exists(tmp_path / 'sim.dat') else None
    return text, dat, err


@pytest.fixture(scope='module')
def clean_run(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    try:
        yield _run(tmp_path_factory.mktemp('clean'), mp, heartbeat_s=5)
    finally:
        mp.undo()


@pytest.mark.data
@pytest.mark.slow
def test_L2_one_generation_line_per_generation(clean_run):
    text, dat, err = clean_run
    assert err is None
    lines = GEN_RE.findall(text)
    n = len(dat['generations'])
    assert n >= 2
    assert [int(l[0]) for l in lines] == list(range(n))
    eta, M = MULTIGEN['eta_sf'], MULTIGEN['M_cl_init'] / M_SUN
    for i, t, m_star, m_gas in lines:
        i = int(i)
        if i >= 1:
            rec = dict((k, v) for k, v, *_ in dat['generations'][i - 1]['phase_transitions'])
            assert abs(float(t) - rec['phase2_to_recollapse'] / MYR_TO_SEC) < 0.01, i
        assert float(m_star) == pytest.approx(eta * (1 - eta) ** i * M, rel=1e-3)
        assert float(m_gas) == pytest.approx((1 - eta) ** (i + 1) * M, rel=1e-3)


@pytest.mark.data
@pytest.mark.slow
def test_L5_run_record_block(clean_run):
    from toddlers.evolution import Evolution
    import inspect
    text, dat, err = clean_run
    block = text.split("==== RUN RECORD ====")[1].split("==== END RUN RECORD ====")[0]
    for name in inspect.signature(Evolution.__init__).parameters:
        if name != 'self':
            assert f"arg {name} =" in block, name
    assert dat['parameters']['feedback_md5'] in block
    assert sys.version.split()[0] in block
    assert text.count("heartbeat: t =") >= 5


@pytest.mark.data
@pytest.mark.slow
def test_S_summary_and_run_record(clean_run):
    text, dat, err = clean_run
    summary = text.split("==== RUN SUMMARY ====")[1]
    assert "result: attempt 1 of 3 succeeded" in summary
    n = len(dat['generations'])
    assert f"generations: {n}" in summary
    for g in dat['generations'][:-1]:
        t = dict((k, v) for k, v, *_ in g['phase_transitions'])['phase2_to_recollapse'] / MYR_TO_SEC
        assert f"{t:.2f}" in summary
    rr = dat['parameters']['run_record']
    assert rr['attempt_used'] == 1 and len(rr['attempts']) == 1
    assert set(rr['solvers_used']) == {'phase1', 'fragmentation', 'phase2'}
    assert sum(rr['rhs_calls'].values()) > 0 and rr['wall_time_s'] > 0
    assert rr['versions']['python'] == sys.version.split()[0]


@pytest.mark.data
@pytest.mark.slow
def test_L4_retry_recorded(tmp_path, monkeypatch):
    text, dat, err = _run(tmp_path, monkeypatch, inject=(RuntimeError, {1}))
    assert err is None
    summary = text.split("==== RUN SUMMARY ====")[1]
    assert "result: attempt 2 of 3 succeeded" in summary
    assert "attempt 1: RuntimeError: probe" in summary
    rr = dat['parameters']['run_record']
    assert rr['attempt_used'] == 2 and rr['attempts'][0]['error_class'] == 'RuntimeError'


@pytest.mark.data
@pytest.mark.slow
def test_F_summary_on_non_retryable_failure(tmp_path, monkeypatch):
    from toddlers.exceptions import ConfigurationError
    text, dat, err = _run(tmp_path, monkeypatch, inject=(ConfigurationError, {1, 2, 3}))
    assert isinstance(err, ConfigurationError)
    summary = text.split("==== RUN SUMMARY ====")[1]
    assert "result: failed in attempt 1 (ConfigurationError, not retried)" in summary
