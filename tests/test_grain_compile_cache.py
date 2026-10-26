"""Compiled Cloudy grain files (.opc) and size distributions (.szd): content-keyed reuse.

- Reuse key: each .opc gets a sidecar <opc>.meta.json holding its key (szd md5, rfi md5,
  n_bins, ratio at full precision, material, md5 of the Cloudy executable) and the Cloudy
  version line read from the .opc header. Reuse needs a readable sidecar whose md5 matches
  the file and whose key equals the request's key. A file without a sidecar is compiled
  once, then reused. A change of the rfi file or of the Cloudy executable triggers one
  recompile.
- .szd writes: unchanged content leaves the file untouched (same inode, same mtime);
  changed content goes through a temporary file renamed into place.
- Compile checks: a non-zero Cloudy exit or a missing .opc raises RuntimeError; the working
  directory is restored when the compile raises; a ratio off the 1e-3 name grid raises
  ValueError, so two ratios never share a file name (names of grid ratios:
  0.05/0.10/0.20/0.40 -> ratio050/100/200/400); a ratio below 0.05 uses Cloudy's Orion
  .opc, warns, and is recorded on the generator and in a comment line of the commands.
- Released bytes: the four .szd files regenerate with the md5s of the released files
  (Cloudy data directory).

Cloudy itself is replaced by a stub script.
"""
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import warnings

import pytest

from toddlers.cloudy_grains_generator import CloudyGrainsGenerator

RELEASED_SZD_MD5 = {
    0.05: ('ratio050_exp3.szd', 'ee3476c78d13efb7249400ee7d507b40'),
    0.10: ('ratio100_exp3.szd', '120d5ff660a26bdab1a4c0e885ffeaf5'),
    0.20: ('ratio200_exp3.szd', '0835dfca3d484e0d97b9fba629ca6f10'),
    0.40: ('ratio400_exp3.szd', 'e2a1517ed381acf598b4112215106e33'),
}

STUB = '''#!{python}
# stub Cloudy for the grain-compile tests ({tag})
import hashlib, os, re, sys, time
inp = sys.argv[1]
with open(os.environ['STUB_LOG'], 'a') as log:
    log.write(inp + '\\n')
mode = os.environ.get('STUB_MODE', 'ok')
time.sleep(float(os.environ.get('STUB_SLEEP', '0')))
if mode == 'fail':
    sys.exit(1)
if mode == 'noout':
    sys.exit(0)
line = open(inp).read()
rfi, szd, nb = re.match(r'compile grain "(.+?)" "(.+?)" (\\d+)', line).groups()
mat = rfi[:-4]
h = hashlib.md5(open(rfi, 'rb').read() + open(szd, 'rb').read()).hexdigest()
with open(f"{{mat}}_{{szd[:-4]}}_{{nb}}.opc", 'w') as f:
    f.write("# this file was created by Cloudy stub ({tag}) (01Jan26) on Sun Oct 25 00:00:00 2026\\n")
    f.write("# ===========================================\\n#\\n")
    f.write(h + "\\n")
'''


def _write_stub(path, tag='v1'):
    path.write_text(STUB.format(python=sys.executable, tag=tag))
    path.chmod(0o755)


@pytest.fixture
def env(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    data.mkdir()
    for mat in ('silicate', 'graphite'):
        (data / f'{mat}.rfi').write_text(f'{mat} refractive index v1\n')
    stub = tmp_path / 'cloudy_stub.py'
    _write_stub(stub)
    log = tmp_path / 'calls.log'
    log.write_text('')
    monkeypatch.setenv('STUB_LOG', str(log))
    monkeypatch.delenv('STUB_MODE', raising=False)
    gen = CloudyGrainsGenerator(str(stub), str(data))
    return gen, data, stub, log


def _calls(log):
    return [l for l in log.read_text().splitlines() if l]


def _compile(gen, ratio=0.10, material='silicate'):
    szd = gen.create_size_distribution_file(ratio, 3)
    try:
        return gen.compile_grains(szd, material, 10, small_to_large_ratio=ratio)
    except TypeError:  # signature without small_to_large_ratio
        return gen.compile_grains(szd, material, 10)


# ---------------------------------------------------------------- released bytes, names
@pytest.mark.parametrize('ratio', sorted(RELEASED_SZD_MD5))
def test_szd_bytes_match_released(tmp_path, ratio):
    gen = CloudyGrainsGenerator('none', str(tmp_path))
    name = gen.create_size_distribution_file(ratio, 3)
    want_name, want_md5 = RELEASED_SZD_MD5[ratio]
    assert name == want_name
    assert hashlib.md5((tmp_path / name).read_bytes()).hexdigest() == want_md5


# ---------------------------------------------------------------- reuse key
def test_opc_without_sidecar_is_compiled_once_then_reused(env):
    gen, data, stub, log = env
    (data / 'silicate_ratio100_exp3_10.opc').write_text('# made by another build\n')
    assert _compile(gen) == 'silicate_ratio100_exp3_10.opc'
    assert len(_calls(log)) == 1, 'an .opc without a sidecar must be compiled once'
    meta = json.loads((data / 'silicate_ratio100_exp3_10.opc.meta.json').read_text())
    assert meta['md5'] == hashlib.md5((data / 'silicate_ratio100_exp3_10.opc').read_bytes()).hexdigest()
    assert meta['cloudy_version_line'].startswith('# this file was created by Cloudy stub (v1)')
    assert meta['key']['n_bins'] == 10 and meta['key']['small_to_large_ratio'] == 0.10
    _compile(gen)
    assert len(_calls(log)) == 1, 'a matching sidecar must be reused'


def test_rfi_change_triggers_recompile(env):
    gen, data, stub, log = env
    _compile(gen)
    (data / 'silicate.rfi').write_text('silicate refractive index v2\n')
    _compile(gen)
    assert len(_calls(log)) == 2


def test_cloudy_build_change_triggers_recompile(env):
    gen, data, stub, log = env
    _compile(gen)
    _write_stub(stub, tag='v2')
    _compile(gen)
    assert len(_calls(log)) == 2
    meta = json.loads((data / 'silicate_ratio100_exp3_10.opc.meta.json').read_text())
    assert '(v2)' in meta['cloudy_version_line']


def test_edited_opc_triggers_recompile(env):
    gen, data, stub, log = env
    _compile(gen)
    with open(data / 'silicate_ratio100_exp3_10.opc', 'a') as f:
        f.write('tampered\n')
    _compile(gen)
    assert len(_calls(log)) == 2


def test_concurrent_processes_compile_once(env, tmp_path):
    gen, data, stub, log = env
    (data / 'silicate_ratio100_exp3_10.opc').write_text('# made by another build\n')
    gen.create_size_distribution_file(0.10, 3)
    code = textwrap.dedent(f'''
        from toddlers.cloudy_grains_generator import CloudyGrainsGenerator
        g = CloudyGrainsGenerator({str(stub)!r}, {str(data)!r})
        g.compile_grains('ratio100_exp3.szd', 'silicate', 10, small_to_large_ratio=0.10)
    ''')
    envv = dict(os.environ, STUB_SLEEP='1.0')
    procs = [subprocess.Popen([sys.executable, '-c', code], env=envv) for _ in range(3)]
    assert [p.wait(timeout=120) for p in procs] == [0, 0, 0]
    assert len(_calls(log)) == 1


# ---------------------------------------------------------------- .szd writes
def test_szd_untouched_when_unchanged(tmp_path):
    gen = CloudyGrainsGenerator('none', str(tmp_path))
    name = gen.create_size_distribution_file(0.10, 3)
    p = tmp_path / name
    os.utime(p, ns=(1_000_000_000, 1_000_000_000))
    st = p.stat()
    gen.create_size_distribution_file(0.10, 3)
    st2 = p.stat()
    assert (st2.st_ino, st2.st_mtime_ns) == (st.st_ino, st.st_mtime_ns)


def test_szd_changed_content_written_atomically(tmp_path, monkeypatch):
    gen = CloudyGrainsGenerator('none', str(tmp_path))
    p = tmp_path / 'ratio100_exp3.szd'
    p.write_text('old content\n')
    replaced = []
    real_replace = os.replace
    monkeypatch.setattr(os, 'replace', lambda a, b: (replaced.append((str(a), str(b))), real_replace(a, b)))
    gen.create_size_distribution_file(0.10, 3)
    assert [b for a, b in replaced] == [str(p)]
    assert hashlib.md5(p.read_bytes()).hexdigest() == RELEASED_SZD_MD5[0.10][1]


# ---------------------------------------------------------------- compile checks
def test_failed_compile_raises(env, monkeypatch):
    gen, data, stub, log = env
    monkeypatch.setenv('STUB_MODE', 'fail')
    with pytest.raises(RuntimeError):
        _compile(gen)
    assert not (data / 'silicate_ratio100_exp3_10.opc.meta.json').exists()


def test_compile_without_output_raises(env, monkeypatch):
    gen, data, stub, log = env
    (data / 'silicate_ratio100_exp3_10.opc').write_text('# made by another build\n')
    monkeypatch.setenv('STUB_MODE', 'noout')
    with pytest.raises(RuntimeError):
        _compile(gen)


def test_cwd_restored_when_compile_raises(env, monkeypatch, tmp_path):
    gen, data, stub, log = env
    monkeypatch.chdir(tmp_path)

    def boom(input_file, **kw):
        raise KeyboardInterrupt('simulated interrupt inside the Cloudy call')
    monkeypatch.setattr(gen, 'run_cloudy', boom)
    with pytest.raises(KeyboardInterrupt):
        _compile(gen)
    assert os.getcwd() == str(tmp_path)


def test_ratio_off_name_grid_refused(tmp_path):
    gen = CloudyGrainsGenerator('none', str(tmp_path))
    gen.create_size_distribution_file(0.10, 3)
    with pytest.raises(ValueError):
        gen.create_size_distribution_file(0.1004, 3)  # would share ratio100 with 0.10


def test_orion_fallback_warns_and_is_recorded(tmp_path):
    gen = CloudyGrainsGenerator('none', str(tmp_path))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        cmds = gen.generate_grain_commands(1.0, 0.03, 0.03, skip_pah=True)
    assert any('orion' in str(x.message).lower() for x in w)
    assert [r['material'] for r in gen.orion_fallbacks] == ['silicate', 'graphite']
    assert all(r['small_to_large_ratio'] == 0.03 for r in gen.orion_fallbacks)
    grain_lines = [c for c in cmds if not c.startswith('#')]
    assert grain_lines == ['grain "silicate_orion_10.opc" 0.0 log function sublimation',
                           'grain "graphite_orion_10.opc" 0.0 log function sublimation']
    assert any(c.startswith('#') and 'orion' in c.lower() for c in cmds)


def test_infer_sigma_docstring_says_small_to_large():
    doc = CloudyGrainsGenerator.infer_sigma.__doc__
    assert 'small grains to large grains' in doc and 'total grain mass' not in doc
    assert 'total grain mass' not in CloudyGrainsGenerator.calculate_mass_ratio.__doc__


# ---------------------------------------------------------------- compile input per bin count
# The lock is per .opc (whose name has the bin count), so the compile input must be per
# .opc too: two concurrent compiles of one material and .szd with different n_bins each
# keep their own input and compile their own bin count. Bar: the input name carries
# n_bins; both concurrent compiles succeed.
def test_compile_input_name_has_bin_count(env):
    gen, data, stub, log = env
    szd = gen.create_size_distribution_file(0.10, 3)
    gen.compile_grains(szd, 'silicate', 10, small_to_large_ratio=0.10)
    gen.compile_grains(szd, 'silicate', 20, small_to_large_ratio=0.10)
    assert _calls(log) == ['compile_silicate_ratio100_exp3_10.in',
                           'compile_silicate_ratio100_exp3_20.in']


def test_concurrent_compiles_with_different_bin_counts(env):
    gen, data, stub, log = env
    gen.create_size_distribution_file(0.10, 3)
    code = textwrap.dedent(f'''
        import sys
        from toddlers.cloudy_grains_generator import CloudyGrainsGenerator
        g = CloudyGrainsGenerator({str(stub)!r}, {str(data)!r})
        g.compile_grains('ratio100_exp3.szd', 'silicate', int(sys.argv[1]), small_to_large_ratio=0.10)
    ''')
    envv = dict(os.environ, STUB_SLEEP='2.0')
    procs = [subprocess.Popen([sys.executable, '-c', code, nb], env=envv) for nb in ('10', '20')]
    assert [p.wait(timeout=120) for p in procs] == [0, 0]
    for nb in (10, 20):
        assert (data / f'silicate_ratio100_exp3_{nb}.opc').exists()


# ---------------------------------------------------------------- process cwd untouched
def test_compile_does_not_change_the_process_cwd(env, monkeypatch, tmp_path):
    """Pass rule: while a stub grain compile runs in another thread, the
    process working directory seen by this thread never changes (Cloudy gets cwd= instead
    of os.chdir), and the .opc still lands in the data directory."""
    import threading
    import time
    gen, data, stub, log = env
    work = tmp_path / 'work'
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv('STUB_SLEEP', '1.0')
    seen, out = set(), {}
    t = threading.Thread(target=lambda: out.setdefault('opc', _compile(gen)))
    t.start()
    while t.is_alive():
        seen.add(os.getcwd())
        time.sleep(0.01)
    t.join()
    assert seen == {str(work)}, seen
    assert out['opc'] == 'silicate_ratio100_exp3_10.opc' and (data / out['opc']).is_file()
