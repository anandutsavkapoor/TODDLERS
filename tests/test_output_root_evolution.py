"""Evolution output under a user-chosen root; exact inputs in the .dat (2.1.0).

Pass rules:

O0  With no root given and $TODDLERS_OUTPUT_ROOT unset, the .dat and log paths are exactly
    <package parent>/evolution_output/<layout>/<name>.dat and
    <package parent>/evolution_logs/<layout>/<name>.log (the default layout).
O1  A run with output_root=<tmp> (the 9-generation SB99 cloud) and a run with only
    $TODDLERS_OUTPUT_ROOT=<tmp> (one generation) write nothing under the package parent
    directory (file listing before/after, __pycache__ ignored), and their .dat and log sit
    under <tmp>/evolution_output and <tmp>/evolution_logs. The stochastic multi-generation case
    (needs the SN single-star database) also puts _stars.npz and every _gen<i>_feedback.h5
    there.
O2  A second Evolution with the same inputs and root finds the cached .dat (_output_exists).
O3  The .dat copied under another name and directory loads (load_output_file) to the same
    Z, eta_sf, n_cl, M_cl_init, template, imf, star_type, mode and profile as at its standard
    path, with M_cl_init exact (rel 1e-12) for an off-grid mass (3e5 Msun); the Cloudy stage
    writes the same shell input file from both.
O5  A .dat without `inputs` (the format of an earlier release) at its standard path loads,
    with its parameters taken from the path.
"""
import glob
import os
import shutil

import pytest

from toddlers.constants import M_SUN, MYR_TO_SEC

PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__import__('toddlers').__file__)))
MULTIGEN = dict(Z=0.02, eta_sf=0.01, n_cl=2560.0, M_cl_init=1e6 * M_SUN, template='SB99', imf='kroupa100')
OFFGRID = dict(Z=0.02, eta_sf=0.05, n_cl=160.0, M_cl_init=3e5 * M_SUN, template='SB99', imf='kroupa100')


def _listing(root):
    out = set()
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x != '__pycache__']
        out.update(os.path.join(d, f) for f in files)
    return out


def _dir_listing(root):
    """Every directory under root/evolution_output and root/evolution_logs."""
    out = set()
    for top in ('evolution_output', 'evolution_logs'):
        for d, dirs, _ in os.walk(os.path.join(root, top)):
            out.add(d)
    return out


# ---------------------------------------------------------------- O0
@pytest.mark.parametrize("kw", [
    dict(MULTIGEN),
    dict(MULTIGEN, cluster_formation_mode='constant_sfr', formation_timescale=2 * MYR_TO_SEC),
    dict(MULTIGEN, add_cover_frac=True, post_sweep_covering_fraction=0.5),
    dict(MULTIGEN, template='BPASS', imf='chab100', star_type='bin', Z=0.014),
])
def test_O0_default_paths_unchanged(monkeypatch, tmp_path, kw):
    import numpy as np
    from toddlers.evolution import Evolution
    from toddlers.utils import f_dust_label
    monkeypatch.delenv('TODDLERS_OUTPUT_ROOT', raising=False)
    dirs_before = _dir_listing(PKG_PARENT)
    # the default paths are checked without creating them in the package tree
    monkeypatch.setattr(Evolution, 'ensure_output_directories', lambda self: None)
    ev = Evolution(**kw, skip_logger_init=True)
    st = ev.star_type
    layout = os.path.join(f"template_{ev.template}", f"imf_{ev.imf}", f"star_type_{st}",
                          f"cluster_mode_{ev.cluster_formation_mode}", "profile_type_uniform")
    name = f"sim_Z{ev.Z:.3g}_eta{ev.eta_sf:.3g}_n{ev.n_cl:.1f}_logM{np.log10(ev.M_cl_init / M_SUN):.2f}"
    if ev.add_cover_frac:
        name += f"_cover{ev.post_sweep_covering_fraction:.2f}"
    if ev.cluster_formation_mode == 'constant_sfr':
        name += f"_tform{ev.formation_timescale / MYR_TO_SEC:.1f}"
    name += f_dust_label(ev.f_dust)
    assert ev.output_path == os.path.join(PKG_PARENT, 'evolution_output', layout, name + '.dat')
    assert ev.log_path == os.path.join(PKG_PARENT, 'evolution_logs', layout, name + '.log')
    assert _dir_listing(PKG_PARENT) == dirs_before          # nothing made in the package tree
    monkeypatch.undo()
    rooted = Evolution(**kw, skip_logger_init=True, output_root=str(tmp_path))
    assert rooted.output_path == os.path.join(str(tmp_path), 'evolution_output', layout, name + '.dat')
    assert os.path.isdir(os.path.dirname(rooted.output_path))
    assert _dir_listing(PKG_PARENT) == dirs_before


def test_O0_campaign_leaf_follows_the_root(tmp_path):
    import json
    from toddlers.hpc.campaign import _evolution_leaf_from_grid
    g = tmp_path / 'grid.json'
    json.dump({'template': 'SB99', 'imf': 'kroupa100', 'star_type': 'sin'}, open(g, 'w'))
    tail = os.path.join('evolution_output', 'template_SB99', 'imf_kroupa100', 'star_type_sin',
                        'cluster_mode_burst', 'profile_type_uniform')
    assert _evolution_leaf_from_grid(str(g), '/pkg/src') == os.path.join('/pkg/src', tail)
    assert _evolution_leaf_from_grid(str(g), '/scratch/run') == os.path.join('/scratch/run', tail)


# ---------------------------------------------------------------- O1, O2, O3, O5
def _fresh(**kw):
    from toddlers.evolution import Evolution
    ev = Evolution(**kw)
    for f in glob.glob(ev.output_path[:-len('.dat')] + '*'):
        os.remove(f)
    return Evolution(**kw)


@pytest.fixture(scope='module')
def rooted_run(tmp_path_factory):
    root = str(tmp_path_factory.mktemp('root'))
    before = _listing(PKG_PARENT)
    ev = _fresh(**MULTIGEN, output_root=root)
    ev.run_simulation()
    return root, ev, before, _listing(PKG_PARENT)


@pytest.mark.data
@pytest.mark.slow
def test_O1_argument_root_multigen(rooted_run):
    root, ev, before, after = rooted_run
    assert after - before == set(), sorted(after - before)[:5]
    assert ev.output_path.startswith(os.path.join(root, 'evolution_output') + os.sep)
    assert os.path.exists(ev.output_path)
    logs = glob.glob(os.path.join(root, 'evolution_logs', '**', '*.log'), recursive=True)
    assert logs
    import json
    assert len(json.load(open(ev.output_path))['generations']) >= 2


@pytest.mark.data
@pytest.mark.slow
def test_O1_variable_root(tmp_path, monkeypatch):
    monkeypatch.setenv('TODDLERS_OUTPUT_ROOT', str(tmp_path))
    before = _listing(PKG_PARENT)
    ev = _fresh(**OFFGRID)
    ev.run_simulation()
    assert _listing(PKG_PARENT) - before == set()
    assert ev.output_path.startswith(str(tmp_path) + os.sep) and os.path.exists(ev.output_path)


@pytest.mark.data
@pytest.mark.slow
def test_O1_stochastic_multigen_side_files(tmp_path):
    from toddlers.pysb99.stochastic.population import IMFSpec, StochasticSpec, file_md5
    from toddlers._paths import get_database_path
    db = get_database_path()
    from toddlers._paths import data_resources
    want = data_resources()['single_star_tracks']['md5']
    if not os.path.exists(db) or file_md5(db) != want:
        pytest.skip(f"needs the single-star database of this version (md5 {want})")
    before = _listing(PKG_PARENT)
    kw = dict(Z=0.014, eta_sf=0.01, n_cl=2560.0, M_cl_init=1e6 * M_SUN, template='pySB99', imf=None,
              stochastic=StochasticSpec(IMFSpec('kroupa', 0.1, 100.0), base_seed=23), output_root=str(tmp_path))
    ev = _fresh(**kw)
    ev.run_simulation()
    assert _listing(PKG_PARENT) - before == set()
    stem = ev.output_path[:-len('.dat')]
    assert stem.startswith(str(tmp_path) + os.sep)
    gens = glob.glob(stem + '_gen*_feedback.h5')
    assert os.path.exists(stem + '_stars.npz') and len(gens) >= 2


@pytest.mark.data
@pytest.mark.slow
def test_O2_cache_found_under_root(rooted_run):
    from toddlers.evolution import Evolution
    root, ev, *_ = rooted_run
    again = Evolution(**MULTIGEN, output_root=root)
    assert getattr(again, '_output_exists', False)


@pytest.fixture(scope='module')
def offgrid_dat(tmp_path_factory):
    root = str(tmp_path_factory.mktemp('offgrid'))
    ev = _fresh(**OFFGRID, output_root=root)
    ev.run_simulation()
    return ev.output_path


_KEYS = ('Z', 'eta_sf', 'n_cl', 'M_cl_init', 'template', 'imf', 'star_type', 'cluster_formation_mode',
         'profile_type')


@pytest.mark.data
@pytest.mark.slow
def test_O3_relocated_dat_same_inputs_exact_mass(offgrid_dat, tmp_path):
    from toddlers.track_simulation import load_output_file
    moved = str(tmp_path / 'elsewhere' / 'my_run.dat')
    os.makedirs(os.path.dirname(moved))
    shutil.copy(offgrid_dat, moved)
    a, _ = load_output_file(offgrid_dat)
    b, _ = load_output_file(moved)
    for k in _KEYS:
        assert a[k] == b[k], k
    assert b['M_cl_init'] == pytest.approx(3e5, rel=1e-12)


@pytest.mark.data
@pytest.mark.slow
@pytest.mark.cloudy
def test_O3_cloudy_input_same_from_relocated_dat(offgrid_dat, tmp_path, monkeypatch):
    from toddlers.cloudy_simulation_manager import CloudySimulationManager
    from toddlers.cloudy_output_handler import CloudyOutputHandler
    monkeypatch.setenv('TODDLERS_OUTPUT_ROOT', str(tmp_path / 'cloudy_root'))
    moved = str(tmp_path / 'elsewhere' / 'my_run.dat')
    os.makedirs(os.path.dirname(moved))
    shutil.copy(offgrid_dat, moved)
    texts = []
    for dat in (offgrid_dat, moved):
        mgr = CloudySimulationManager(dat, method='adaptive', add_DIG=False, add_logger=False)
        t = mgr.get_time_points()[1]
        with mgr.change_dir(mgr.cloudy_run_dir):
            mgr.get_model_generator('shell').write_input_file(t)
            texts.append(open(CloudyOutputHandler('shell', t).get_file_path('in')).read())
    assert texts[0] == texts[1]


@pytest.mark.data
@pytest.mark.slow
def test_O5_dat_without_inputs_reads_the_path(offgrid_dat, tmp_path):
    import json
    from toddlers.track_simulation import load_output_file
    rel = offgrid_dat.split(os.sep + 'evolution_output' + os.sep, 1)[1]       # the layout + name
    old = str(tmp_path / 'evolution_output' / rel)
    os.makedirs(os.path.dirname(old))
    data = json.load(open(offgrid_dat))
    data['parameters'].pop('inputs')
    json.dump(data, open(old, 'w'))
    p, _ = load_output_file(old)
    assert p['M_cl_init'] == pytest.approx(10 ** 5.48, rel=1e-12)      # from logM5.48 in the name
    assert p['Z'] == 0.02 and p['template'] == 'SB99'
