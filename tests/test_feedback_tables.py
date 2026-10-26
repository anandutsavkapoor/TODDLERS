"""Invariants of the unified feedback table (toddlers.feedback_tables, schema 1).

Synthetic tables only, written to a temporary TODDLERS_DATA, so no shipped data is needed.

PASS = every test below green. Each of five mutations of the writer/reader (finite check
off, floor off, wind-only total, units check off, a timestamp attribute written into the
file) turns at least one test red.

(a) negative / NaN / inf input refused, nothing written;
(b) the floor 1e-50 applied and counted per quantity;
(c) totals = sum of the channels, a further channel summed;
(d) a quantity listed as missing loads and raises on access; an absent, unlisted one is
    refused by the writer and, in a file, by the reader;
(e) every dataset and axis carries units; a file with L_bol in 'erg' is refused, 'erg/s'
    accepted (parsed-unit equality);
(f) an unknown schema_version is refused; a template/imf attribute mismatch is refused;
(g) the same content written twice has the same md5; a value changed after writing is refused
    (content md5);
(h) n_Z = 1 loads and refuses another Z; Z must be a node;
(i) a template name unknown to the code loads in StellarFeedback (metadata routing) and
    constructs an Evolution; Cloudy refuses it with a 'spectral loader' message;
(j) load_table: get() returns astropy Quantities in the schema units; derived quantities
    (hardness, mean ionizing photon energy) follow their one definition;
(k) a time grid ending before 30 Myr is refused, and written only as a legacy table that
    records the failed check;
(l) U-IMF: the record integrates to the normalization mass (independent quadrature,
    rel 1e-6); a broken record is refused; the public reader exposes .imf.
"""
import contextlib
import io
import os

import h5py
import numpy as np
import pytest
from scipy.integrate import quad

from toddlers.constants import M_SUN, MYR_TO_SEC
from toddlers.feedback_tables import (FLOOR, SCHEMA, build_feedback_table, broken_power_law, file_md5,
                                      load_table, read_table_file, table_filename, write_feedback_table)
from toddlers.stellar_feedback import StellarFeedback, feedback_table_md5, feedback_table_path

T = np.concatenate([np.linspace(0.0, 1.9, 20), np.arange(2.0, 6.01, 0.1), np.linspace(7.0, 40.0, 34)])
ZN = np.array([0.004, 0.02])


def _quantities(Z=ZN):
    t = T[None, :]
    z = (np.asarray(Z) / 0.02)[:, None]
    one = np.ones_like(z)
    return {
        'L_bol': 1e42 * (1 + t) ** -1.0 * z ** 0.1,
        'Q_HI': 1e52 * np.exp(-t / 4.0) * z ** -0.1,
        'Q_HeI': 1e51 * np.exp(-t / 3.0) * one,
        'Q_HeII': 1e48 * np.exp(-t / 2.0) * one,
        'L_ion': 4e41 * np.exp(-t / 4.0) * one,
        'L_LyW': 1e38 * (1 + t) ** -0.5 * one,
        'L_mech_wind': 1e39 * np.exp(-t / 5.0) * z ** 0.5,
        'L_mech_SN': np.where(t > 3.0, 1e40, 0.0) * one,
        'F_ram_wind': 1e31 * np.exp(-t / 5.0) * z ** 0.5,
        'F_ram_SN': np.where(t > 3.0, 3e31, 0.0) * one,
    }


IMF = broken_power_law([1.3, 2.3], [0.1, 0.5, 100.0], 1e6)
META = {'template': 'TESTTPL', 'imf': 'myimf', 'star_type': None,
        'source': {'code': 'synthetic', 'version': '0'},
        'normalization': {'rule': 'per 1e6 Msun of the full IMF', 'per_mass_msun': 1e6,
                          'population_mass_msun': 1e6, 'm_low_msun': 0.1, 'm_up_msun': 100.0},
        'imf_definition': IMF,
        'conventions': {'E_SN_erg': 1e51}}


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv('TODDLERS_DATA', str(tmp_path))
    return tmp_path


def _write(d, q=None, meta=None, **kw):
    return write_feedback_table(T, ZN, _quantities() if q is None else q,
                                dict(META) if meta is None else meta, str(d), **kw)


def _sf(**kw):
    args = dict(template='TESTTPL', imf='myimf', Z=0.02, M_cl_init=1e6 / 0.05 * M_SUN, eta_sf=0.05,
                t_list_collapse=[])
    args.update(kw)
    with contextlib.redirect_stdout(io.StringIO()):
        return StellarFeedback(**args)


# (a)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0])
def test_bad_values_refused(data_dir, bad):
    q = _quantities()
    q['L_bol'][1, 5] = bad
    with pytest.raises(ValueError, match="L_bol"):
        _write(data_dir, q=q)
    assert not os.listdir(data_dir)


def test_unknown_quantity_and_shape_refused(data_dir):
    q = _quantities()
    q['L_bolo'] = q.pop('L_bol')
    with pytest.raises(ValueError):
        _write(data_dir, q=q)
    q = _quantities()
    q['Q_HI'] = q['Q_HI'][:, :-1]
    with pytest.raises(ValueError, match="shape"):
        _write(data_dir, q=q)


# (b)
def test_floor_applied_and_counted(data_dir):
    tab = _write(data_dir)
    n_zero = int(np.sum(_quantities()['L_mech_SN'] == 0))
    assert n_zero > 0
    back = read_table_file(tab.path)
    assert np.all(back.values['L_mech_SN'][_quantities()['L_mech_SN'] == 0] == FLOOR)
    assert back.metadata['floor']['value'] == FLOOR
    assert back.metadata['floor']['floored_nodes']['L_mech_SN'] == n_zero
    assert back.metadata['floor']['floored_nodes']['L_bol'] == 0
    with h5py.File(tab.path) as f:
        assert f['quantities/L_mech_SN'].attrs['floored_nodes'] == n_zero


# (c)
def test_totals_are_channel_sums(data_dir):
    q = _quantities()
    q['L_mech_jet'] = 5e38 * np.ones_like(q['L_bol'])
    tab = read_table_file(_write(data_dir, q=q).path)
    L = np.maximum(q['L_mech_wind'], FLOOR) + np.maximum(q['L_mech_SN'], FLOOR) + q['L_mech_jet']
    F = np.maximum(q['F_ram_wind'], FLOOR) + np.maximum(q['F_ram_SN'], FLOOR)
    np.testing.assert_allclose(tab.log_nodes('L_mech'), np.log10(L), rtol=1e-12)
    np.testing.assert_allclose(tab.log_nodes('F_ram'), np.log10(F), rtol=1e-12)
    assert tab.channels('L_mech') == ['L_mech_SN', 'L_mech_jet', 'L_mech_wind']
    with h5py.File(tab.path) as f:
        assert 'L_mech' not in f['quantities'] and 'F_ram' not in f['quantities']


# (d)
def test_missing_listed_loads_and_raises(data_dir):
    q = _quantities()
    q.pop('Q_HeII')
    tab = read_table_file(_write(data_dir, q=q, missing={'Q_HeII': 'not in the source'}).path)
    assert tab.missing == {'Q_HeII': 'not in the source'}
    with pytest.raises(KeyError, match="missing"):
        tab.get('Q_HeII', 1.0, Z=0.02)
    with pytest.raises(KeyError, match="missing"):
        tab.get('hardness_HeII', 1.0, Z=0.02)
    assert 'hardness_HeII' not in tab.quantities and 'hardness_HeI' in tab.quantities


def test_absent_unlisted_refused(data_dir):
    q = _quantities()
    q.pop('Q_HeII')
    with pytest.raises(ValueError, match="Q_HeII"):
        _write(data_dir, q=q)
    path = _write(data_dir).path
    with h5py.File(path, 'a') as f:
        del f['quantities/Q_HeII']
    with pytest.raises(ValueError, match="Q_HeII"):
        read_table_file(path)


# (e)
def test_every_dataset_and_axis_has_units(data_dir):
    import astropy.units as u
    path = _write(data_dir).path
    with h5py.File(path) as f:
        for g in ('axes', 'quantities'):
            for k in f[g]:
                s = f[g][k].attrs['units']
                assert u.Unit(u.Unit(s).to_string()) == u.Unit(s)


@pytest.mark.parametrize("stored,ok", [('erg', False), ('erg/s', True), ('erg s-1', True), ('W', False)])
def test_unit_mismatch_refused(data_dir, stored, ok):
    path = _write(data_dir).path
    with h5py.File(path, 'a') as f:
        f['quantities/L_bol'].attrs['units'] = stored
        del f.attrs['content_md5']
    if ok:
        read_table_file(path)
    else:
        with pytest.raises(ValueError, match="unit"):
            read_table_file(path)
        with pytest.raises(ValueError, match="unit"):
            _sf()


def test_axis_without_units_refused(data_dir):
    path = _write(data_dir).path
    with h5py.File(path, 'a') as f:
        del f['axes/t'].attrs['units']
    with pytest.raises(ValueError, match="units"):
        read_table_file(path)


# (f)
def test_schema_version_and_names_checked(data_dir):
    path = _write(data_dir).path
    with h5py.File(path, 'a') as f:
        f.attrs['schema_version'] = 99
    with pytest.raises(ValueError, match="schema_version"):
        read_table_file(path)
    path = _write(data_dir).path
    with h5py.File(path, 'a') as f:
        f.attrs['imf'] = 'otherimf'
        del f.attrs['content_md5']
    with pytest.raises(ValueError, match="records"):
        _sf()


# (g)
def test_same_content_same_md5_and_tamper_refused(data_dir):
    a = _write(data_dir).md5
    b = _write(data_dir).md5
    assert a == b == feedback_table_md5('TESTTPL', 'myimf')
    path = feedback_table_path('TESTTPL', 'myimf')
    with h5py.File(path, 'a') as f:
        d = f['quantities/L_bol'][()]
        d[0, 3] *= 1.01
        f['quantities/L_bol'][...] = d
    with pytest.raises(ValueError, match="content md5"):
        read_table_file(path)


# (h)
def test_single_metallicity_table(data_dir):
    q = _quantities(Z=[0.014])
    tab = write_feedback_table(T, [0.014], q, dict(META), str(data_dir))
    sf = _sf(Z=0.014)
    assert sf.L_bolo_interpolant(3.0, 0.014) == pytest.approx(np.log10(np.interp(3.0, T, q['L_bol'][0])), abs=1e-3)
    with pytest.raises(ValueError):
        _sf(Z=0.02)
    with pytest.raises(ValueError):
        sf.L_bolo_interpolant(3.0, 0.02)
    assert read_table_file(tab.path).get('Q_HI', 2.0).unit.to_string() == '1 / s'


def test_off_node_metallicity_refused(data_dir):
    _write(data_dir)
    assert _sf(Z=0.004).Z == 0.004
    with pytest.raises(ValueError, match="not a node"):
        _sf(Z=0.01)


# (i)
def test_unknown_template_routes_by_table(data_dir, tmp_path, monkeypatch):
    _write(data_dir)
    sf = _sf()
    q = _quantities()
    j = int(np.argmin(np.abs(T - 3.0)))
    assert sf.get_ionizing_photon_rate(T[j] * MYR_TO_SEC) == pytest.approx(q['Q_HI'][1, j], rel=1e-6)
    assert sf.get_bolometric_luminosity(T[j] * MYR_TO_SEC) == pytest.approx(q['L_bol'][1, j], rel=1e-6)
    L = q['L_mech_wind'][1, j] + q['L_mech_SN'][1, j]
    assert sf.get_mechanical_luminosity(T[j] * MYR_TO_SEC) == pytest.approx(L, rel=1e-6)
    from toddlers.evolution import Evolution
    monkeypatch.setenv("TODDLERS_OUTPUT_ROOT", str(tmp_path / 'out'))
    with contextlib.redirect_stdout(io.StringIO()):
        evo = Evolution(Z=0.02, eta_sf=0.05, n_cl=320.0, M_cl_init=1e5 * M_SUN, template='TESTTPL',
                        imf='myimf')
    assert evo.feedback_md5 == file_md5(feedback_table_path('TESTTPL', 'myimf'))
    with pytest.raises(ValueError, match="needs imf"):
        Evolution(Z=0.02, eta_sf=0.05, n_cl=320.0, M_cl_init=1e5 * M_SUN, template='TESTTPL')
    import toddlers.cloudy_stellar_spectra_generator as csg
    with pytest.raises(ValueError, match="spectral loader"):
        csg.SpectralTableGenerator(template='TESTTPL', imf='myimf', star_type=None)


def test_missing_table_for_unknown_template(data_dir):
    with pytest.raises(ValueError, match="Unsupported template"):
        _sf(template='NOPE', imf='x')


# (j)
def test_load_table_get_units_and_derived(data_dir):
    import astropy.units as u
    _write(data_dir)
    tab = load_table('TESTTPL', 'myimf')
    assert tab.metadata['schema'] == SCHEMA
    q = _quantities()
    j = int(np.argmin(np.abs(T - 3.0)))
    t = T[j]
    assert tab.get('L_bol', t, Z=0.02).unit == u.erg / u.s
    assert tab.get('F_ram_SN', t, Z=0.02).unit == u.dyn
    assert tab.get('L_LyW', t, Z=0.02).unit == u.erg / u.s / u.AA
    assert tab.get('Q_HI', t, Z=0.02).value == pytest.approx(q['Q_HI'][1, j], rel=1e-12)
    assert tab.get('hardness_HeI', t, Z=0.02).value == pytest.approx(q['Q_HeI'][1, j] / q['Q_HI'][1, j], rel=1e-12)
    e = tab.get('mean_ionizing_photon_energy', t, Z=0.02)
    assert e.unit == u.erg and e.value == pytest.approx(q['L_ion'][1, j] / q['Q_HI'][1, j], rel=1e-12)
    ev = tab.get('mean_ionizing_photon_energy_eV', t, Z=0.02)
    assert ev.unit == u.eV and ev.to(u.erg).value == pytest.approx(e.value, rel=1e-9)
    assert tab.get('L_mech', [1.0, 3.0, 10.0], Z=0.004).shape == (3,)
    for k in ('L_bol', 'L_ion', 'Q_HI', 'Q_HeI', 'Q_HeII', 'L_LyW', 'L_mech_wind', 'L_mech_SN',
              'F_ram_wind', 'F_ram_SN', 'L_mech', 'F_ram', 'hardness_HeI', 'hardness_HeII',
              'mean_ionizing_photon_energy', 'mean_ionizing_photon_energy_eV'):
        assert k in tab.quantities
    with pytest.raises(ValueError, match="Z is required"):
        tab.get('L_bol', 1.0)
    assert load_table(tab.path).md5 == tab.md5


# (k)
def test_time_grid_checks(data_dir):
    keep = T <= 25.0
    q = {k: v[:, keep] for k, v in _quantities().items()}
    with pytest.raises(ValueError, match="30 Myr"):
        write_feedback_table(T[keep], ZN, q, dict(META), str(data_dir))
    tab = write_feedback_table(T[keep], ZN, q, dict(META), str(data_dir), legacy=True)
    m = read_table_file(tab.path).metadata
    assert m['time_grid_checks']['last_node_beyond_30_myr'] is False and m['legacy'] is True


# (l) U-IMF
def test_imf_record_integrates_to_normalization_mass(data_dir):
    tab = load_table(_write(data_dir).path)
    rec = tab.imf
    I = sum(quad(lambda m, s=s: m * s['coefficient'] * m ** -s['slope'], s['m_low_msun'], s['m_up_msun'],
                 epsrel=1e-12, epsabs=0)[0] for s in rec['definition']['segments'])
    assert abs(I / rec['normalization_mass_msun'] - 1) <= 1e-6
    assert rec['normalization']['m_low_msun'] == 0.1 and rec['name'] == 'myimf'


def test_broken_imf_record_refused(data_dir):
    meta = dict(META)
    bad = dict(IMF)
    bad['segments'] = [dict(s, coefficient=s['coefficient'] * 1.01) for s in IMF['segments']]
    meta['imf_definition'] = bad
    with pytest.raises(ValueError, match="integrates"):
        _write(data_dir, meta=meta)
    meta['imf_definition'] = IMF
    meta['normalization'] = dict(META['normalization'], m_low_msun=0.8)
    with pytest.raises(ValueError, match="m_low"):
        _write(data_dir, meta=meta)


def test_table_filename():
    assert table_filename('BPASS', 'chab100', 'bin') == 'BPASS_chab100_bin_feedback.h5'
    assert table_filename('SB99', 'kroupa100') == 'SB99_kroupa100_feedback.h5'
    assert os.path.basename(feedback_table_path('BPASSv2.2.1', None, None)) == 'BPASS_chab100_bin_feedback.h5'
    assert os.path.basename(feedback_table_path('SB99', None, 'sin')) == 'SB99_kroupa100_feedback.h5'
