"""Units round trip: every dataset and axis of every unified feedback table
in the data dir, of the single-star database (when present and schema-aligned) and of every
star-table sidecar in CLOUDY_DATA_DIR that carries a sidecar schema version, has a unit string
that astropy parses and that round-trips: Unit(Unit(s).to_string()) == Unit(s)."""
import glob
import json
import os

import astropy.units as u
import h5py
import pytest

from toddlers._paths import get_data_dir, get_database_path
from toddlers.feedback_tables import AXES, QUANTITIES, quantity_units, units_equal


def _round_trips(s):
    unit = u.Unit(s)
    return u.Unit(unit.to_string()) == unit


def _walk(group, out, prefix=''):
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            out.append((prefix + k, v.attrs.get('units')))
        else:
            _walk(v, out, prefix + k + '/')


TABLES = sorted(glob.glob(os.path.join(get_data_dir(), '*_feedback.h5')))


def test_shipped_tables_present():
    names = {os.path.basename(p) for p in TABLES}
    for n in ('SB99_kroupa100_feedback.h5', 'pySB99_kroupa_100_feedback.h5'):
        assert n in names


@pytest.mark.parametrize("path", TABLES, ids=[os.path.basename(p) for p in TABLES])
def test_table_units_round_trip(path):
    items = []
    with h5py.File(path) as f:
        _walk(f, items)
    assert items
    for name, s in items:
        assert s is not None, f"{name} has no units"
        assert _round_trips(s), name
        if name.startswith('quantities/'):
            assert units_equal(s, quantity_units(name.split('/')[1])), name
        if name.startswith('axes/'):
            assert units_equal(s, AXES[name.split('/')[1]][0]), name


@pytest.mark.data
def test_single_star_database_units():
    p = get_database_path()
    if not os.path.exists(p):
        pytest.skip(f"no single-star database at {p}")
    with h5py.File(p) as f:
        if f.attrs.get('schema') != 'toddlers.single_star_library':
            pytest.skip("single-star database without the aligned schema (pre-2.1 file)")
        items = []
        _walk(f, items)
        for name, s in items:
            assert s is not None, f"{name} has no units"
            assert _round_trips(s), name
        for k, s in json.loads(f.attrs['attribute_units']).items():
            assert _round_trips(s), k


def test_star_table_sidecar_units():
    from toddlers.constants import CLOUDY_DATA_DIR
    n = 0
    for p in glob.glob(os.path.join(CLOUDY_DATA_DIR, '*.ascii.meta.json')):
        try:
            side = json.load(open(p))
        except (OSError, ValueError):
            continue
        if 'sidecar_schema_version' not in side:
            continue
        n += 1
        for k, s in side['units'].items():
            if k.endswith('_note') or s == 'unspecified':
                continue
            assert _round_trips(s), (p, k, s)
    if n == 0:
        pytest.skip("no star-table sidecar with a schema version in CLOUDY_DATA_DIR")


@pytest.mark.parametrize("path", TABLES, ids=[os.path.basename(p) for p in TABLES])
def test_shipped_imf_records(path):
    """IMF record check on the shipped tables: a broken-power-law record integrates (independent
    quadrature) to the normalization mass within rel 1e-6; a custom population's stars sum to
    its per-metallicity mass; a record read from the source names its files with md5."""
    from scipy.integrate import quad
    from toddlers.feedback_tables import load_table
    tab = load_table(path)
    rec = tab.imf
    d = rec['definition']
    assert rec['name'] == tab.metadata['imf'] and rec['per_mass_msun'] == 1e6
    if d['kind'] == 'broken_power_law':
        I = sum(quad(lambda m, s=s: m * s['coefficient'] * m ** -s['slope'], s['m_low_msun'], s['m_up_msun'],
                     epsrel=1e-12, epsabs=0)[0] for s in d['segments'])
        assert abs(I / rec['normalization_mass_msun'] - 1) <= 1e-6
        assert rec['normalization']['m_low_msun'] == d['m_low_msun'] == 0.1
    elif d['kind'] == 'custom_population':
        for z, stars in d['star_numbers_msun_per_Z'].items():
            M = sum(float(m) * n for m, n in stars.items())
            assert abs(M / rec['normalization']['population_mass_msun_per_Z'][z] - 1) <= 1e-6
    else:
        assert d['kind'] == 'read_from_source' and d['source_files_md5']
