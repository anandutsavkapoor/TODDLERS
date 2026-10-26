"""The public reader applies the identity and mandatory-quantity rules.

load_table by names compares the file's template / imf / star_type attributes with the
request, as StellarFeedback does, and read_table_file refuses a file that the writer would
refuse: L_bol, L_ion, Q_HI or L_LyW listed missing, or no L_mech_* / F_ram_* channel stored.
Bar: load_table('SB99', 'otherimf') on a copy of the SB99 kroupa100 table refuses, naming
both records; a file without L_bol (listed missing), or without any L_mech_* channel, is
refused on read; the shipped tables all read.
The edited copies have their content_md5 attribute removed, so the content check (which
would refuse them for a different reason) is not what is tested.
"""
import glob
import json
import os
import shutil

import h5py
import pytest

from toddlers.feedback_tables import load_table, read_table_file

SHIPPED = os.path.join(os.path.dirname(__file__), '..', 'src', 'toddlers', 'database')
SB99 = os.path.join(SHIPPED, 'SB99_kroupa100_feedback.h5')


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv('TODDLERS_DATA', str(tmp_path))
    return tmp_path


def _edited_copy(dst, drop=(), missing=None):
    shutil.copy(SB99, dst)
    with h5py.File(dst, 'r+') as f:
        del f.attrs['content_md5']
        for name in drop:
            del f['quantities'][name]
        if missing:
            m = json.loads(f['quantities'].attrs.get('missing', '{}'))
            m.update(missing)
            f['quantities'].attrs['missing'] = json.dumps(m)
    return str(dst)


def test_renamed_copy_refused_by_name(data_dir):
    shutil.copy(SB99, data_dir / 'SB99_otherimf_feedback.h5')
    with pytest.raises(ValueError, match="otherimf"):
        load_table('SB99', 'otherimf')


def test_mandatory_quantity_listed_missing_refused(data_dir):
    p = _edited_copy(data_dir / 'x_feedback.h5', drop=['L_bol'], missing={'L_bol': 'test'})
    with pytest.raises(ValueError, match="L_bol"):
        read_table_file(p)


def test_no_mechanical_channel_refused(data_dir):
    p = _edited_copy(data_dir / 'y_feedback.h5', drop=['L_mech_SN', 'L_mech_wind'],
                     missing={'L_mech_SN': 'test', 'L_mech_wind': 'test'})
    with pytest.raises(ValueError, match="L_mech"):
        read_table_file(p)


@pytest.mark.parametrize('path', sorted(glob.glob(os.path.join(SHIPPED, '*_feedback.h5'))),
                         ids=os.path.basename)
def test_shipped_tables_still_read_by_name(path):
    t = read_table_file(path)
    m = t.metadata
    tab = load_table(m['template'], m['imf'], m['star_type'] or None)
    assert tab.md5 == t.md5
