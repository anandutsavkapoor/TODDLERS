"""A Cloudy star-table index (.idx, and any .mod) never outlives its table.

Cloudy compiles <name>.ascii into <name>.idx (stars.cpp:2841 at 013dc0bf, 2402 at
01c4cfc6) and, when the index is present with the right magic number, reads its byte
offsets into the .ascii without checking that the .ascii is the one it was built from
(InitGrid, stars.cpp:2914-2930 at 013dc0bf). A regenerated table next to an old index is
then read at the wrong offsets.

Bars: moving a table aside moves its .idx and .mod with it; writing a table moves aside any
.idx/.mod already next to that name. Both land in stale_tables/<UTC time>/ (never deleted).
"""
import numpy as np

import toddlers.cloudy_stellar_spectra_generator as csg


def _stale_names(d):
    return sorted(p.name for p in (d / 'stale_tables').rglob('*') if p.is_file())


def test_move_aside_takes_the_cloudy_index(tmp_path):
    t = tmp_path / 'X_burst.ascii'
    for p in (t, tmp_path / 'X_burst.ascii.meta.json', tmp_path / 'X_burst.idx',
              tmp_path / 'X_burst.mod'):
        p.write_text('x\n')
    csg.move_star_table_aside(str(t), str(tmp_path))
    assert not (tmp_path / 'X_burst.idx').exists() and not (tmp_path / 'X_burst.mod').exists()
    assert _stale_names(tmp_path) == ['X_burst.ascii', 'X_burst.ascii.meta.json',
                                      'X_burst.idx', 'X_burst.mod']


def test_writing_a_table_moves_an_old_index_aside(tmp_path):
    t = tmp_path / 'Y_burst.ascii'
    (tmp_path / 'Y_burst.idx').write_text('old index\n')
    csg.write_cloudy_star_table(str(t), [0.0, 1e6], [0.02], [100.0, 1000.0],
                                np.ones((1, 2, 2)), {'origin': 'generated', 'key': 'k'})
    assert t.exists()
    assert not (tmp_path / 'Y_burst.idx').exists()
    assert _stale_names(tmp_path) == ['Y_burst.idx']
