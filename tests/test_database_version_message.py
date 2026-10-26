"""A single-star database of another version: the error says how to get the 2.1.0 one.

Bar: the message names the 2.1.0 database, the download command and the local build, and
does not offer include_supernovae=False, which no user-facing class (StochasticSpec,
StellarFeedback, Evolution) exposes. No data files: a toy h5 without the SN attributes of the 2.1.0 schema.
"""
import h5py
import numpy as np
import pytest

from toddlers.pysb99.stochastic.database import get_supernova_table


def test_database_of_another_version_names_how_to_get_the_new_one(tmp_path):
    db = str(tmp_path / 'other_version.h5')
    with h5py.File(db, 'w') as f:
        f.create_group('metadata')
        z = f.create_group('Z_0.0140')
        z.attrs['Z_value'] = 0.014
        z.create_dataset('mass_grid', data=np.array([20.0]))
    with pytest.raises(ValueError) as e:
        get_supernova_table(db, 'MW')
    msg = str(e.value)
    assert 'not the single-star database of this version' in msg
    assert 'download_data.py --stochastic-tracks --overwrite' in msg
    assert 'build_stochastic_database.py' in msg and '2.1.0' in msg
    assert 'include_supernovae' not in msg
