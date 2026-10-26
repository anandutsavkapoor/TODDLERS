"""A reused stochastic star table gets its feedback pairing refreshed.

ensure_star_table reuses a current table (same star list, database, Z, age grid) without
rewriting it, while the generation's feedback table can change under the same stars (new
feedback content, e.g. the SN reading rule). Bar: on reuse with a paired_feedback_table, the
sidecar field ``paired_feedback_content_md5`` names that table's content md5; the star
table itself is not rewritten and stays current.

Multi-generation: two generations of one run, each refreshed to its own md5.
No data files: the population, spec and spectra are synthetic.
"""
import json
import types

import numpy as np
import pytest

from toddlers.pysb99.stochastic import population as P
from toddlers.pysb99.stochastic import interpolants as I


def _pop(gen):
    return types.SimpleNamespace(
        masses_msun=np.array([10.0, 30.0, 60.0]) * (1 + gen), birth_times_myr=np.zeros(3),
        track_min_mass_msun=1.0, metallicity_Z=0.014, ages_included=False, base_seed=7,
        generation_index=gen, database_md5='x', arrays_md5=lambda: f'a{gen}')


@pytest.fixture
def setup(tmp_path, monkeypatch):
    db = tmp_path / 'db.h5'
    db.write_bytes(b'synthetic database')
    spec = types.SimpleNamespace(label='Test', interpolation_mode='snap', database_path=str(db),
                                 n_processes=1)
    calls = []

    def fake_spectra(masses, initial_ages, time_grid_myr, database_path, metallicity,
                     rotation, n_processes, verbose):
        calls.append(1)
        lam = np.array([100.0, 912.0, 5000.0])
        return lam, np.ones((len(time_grid_myr), lam.size))
    monkeypatch.setattr(I, 'compute_time_delayed_spectra', fake_spectra)
    return spec, calls, tmp_path / 'out'


def _ft(md5):
    return types.SimpleNamespace(content_md5=lambda: md5)


def _side(out, name):
    return json.loads((out / (name + '.meta.json')).read_text())


@pytest.mark.parametrize('generations', [(0,), (0, 1)])
def test_reuse_refreshes_pairing(setup, generations):
    spec, calls, out = setup
    names = {}
    for g in generations:
        names[g] = P.ensure_star_table(_pop(g), spec, output_dir=str(out),
                                       paired_feedback_table=_ft(f'old-md5-{g}'))
    table_bytes = {g: (out / n).read_bytes() for g, n in names.items()}
    n_written = len(calls)
    for g in generations:
        assert P.ensure_star_table(_pop(g), spec, output_dir=str(out),
                                   paired_feedback_table=_ft(f'new-md5-{g}')) == names[g]
    assert len(calls) == n_written  # reused, not recomputed
    for g, n in names.items():
        assert _side(out, n)['paired_feedback_content_md5'] == f'new-md5-{g}'
        assert (out / n).read_bytes() == table_bytes[g]
    assert not (out / 'stale_tables').exists()
    # still current on a third call (the sidecar md5 of the table was kept)
    for g in generations:
        P.ensure_star_table(_pop(g), spec, output_dir=str(out),
                            paired_feedback_table=_ft(f'new-md5-{g}'))
    assert len(calls) == n_written
