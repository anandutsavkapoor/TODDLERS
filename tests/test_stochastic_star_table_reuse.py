"""A generation's Cloudy star table is reused only on a matching sidecar key.

ensure_star_table reuses a file at the table's name only under the sidecar rule that
SpectralTableGenerator and generate_cloudy_spectral_table apply (star_table_is_current:
sidecar md5 = file md5, sidecar key = request key, grid lines equal), so a partial or
edited table at that name never reaches Cloudy; anything else is moved to stale_tables/ and
the table is written again. A current table is reused without recomputing the spectra or
the feedback table.

No data files: the population, spec and spectra are synthetic.
"""
import os
import types

import numpy as np

from toddlers.pysb99.stochastic import population as P
from toddlers.pysb99.stochastic import interpolants as I


def _setup(tmp_path, monkeypatch):
    db = tmp_path / 'db.h5'
    db.write_bytes(b'synthetic database')
    pop = types.SimpleNamespace(
        masses_msun=np.array([10.0, 30.0, 60.0]), birth_times_myr=np.zeros(3),
        track_min_mass_msun=1.0, metallicity_Z=0.014, ages_included=False, base_seed=7,
        generation_index=0, database_md5='x', arrays_md5=lambda: 'a')
    spec = types.SimpleNamespace(label='Test', interpolation_mode='snap', database_path=str(db),
                                 n_processes=1)
    calls = []

    def fake_spectra(masses, initial_ages, time_grid_myr, database_path, metallicity,
                     rotation, n_processes, verbose):
        calls.append(1)
        lam = np.array([100.0, 912.0, 5000.0])
        return lam, np.ones((len(time_grid_myr), lam.size))
    monkeypatch.setattr(I, 'compute_time_delayed_spectra', fake_spectra)
    ft = types.SimpleNamespace(content_md5=lambda: 'feedback-md5')
    return pop, spec, ft, calls


def test_current_table_is_reused(tmp_path, monkeypatch):
    pop, spec, ft, calls = _setup(tmp_path, monkeypatch)
    out = tmp_path / 'out'
    name = P.ensure_star_table(pop, spec, output_dir=str(out), paired_feedback_table=ft)
    assert len(calls) == 1 and (out / (name + '.meta.json')).exists()
    assert P.ensure_star_table(pop, spec, output_dir=str(out), paired_feedback_table=ft) == name
    assert len(calls) == 1


def test_foreign_file_at_the_name_is_replaced(tmp_path, monkeypatch):
    pop, spec, ft, calls = _setup(tmp_path, monkeypatch)
    out = tmp_path / 'out'
    out.mkdir()
    name = P.star_table_name(pop, spec)
    (out / name).write_text('a partial or foreign table\n')
    assert P.ensure_star_table(pop, spec, output_dir=str(out), paired_feedback_table=ft) == name
    assert len(calls) == 1
    assert (out / name).read_text().startswith('20060612\n')
    moved = [p.name for p in (out / 'stale_tables').rglob(name)]
    assert moved == [name]
