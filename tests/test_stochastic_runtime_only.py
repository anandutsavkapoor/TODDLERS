"""Stochastic runs draw at run time only; configuration errors are not retried
(2.1.0).

Pass rules:

R1  A stochastic table named as input is refused before any integration: a unified table
    with a /population record and a legacy-layout pySB99 'stochastic_*' table, each named as
    imf, raise ConfigurationError (with the StochasticSpec pointer) from StellarFeedback and
    from the Evolution constructor.
R2  In a multi-generation run (Z 0.02, eta_sf 0.01, n_cl 2560, M_cl 1e6 Msun, SB99: first
    recollapse at 2.49 Myr, 9 generations), a ConfigurationError raised at the first
    recollapse ends the run after exactly one attempt; a RuntimeError raised at the same
    point is still retried (three attempts; control).
E   An OSError from the output write at the end of the same multi-
    generation run ends it after one attempt; a TimeoutError (an OSError subclass, the
    stall watchdog's) is still retried three times.
"""
import os
import pickle

import numpy as np
import pytest

from toddlers.constants import M_SUN
from toddlers.exceptions import ConfigurationError
from toddlers.pysb99.stochastic.population import (
    IMFSpec, GenerationPopulation, birth_times, draw_stop_after, generation_rng,
    generation_table_from_quantities)
from toddlers.stellar_feedback import StellarFeedback

KROUPA = IMFSpec('kroupa', 0.1, 100.0)
Z = 0.014


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv('TODDLERS_DATA', str(tmp_path))
    return str(tmp_path)


def _unified_stochastic_table(name, data_dir, target=1e3, seed=7):
    """A unified table with a /population record, written as pySB99_<name>_feedback.h5."""
    from toddlers.feedback_tables import FeedbackTable, write_table_file, table_filename
    m = draw_stop_after(KROUPA, target, generation_rng(seed, 0))
    pop = GenerationPopulation(
        masses_msun=m, birth_times_myr=birth_times(m, 'burst', target), imf_name='kroupa',
        imf_m_low_msun=0.1, imf_m_up_msun=100.0, base_seed=seed, generation_index=0,
        target_mass_msun=target, sf_mode='burst', tau_sf_myr=None, metallicity_Z=Z,
        track_min_mass_msun=0.8, interpolation_mode='snap', database_md5='synthetic',
        seed_spawn_key=[0])
    t = np.arange(0, 1205) * 0.025
    shape = np.where(t < 4.0, 0.0, -2.0 * (t - 4.0) / 26.0)
    q = {'L_mech_wind': 10 ** (40 + shape), 'L_mech_SN': np.zeros_like(t), 'F_ram_wind': 10 ** (31 + shape),
         'F_ram_SN': np.zeros_like(t), 'L_bol': 10 ** (42 + shape), 'L_ion': 10 ** (41.5 + shape),
         'Q_HI': 10 ** (52 + shape), 'Q_HeI': 10 ** (51 + shape), 'Q_HeII': 10 ** (48 + shape),
         'L_LyW': 10 ** (38 + shape)}
    ft = generation_table_from_quantities(pop, t, {k: v[None, :] for k, v in q.items()}).feedback_table
    meta = dict(ft.metadata, imf=name, name=table_filename('pySB99', name)[:-3])
    write_table_file(FeedbackTable(ft.t, ft.Z, ft.values, ft.missing, meta, population=ft.population),
                     os.path.join(data_dir, table_filename('pySB99', name)))
    return pop


def _legacy_stochastic_table(name, data_dir):
    """A legacy-layout pySB99 table (seven Interp2DLinear entries + the Lyman-Werner file)."""
    from toddlers._interp_compat import Interp2DLinear
    t = np.logspace(-2, np.log10(50), 50)
    Zs = np.array([0.006, 0.014, 0.020])
    f = lambda v: Interp2DLinear(t, Zs, np.full((3, 50), v))
    with open(os.path.join(data_dir, f'pySB99interpolation_{name}.obj'), 'wb') as fh:
        pickle.dump([f(40.0), f(31.0), f(42.0), f(41.5), f(52.0), f(0.0), f(0.0)], fh)
    with open(os.path.join(data_dir, f'pySB99interpolation_{name}_LumLymanWerner.obj'), 'wb') as fh:
        pickle.dump(f(38.0), fh)


# ---------------------------------------------------------------- R1
def test_R1_unified_stochastic_table_refused(data_dir):
    pop = _unified_stochastic_table('synth1e3', data_dir)
    with pytest.raises(ConfigurationError, match="StochasticSpec"):
        StellarFeedback('pySB99', Z, pop.drawn_mass_msun / 0.1 * M_SUN, 0.1, [], imf='synth1e3')


def test_R1_legacy_stochastic_table_refused(data_dir):
    # a legacy 'stochastic*' table not in the stochastic_seed<N>_M<mass> form, so the file-based
    # check is exercised (that form is refused by name first; test_evolution_campaign)
    _legacy_stochastic_table('stochastic_custom_M1000', data_dir)
    with pytest.warns(FutureWarning):
        with pytest.raises(ConfigurationError, match="StochasticSpec"):
            StellarFeedback('pySB99', Z, 2e4 * M_SUN, 0.05, [], imf='stochastic_custom_M1000')


def test_R1_refused_at_evolution_construction(data_dir, tmp_path, monkeypatch):
    from toddlers.evolution import Evolution
    monkeypatch.setattr(Evolution, 'get_output_paths',
                        lambda self: (str(tmp_path / 'sim.dat'), str(tmp_path / 'sim.log')))
    monkeypatch.setattr(Evolution, 'simulation_attempt',
                        lambda self, *a: pytest.fail("integration started"))
    pop = _unified_stochastic_table('synth1e3', data_dir)
    with pytest.raises(ConfigurationError, match="StochasticSpec"):
        Evolution(Z=Z, eta_sf=0.1, n_cl=160.0, M_cl_init=pop.drawn_mass_msun / 0.1 * M_SUN,
                  template='pySB99', imf='synth1e3')


def test_R1_deterministic_pysb99_table_still_read():
    sf = StellarFeedback('pySB99', Z, 1e6 * M_SUN, 0.05, [], imf='kroupa_100')
    assert sf.stochastic is None and sf.M_stellar_list[0] == pytest.approx(5e4)


# ---------------------------------------------------------------- R2, E
MULTIGEN = dict(Z=0.02, eta_sf=0.01, n_cl=2560.0, M_cl_init=1e6 * M_SUN, template='SB99', imf='kroupa100')


def _counted_run(tmp_path, monkeypatch, patch):
    """Run the multi-generation cloud with one injected failure; return (attempts, error)."""
    from toddlers.evolution import Evolution
    monkeypatch.setattr(Evolution, 'get_output_paths',
                        lambda self: (str(tmp_path / 'sim.dat'), str(tmp_path / 'sim.log')))
    calls = []
    orig = Evolution.simulation_attempt

    def counted(self, solvers, rtol):
        calls.append(rtol)
        return orig(self, solvers, rtol)
    monkeypatch.setattr(Evolution, 'simulation_attempt', counted)
    patch(Evolution)
    ev = Evolution(**MULTIGEN)
    with pytest.raises(Exception) as err:
        ev.run_simulation()
    return len(calls), err.value


def _raise_at_first_recollapse(monkeypatch, exc):
    def patch(Evolution):
        def boom(self):
            raise exc("probe: injected at the first recollapse")
        monkeypatch.setattr(Evolution, 'add_new_generation', boom)
    return patch


@pytest.mark.data
@pytest.mark.slow
def test_R2_configuration_error_at_recollapse_not_retried(tmp_path, monkeypatch):
    n, err = _counted_run(tmp_path, monkeypatch, _raise_at_first_recollapse(monkeypatch, ConfigurationError))
    assert n == 1 and isinstance(err, ConfigurationError) and 'probe' in str(err)


@pytest.mark.data
@pytest.mark.slow
def test_R2_control_runtime_error_at_recollapse_retried(tmp_path, monkeypatch):
    n, err = _counted_run(tmp_path, monkeypatch, _raise_at_first_recollapse(monkeypatch, RuntimeError))
    assert n == 3 and isinstance(err, RuntimeError) and 'probe' in repr(err.__cause__)


@pytest.mark.data
@pytest.mark.slow
def test_E_output_write_oserror_not_retried(tmp_path, monkeypatch):
    from toddlers.track_simulation import TrackSimulation

    def patch(Evolution):
        def boom(self, *a, **k):
            raise OSError("probe: disk full at the output write")
        monkeypatch.setattr(TrackSimulation, 'write_output_file', boom)
    n, err = _counted_run(tmp_path, monkeypatch, patch)
    assert n == 1 and isinstance(err, OSError) and 'probe' in str(err)


@pytest.mark.data
def test_E_control_timeout_still_retried(tmp_path, monkeypatch):
    from toddlers.evolution import Evolution
    monkeypatch.setattr(Evolution, 'get_output_paths',
                        lambda self: (str(tmp_path / 'sim.dat'), str(tmp_path / 'sim.log')))
    calls = []

    def stall(self, solvers, rtol):
        calls.append(rtol)
        raise TimeoutError("probe: watchdog")
    monkeypatch.setattr(Evolution, 'simulation_attempt', stall)
    monkeypatch.setattr(Evolution, 'cleanup_after_attempt', lambda self: None)
    ev = Evolution(**MULTIGEN)
    with pytest.raises(RuntimeError) as err:
        ev.run_simulation()
    assert len(calls) == 3 and isinstance(err.value.__cause__, TimeoutError)
