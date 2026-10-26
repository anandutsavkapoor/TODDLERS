"""The pySB99 table's L_ion is the < 912 A luminosity L_HI.

pySB99's ionizing bands are nested (pysb99_core: each band integrates wave <= its edge), so
L_HI already contains the He-ionizing luminosity, and a sum L_HI + L_HeI + L_HeII would
count He-ionizing photons two or three times. Rule: the generator's L_ion equals L_HI per
node to rel 1e-12, and the shipped kroupa_100 table records it in its conventions.
"""
import contextlib
import io
import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.data


def test_generator_L_ion_is_L_HI():
    from toddlers.pysb99.generate_pysb99_interpolants import PySB99InterpolantGenerator
    warnings.filterwarnings('ignore')
    g = PySB99InterpolantGenerator(metallicities=['MW', 'MWC'], time_start_myr=0.01, time_end_myr=6.0,
                                   time_step_myr=0.5)
    with contextlib.redirect_stdout(io.StringIO()):
        g.run_all_metallicities(total_mass=1e6, imf_exponents=[1.3, 2.3], imf_mass_limits=(0.1, 0.5, 100.0),
                                auto_extend_imf=False, verbose=False)
    q = g.table_quantities()[0]
    L_HI = np.vstack([10 ** g.results_dict[m].hi_ionizing_luminosity for m in g.metallicities])
    L_He = np.vstack([10 ** g.results_dict[m].hei_ionizing_luminosity for m in g.metallicities])
    assert np.max(np.abs(q['L_ion'] / L_HI - 1)) <= 1e-12
    assert np.all(L_He <= L_HI * (1 + 1e-12))      # nested bands


def test_shipped_table_records_the_convention():
    from toddlers.feedback_tables import load_table
    tab = load_table('pySB99', 'kroupa_100')
    assert '912' in tab.metadata['conventions']['L_ion']
    assert np.all(tab.values['L_ion'] <= tab.values['L_bol'])
