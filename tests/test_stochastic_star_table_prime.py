"""S7': per-star feedback and star-table spectra use ONE age-interpolation convention (data, slow).

Pass rule: same cases and
nodes as S7 (tests/test_stochastic_star_table.py): base seed 77, Z = 0.014, eta_sf = 0.1,
M_cl = 3e4 Msun, two generations (second at 5 Myr), burst and constant SFR over 2 Myr;
star-table age nodes nearest 1, 3 and 10 Myr; every generation. PASS if at every node
    |log10(L_table / L_fb)| <= 0.01 dex   and   |log10(Q_HI,table / Q_HI,fb)| <= 0.01 dex,
with L_table = 1e20 * int F_lambda dlambda, Q_HI,table = 1e20 * int F_lambda lambda/(h c)
dlambda below 911.76 A, and L_fb, Q_HI,fb the generation's feedback (linear components of
its table, linear in time between the 0.025 Myr nodes).
"""
import os

import numpy as np
import pytest

from toddlers.constants import M_SUN, MYR_TO_SEC
from toddlers.pysb99.stochastic import population as P
from toddlers.stellar_feedback import StellarFeedback
from test_stochastic_star_table import read_star_table, H_C_ERG_A, LYMAN_EDGE_A

pytestmark = [pytest.mark.data, pytest.mark.slow]
BAR_DEX = 0.01


@pytest.mark.parametrize("mode", ["burst", "constant_sfr"])
def test_S7prime_one_age_convention(mode, tmp_path):
    spec = P.StochasticSpec(P.IMFSpec('kroupa', 0.1, 100.0), 77, n_processes=8)
    kw = dict(mode=mode, formation_timescale=2 * MYR_TO_SEC) if mode == 'constant_sfr' else {}
    P._TABLE_CACHE.clear()
    sf = StellarFeedback('pySB99', 0.014, 3e4 * M_SUN, 0.1, [5 * MYR_TO_SEC], stochastic=spec, **kw)
    bad = []
    for pop in sf.populations:
        name = P.ensure_star_table(pop, spec, output_dir=str(tmp_path))
        ages, wave, flux = read_star_table(os.path.join(str(tmp_path), name))
        tab = P.build_generation_table(pop, spec)
        for target in (1.0, 3.0, 10.0):
            k = int(np.argmin(np.abs(ages - target)))
            a = ages[k]
            f = flux[k] * 1e20
            L_tab = np.trapz(f, wave)
            ion = wave <= LYMAN_EDGE_A
            Q_tab = np.trapz(f[ion] * wave[ion] / H_C_ERG_A, wave[ion])
            L_fb = np.interp(a, tab.time_grid_myr, tab.linear_components['L_bol'])
            Q_fb = np.interp(a, tab.time_grid_myr, tab.linear_components['Q_HI'])
            dl, dq = np.log10(L_tab / L_fb), np.log10(Q_tab / Q_fb)
            print(f"S7' {mode} gen {pop.generation_index} age {a:.3f} Myr: "
                  f"dlog L {dl:+.5f} dex, dlog Q_HI {dq:+.5f} dex")
            if abs(dl) > BAR_DEX or abs(dq) > BAR_DEX:
                bad.append((pop.generation_index, round(a, 3), dl, dq))
    assert not bad, bad
