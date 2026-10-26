"""S7: each generation's Cloudy star table pairs with that generation's feedback (data, slow).

Pass rule: for every generation tested, at the star-table age nodes nearest
1, 3 and 10 Myr,
    |log10(L_fb * (Q_HI/L)_table / Q_HI_fb)| <= 0.01 dex   and   |L_table / L_fb - 1| <= 1%,
where L_table = 1e20 * int F_lambda dlambda over the table, Q_HI,table = 1e20 * int
F_lambda lambda/(h c) dlambda below 911.76 A (table fluxes x 1e20 give erg/s/A), and L_fb,
Q_HI,fb are the generation's feedback (all stars >= track minimum, birth times included)
at the same age. Snap mode (the only mode the Cloudy stage accepts). Cases: a burst
generation and a constant-SFR (2 Myr) generation of a two-generation run, so a
generation-1 table is tested too, and the two generations' tables must differ.
"""
import os

import numpy as np
import pytest

from toddlers.constants import M_SUN, MYR_TO_SEC
from toddlers.pysb99.stochastic import population as P
from toddlers.stellar_feedback import StellarFeedback

pytestmark = [pytest.mark.data, pytest.mark.slow]

H_C_ERG_A = 6.62607015e-27 * 2.99792458e18     # h c in erg Angstrom
LYMAN_EDGE_A = 911.76


def read_star_table(path):
    with open(path) as f:
        lines = f.read().split('\n')
    n_models = int(lines[5])
    n_wave = int(lines[6])
    i = 11
    ages = np.array([float(lines[i + k].split()[0]) for k in range(n_models)]) / 1e6
    i += n_models
    vals = []
    while len(vals) < n_wave * (1 + n_models):
        vals.extend(float(x) for x in lines[i].split())
        i += 1
    vals = np.array(vals)
    wave = vals[:n_wave]
    flux = vals[n_wave:].reshape(n_models, n_wave)
    return ages, wave, flux


def _pairing(pop, spec, out_dir):
    name = P.ensure_star_table(pop, spec, output_dir=out_dir)
    ages, wave, flux = read_star_table(os.path.join(out_dir, name))
    tab = P.build_generation_table(pop, spec)
    lines = []
    for target in (1.0, 3.0, 10.0):
        k = int(np.argmin(np.abs(ages - target)))
        a = ages[k]
        f = flux[k] * 1e20
        L_tab = np.trapz(f, wave)
        ion = wave <= LYMAN_EDGE_A
        Q_tab = np.trapz(f[ion] * wave[ion] / H_C_ERG_A, wave[ion])
        L_fb = np.interp(a, tab.time_grid_myr, tab.linear_components['L_bol'])
        Q_fb = np.interp(a, tab.time_grid_myr, tab.linear_components['Q_HI'])
        dq = np.log10(L_fb * (Q_tab / L_tab) / Q_fb)
        dl = L_tab / L_fb - 1
        lines.append((pop.generation_index, a, dq, dl))
        print(f"S7 gen {pop.generation_index} {pop.sf_mode} age {a:.3f} Myr: "
              f"dlog(Q pairing) {dq:+.5f} dex, L_table/L_fb - 1 {dl:+.5f}")
    return name, lines


@pytest.mark.parametrize("mode", ["burst", "constant_sfr"])
def test_S7_star_table_pairs_with_feedback(mode, tmp_path):
    spec = P.StochasticSpec(P.IMFSpec('kroupa', 0.1, 100.0), 77, n_processes=8)
    kw = dict(mode=mode, formation_timescale=2 * MYR_TO_SEC) if mode == 'constant_sfr' else {}
    sf = StellarFeedback('pySB99', 0.014, 3e4 * M_SUN, 0.1, [5 * MYR_TO_SEC], stochastic=spec, **kw)
    names = []
    for pop in sf.populations:
        name, lines = _pairing(pop, spec, str(tmp_path))
        names.append(name)
        for g, a, dq, dl in lines:
            assert abs(dq) <= 0.01, (g, a, dq)
            assert abs(dl) <= 0.01, (g, a, dl)
    assert names[0] != names[1]
