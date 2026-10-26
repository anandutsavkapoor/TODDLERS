"""SB99 and BPASS adapters of toddlers.feedback_tables on tiny synthetic source files.

The reproduction of the shipped tables from the real source data needs the
full Starburst99 runs and the BPASS release and is run outside the test suite.
"""
import json
import os

import numpy as np
import pytest

from toddlers.feedback_tables import sb99, bpass, read_table_file

T_YR = np.arange(0.01, 35.0, 0.1) * 1e6  # 350 nodes, as the shipped SB99 runs


def _sb99_run(d, model, z):
    os.makedirs(d, exist_ok=True)
    n = T_YR.size
    rng = np.random.default_rng(int(z * 1e4))
    pw = np.round(39.7 + 0.3 * rng.random(n), 3)
    psn = np.where(T_YR > 3.5e6, np.round(40.1 + 0.1 * rng.random(n), 3), -30.0)
    tot = np.round(np.log10(10 ** pw + 10 ** psn), 3)
    head = " MODEL DESIGNATION: x\n MODEL GENERATED: y\n\n  RESULTS\n\n  HEADER\n"

    def table(cols):
        return "".join("  ".join(f"{c:.5E}" if j == 0 else f"{c:.3f}" for j, c in enumerate(row)) + "\n"
                       for row in np.column_stack(cols))

    with open(os.path.join(d, f"{model}.power1"), 'w') as f:
        f.write(head + "    TIME   ALL ...\n" + table([T_YR, pw] + [pw] * 5 + [pw - 8] * 5))
    with open(os.path.join(d, f"{model}.snr1"), 'w') as f:
        z0 = np.zeros(n)
        f.write(head + "    TIME   TOTAL ...\n" + table([T_YR, z0, psn, z0, z0, z0, z0, z0, z0, tot, z0]))
    with open(os.path.join(d, f"{model}.yield1"), 'w') as f:
        mw = np.full(n, -2.5)
        msn = np.where(T_YR > 3.5e6, -1.5, -30.0)
        f.write(head + "    TIME   H ...\n" + table([T_YR] + [mw] * 9 + [mw, msn, mw, mw]))
    with open(os.path.join(d, f"{model}.quanta1"), 'w') as f:
        f.write(head + "   TIME   H I\n              [s^-1]  % OF L\n"
                + table([T_YR, np.full(n, 52.5), np.full(n, -0.4), np.full(n, 51.9), np.full(n, -0.9),
                         np.full(n, 48.6), np.full(n, -3.9), np.full(n, 42.5)]))
    lam = np.array([900.0, 995.0, 1000.0, 1005.0, 1100.0])
    with open(os.path.join(d, f"{model}.spectrum1"), 'w') as f:
        f.write(head + " TIME [YR]    WAVELENGTH [A] ...\n")
        for t in T_YR:
            for k, l in enumerate(lam):
                f.write(f" {t:.5E}  {l:.2f}  {38 + k * 0.01:.5f}  {38 + k * 0.01:.5f}  -15.00000\n")
    with open(os.path.join(d, f"{model}.input1"), 'w') as f:
        f.write("MODEL DESIGNATION:    [NAME]\nx\nFIXED MASS (<=0): [ISF]\n-1\nTOTAL STELLAR MASS: [TOMA]\n1.000\n"
                "IMF EXPONENTS (KROUPA=1.3,2.3):     [XPONENT]\n1.300,2.300\n"
                "MASS BOUNDARIES: [XMASLIM]\n0.100,0.500,100.000\nMETALLICITY + TRACKS: [IZ]\n"
                "GENEVA STD: 11=0.001\nGENEVA HIGH:21=0.001\n24\n")
    return pw, psn, tot


def test_sb99_adapter(tmp_path):
    runs, src = {}, {}
    for z in (0.004, 0.02):
        d = str(tmp_path / f"Z{z}")
        src[z] = _sb99_run(d, 'm', z)
        runs[z] = (d, 'm')
    tab = read_table_file(sb99.build_sb99_table(runs, 'testimf', str(tmp_path / 'out')).path)
    assert os.path.basename(tab.path) == 'SB99_testimf_feedback.h5'
    for i, z in enumerate((0.004, 0.02)):
        pw, psn, tot = src[z]
        np.testing.assert_allclose(tab.log_nodes('L_mech')[i], tot, rtol=1e-12)     # published total kept
        mdot = lambda x: 10 ** x * (sb99.MSUN_G / sb99.YR_S)
        F = np.sqrt(2 * 10 ** pw * mdot(-2.5)) + np.sqrt(
            2 * 10 ** psn * mdot(np.where(T_YR > 3.5e6, -1.5, -30.0)))
        np.testing.assert_allclose(tab.log_nodes('F_ram')[i], np.log10(F), rtol=1e-12)
        np.testing.assert_allclose(tab.log_nodes('L_ion')[i], np.log10(10 ** -0.4 * 10 ** 42.5), rtol=1e-12)
        np.testing.assert_allclose(tab.values['Q_HeI'][i], 10 ** 51.9, rtol=1e-12)
        np.testing.assert_allclose(tab.values['Q_HeII'][i], 10 ** 48.6, rtol=1e-12)
    np.testing.assert_allclose(tab.log_nodes('L_LyW')[0], np.log10(np.mean(10 ** np.array([38.01, 38.02, 38.03]))),
                               rtol=1e-12)
    meta = tab.metadata
    assert meta['legacy'] is True and tab.missing == {}
    assert meta['source']['run_settings']['0.02']['IZ'] == '24'
    d = tab.imf['definition']
    assert d['exponents'] == [1.3, 2.3] and d['mass_limits_msun'] == [0.1, 0.5, 100.0]
    assert any(k.endswith('.input1') for k in d['source']['input_files_md5'])
    assert tab.imf['normalization_mass_msun'] == 1e6
    assert 0 < meta['conventions']['max_abs_log10_components_over_published_total'] < 0.0025


def _bpass_release(d, star, imf):
    os.makedirs(d, exist_ok=True)
    la = np.arange(6.0, 11.1, 0.1)
    lam = np.arange(1, 2001, dtype=float)
    for i, zs in enumerate(bpass.ZFILE):
        rng = np.random.default_rng(i)
        y = np.column_stack([la] + [1e-3 * (1 + rng.random(la.size))] * 3
                            + [np.where(la < 9, 1e20, 0.0), np.where(la > 6.5, 1e34, 0.0)]
                            + [np.where(la > 6.5, 1e-2, 0.0)] * 3)
        np.savetxt(os.path.join(d, f"yields-{star}-imf_{imf}.z{zs}.dat"), y)
        ion = np.column_stack([la, 53 - 0.5 * (la - 6), np.ones_like(la), np.ones_like(la), np.ones_like(la)])
        np.savetxt(os.path.join(d, f"ionizing-{star}-imf_{imf}.z{zs}.dat"), ion)
        spec = np.column_stack([lam] + [1e3 * (1 + 0.1 * k) * (lam / 1000.0) ** 2 for k in range(51)])
        np.savetxt(os.path.join(d, f"spectra-{star}-imf_{imf}.z{zs}.dat"), spec)


def _bpass_ionizing_matching_spectrum(d, star, imf):
    """Rewrite the ionizing files with Q_HI = the spectrum photon integral up to and including the 912 A bin."""
    for zs in bpass.ZFILE:
        s = np.loadtxt(os.path.join(d, f"spectra-{star}-imf_{imf}.z{zs}.dat"))
        lam, spec = s[:, 0], s[:, 1:].T
        q = np.sum(bpass.LSUN * spec[:, lam <= 912] * lam[lam <= 912] * 1e-8 / (bpass.H_PLANCK * bpass.C_LIGHT), axis=1)
        p = os.path.join(d, f"ionizing-{star}-imf_{imf}.z{zs}.dat")
        ion = np.loadtxt(p)
        ion[:, 1] = np.log10(q)
        np.savetxt(p, ion)


@pytest.mark.parametrize("matching", [False, True])
def test_bpass_adapter(tmp_path, matching):
    d = str(tmp_path / 'rel')
    _bpass_release(d, 'bin', 'chab100')
    if matching:
        _bpass_ionizing_matching_spectrum(d, 'bin', 'chab100')
    tab = read_table_file(bpass.build_bpass_table(d, 'chab100', 'bin', str(tmp_path / 'out')).path)
    assert os.path.basename(tab.path) == 'BPASS_chab100_bin_feedback.h5'
    assert tab.t.size == bpass.T_MYR.size
    cc = tab.metadata['conventions']['Q_HI_crosscheck']
    if matching:     # cross-check passes: He rates stored
        assert cc['pass'] and cc['worst_abs_ratio_minus_1'] < 1e-10 and tab.missing == {}
        assert 'hardness_HeI' in tab.quantities
    else:            # synthetic spectrum and ionizing file disagree: He rates missing, reason given
        assert not cc['pass'] and set(tab.missing) == {'Q_HeI', 'Q_HeII'}
        assert 'Q_HI' in tab.missing['Q_HeI']
    # at a source node (log age 7.0 = 10 Myr), the total is the source total
    j = int(np.argmin(np.abs(bpass.T_MYR - 10.0)))
    la, z = np.log10(bpass.T_MYR[j]) + 6, 0
    y = np.loadtxt(os.path.join(d, 'yields-bin-imf_chab100.zem5.dat'))
    k = int(np.argmin(np.abs(y[:, 0] - 7.0)))
    E = bpass.JYR2ERGS * (y[k, 5] + bpass.WIND_KG * y[k, 4])
    expect = np.interp(la, y[:, 0], np.log10(np.where(
        bpass.JYR2ERGS * (y[:, 5] + bpass.WIND_KG * y[:, 4]) == 0, 1e-16,
        bpass.JYR2ERGS * (y[:, 5] + bpass.WIND_KG * y[:, 4]))))
    assert tab.log_nodes('L_mech')[z, j] == pytest.approx(expect, rel=1e-12)
    assert abs(tab.log_nodes('L_mech')[z, j] - np.log10(E)) < 0.05
    assert tab.metadata['conventions']['wind_energy_factor_kg'] == 2e30
    d_imf = tab.imf['definition']
    assert d_imf['kind'] == 'read_from_source' and d_imf['m_up_msun'] == 100.0
    assert len(d_imf['source_files_md5']) == 3 * len(bpass.ZFILE)
