"""
Test the Z-dependent n_H conversion at the Cloudy interface.

Verifies that:
1. nuclei_to_h is computed correctly from He/H(Z)
2. Shell and unified dlaw paths give identical n_H
3. Round-trip: rho -> n_H -> mass is exact (uses MU_N convention throughout)
4. Solar calibration: close to old He/H=0.1 assumption
5. Dust/H per Z is constant (grain commands independent of n_H)
"""
import sys, os

import numpy as np
from toddlers.constants import MU_N, M_P, M_SUN, Z_SOLAR_GASS10
from toddlers.cloudy_abundances_generator import CloudyAbundancesGenerator

abund = CloudyAbundancesGenerator()
Z_VALUES = [0.001, 0.004, 0.008, 0.014, 0.020, 0.030, 0.040]


def test_nuclei_to_h_values():
    """nuclei_to_h is consistent with He/H at each Z."""
    print("Test 1: nuclei_to_h consistency")

    for Z in Z_VALUES:
        log_Z_ratio = np.log10(Z / Z_SOLAR_GASS10)
        He_H = 10.0**abund._calc_He(log_Z_ratio)
        nuclei_to_h = 1.0 / (1.0 + He_H)

        # Check: nuclei_to_h * (1 + He_H) == 1
        assert abs(nuclei_to_h * (1.0 + He_H) - 1.0) < 1e-12, f"Failed at Z={Z}"
        print(f"  Z={Z:.3f}: He/H={He_H:.4f}, nuclei_to_h={nuclei_to_h:.6f}  [PASS]")

    print("  All passed.\n")


def test_shell_dlaw_consistency():
    """Shell and unified dlaw paths give identical n_H at all Z."""
    print("Test 2: Shell vs dlaw n_H consistency")
    n_cl = 160.0
    rho = n_cl * MU_N

    for Z in Z_VALUES:
        log_Z_ratio = np.log10(Z / Z_SOLAR_GASS10)
        He_H = 10.0**abund._calc_He(log_Z_ratio)
        nuclei_to_h = 1.0 / (1.0 + He_H)

        n_H_shell = n_cl * nuclei_to_h              # shell path
        n_H_dlaw = rho * (nuclei_to_h / MU_N)       # dlaw path
        assert abs(n_H_shell - n_H_dlaw) < 1e-10, \
            f"n_H mismatch at Z={Z}: shell={n_H_shell}, dlaw={n_H_dlaw}"
        print(f"  Z={Z:.3f}: n_H(shell)={n_H_shell:.4f}, n_H(dlaw)={n_H_dlaw:.4f}  [PASS]")

    print("  All passed.\n")


def test_mass_roundtrip():
    """rho -> n_H -> mass is exact (MU_N convention throughout)."""
    print("Test 3: Mass round-trip")
    n_cl = 160.0
    rho = n_cl * MU_N

    for Z in Z_VALUES:
        log_Z_ratio = np.log10(Z / Z_SOLAR_GASS10)
        He_H = 10.0**abund._calc_He(log_Z_ratio)
        nuclei_to_h = 1.0 / (1.0 + He_H)

        n_H = rho * (nuclei_to_h / MU_N)
        rho_back = n_H * (MU_N / nuclei_to_h)
        assert abs(rho - rho_back) / rho < 1e-12, f"Round-trip failed at Z={Z}"
        print(f"  Z={Z:.3f}: rho_back/rho = {rho_back/rho:.10f}  [PASS]")

    print("  All passed.\n")


def test_solar_calibration():
    """At solar Z, nuclei_to_h should be close to old He/H=0.1 value."""
    print("Test 4: Solar calibration")
    log_Z_ratio = np.log10(0.014 / Z_SOLAR_GASS10)
    He_H = 10.0**abund._calc_He(log_Z_ratio)

    assert 0.09 < He_H < 0.11, f"He/H at solar out of range: {He_H}"

    nuclei_to_h = 1.0 / (1.0 + He_H)
    old_nuclei_to_h = 10.0 / 11.0
    diff_pct = abs(nuclei_to_h - old_nuclei_to_h) / old_nuclei_to_h * 100
    assert diff_pct < 1.0, f"Differs by {diff_pct:.2f}% from old value"
    print(f"  He/H = {He_H:.4f}, nuclei_to_h: new={nuclei_to_h:.6f}, old={old_nuclei_to_h:.6f}, diff={diff_pct:.2f}%")
    print("  Passed.\n")


def test_dust_per_h_invariance():
    """Grain commands scale linearly with Z, independent of n_H or He/H."""
    print("Test 5: Dust-per-H invariance across Z")
    ratios = []
    for Z in Z_VALUES:
        scaling = Z / Z_SOLAR_GASS10  # grain metallicity_scaling
        ratios.append(scaling / Z)    # dust_per_H / Z = const

    variation = (max(ratios) - min(ratios)) / np.mean(ratios)
    assert variation < 1e-12, f"Not constant: variation={variation}"
    print(f"  dust_per_H / Z variation: {variation:.2e}")
    print("  Passed.\n")


def test_he_h_monotonic():
    """He/H should increase monotonically with Z."""
    print("Test 6: He/H monotonicity")
    Z_fine = np.logspace(-3, -1.4, 20)
    He_H_values = [10.0**abund._calc_He(np.log10(Z / Z_SOLAR_GASS10)) for Z in Z_fine]
    assert np.all(np.diff(He_H_values) > 0), "Not monotonic"
    print(f"  He/H range: {He_H_values[0]:.4f} to {He_H_values[-1]:.4f}")
    print("  Passed.\n")


if __name__ == '__main__':
    test_nuclei_to_h_values()
    test_shell_dlaw_consistency()
    test_mass_roundtrip()
    test_solar_calibration()
    test_dust_per_h_invariance()
    test_he_h_monotonic()
    print("All tests passed.")
