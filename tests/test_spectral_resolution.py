"""Wavelength-resolution reduction of the Cloudy star tables.

'decimate' (the default, which the released v2 tables used) keeps every factor-th point;
'average' replaces each block of factor points by its cell-weighted mean, which keeps the
integral of the spectrum. Averaged tables carry their own filename suffix (_resAvgN).
"""
import numpy as np
import pytest

from toddlers.cloudy_stellar_spectra_generator import reduce_resolution, IONIZATION_EDGES_ANGSTROM
from toddlers.utils import generate_spectral_table_filename


def _blocks(lam, factor):
    cuts = np.unique(np.concatenate(([0], np.searchsorted(lam, IONIZATION_EDGES_ANGSTROM), [lam.size])))
    return [slice(i, min(i + factor, c1)) for c0, c1 in zip(cuts[:-1], cuts[1:]) for i in range(c0, c1, factor)]


def _cells(lam):
    edges = np.concatenate(([lam[0]], 0.5 * (lam[1:] + lam[:-1]), [lam[-1]]))
    return np.diff(edges)


@pytest.fixture
def spectrum():
    rng = np.random.default_rng(3)
    lam = np.cumsum(rng.uniform(5.0, 20.0, 103)) + 100.0  # non-uniform grid across all three edges
    flux = rng.uniform(0.1, 5.0, (2, 4, lam.size))          # (Z, age, wavelength)
    return lam, flux


def test_decimate_is_plain_slicing(spectrum):
    lam, flux = spectrum
    l2, f2 = reduce_resolution(lam, flux, 10, 'decimate')
    assert np.array_equal(l2, lam[::10]) and np.array_equal(f2, flux[..., ::10])


def test_average_keeps_the_cell_integral(spectrum):
    lam, flux = spectrum
    l2, f2 = reduce_resolution(lam, flux, 10, 'average')
    w = _cells(lam)
    blocks = _blocks(lam, 10)
    assert l2.size == len(blocks)
    W = np.array([w[b].sum() for b in blocks])
    np.testing.assert_allclose((f2 * W).sum(axis=-1), (flux * w).sum(axis=-1), rtol=1e-12)
    np.testing.assert_allclose(l2, [np.sum(w[b] * lam[b]) / w[b].sum() for b in blocks], rtol=1e-12)


def test_factor_one_is_identity(spectrum):
    lam, flux = spectrum
    for mode in ('decimate', 'average'):
        l2, f2 = reduce_resolution(lam, flux, 1, mode)
        assert np.array_equal(l2, lam) and np.array_equal(f2, flux)


def test_unknown_mode_raises(spectrum):
    with pytest.raises(ValueError):
        reduce_resolution(*spectrum, 10, 'median')


def test_filename_suffix():
    assert generate_spectral_table_filename('BPASSv2.2.1', 'chab100', 'bin', wavelength_resolution_factor=10) \
        == 'BPASS_chab100_bin_burst_resFac10.ascii'
    assert generate_spectral_table_filename('BPASSv2.2.1', 'chab100', 'bin', wavelength_resolution_factor=10,
                                            resample='average') == 'BPASS_chab100_bin_burst_resAvg10.ascii'
    assert generate_spectral_table_filename('SB99', 'kroupa100', 'sin', resample='average') \
        == 'SB99_kroupa100_sin_burst.ascii'


@pytest.mark.parametrize("edge", IONIZATION_EDGES_ANGSTROM)
def test_average_never_mixes_across_an_edge(spectrum, edge):
    """A step spectrum (1 below the edge, 0 above) must stay a pure step after averaging:
    every output value is exactly 0 or 1, and each output point lies on its value's side."""
    lam, _ = spectrum
    assert lam[0] < edge < lam[-1]
    step = (lam < edge).astype(float)[None, None, :]
    l2, f2 = reduce_resolution(lam, step, 10, 'average')
    assert set(np.unique(f2)) <= {0.0, 1.0}
    assert np.all((l2 < edge) == (f2[0, 0] == 1.0))


def test_input_checks(spectrum):
    lam, flux = spectrum
    with pytest.raises(ValueError):
        reduce_resolution(lam[::-1], flux, 10, 'average')    # descending grid
    with pytest.raises(ValueError):
        generate_spectral_table_filename('BPASSv2.2.1', 'chab100', 'bin', wavelength_resolution_factor=10,
                                         resample='median')
