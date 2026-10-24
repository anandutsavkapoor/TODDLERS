"""pySB99 metadata parsing: current and legacy keys give the same population."""
from toddlers.cloudy_stellar_spectra_generator import parse_pysb99_population_metadata


def test_current_keys_imf():
    meta = {'population_type': 'imf', 'imf_exponents': [1.3, 2.0], 'imf_mass_limits': [0.1, 0.5, 100.0]}
    assert parse_pysb99_population_metadata(meta) == ('imf', None, [1.3, 2.0], (0.1, 0.5, 100.0))


def test_legacy_keys_topheavy_not_kroupa():
    meta = {'imf_type': 'power_law', 'exponents': [1.3, 2.0], 'mass_limits': [0.1, 0.5, 100.0]}
    pop_type, custom, exps, limits = parse_pysb99_population_metadata(meta)
    assert pop_type == 'imf' and custom is None
    assert exps == [1.3, 2.0] and limits == (0.1, 0.5, 100.0)


def test_legacy_custom_population():
    meta = {'imf_type': 'custom', 'custom_star_numbers': {'20.0': 5, '40.0': 2}}
    assert parse_pysb99_population_metadata(meta) == ('custom', {20.0: 5, 40.0: 2}, None, None)


def test_custom_numbers_without_type_key():
    meta = {'custom_star_numbers': {'80.0': 1}}
    assert parse_pysb99_population_metadata(meta)[0] == 'custom'
