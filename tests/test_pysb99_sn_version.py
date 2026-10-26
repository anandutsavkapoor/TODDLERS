"""pySB99 SN metadata names the PCHIP death-time rule.

The SN death time and final mass between the Geneva tracks are interpolated by monotone
PCHIP in log-log, and the SN metadata names that rule, so products built with different
death-time rules are told apart. Bar: the generator's supernova_metadata records
pysb99_sn_version 2 and sn_death_time_interpolation 'pchip_loglog' (with SNe), and
product_includes_supernovae still reads it. No code branches on the version (searched:
src, tests, examples).
"""
from toddlers.pysb99.generate_pysb99_interpolants import (product_includes_supernovae,
                                                          supernova_metadata)


def test_sn_metadata_version_2():
    md = supernova_metadata(True)
    assert md['pysb99_sn_version'] == 2
    assert md['sn_death_time_interpolation'] == 'pchip_loglog'
    assert product_includes_supernovae(md)
    off = supernova_metadata(False)
    assert off['pysb99_sn_version'] == 2 and not product_includes_supernovae(off)
    assert 'sn_death_time_interpolation' not in off
