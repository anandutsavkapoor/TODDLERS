"""The v1 line list is legacy: no code path can select it.

The v1 list comes from the first TODDLERS build and is kept for comparison only. Bars:
get_output_options(line_list_version='v1') raises ValueError saying the list is legacy;
no module in the package names the v1 file; one reference copy lives in docs/legacy/
(outside the package, with the md5 of the first build's list) and none under src/.
"""
import hashlib
from pathlib import Path

import pytest

from toddlers.cloudy_base_input_generator import BaseCloudyInputGenerator

ROOT = Path(__file__).resolve().parents[1]
V1_MD5 = 'f3a684e93385df7eac53b3956bf3db3e'  # the v1 list of the first TODDLERS build


def test_v1_option_refused():
    with pytest.raises(ValueError, match='legacy'):
        BaseCloudyInputGenerator.get_output_options(None, line_list_version='v1')


def test_no_package_code_names_the_v1_list():
    hits = [str(p.relative_to(ROOT)) for p in (ROOT / 'src').rglob('*.py')
            if 'TODDLERS_v1' in p.read_text(errors='replace')]
    assert hits == []


def test_one_reference_copy_outside_the_package():
    ref = ROOT / 'docs' / 'legacy' / 'cloudy_lines_TODDLERS_v1.dat'
    assert hashlib.md5(ref.read_bytes()).hexdigest() == V1_MD5
    assert list((ROOT / 'src').rglob('cloudy_lines_TODDLERS_v1*')) == []
