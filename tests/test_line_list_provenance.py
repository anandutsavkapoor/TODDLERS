"""The shipped Cloudy line list is reproducible and matches what Cloudy writes back.

Two locks:

1. get_lines_list.py, run on the shipped Cloudy LineList_*.dat in a scratch directory,
   regenerates cloudy_lines_TODDLERS_v2.dat, _v2_with_comments, _v2_sorted and
   _v2_removed byte for byte.
2. The line keys the reader builds from the list (_read_line_list) equal the keys built
   from the header of a .cum file Cloudy wrote for that list (same parsing as
   get_line_luminosities; 148 labels, 147 keys after the documented mg 7 9.03094 skip).
   Fixtures are header lines only:
     shell_cloudy_4a6d38be_modified.cum_header:
       cloudy_output/template_BPASS/imf_chab100/star_type_bin/.../Z0.014_eta0.05_n160.0_logM6.00/
       shell_1.38.cum (file md5 48acc7c796f3719595869ce2572fcc63), Cloudy (master, 4a6d38be, modified)
     shell_cloudy_01c4cfc6_modified.cum_header: src/cloudy_output/
       template_SB99/imf_kroupa100/star_type_sin/.../Z0.014_eta0.05_n160.0_logM6.00/shell_0.10.cum
       (file md5 eca19e126e55f61e4efec7e701284535), Cloudy (master, 01c4cfc6, modified)
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import toddlers.cloudy_output_handler as coh

LISTS = Path(coh.__file__).resolve().parent / 'database' / 'lines_list'
FIXTURES = Path(__file__).resolve().parent / 'data' / 'cum_headers'
OUTPUTS = ['cloudy_lines_TODDLERS_v2.dat', 'cloudy_lines_TODDLERS_v2_with_comments.dat',
           'cloudy_lines_TODDLERS_v2_sorted.dat', 'cloudy_lines_TODDLERS_v2_removed.dat']


def test_list_regenerates_byte_for_byte(tmp_path):
    for f in ['get_lines_list.py', 'LineList_HII.dat', 'LineList_HeH.dat', 'LineList_PDR_H2.dat']:
        shutil.copy(LISTS / f, tmp_path / f)
    subprocess.run([sys.executable, 'get_lines_list.py'], cwd=tmp_path, check=True,
                   capture_output=True)
    for f in OUTPUTS:
        assert (tmp_path / f).read_bytes() == (LISTS / f).read_bytes(), f


@pytest.mark.parametrize('fixture', sorted(p.name for p in FIXTURES.glob('*.cum_header')))
def test_list_keys_equal_cum_header_keys(tmp_path, monkeypatch, fixture):
    header = (FIXTURES / fixture).read_text().rstrip('\n')
    n = len(header.split('\t')) - 1
    assert n == 148
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'shell_0.10.cum').write_text(header + '\n' + '\t'.join(['1.0'] * (n + 1)) + '\n')
    h = coh.CloudyOutputHandler.__new__(coh.CloudyOutputHandler)
    h.skip_lines = [('mg', 7, 9.03094)]
    h.model_prefix = 'shell'
    monkeypatch.setattr(h, 'get_file_path', lambda suffix: str(run / f'shell_0.10.{suffix}'),
                        raising=False)
    cum_keys = set(h.get_line_luminosities())
    list_keys = {(d['element'], d['ion'], h._parse_wavelength(d['wavelength']))
                 for d in h._read_line_list()}
    assert len(list_keys) == 147
    assert cum_keys == list_keys
