"""A ``table star`` line requests log Z exactly as the star table stores it.

Every table TODDLERS writes stores log10 Z to 6 decimals. A request at full float precision
lies just outside a one-node table (log Z = -1.853871964... vs the node -1.853872 at
Z = 0.014), where Cloudy stops in InterpolateRectGrid ("not within the range -1.85 to
-1.85"); every stochastic generation table is such a table. Bars: the request text is the
node text for every Z node of every shipped feedback table, one-node and edge nodes
included, and both input writers use the one formatter.
"""
import glob
import inspect
import os
import types

import numpy as np
import pytest

import toddlers.cloudy_stellar_spectra_generator as csg
from toddlers.cloudy_base_input_generator import BaseCloudyInputGenerator
from toddlers.cloudy_dig_model_generator import DIGModelGenerator
from toddlers.feedback_tables import read_table_file

DB = os.path.join(os.path.dirname(csg.__file__), 'database')
SHIPPED = sorted(glob.glob(os.path.join(DB, '*_feedback.h5')))


def _node_texts(Z_values):
    return [line.split()[1] for line in csg._grid_lines([1e6], Z_values)]


@pytest.mark.parametrize('Z', [0.014, 0.0004, 0.02, 1e-5, 0.04])
def test_one_node_table_request_is_its_node(Z):
    node = _node_texts([Z])[0]
    assert csg.cloudy_logZ_text(Z) == node
    # a full-precision request lies off the node (outside a one-node table)
    assert f"{np.log10(Z)}" != node or float(node) == np.log10(Z)


@pytest.mark.parametrize('path', SHIPPED, ids=os.path.basename)
def test_request_is_a_node_of_every_shipped_table(path):
    Z = read_table_file(path).Z
    nodes = _node_texts(Z)
    lo, hi = min(map(float, nodes)), max(map(float, nodes))
    for z, node in zip(Z, nodes):
        req = csg.cloudy_logZ_text(z)
        assert req == node
        assert lo <= float(req) <= hi


def test_written_one_metallicity_table_accepts_the_request(tmp_path):
    t = tmp_path / 'one_Z_burst.ascii'
    csg.write_cloudy_star_table(str(t), [0.0, 1e6], [0.014], [100.0, 1000.0],
                                np.ones((1, 2, 2)), {'origin': 'generated', 'key': 'k'})
    nodes = {line.split()[1] for line in csg._read_grid_lines(str(t))}
    assert nodes == {csg.cloudy_logZ_text(0.014)}


def test_both_input_writers_use_the_formatter():
    dig = DIGModelGenerator.__new__(DIGModelGenerator)
    dig.base = types.SimpleNamespace(Z=0.014)
    assert dig.get_old_stellar_source().endswith(f"logZ={csg.cloudy_logZ_text(0.014)}")
    src = inspect.getsource(BaseCloudyInputGenerator)
    assert 'logZ={cloudy_logZ_text(self.Z)}' in src
    assert 'logZ={np.log10(' not in src
