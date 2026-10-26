"""Cloudy time grid: a grid point placed at the dissolution time keeps its exact value, so the
model choice there (t >= dissolution time -> dissolved) does not depend on round-off."""
import numpy as np
import pytest

from toddlers.cloudy_timegrid_generator import TimeGridGenerator
from toddlers.constants import MYR_TO_SEC

T_DISS = 895694380195746.6          # 28.383 Myr; rounding it to 2 decimals (in s) lands below it


@pytest.mark.parametrize('n_points', [4, 6, 10])
def test_uniform_grid_keeps_dissolution_time_exact(n_points):
    gen = TimeGridGenerator(t_start=0.1 * MYR_TO_SEC, dissolution_time=T_DISS)
    for cont in (True, False):
        grid = gen.generate_grid(method='uniform', n_points=n_points, continue_after_dissolution=cont)
        assert T_DISS in grid
        assert np.sum(grid < T_DISS) == (n_points // 2 - 1 if cont else n_points - 1)
