"""The evolution resolves every supernova point event.

A point event reaches the evolution as a hat one precalculation step wide either side of
its node (StellarFeedback, PRECALC_DT_MYR = 0.025 Myr). The ODE solver resolves it only if
its maximum steps stay below that step; the configured bounds are 3-5 kyr. Bar: every maximum
step is at most a fifth of the precalculation step.
"""
from toddlers import constants as C
from toddlers.stellar_feedback import PRECALC_DT_MYR


def test_max_solver_steps_resolve_the_precalc_step():
    step = PRECALC_DT_MYR * C.MYR_TO_SEC
    for name in ('MAX_STEP_PH1', 'MAX_STEP_FRAG', 'MAX_STEP_PH2'):
        assert getattr(C, name) <= step / 5, name
