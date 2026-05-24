"""Fast precomputed feedback queries (needs feedback data).

End users do not re-run the dynamics for every evaluation: feedback is read from the
precomputed interpolation tables. This makes repeated evaluation cheap, which matters
when TODDLERS is called many times (e.g. inside an MCMC loop). Here we load a
population's feedback interpolants and time a large number of queries.

Requires: `python scripts/download_data.py`.
"""
import time

import numpy as np

from toddlers.stellar_feedback import StellarFeedback
from toddlers.constants import M_SUN, MYR_TO_SEC

sf = StellarFeedback(template="SB99", imf="kroupa100", star_type="sin",
                     Z=0.02, M_cl_init=1e6 * M_SUN, eta_sf=0.05, t_list_collapse=[])

ages = np.random.uniform(0.1, 20.0, 100000) * MYR_TO_SEC
t0 = time.time()
q = np.array([sf.get_ionizing_photon_rate(t) for t in ages])
dt = time.time() - t0

print(f"queried Q(H) at {len(ages):,} ages in {dt:.3f} s "
      f"({len(ages) / dt:,.0f} queries/s, {1e6 * dt / len(ages):.2f} us/query)")
print(f"  Q(H) range: {q.min():.2e} - {q.max():.2e} s^-1")
print("A few thousand MCMC evaluations therefore cost well under a second of lookup.")
