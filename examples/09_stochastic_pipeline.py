"""Stochastic discrete sampling from the stellar database (needs the database).

For low-mass clusters the IMF is not fully sampled, so feedback is built from an
explicit draw of individual stars, each snapped to the discrete grid of evolutionary
tracks. This shows the discrete population and how to feed it into the interpolant
builder (which then plugs into the pySB99 template, exactly as in example 03).

Requires the single-star database (`python scripts/download_data.py`).
"""
import numpy as np

from toddlers.pysb99.stochastic.sampling import sample_imf_discrete, sample_ages
from toddlers._paths import get_database_path

DB = get_database_path()
TOTAL_MASS = 1e3        # Msun (under-sampled regime)
M_UPPER = 100.0

# Draw a discrete population: continuous IMF masses snapped to the track grid.
masses = sample_imf_discrete(total_mass=TOTAL_MASS, database_path=DB,
                             metallicity="MW", m_upper=M_UPPER,
                             imf_name="kroupa", seed=0)
ages = sample_ages(len(masses), t_sf_myr=0.0, mode="burst", seed=0)

print(f"sampled {len(masses)} stars, total {masses.sum():.0f} Msun "
      f"(target {TOTAL_MASS:.0f}), most massive {masses.max():.0f} Msun")
print(f"  distinct grid masses used: {len(np.unique(masses))} "
      f"(stars are snapped to the discrete track grid)")
print(f"  massive stars (m > 8 Msun): {(masses > 8).sum()}")

print("\nNext step (heavy: computes feedback per sampled star over the database):")
print("  from toddlers.pysb99.stochastic.interpolants import build_stochastic_interpolants_2d")
print("  interp = build_stochastic_interpolants_2d(masses, ages, DB, metallicities=['MW'])")
print("  save_stochastic_interpolants(interp, ...)   # then load via the pySB99 template")
