"""Cloud density profiles (no data required).

Instantiates the available cloud density profiles at a fixed mass and mean density
and plots their radial density and enclosed-mass fraction. All profiles share the
same total mass and average density, differing only in concentration.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toddlers.cloud_density_profiles import (
    UniformDensity, BonnorEbertSphere, ModifiedBonnorEbertProfile,
)
from toddlers.constants import M_SUN, PC_TO_CM

M_CL = 1e6 * M_SUN
N_AVG = 160.0

profiles = {
    "uniform": UniformDensity(M_CL, N_AVG),
    "Bonnor-Ebert": BonnorEbertSphere(M_CL, N_AVG),
    "modified BE": ModifiedBonnorEbertProfile(M_CL, N_AVG, alpha=1.0),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.4))
for name, p in profiles.items():
    R_cl = p.get_parameters()["R_cl"]
    r = np.linspace(0.02 * R_cl, R_cl, 200)
    rho = np.array([p.density(ri) for ri in r])
    menc = np.array([p.mass_enclosed(ri) for ri in r]) / M_CL
    ax1.semilogy(r / PC_TO_CM, rho, label=name)
    ax2.plot(r / PC_TO_CM, menc, label=name)

ax1.set_xlabel("r [pc]"); ax1.set_ylabel(r"$\rho$ [g cm$^{-3}$]")
ax2.set_xlabel("r [pc]"); ax2.set_ylabel(r"$M(<r)/M_{\rm cl}$")
ax1.legend(frameon=False, fontsize=8)
fig.suptitle(f"Cloud density profiles  (M_cl=1e6 Msun, n=160 cm^-3)")
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "01_density_profiles.png")
fig.savefig(out, dpi=130)
print("wrote", out)
for name, p in profiles.items():
    print(f"  {name:14s}: R_cl = {p.get_parameters()['R_cl'] / PC_TO_CM:.1f} pc")
