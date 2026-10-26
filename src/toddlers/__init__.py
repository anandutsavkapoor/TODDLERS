# Package metadata
__version__ = '2.1.0'
__author__ = 'Anand Utsav Kapoor'
__email__ = 'anandutsavkapoor@gmail.com'

# The feedback interpolants, built from the HDF5 feedback tables, are
# RegularGridInterpolator-backed Interp2DLinear objects (toddlers._interp_compat), which
# take the place of scipy.interpolate.interp2d (removed in SciPy 1.14) and work on any SciPy;
# tables in the earlier .obj format also load on any SciPy.

# numpy 2.0 renamed np.trapz -> np.trapezoid and later removed the old name. The package
# uses np.trapz in many places; restore it process-wide here (runs on any `import toddlers`)
# so the code works on both numpy <2 and >=2 without editing every call site.
import numpy as _np
if not hasattr(_np, "trapz") and hasattr(_np, "trapezoid"):
    _np.trapz = _np.trapezoid