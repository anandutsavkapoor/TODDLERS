# Package metadata
__version__ = '2.0'
__author__ = 'Anand Utsav Kapoor'
__email__ = 'anandutsavkapoor@gmail.com'

# The feedback interpolants are pickled scipy.interpolate.interp2d objects (deprecated
# in SciPy >= 1.10). Silence that specific deprecation here as an interim measure; the
# proper fix is to regenerate the interpolants with RegularGridInterpolator.
import warnings as _warnings
_warnings.filterwarnings("ignore", message=".*interp2d.*", category=DeprecationWarning)

# numpy 2.0 renamed np.trapz -> np.trapezoid and later removed the old name. The package
# uses np.trapz in many places; restore it process-wide here (runs on any `import toddlers`)
# so the code works on both numpy <2 and >=2 without editing every call site.
import numpy as _np
if not hasattr(_np, "trapz") and hasattr(_np, "trapezoid"):
    _np.trapz = _np.trapezoid