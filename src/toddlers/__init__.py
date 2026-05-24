# Package metadata
__version__ = '2.0'
__author__ = 'Anand Utsav Kapoor'
__email__ = 'anandutsavkapoor@gmail.com'

# The feedback interpolants are pickled scipy.interpolate.interp2d objects (deprecated
# in SciPy >= 1.10). Silence that specific deprecation here as an interim measure; the
# proper fix is to regenerate the interpolants with RegularGridInterpolator.
import warnings as _warnings
_warnings.filterwarnings("ignore", message=".*interp2d.*", category=DeprecationWarning)