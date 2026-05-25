"""Future-proof, picklable replacement for ``scipy.interpolate.interp2d``.

``interp2d`` was deprecated in SciPy 1.10 and **removed in SciPy 1.14**, which broke the
shipped feedback interpolants (pickled ``interp2d`` objects) on any modern SciPy. This
module provides :class:`Interp2DLinear`, a small picklable callable backed by
:class:`scipy.interpolate.RegularGridInterpolator` that reproduces ``interp2d``'s linear
call semantics exactly:

* constructed as ``Interp2DLinear(x, y, z)`` with 1-D ``x`` (e.g. time), 1-D ``y`` (e.g.
  metallicity), and ``z`` of shape ``(len(y), len(x))`` -- the same argument order and
  ``z`` layout as ``interp2d(x, y, z)``;
* called as ``f(x, y)`` returning values on the grid with shape ``(len(y), len(x))``,
  and a length-1 array for scalar inputs (matching ``interp2d``);
* linear ``interp2d`` (a ``RectBivariateSpline`` with ``kx=ky=1, s=0``) is exactly
  bilinear interpolation, i.e. identical to a linear ``RegularGridInterpolator``, so the
  replacement is numerically equivalent (exact at nodes).

It is used both for newly built interpolants and to regenerate the shipped ``.obj`` feedback
libraries from the nodes recovered out of the legacy ``interp2d`` pickles. Because instances
are plain attributes plus one ``RegularGridInterpolator``, they pickle and unpickle on any
SciPy version with no ``interp2d`` dependency.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class Interp2DLinear:
    """Drop-in linear replacement for ``scipy.interpolate.interp2d`` (see module docstring)."""

    def __init__(self, x, y, z, fill_value=None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        if z.shape != (y.size, x.size):
            raise ValueError(
                f"z must have shape (len(y), len(x)) = ({y.size}, {x.size}); got {z.shape}"
            )
        # RegularGridInterpolator needs strictly-ascending axes.
        xo = np.argsort(x, kind="stable")
        yo = np.argsort(y, kind="stable")
        self._x = x[xo]
        self._y = y[yo]
        self._z = z[np.ix_(yo, xo)]
        # Out-of-grid policy. Default (fill_value=None) is CLAMP-TO-EDGE: queries are
        # clipped to the grid bounds so the boundary value is held, never extrapolated --
        # the safe choice for the log feedback quantities stored here (linear extrapolation
        # could produce absurd luminosities/photon rates). In practice queries are in-range
        # (Z is validated upstream, the time grid spans the full evolution), so this only
        # guards the edges. Pass a numeric fill_value to return that constant outside instead.
        self._fill_value = fill_value
        self._rgi = RegularGridInterpolator(
            (self._y, self._x), self._z,
            method="linear", bounds_error=False, fill_value=fill_value,
        )

    def __call__(self, x, y):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        # interp2d sorts its inputs and returns values on the sorted grid.
        xi = np.sort(x)
        yi = np.sort(y)
        if getattr(self, "_fill_value", None) is None:
            # clamp to grid edges (hold the boundary value) rather than extrapolate
            xi = np.clip(xi, self._x[0], self._x[-1])
            yi = np.clip(yi, self._y[0], self._y[-1])
        Y, X = np.meshgrid(yi, xi, indexing="ij")  # both (len(y), len(x))
        pts = np.column_stack([Y.ravel(), X.ravel()])
        out = self._rgi(pts).reshape(yi.size, xi.size)
        # interp2d squeezes singleton dimensions; a scalar call returns shape (1,).
        return out.ravel() if out.size == 1 else np.squeeze(out)
