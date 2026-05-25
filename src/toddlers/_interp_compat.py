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
        self._build()

    def _build(self):
        """(Re)construct the RegularGridInterpolator from the stored node arrays."""
        self._rgi = RegularGridInterpolator(
            (self._y, self._x), self._z,
            method="linear", bounds_error=False, fill_value=self._fill_value,
        )

    # Pickle only the raw node data, never the RegularGridInterpolator itself: scipy's
    # interpolator internals change between versions (e.g. the `_spline` attribute), so a
    # pickled RGI built under one scipy fails under another. Storing plain numpy arrays and
    # rebuilding the RGI on load makes the .obj libraries portable across scipy versions.
    def __getstate__(self):
        return {"_x": self._x, "_y": self._y, "_z": self._z, "_fill_value": self._fill_value}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build()

    def __call__(self, x, y):
        # Fast path for the scalar call (the hot path in the evolution loops): do the
        # bilinear blend directly and skip the RegularGridInterpolator / meshgrid overhead,
        # which is ~30-100x slower per call. Numerically identical to linear RGI / interp2d.
        if np.ndim(x) == 0 and np.ndim(y) == 0:
            return np.array([self._eval_scalar(float(x), float(y))])
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

    def _eval_scalar(self, xv, yv):
        """Bilinear value at a single (x, y); honors the clamp-to-edge / fill_value policy."""
        xg, yg, z = self._x, self._y, self._z   # z is (len(y), len(x))
        if self._fill_value is None:
            xv = xg[0] if xv < xg[0] else (xg[-1] if xv > xg[-1] else xv)
            yv = yg[0] if yv < yg[0] else (yg[-1] if yv > yg[-1] else yv)
        elif xv < xg[0] or xv > xg[-1] or yv < yg[0] or yv > yg[-1]:
            return float(self._fill_value)
        ix = min(max(int(np.searchsorted(xg, xv) - 1), 0), xg.size - 2)
        iy = min(max(int(np.searchsorted(yg, yv) - 1), 0), yg.size - 2)
        x0, x1 = xg[ix], xg[ix + 1]
        y0, y1 = yg[iy], yg[iy + 1]
        tx = 0.0 if x1 == x0 else (xv - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (yv - y0) / (y1 - y0)
        return (z[iy, ix]     * (1 - tx) * (1 - ty) + z[iy, ix + 1]     * tx * (1 - ty)
              + z[iy + 1, ix] * (1 - tx) * ty       + z[iy + 1, ix + 1] * tx * ty)
