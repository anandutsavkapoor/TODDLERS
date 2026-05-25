"""One-time migration: convert the shipped feedback ``.obj`` libraries from legacy
``scipy.interpolate.interp2d`` to the future-proof :class:`toddlers._interp_compat.Interp2DLinear`.

``interp2d`` was removed in SciPy 1.14, so the pickled interp2d objects are not callable on
modern SciPy. This script must therefore be run on an environment where interp2d still works
(SciPy < 1.14). For each ``.obj`` in the data directory it:

  1. loads the (possibly nested) pickle,
  2. for every ``interp2d`` found, recovers its node grid (``unique(x)``, ``unique(y)``) and
     the node values by evaluating the interp2d on its own nodes (linear interp2d is exact at
     nodes, so this is lossless),
  3. builds an ``Interp2DLinear`` with the same ``(x, y, z)`` layout,
  4. verifies the replacement against the original at random in-range points,
  5. backs up the original (outside the repo) and rewrites the ``.obj`` with the new objects.

Usage:  python scripts/regenerate_interp2d.py [--check-only]
"""
import argparse
import glob
import os
import pickle
import shutil

import numpy as np
from scipy.interpolate import interp2d   # requires SciPy < 1.14

from toddlers._interp_compat import Interp2DLinear
from toddlers._paths import get_data_dir

BACKUP_DIR = "/tmp/toddlers_interp2d_backups"
_rng = np.random.default_rng(0)


def _convert(obj):
    """Recursively replace interp2d with Interp2DLinear; return (new_obj, max_rel_err)."""
    if isinstance(obj, interp2d):
        xs = np.unique(np.asarray(obj.x, dtype=float))
        ys = np.unique(np.asarray(obj.y, dtype=float))
        z = np.asarray(obj(xs, ys), dtype=float)          # (len(ys), len(xs)), exact at nodes
        new = Interp2DLinear(xs, ys, z)
        # verify at random in-range points (no extrapolation)
        tt = _rng.uniform(xs.min(), xs.max(), 300)
        zz = _rng.uniform(ys.min(), ys.max(), 8)
        err = 0.0
        for t in tt:
            for Z in zz:
                a = float(np.ravel(obj(t, Z))[0])
                b = float(np.ravel(new(t, Z))[0])
                err = max(err, abs(a - b) / (abs(a) + 1e-300))
        return new, err
    if isinstance(obj, (list, tuple)):
        outs, me = [], 0.0
        for e in obj:
            c, err = _convert(e)
            outs.append(c)
            me = max(me, err)
        return (type(obj)(outs), me)
    return obj, 0.0   # leave non-interp2d objects untouched


def main():
    ap = argparse.ArgumentParser(description="Migrate feedback .obj interp2d -> Interp2DLinear")
    ap.add_argument("--check-only", action="store_true",
                    help="report conversion + max error but do not overwrite the .obj")
    args = ap.parse_args()

    data_dir = get_data_dir()
    objs = sorted(glob.glob(os.path.join(data_dir, "*.obj")))
    if not objs:
        print(f"No .obj files in {data_dir}")
        return
    os.makedirs(BACKUP_DIR, exist_ok=True)

    for path in objs:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        new, err = _convert(obj)
        name = os.path.basename(path)
        if err == 0.0 and new is obj:
            print(f"{name:48s} no interp2d found, skipped")
            continue
        print(f"{name:48s} converted, max rel err = {err:.2e}")
        if args.check_only:
            continue
        shutil.copy2(path, os.path.join(BACKUP_DIR, name + ".interp2d.bak"))
        with open(path, "wb") as f:
            pickle.dump(new, f)

    print("done" + (" (check-only)" if args.check_only else f"; backups in {BACKUP_DIR}"))


if __name__ == "__main__":
    main()
