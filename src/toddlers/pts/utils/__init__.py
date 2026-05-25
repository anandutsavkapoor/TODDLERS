"""Minimal vendored shim of ``pts.utils`` (AGPL-3.0; see ../LICENSE.txt).

Only ``absPath`` is provided -- the single utility the StoredTable I/O uses.
Reproduced from PTS ``pts/utils/path.py``.
"""
import pathlib


def absPath(inpath):
    """Absolute, canonical path for ``inpath`` (expanding ~ and resolving)."""
    path = pathlib.Path(inpath).expanduser()
    if not path.is_absolute():
        path = pathlib.Path.cwd() / path
    return path.resolve()
