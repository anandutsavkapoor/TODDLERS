"""Resolution of TODDLERS data-file locations.

The package ships only small parameter/grid inputs. Large stellar-atmosphere and
track libraries are downloaded (see ``scripts/download_data.py``), and heavy
synthesized products (the single-star feedback database, SEDs, interpolants) are
built on first use. All of these live under a single data directory, resolved here.

Override the location by setting the ``TODDLERS_DATA`` environment variable;
otherwise it defaults to ``<package>/database``.
"""
import os

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))   # .../toddlers


def get_data_dir():
    """Directory holding downloaded/built data (``$TODDLERS_DATA`` if set and not empty)."""
    return os.environ.get("TODDLERS_DATA") or os.path.join(_PKG_DIR, "database")


def get_database_path():
    """Path to the single-star feedback database (``single_star_tracks.h5``)."""
    return os.path.join(get_data_dir(), "single_star_tracks.h5")


def get_pysb99_files_dir():
    """Directory holding the pySTARBURST99 track/atmosphere libraries."""
    return os.path.join(_PKG_DIR, "pysb99", "pySB99_files")


def data_resources():
    """The versioned data resources this code version downloads (data_resources.json next to
    this module): {name: {'version', 'kind', 'file', 'drive_id', 'md5', 'size_mb', 'install_to'}}.
    A new data version is a new Drive file and a new entry; released code keeps its own."""
    import json
    with open(os.path.join(_PKG_DIR, 'data_resources.json')) as f:
        return json.load(f)['resources']
