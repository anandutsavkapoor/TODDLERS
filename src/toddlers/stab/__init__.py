"""TODDLERS STAB-building subpackage.

Reusable tooling to turn TODDLERS evolution + Cloudy outputs into SKIRT
StoredTable (``.stab``) libraries: interpolant generation, cloud-family and
SFR-normalized STAB writers, recollapse weighting, SED resampling and the
associated I/O helpers.

Population selection (template / IMF / star type) and the parameter-space grid
axes live in :mod:`toddlers.stab.config` and may be overridden via the
``TODDLERS_STAB_*`` environment variables. Submodules are imported explicitly
by callers (e.g. ``from toddlers.stab import interpolants``) to avoid pulling in
matplotlib/h5py at package-import time.
"""
