"""Vendored ``pts.storedtable`` (AGPL-3.0; see ../LICENSE.txt).

Trimmed to the StoredTable read/write API used by the TODDLERS STAB generators.
The ``conversionspec`` module (and its convert_* helpers, which pull in the heavy
PTS submodules) is intentionally omitted.
"""
from .io import (listStoredTableInfo, readStoredTable, writeStoredTable,
                 listStoredColumnsInfo, readStoredColumns, writeStoredColumns)
from .tokenizedfile import TokenizedFile
