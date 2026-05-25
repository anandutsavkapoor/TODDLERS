"""Vendored minimal subset of PTS (the SKIRT Python Toolkit).

ONLY the pieces needed to read/write SKIRT StoredTable (``.stab``) files are
bundled here: ``pts.storedtable`` (verbatim ``io.py`` + ``tokenizedfile.py``) plus
tiny ``pts.utils.absPath`` and ``pts.simulation.unit`` shims that the storedtable
writer/reader call. The heavy PTS submodules (band, simulation, visual, …) and
their dependencies (e.g. lxml) are intentionally NOT included.

LICENSE: PTS is distributed under the GNU Affero General Public License v3
(AGPL-3.0); the original license text is preserved in ``LICENSE.txt`` in this
directory. This vendored subset therefore carries the AGPL-3.0 license,
independently of the MIT license of the rest of TODDLERS. Source:
https://github.com/SKIRT/PTS
"""
