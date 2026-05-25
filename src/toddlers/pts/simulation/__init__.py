"""Minimal vendored shim of ``pts.simulation`` (AGPL-3.0; see ../LICENSE.txt).

Only ``unit`` is provided -- the single function the StoredTable I/O uses to turn
a unit string into an astropy unit. Reproduced verbatim from PTS
``pts/simulation/units.py`` so that StoredTable units parse identically.
"""
import warnings

import astropy.units as u


def unit(unitlike):
    if isinstance(unitlike, str):
        # handle SKIRT-specific pressure equivalence: 1 K/m3 <==> k_B Pa
        if unitlike == "K/m3":
            unitlike = "1.3806488e-23 Pa"
        # parse string ignoring warnings about multiple divisions, as in "W/m2/sr"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=u.UnitsWarning)
            # astropy does not support multiple division for units starting with
            # "1/" or "/", as in "1/s/keV"; replace the leading "1" by an arbitrary
            # unit and then remove it again
            if unitlike.startswith("1/"):
                return u.Unit("A" + unitlike[1:]) / u.Unit("A")
            if unitlike.startswith("/"):
                return u.Unit("A" + unitlike) / u.Unit("A")
            # in other cases, directly use the astropy parser
            return u.Unit(unitlike)
    if isinstance(unitlike, u.Quantity):
        return unitlike.unit
    if isinstance(unitlike, u.UnitBase):
        return unitlike
    raise ValueError("Unsupported unit-like type: {}".format(type(unitlike)))
