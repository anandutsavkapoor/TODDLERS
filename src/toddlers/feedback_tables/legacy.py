"""Deprecated read-only access to the TODDLERS 2.0 feedback tables (pickled ``.obj``), and a
converter to the unified HDF5 table.

2.0 layout, per (template, imf[, star_type]):

``<prefix>.obj``
    list of 7 Interp2DLinear(t [Myr], Z): log10 L_mech [erg/s], log10 F_ram [dyn],
    log10 L_bol [erg/s], log10 L_ion [erg/s], log10 Q_HI [1/s], dF_ram/dt, dL_mech/dt.
``<prefix>_LumLymanWerner.obj``
    Interp2DLinear of log10 L_LyW [erg/s/A].

with ``<prefix>`` = ``BPASSinterpolation_<star_type>_<imf>``, ``SB99interpolation_<imf>``,
``pySB99interpolation_<imf>`` or ``<template>interpolation_[<star_type>_]<imf>``.

The loader never writes. 2.1.0 ships HDF5 tables only; this path serves users' own 2.0 files.
"""
import os
import pickle
import warnings

import numpy as np

LEGACY_ENTRIES = ['L_mech', 'F_ram', 'L_bolo', 'L_ion', 'Q_ion', 'F_ram_dot', 'L_mech_dot']


def legacy_files(template, imf, star_type=None, data_dir=None):
    """Paths of a 2.0 table: (main .obj, Lyman-Werner .obj)."""
    from .._paths import get_data_dir
    if template.startswith('BPASS'):
        prefix = f"BPASSinterpolation_{star_type}_{imf}"
    elif template.startswith('SB99'):
        prefix = f"SB99interpolation_{imf}"
    elif template == 'pySB99':
        prefix = f"pySB99interpolation_{imf}"
    elif template and imf:
        prefix = f"{template}interpolation_{star_type}_{imf}" if star_type else f"{template}interpolation_{imf}"
    else:
        raise ValueError(f"Unsupported template: {template}")
    d = data_dir or get_data_dir()
    return os.path.join(d, f"{prefix}.obj"), os.path.join(d, f"{prefix}_LumLymanWerner.obj")


def legacy_md5(template, imf, star_type=None):
    """The 2.0 fingerprint: md5 over the bytes of the two files (kept for 2.0 tables so a
    user's cached 2.0 tracks stay valid while they keep their .obj tables)."""
    import hashlib
    h = hashlib.md5()
    for path in legacy_files(template, imf, star_type):
        with open(path, 'rb') as f:
            h.update(f.read())
    return h.hexdigest()


def load_legacy_obj(template, imf, star_type=None, warn=True):
    """The 2.0 interpolants: dict entry name -> Interp2DLinear (log10 values; the two
    derivative entries are returned but unused), plus 'L_LyW'. Emits a FutureWarning: the
    deprecation is addressed to users, and Python hides a DeprecationWarning raised inside a
    package under its default filters."""
    main_file, lw_file = legacy_files(template, imf, star_type)
    if warn:
        warnings.warn(f"{os.path.basename(main_file)}: TODDLERS 2.0 .obj feedback tables are deprecated "
                      f"(read-only). Convert with toddlers.feedback_tables.legacy.convert_legacy_obj.",
                      FutureWarning, stacklevel=3)
    with open(main_file, 'rb') as f:
        obj = pickle.load(f)
    with open(lw_file, 'rb') as f:
        lyw = pickle.load(f)
    out = dict(zip(LEGACY_ENTRIES, obj[:7]))
    out['L_LyW'] = lyw
    return out


def legacy_nodes(template, imf, star_type=None):
    """(t, Z, dict name -> (n_Z, n_t) log10 nodes) of entries 0-4 and the LyW file."""
    o = load_legacy_obj(template, imf, star_type, warn=False)
    t, Z = np.asarray(o['L_bolo']._x, float), np.asarray(o['L_bolo']._y, float)
    nodes = {}
    for k in ('L_mech', 'F_ram', 'L_bolo', 'L_ion', 'Q_ion', 'L_LyW'):
        it = o[k]
        if not (np.array_equal(it._x, t) and np.array_equal(it._y, Z)):
            raise ValueError(f"{k}: axes differ from L_bolo's; cannot convert")
        nodes[k] = np.asarray(it._z, float)
    return t, Z, nodes


def convert_legacy_obj(template, imf, star_type=None, out_dir=None, *, metadata):
    """
    Convert a 2.0 table to the unified HDF5 table.

    The 2.0 files hold the mechanical totals only, so they become one channel each,
    ``L_mech_unsplit`` and ``F_ram_unsplit`` (marked in ``conventions``); the wind and SN
    components and the He-ionizing rates Q_HeI, Q_HeII are stored as missing. Node values are
    10**(2.0 log10 nodes); the time-derivative entries are not converted (not stored in 2.1).

    Args:
        metadata (dict): ``imf_definition`` and ``normalization`` of the population (the 2.0
            files do not record them), optionally ``source`` and ``conventions`` entries.
    Returns:
        FeedbackTable (written to out_dir/<template>_<imf>[_<star_type>]_feedback.h5).
    """
    from .table import write_feedback_table
    t, Z, nodes = legacy_nodes(template, imf, star_type)
    main_file, lw_file = legacy_files(template, imf, star_type)
    from .table import file_md5
    q = {'L_bol': 10 ** nodes['L_bolo'], 'L_ion': 10 ** nodes['L_ion'], 'Q_HI': 10 ** nodes['Q_ion'],
         'L_LyW': 10 ** nodes['L_LyW'], 'L_mech_unsplit': 10 ** nodes['L_mech'],
         'F_ram_unsplit': 10 ** nodes['F_ram']}
    reason = 'not in the 2.0 table (mechanical total only)'
    missing = {'L_mech_wind': reason, 'L_mech_SN': reason, 'F_ram_wind': reason, 'F_ram_SN': reason,
               'Q_HeI': 'not in the 2.0 table', 'Q_HeII': 'not in the 2.0 table'}
    family = 'BPASS' if template.startswith('BPASS') else ('SB99' if template.startswith('SB99') else template)
    meta = {
        'template': family, 'imf': imf, 'star_type': star_type if family not in ('SB99', 'pySB99') else None,
        'normalization': metadata['normalization'], 'imf_definition': metadata['imf_definition'],
        'source': {'code': 'convert_legacy_obj', 'legacy_files_md5': {os.path.basename(main_file): file_md5(main_file),
                                                                       os.path.basename(lw_file): file_md5(lw_file)},
                   **metadata.get('source', {})},
        'conventions': {'mechanical_channels': 'unsplit: the 2.0 table stores the wind + SN total only',
                        **metadata.get('conventions', {})},
    }
    out_dir = out_dir or os.path.dirname(main_file)
    return write_feedback_table(t, Z, q, meta, out_dir, missing=missing, legacy=True)
