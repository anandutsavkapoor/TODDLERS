"""Stellar feedback tables: one HDF5 schema for every template (TODDLERS 2.1).

A table holds, on a (Z, age) grid and per 1e6 Msun of its reference population, the same
quantities for every template: L_bol, L_ion, Q_HI, Q_HeI, Q_HeII, L_LyW and the mechanical
channels L_mech_wind, L_mech_SN, F_ram_wind, F_ram_SN, each with an astropy unit string; see
:mod:`toddlers.feedback_tables.table` for the layout.

Reading::

    from toddlers.feedback_tables import load_table
    tab = load_table('SB99', 'kroupa100')            # or ('BPASS', 'chab100', 'bin'), ('pySB99', 'kroupa_100')
    tab.get('Q_HI', [1.0, 3.0, 10.0], Z=0.02)       # astropy Quantity [1/s]
    tab.get('mean_ionizing_photon_energy_eV', 3.0, Z=0.02)
    tab.quantities, tab.missing, tab.imf, tab.metadata

``load_table(path)`` opens any table file, including the per-generation tables a stochastic
evolution run writes next to its output; :func:`load_run_generations` returns every
generation's table and star list of such a run.

Writing: :func:`build_feedback_table` / :func:`write_feedback_table`; source adapters in
:mod:`toddlers.feedback_tables.sb99`, :mod:`toddlers.feedback_tables.bpass`, and the pySB99
generator (:mod:`toddlers.pysb99.generate_pysb99_interpolants`). The TODDLERS 2.0 ``.obj``
tables are read (deprecated) by :mod:`toddlers.feedback_tables.legacy`, which also converts them.
"""
from .table import (SCHEMA, SCHEMA_VERSION, WRITER_VERSION, FLOOR, PER_MASS_MSUN, INLINE_STAR_LIMIT,
                    QUANTITIES, TOTALS, DERIVED, AXES, FIRST_QUERY_MYR, FeedbackTable, build_feedback_table,
                    write_feedback_table, write_table_file, read_table_file, load_table, find_table, check_table_names,
                    table_filename, resolve_names, broken_power_law, imf_mass_integral, check_imf_record,
                    time_grid_checks, quantity_units, units_equal, file_md5)
from .runs import load_run_generations

__all__ = ['SCHEMA', 'SCHEMA_VERSION', 'WRITER_VERSION', 'FLOOR', 'PER_MASS_MSUN', 'INLINE_STAR_LIMIT',
           'QUANTITIES', 'TOTALS', 'DERIVED', 'AXES', 'FIRST_QUERY_MYR', 'FeedbackTable',
           'build_feedback_table', 'write_feedback_table', 'write_table_file', 'read_table_file',
           'load_table', 'find_table', 'check_table_names', 'table_filename', 'resolve_names', 'broken_power_law',
           'imf_mass_integral', 'check_imf_record', 'time_grid_checks', 'quantity_units', 'units_equal',
           'file_md5', 'load_run_generations']
