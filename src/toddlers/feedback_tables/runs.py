"""Per-generation feedback tables and star lists of a stochastic evolution run."""
import json
import os


def load_run_generations(dat_path, database_path=None):
    """
    Every generation of a stochastic evolution run: its feedback table and its stars.

    The table is read from ``<output stem>_gen<i>_feedback.h5`` when the run saved it
    (Evolution(save_generation_tables=True), the default) and its star list matches the run's;
    otherwise it is rebuilt from the saved star list, which is first checked against its md5
    and against a fresh draw from the recorded seed. Both give the same ``get`` values.

    Args:
        dat_path (str): the evolution output (.dat).
        database_path (str, optional): single-star database for a rebuild (default: the path
            recorded in the run).

    Returns:
        list of dict, one per generation: 'table' (FeedbackTable), 'masses_msun',
        'birth_times_myr', 'population' (GenerationPopulation), 'source' ('file' or 'rebuilt'),
        'path' (the table file, or None).
    """
    from ..pysb99.stochastic.population import (StochasticSpec, populations_from_records,
                                                 build_generation_table, check_redraw)
    from .table import read_table_file
    with open(dat_path) as f:
        params = json.load(f).get('parameters', {})
    rec = params.get('stochastic')
    if rec is None:
        raise ValueError(f"{dat_path} is not a stochastic run (no 'stochastic' record)")
    spec = StochasticSpec.from_run_record(rec, database_path)
    pops = populations_from_records(rec['generations'], dat_path)
    stem = dat_path[:-len('.dat')] if dat_path.endswith('.dat') else dat_path
    out = []
    for p in pops:
        path = f"{stem}_gen{p.generation_index}_feedback.h5"
        table, source = None, 'rebuilt'
        if os.path.exists(path):
            t = read_table_file(path)
            if _saved_table_is_current(t, p, rec):
                table, source = t, 'file'
        if table is None:
            check_redraw(p, spec)
            if rec.get('database_md5') and spec.database_md5() != rec['database_md5']:
                raise ValueError(f"{spec.database_path} is not the single-star database the run used "
                                 f"(md5 {rec['database_md5']})")
            table = build_generation_table(p, spec).feedback_table
            path = None
        out.append({'table': table, 'masses_msun': p.masses_msun, 'birth_times_myr': p.birth_times_myr,
                    'population': p, 'source': source, 'path': path})
    return out


def _saved_table_is_current(table, pop, rec):
    """A saved generation table is used only if it holds this population (star-list md5),
    was built from the run's single-star database by the current table rule, and records
    its supernova deposition mode (tables written before 2.1.0 do not; they are rebuilt)."""
    from ..pysb99.stochastic.population import TABLE_RULE_VERSION
    if table.population is None or table.population['scalars'].get('population_md5') != pop.arrays_md5():
        return False
    src = table.metadata.get('source', {})
    db = src.get('single_star_database', {}).get('database_md5')
    if rec.get('database_md5') and db != rec['database_md5']:
        return False
    if src.get('table_rule_version') != TABLE_RULE_VERSION:
        return False
    return 'sn_time_deposition_mode' in table.metadata.get('conventions', {})
