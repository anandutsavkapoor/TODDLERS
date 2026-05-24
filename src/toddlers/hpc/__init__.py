"""Universal SLURM HPC interface for TODDLERS.

A small, scheduler-agnostic toolkit for running the two expensive TODDLERS work
units on any SLURM cluster:

  * **evolution** -- the 1D shell dynamics (one task per parameter combination), and
  * **cloudy**    -- the photoionization post-processing (one task per
                      simulation x timepoint x model phase).

The design is the worker-pool pattern: a flat task file (JSON lines) is produced
once with :mod:`toddlers.hpc.generate_tasks`, then one persistent
:mod:`toddlers.hpc.worker_loop` process per allocated core consumes a modular
slice of it (``row_index % n_workers == worker_id``). This amortises Python
startup across thousands of tasks and needs no cluster-specific job framework.

Nothing here hard-codes a cluster: accounts, partitions, modules, data paths and
the Cloudy executable are all supplied through the fill-in SLURM templates in
``toddlers/hpc/templates/`` or via command-line flags. See ``hpc/README.md``.
"""

from . import runner  # noqa: F401

__all__ = ["runner"]
