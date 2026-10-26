TODDLERS
========

**TODDLERS** (Time evolution of Observables including Dust Diagnostics and Line
Emission from Regions containing young Stars) models the coupled gas/dust shell
dynamics and Cloudy photoionization of star-forming clouds, producing time-resolved
SEDs and SKIRT ``.stab`` SED libraries.

This is the API reference for the ``toddlers`` package. For installation, a quickstart,
and runnable end-to-end examples, see the project `README
<https://github.com/anandutsavkapoor/TODDLERS>`_ and the ``examples/`` directory.

Getting started
---------------

.. code-block:: bash

   pip install -e .
   python scripts/download_data.py --list               # versioned data of this release
   python scripts/download_data.py                      # Cloudy stellar-continua tables (needs Cloudy)
   python scripts/download_data.py --stochastic-tracks  # single-star database (stochastic IMFs)

The stellar feedback tables ship with the package, so the dynamics run without a download.

.. code-block:: python

   from toddlers.evolution import Evolution
   from toddlers.constants import M_SUN, MYR_TO_SEC

   ev = Evolution(Z=0.02, eta_sf=0.05, n_cl=160.0, M_cl_init=1e6 * M_SUN,
                  template="SB99", imf="kroupa100", star_type="sin", profile_type="uniform")
   results = ev.run_simulation()
   g = results[0]
   print(g["time"] / MYR_TO_SEC, g["radius"])   # shell radius vs time

Usage notes
-----------

* **Outputs.** Evolution and Cloudy outputs go under one root: the ``output_root``
  argument of ``Evolution``, else ``$TODDLERS_OUTPUT_ROOT``, else the directory that holds the package
  (``src/`` in a source checkout).
  The Cloudy stage writes next to the evolution ``.dat`` it reads. A ``.dat`` records its
  inputs, the feedback table and the code version, and is reused only when they match.
* **Feedback tables.** ``toddlers.feedback_tables.load_table(template, imf, star_type)``
  reads any shipped table (HDF5, with units and provenance). Tables in the earlier
  ``.obj`` format still load, with a warning, and
  ``toddlers.feedback_tables.legacy.convert_legacy_obj`` converts them.
* **Stochastic populations.** Pass the IMF and a seed,
  ``stochastic=StochasticSpec(IMFSpec("kroupa", 0.1, 100.0), base_seed=17)`` with
  ``template="pySB99", imf=None``; each star-forming generation draws its own stars.
  The upper mass is limited by the single-star database's most massive track at the run's
  metallicity: 120 Msun at Z = 0.002, 299.7 at 0.006, 498.5 at 0.014, 298.8 at 0.020.
  The Kroupa form needs a lower mass of at least 0.08 Msun.
* **Custom pySB99 tables with supernova point events** need a time grid whose steps are
  whole multiples of 0.025 Myr.

API reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api/toddlers

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
