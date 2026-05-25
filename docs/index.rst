TODDLERS
========

**TODDLERS** (Time evolution of Observables including Dust Diagnostics and Line
Emission from Regions containing young Stars) models the coupled gas/dust shell
dynamics and Cloudy photoionization of star-forming clouds, producing time-resolved
SEDs and SKIRT ``.stab`` SED libraries.

This is the API reference for the ``toddlers`` package. For installation, a quickstart,
and runnable end-to-end examples, see the project `README
<https://github.com/anandutsavkapoor/toddlers-public>`_ and the ``examples/`` directory.

Getting started
---------------

.. code-block:: bash

   pip install -e .
   python scripts/download_data.py                      # base libraries + BPASS tables
   python scripts/download_data.py --stochastic-tracks  # stochastic sampler database

.. code-block:: python

   from toddlers.evolution import Evolution
   from toddlers.constants import M_SUN, MYR_TO_SEC

   ev = Evolution(Z=0.02, eta_sf=0.05, n_cl=160.0, M_cl_init=1e6 * M_SUN,
                  template="SB99_100", profile_type="uniform")
   results = ev.run_simulation()
   g = results[0]
   print(g["time"] / MYR_TO_SEC, g["radius"])   # shell radius vs time

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
