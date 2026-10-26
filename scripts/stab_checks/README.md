# STAB pipeline checks

Manual check scripts for the STAB assembly (`toddlers.stab`). They are not part of the test suite:
each needs generated data that a fresh checkout does not have.

| Script | Needs | Run |
|---|---|---|
| `check_recollapse_sim.py` | the recollapse HDF5 (`hdf5/recollapse_data_SB99_kroupa100_sin.hdf5`, relative to the working directory) | `python check_recollapse_sim.py` |
| `check_dissolution_interp.py` | a dissolution-time interpolant (`dissolution_time_interp.pkl`) | `python check_dissolution_interp.py` |
| `check_sed_res.py` | a Cloudy model output directory | `python check_sed_res.py <cloudy_output_dir> [time_Myr] [model]` |
