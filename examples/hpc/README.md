# Running a STAB campaign on a SLURM cluster

## Do you even need this?

Most users do **not** run their own grids:

- **Use the precomputed STAB libraries** (in SKIRT, or for MCMC fitting): pure table lookup,
  no Cloudy and no cluster — nothing here applies.
- **Run a handful of bespoke regions**: one workstation is enough. See the local examples
  (`examples/04_cloudy_pipeline.py`, `examples/11_bpt_diagram.py`), which call
  `Evolution().run_simulation()` then `CloudySimulationManager(...).run_full_simulation()`.
  No scheduler.
- **Generate a new large grid** (new IMF/template/parameter range): this is the only case
  you need the pipeline below. SLURM is the common case; the worker pool itself is
  scheduler-agnostic (only the submit templates are SLURM-specific, see
  `src/toddlers/hpc/README.md`), so PBS/SGE/LSF/cloud users reuse it with their own
  one-line launcher.

## The driver

`toddlers.hpc.campaign` is the one packaged driver for the complete TODDLERS → SKIRT STAB
workflow on any SLURM cluster:

```
[evolution] → Cloudy (shell → unified → dissolved [→ dig]) → SED interpolant
→ STAB libraries (cloud-family + SFR-normalised)
```

It reuses the fill-in templates in `toddlers/hpc/templates/`, submits the phases as an
`afterok` chain, and runs a final post-processing job (interpolant + STABs). The STAB grid
axes (Z, SFE, n_cl) **and** the population (template/IMF/star-type) are **derived
automatically** from the evolution path/sims and passed to the STAB stage as
`TODDLERS_STAB_*` environment overrides read by `toddlers.stab.config`, so nothing is
hand-edited.

Run it on the cluster's login node (it calls `sbatch`). Always preview with `--dry-run`
first: it prints every `generate_tasks` / `sbatch` command and submits nothing.

## Two campaign types (DTM is the only difference)

* **Fiducial (no DTM axis)** — omit `--dust-to-metal`: a single fiducial-DTM run.
* **Variable-DTM (paper)** — `--dust-to-metal 0.02 0.10 0.20 0.40 0.60 0.80 1.00`: the
  Cloudy DTM sweep that produces the 5D (DTM-axis) SFR-normalised STAB.

DIG is off by default (the paper's variable-DTM grid has none). The Cloudy time grid
defaults to `toddlers_v1`, which the SED interpolant requires.

### The two campaigns behind the paper

Both are the *same* driver; they map onto the two recipes below:

1. **Production grid, from scratch** (e.g. the BPASS chab100 binary grid): generate the grid
   and run evolution → Cloudy → STAB in one dependent chain — see **From scratch** below.
   The grid axes live in `toddlers.stab.config` per template, so regenerate the grid JSON
   with `make_grid.py` (it reads those axes) rather than hand-editing.
2. **Variable-DTM 5D library**: start from the already-computed evolution and sweep DTM at
   the Cloudy stage, scaled across nodes — see **From existing evolution runs** + **Large
   workloads** below. This is the campaign that builds the paper's DTM-axis SFR-normalised
   STAB.

## From scratch (run evolution too)

`stab_campaign_grid.json` here is a small rectangular demo grid (2×2×2×2 = 16 runs:
Z{0.008,0.02} × SFE{0.025,0.05} × n_cl{80,160} × logM{5,6}).

```bash
python -m toddlers.hpc.campaign \
    --grid examples/hpc/stab_campaign_grid.json \
    --work-dir runs --stab-dir examples/stab \
    --account <your_account> --partition <partition> --ntasks 128 \
    --python-module <SciPy-bundle/...> \
    --toddlers-src $PWD/src \
    --cloudy-exe /path/to/cloudy.exe --cloudy-data /path/to/cloudy/data \
    --dry-run
```

The grid JSON fixes the template/IMF/star-type/profile, so the driver derives the evolution
output leaf and submits a dependent **stage-2** job that runs the Cloudy chain + STABs once
evolution finishes (Cloudy enumeration needs the evolution output, which does not exist at
submit time). This requires the cluster to allow `sbatch` from within a job.

## From existing evolution runs

If the evolution `.dat` already exist (e.g. produced elsewhere), skip evolution and point
the driver at the output leaf — everything downstream runs in one shot:

```bash
python -m toddlers.hpc.campaign \
    --evolution-dir evolution_output/template_SB99/imf_kroupa100/star_type_sin/cluster_mode_burst/profile_type_uniform \
    --work-dir runs --stab-dir examples/stab \
    --account <your_account> --partition <partition> --ntasks 128 \
    --python-module <SciPy-bundle/...> --toddlers-src $PWD/src \
    --cloudy-exe /path/to/cloudy.exe --cloudy-data /path/to/cloudy/data \
    --dust-to-metal 0.02 0.10 0.20 0.40 0.60 0.80 1.00
```

The existing `.dat` must sit in the standard
`template_*/imf_*/star_type_*/cluster_mode_*/profile_type_*/` layout. The STAB axes are read
back from the `.dat` filenames, so the libraries match whatever sims you bring.

## Large workloads: scale across nodes

`--ntasks` is the cores **per node**; `--max-nodes` caps how many nodes a phase scales to.
Each Cloudy phase is sized to its own task count: `nodes = ceil(tasks / ntasks)` capped at
`--max-nodes`. Small phases stay on one node (no wasted cores); large ones (e.g. a DTM
sweep with 10⁴–10⁵ Cloudy tasks) spread across nodes for wall-clock speed at the **same**
CPU-hours (more workers → fewer waves). For a big variable-DTM run:

```bash
python -m toddlers.hpc.campaign --evolution-dir <leaf> ... \
    --dust-to-metal 0.02 0.10 0.20 0.40 0.60 0.80 1.00 \
    --ntasks 128 --max-nodes 8
```

(Default `--max-nodes 1` keeps everything on a single node.)

## The grid files here

| file | what |
|---|---|
| `stab_campaign_grid.json` | small rectangular demo grid (16 runs) for an end-to-end campaign |
| `evolution_grid.json` | another small evolution grid (for the `generate_tasks evolution` examples in the package HPC README) |
| `sb99_2023_grid.json` | the full SB99 parameter grid from Kapoor+2023 (Paper I): Z×SFE×n_cl (large) |
| `make_grid.py` | regenerate `sb99_2023_grid.json` (the SB99 v1 axes) from `toddlers.stab.config` |

## Lower-level pieces

For the individual building blocks (`generate_tasks`, `worker_loop`, `check_status`, the
SLURM templates, and the manual phase-by-phase `afterok` chain that `campaign` automates),
see `src/toddlers/hpc/README.md`.
