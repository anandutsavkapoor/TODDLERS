# Running TODDLERS on a SLURM cluster

`toddlers.hpc` is a small, scheduler-agnostic toolkit for the two expensive
TODDLERS work units on **any SLURM cluster**:

| stage | one task = | typical count |
|-------|------------|---------------|
| `evolution` | one parameter combination (1D shell dynamics) | 10s–1000s |
| `cloudy`    | one simulation × timepoint × model phase (photoionization) | 10⁴–10⁶ |

It uses the **worker-pool** pattern: a flat task file is produced once, then one
persistent worker process per allocated core consumes a modular slice of it
(`line_index % n_workers == worker_id`). This pays Python startup once per worker
instead of once per task, and needs no cluster-specific job framework.

Nothing here hard-codes a cluster. Account, partition, modules, data paths and the
Cloudy executable are supplied through the fill-in templates in `templates/` or via
flags / environment variables.

## The pieces

- `generate_tasks.py` — write the task file(s) (JSON-lines).
- `worker_loop.py` — the per-core worker (launched by the templates).
- `runner.py` — the inline per-task functions (`run_evolution_task`, `run_cloudy_task`).
- `enumerate_cloudy.py` — decides which Cloudy phases run at which timepoints.
- `check_status.py` — count OK/failed/pending and emit a resume task file.
- `templates/submit_evolution.sh`, `templates/submit_cloudy.sh` — fill-in SLURM scripts.

## 1. Install on the cluster

```bash
git clone <repo> && cd toddlers
module load <a SciPy bundle>           # provides numpy/scipy/h5py/...
pip install --user -e .                # or into a conda/venv env
python scripts/download_data.py        # large libraries (tracks, atmospheres, db)
```

Point the data resolver at the downloaded data (if not the package default):

```bash
export TODDLERS_DATA=/path/to/toddlers/data
```

## 2. Evolution stage

```bash
# expand a parameter grid into one task per combination
python -m toddlers.hpc.generate_tasks evolution \
    --grid examples/hpc/evolution_grid.json -o tasks
# -> tasks/evolution.tasks
```

Copy `templates/submit_evolution.sh`, replace every `@PLACEHOLDER@`
(`@ACCOUNT@`, `@PARTITION@`, `@NTASKS@`, `@WALLTIME@`, `@PYTHON_MODULE@`,
`@ACTIVATE_ENV@`, `@TODDLERS_SRC@`, `@TODDLERS_DATA@`, `@TASKFILE@`,
`@RESULTSDIR@`), then:

```bash
sbatch submit_evolution.sh
```

## 3. Cloudy stage

```bash
# enumerate work items from the evolution .dat outputs, one file per phase
python -m toddlers.hpc.generate_tasks cloudy \
    --input-dir evolution_output --pattern '*.dat' -o tasks --add-dig
# -> tasks/cloudy_shell.tasks, tasks/cloudy_unified.tasks, tasks/cloudy_dig.tasks
```

**Phase ordering matters, and it is linear**: the `unified` model reads the
`shell` model's Cloudy output (its density structure), and a `dig` model uses the
transmitted continuum of its inner (`shell`/`unified`) model. So the phases must run
in sequence `shell -> unified -> dig`, each depending on the previous with
`afterok` (not in parallel):

```bash
sh=$(sbatch --parsable --export=ALL,PHASE=shell                       submit_cloudy.sh)
un=$(sbatch --parsable --dependency=afterok:$sh --export=ALL,PHASE=unified submit_cloudy.sh)
sbatch          --dependency=afterok:$un --export=ALL,PHASE=dig        submit_cloudy.sh
```

`submit_cloudy.sh` sets `TMPDIR=/dev/shm` (Cloudy I/O in RAM) and passes the
cluster's `cloudy.exe` via `CLOUDY_EXE`.

## 4. Check status / resume

```bash
python -m toddlers.hpc.check_status \
    --task-file tasks/cloudy_shell.tasks \
    --results 'results/cloudy_shell_*.results' \
    -o tasks/cloudy_shell.resume.tasks
```

This prints OK / failed / pending counts and writes every task not confirmed `OK`
to a fresh resume task file. Resubmit the worker pool against that file (with a new
results file) to finish the remaining work. Exit code is non-zero while anything is
unfinished, so it composes in scripted retry loops.

### Automatic repair of known Cloudy failures

Cloudy tasks self-heal common, well-understood failures: when a model fails with a
recognised signature (non-convergence, temperature/pressure floor, zone limit),
`run_cloudy_task` classifies it (`toddlers.hpc.error_recovery`), applies the matching
input fix (add turbulence, set/raise cosmic-ray background, raise the zone cap) and
reruns it in place, up to `max_repair_attempts` times (default 2; set
`auto_repair=False` to disable). The repair is logged (`[repair] …`) and noted in the
`OK` result. Unrecognised failures are left as `FAIL` for manual review. This keeps a
production grid from accumulating permanent failures on issues that a small input
tweak resolves.

## Example: the variable-DTM campaign

This is the paper's variable dust-to-metal campaign: a
grid of evolution runs at the **fiducial** DTM, then Cloudy post-processing that
**sweeps DTM** (the dust-to-metal scaling enters only the Cloudy grain abundances, so
the shell dynamics are computed once and reused). DIG is **not** included.

```bash
# 1) evolution grid at fiducial DTM (no DTM axis). ~2500 runs for the paper axes.
python -m toddlers.hpc.generate_tasks evolution \
    --grid examples/hpc/sb99_2023_grid.json -o tasks
sbatch submit_evolution.sh                       # worker pool, resume as needed

# 2) Cloudy (no DIG) sweeping DTM. One Cloudy task per (sim, timepoint, DTM);
#    each DTM lands in its own `..._dtm<value>` run directory, so they never collide.
python -m toddlers.hpc.generate_tasks cloudy \
    --input-dir evolution_output --pattern '*.dat' -o tasks \
    --dust-to-metal 0.02 0.10 0.20 0.40 0.60 0.80 1.00
#    -> tasks/cloudy_shell.tasks, tasks/cloudy_unified.tasks   (no dig)

sh=$(sbatch --parsable --export=ALL,PHASE=shell                       submit_cloudy.sh)
sbatch          --dependency=afterok:$sh --export=ALL,PHASE=unified    submit_cloudy.sh
```

The `--dust-to-metal <list>` flag overrides the default "read each file's own DTM"
behaviour and is what produces the DTM axis of the 5D STAB. Sizing note: the `unified`
models are the expensive ones (a wide density law at fine resolution), so give the
unified phase the most walltime; shell is cheap. Use whole nodes (`--ntasks` = node
core count) and let `check_status` + resume absorb any walltime cutoff.

If you instead want DTM to affect the **dynamics** (it scales the dust opacity /
IR trapping), make `dust_to_metal` an axis in the evolution `grid.json` rather than a
Cloudy sweep; the evolution output paths disambiguate with a `_dtm` suffix.

## Disk and inode quotas (large grids)

A finished Cloudy model leaves ~100 small files per parameter directory, most of them
diagnostic dumps (`.ovr`, `.heat`, `.cool`, `.grTemp`, ...) the STAB build never reads.
A big grid (thousands of models × a DTM sweep) therefore hits the cluster's **file-count
(inode) quota** well before its byte quota. Check both:

```bash
# Lustre / VSC dodrio: bytes AND files vs their limits
lfs quota -g <your_group> /path/to/scratch     # 'files' / 'ilimit' columns
```

To stay under the inode ceiling, `toddlers.hpc.archive_cloudy_output` packs each
parameter directory's pure-diagnostic files (`.grAbund`, `.grCont`, `.grDGrat`,
`.grTemp`, `.heat`, `.cool`) into one `output_archive.tar` and removes the originals. It
keeps loose every file the SED interpolant + STAB build reads: `.cont` and
`.diffContUnatt` (continuum, incl. the unattenuated nebular continuum for the noDust /
variable-DTM SEDs), `.cum`/`.cumEmer` (cumulative line luminosities for the HR STABs),
`.ovr`/`.phy`/`.rad` (ionization / density / radial structure), plus `.in`/`.out`. So it
is safe to run *before* the build (≈25% fewer files per directory). It is fully
reversible:

```bash
python -m toddlers.hpc.archive_cloudy_output cloudy_output              # archive (build-safe)
python -m toddlers.hpc.archive_cloudy_output cloudy_output --dry-run    # preview
python -m toddlers.hpc.archive_cloudy_output cloudy_output --aggressive # storage (post-build)
python -m toddlers.hpc.archive_cloudy_output cloudy_output --untar      # restore
```

For **long-term storage** of a grid you have already turned into STABs, `--aggressive`
also tars the line (`.cum`/`.cumEmer`), nebular (`.diffContUnatt`), overview and structure
files, keeping only `{.in,.out,.cont,.phy,.rad}` (~50% fewer files). It is *not* safe
before a build: `--untar` first if you rebuild from an aggressively-archived grid.

The `campaign.py` orchestrator runs the build-safe mode automatically once the Cloudy
grid is complete (after the resume gate, before the build); pass `--no-archive-cloudy` to
keep all files loose. Only models confirmed successful are archived, so a failed model's
files stay in place for inspection or resume. Note this step runs after the grid is
complete, so it lowers the build-time and steady-state footprint but **not** the transient
inode peak *during* the Cloudy stage (every model exists before anything is archived) —
size scratch for the full uncompressed grid. On shared **project** scratch, watch that the
byte and inode budgets are co-tenant with other users' runs, and always point
`cloudy_output` at scratch (not a small `$HOME`/`$DATA` quota).

## Cluster gotchas (learned the hard way)

- **Launch the worker pool with `srun`, never a bash `... &` loop.** On some
  clusters (e.g. VSC Hortense) the batch step's CPU affinity is pinned to one core,
  so background children all time-share that single core: a 128-worker job then runs
  at ~0.9% CPU each and times out with zero completions. `srun --cpu-bind=cores`
  binds each worker to its own core (the submit templates do this; `worker_loop`
  reads its slot from `$SLURM_PROCID`/`$SLURM_NTASKS`). Symptom of getting it wrong:
  wall time >> Cloudy `ExecTime`.
- Use `--nodes=1 --ntasks=N`, **not** `--ntasks-per-node=` / `--cpus-per-task=`
  (the latter trigger "partition configuration not available" on some clusters).
- **Multiple nodes work too**: set `--nodes=M --ntasks=N` (N total). `srun` makes
  `$SLURM_PROCID` a global rank and `$SLURM_NTASKS` the global total, so the modular
  task slicing stays globally disjoint, and each worker writes its own results file
  (`<results>.<id>`) — no cross-node locking. `check_status --results '<results>.*'`
  aggregates them.
- `export PYTHONPATH=...:${PYTHONPATH:-}` — **prepend**; overwriting wipes the
  module's site-packages and numpy import fails.
- `source /etc/profile.d/modules.sh` in the script — non-interactive shells don't
  get it for free.
- Pass `--cloudy-exe` / `CLOUDY_EXE` explicitly; the default `cloudy.exe`-on-PATH
  rarely matches the cluster install.
- **Login-node benchmarks lie**: Cloudy runs ~4× slower on a saturated node than on
  an idle login node (cache/bandwidth contention). Size production batches from a
  short on-cluster run, reading the per-worker `[wNN] DONE: N tasks in Ts` lines.
- Clear stale `__pycache__` after updating code on the cluster.
- **A `[patch-check]` line warns if `cloudy.exe` lacks the unattenuated-continuum patch.**
  The patch (`cloudy_patches/`) is optional; without it Cloudy writes no `.diffContUnatt`,
  so the noDust / variable-DTM SEDs fall back to the standard attenuated diffuse continuum.
  Each worker notes this once (greppable as `[patch-check]`) if the patch is absent, so you
  know which behaviour a grid was built with. No warning means the patch is present.
- **After rebuilding or re-patching `cloudy.exe`, clear (or force-regenerate) the
  affected `cloudy_output`.** The "already run, skipping" guard (`check_cloudy_success`)
  only checks that a model's `.out` says "Cloudy exited OK"; it does **not** track which
  binary wrote the file, so output from an *older* `cloudy.exe` is silently reused. If a
  result looks "very strange," check the output mtime against the `cloudy.exe` build time
  before debugging anything else.
