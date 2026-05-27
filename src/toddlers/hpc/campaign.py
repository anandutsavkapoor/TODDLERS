"""End-to-end STAB campaign orchestrator for SLURM clusters.

One packaged, parameterised driver for the *complete* TODDLERS -> SKIRT STAB workflow,
so you do not hand-roll per-run submit scripts:

    [evolution] -> Cloudy (shell -> unified -> dissolved [-> dig]) -> SED interpolant
    -> STAB libraries (cloud-family + SFR-normalised)

It reuses the fill-in templates in ``templates/`` (the single source of truth for the
worker-pool SLURM jobs), submits the phases as an ``afterok`` chain, and finally submits
one post-processing job (depending on the Cloudy phases) that builds the interpolant and
the STABs.

Dust-to-metal is **optional**, which selects the two standard campaigns:

* **omit ``--dust-to-metal``**  -> a single fiducial-DTM run, no DTM axis (the standard
  STAB campaign).
* **``--dust-to-metal 0.02 0.10 0.20 0.40 0.60 0.80 1.00``** -> the Cloudy DTM sweep that
  produces the 5D (DTM-axis) SFR-normalised STAB (the paper's variable-DTM run).

The Cloudy time grid defaults to ``toddlers_v1`` (the grid the SED interpolant requires);
DIG is off by default (the paper's variable-DTM grid has no DIG).

Examples
--------
Start from existing evolution runs, fiducial DTM, both STAB families::

    python -m toddlers.hpc.campaign \
        --evolution-dir evolution_output/template_SB99/imf_kroupa100/star_type_sin/cluster_mode_burst/profile_type_uniform \
        --work-dir runs --stab-dir examples/stab \
        --account my_account --partition cpu_milan_rhel9 --ntasks 128 \
        --python-module SciPy-bundle/2024.05-gfbf-2024a \
        --toddlers-src $PWD/src --cloudy-exe /path/cloudy.exe --cloudy-data /path/cloudy/data

Variable-DTM (paper), preview the SLURM plan first::

    python -m toddlers.hpc.campaign ... --dust-to-metal 0.02 0.10 0.20 0.40 0.60 0.80 1.00 --dry-run

Run it on the cluster's login node (it calls ``sbatch``). Use ``--dry-run`` to print the
exact ``generate_tasks`` and ``sbatch`` commands without submitting anything.
"""
import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

HPC_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = HPC_DIR / "templates"

# Cloudy phases, in dependency order. unified reads shell's density structure; dissolved
# holds the post-dissolution models (present when the toddlers_v1 grid continues past
# dissolution); dig (optional) reads the inner model's transmitted continuum.
CLOUDY_PHASES = ("shell", "unified", "dissolved", "dig")


def _fill_template(template_path, replacements):
    text = template_path.read_text()
    for key, value in replacements.items():
        text = text.replace(key, str(value))
    return text


def _write_script(work_dir, name, text):
    path = Path(work_dir) / name
    path.write_text(text)
    path.chmod(0o755)
    return path


def _sbatch(script_path, *, dependency=None, dependency_type="afterok", exports=None,
            nodes=None, ntasks=None, dry_run=False):
    """Submit a job, return its job id. Strips the VSC ``jobid;cluster`` suffix that
    ``--parsable`` emits on multi-cluster sites (otherwise ``--dependency`` breaks).
    ``nodes``/``ntasks`` (when given) override the template's #SBATCH for per-phase sizing.
    ``dependency_type`` is ``afterok`` (default) or ``afterany`` (e.g. the resume gate, which
    must run even if a Cloudy phase timed out)."""
    cmd = ["sbatch", "--parsable"]
    if nodes is not None:
        cmd.append(f"--nodes={nodes}")
    if ntasks is not None:
        cmd.append(f"--ntasks={ntasks}")
    if dependency:
        cmd.append(f"--dependency={dependency_type}:{dependency}")
    if exports:
        cmd.append("--export=" + ",".join(["ALL"] + [f"{k}={v}" for k, v in exports.items()]))
    cmd.append(str(script_path))
    if dry_run:
        print("  " + " ".join(cmd))
        return "DRYRUN"
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    return out.split(";")[0]  # VSC dodrio: "<jobid>;<cluster>"


def _count_lines(path):
    try:
        with open(path) as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def _size_phase(task_count, cores_per_node, max_nodes):
    """Size a worker pool to the work: enough nodes to cover the tasks (so big phases
    finish in fewer waves / less wall time), capped at --max-nodes, and never more nodes
    than there are tasks. Returns (nodes, ntasks=nodes*cores_per_node). The node count is
    the real CPU-hour lever on node-exclusive clusters; idle cores in the final wave are
    inherent to static slicing (see hpc-cpu-hour-efficiency note)."""
    if task_count <= 0:
        return 1, cores_per_node
    nodes = (task_count + cores_per_node - 1) // cores_per_node  # ceil
    nodes = max(1, min(nodes, max_nodes))
    return nodes, nodes * cores_per_node


def _run(cmd, *, dry_run=False, **kw):
    if dry_run:
        print("  " + (cmd if isinstance(cmd, str) else " ".join(cmd)))
        return
    subprocess.run(cmd, check=True, **kw)


def _common_replacements(args):
    return {
        "@ACCOUNT@": args.account,
        "@PARTITION@": args.partition,
        "@NTASKS@": args.ntasks,
        "@WALLTIME@": args.walltime,
        "@PYTHON_MODULE@": args.python_module,
        "@ACTIVATE_ENV@": args.activate_env,
        "@TODDLERS_SRC@": args.toddlers_src,
        "@TODDLERS_DATA@": args.toddlers_data or args.toddlers_src.replace("/src", "/src/toddlers/database"),
    }


def _gen_evolution_tasks(args):
    out = Path(args.work_dir) / "tasks"
    _run([sys.executable, "-m", "toddlers.hpc.generate_tasks", "evolution",
          "--grid", args.grid, "-o", str(out)], dry_run=args.dry_run)
    return out / "evolution.tasks"


def _gen_cloudy_tasks(args):
    out = Path(args.work_dir) / "tasks"
    cmd = [sys.executable, "-m", "toddlers.hpc.generate_tasks", "cloudy",
           "--input-dir", args.evolution_dir, "--pattern", args.pattern,
           "-o", str(out), "--method", args.time_method]
    if not args.continue_after_dissolution:
        cmd.append("--no-continue-after-dissolution")
    if args.add_dig:
        cmd.append("--add-dig")
    if args.dust_to_metal:
        cmd += ["--dust-to-metal"] + [str(d) for d in args.dust_to_metal]
    _run(cmd, dry_run=args.dry_run)
    return out


def _submit_evolution(args, taskfile):
    repl = _common_replacements(args)
    repl["@TASKFILE@"] = taskfile
    repl["@RESULTSDIR@"] = str(Path(args.work_dir) / "results")
    text = _fill_template(TEMPLATES_DIR / "submit_evolution.sh", repl)
    script = _write_script(args.work_dir, "campaign_evolution.sh", text)
    nodes, ntasks = _size_phase(_count_lines(taskfile), args.ntasks, args.max_nodes)
    jid = _sbatch(script, nodes=nodes, ntasks=ntasks, dry_run=args.dry_run)
    print(f"evolution: job {jid} ({nodes} node(s) x {args.ntasks})")
    return jid


def _submit_cloudy_chain(args, taskdir, after=None):
    """Submit the Cloudy phases as an afterok chain; return {phase: jobid}."""
    repl = _common_replacements(args)
    repl["@TASKDIR@"] = str(taskdir)
    repl["@RESULTSDIR@"] = str(Path(args.work_dir) / "results")
    repl["@CLOUDY_EXE@"] = args.cloudy_exe
    text = _fill_template(TEMPLATES_DIR / "submit_cloudy.sh", repl)
    script = _write_script(args.work_dir, "campaign_cloudy.sh", text)

    phases = [p for p in CLOUDY_PHASES if (p != "dig" or args.add_dig)]
    jobs = {}
    # shell depends on evolution (if any); unified/dissolved depend on shell; dig on unified.
    for phase in phases:
        if phase == "shell":
            dep = after
        elif phase in ("unified", "dissolved"):
            dep = jobs.get("shell")
        else:  # dig
            dep = jobs.get("unified")
        taskfile = Path(taskdir) / f"cloudy_{phase}.tasks"
        if not args.dry_run and not (taskfile.exists() and taskfile.stat().st_size > 0):
            print(f"cloudy/{phase}: no tasks, skipping")
            continue
        nodes, ntasks = _size_phase(_count_lines(taskfile), args.ntasks, args.max_nodes)
        jobs[phase] = _sbatch(script, dependency=dep, exports={"PHASE": phase},
                              nodes=nodes, ntasks=ntasks, dry_run=args.dry_run)
        print(f"cloudy/{phase}: job {jobs[phase]} ({nodes} node(s) x {args.ntasks}, afterok {dep})")
    return jobs


def _stab_axes_from_dats(evolution_dir):
    """Distinct (Z, SFE, n_cl) in the evolution .dat filenames -> the STAB grid axes.

    Parses ``sim_Z<z>_eta<e>_n<n>_logM<m>...dat``. Lets the STAB stage match whatever
    sims were actually run (the test grid, or a collaborator's own runs) without editing
    toddlers.stab.config."""
    zs, es, ns = set(), set(), set()
    for f in glob.glob(os.path.join(evolution_dir, "*.dat")):
        m = re.search(r"Z([0-9.]+)_eta([0-9.]+)_n([0-9.]+)_logM", os.path.basename(f))
        if m:
            zs.add(float(m.group(1))); es.add(float(m.group(2))); ns.add(float(m.group(3)))
    return sorted(zs), sorted(es), sorted(ns)


def _stab_population_from_dir(evolution_dir):
    """Extract (template, imf, star_type) from the evolution-dir path.

    The evolution output path contains ``template_<T>/imf_<I>/star_type_<S>/``; these
    select the stellar population so the STAB stage (toddlers.stab.config) matches the
    sims that were run, via the TODDLERS_STAB_TEMPLATE/IMF/STARTYPE env overrides."""
    template = imf = star_type = None
    for part in Path(evolution_dir).parts:
        if part.startswith("template_"):
            template = part[len("template_"):]
        elif part.startswith("imf_"):
            imf = part[len("imf_"):]
        elif part.startswith("star_type_"):
            star_type = part[len("star_type_"):]
    return template, imf, star_type


def _submit_postprocess(args, taskdir, after_jobs):
    """Submit the resume-gate + STAB build, depending (afterany) on the Cloudy phases.

    This job FIRST gates on Cloudy completeness: it runs check_status per phase and, for any
    phase with unfinished tasks (failed models that auto-repair could not fix, or a phase
    that hit the walltime), resubmits just those tasks as a resized worker pool and re-arms
    ITSELF (afterany the resumes), up to --max-resume-rounds. Only when the grid is complete
    does it build the interpolant + STABs (see examples/stab/README.md):
      python -m toddlers.stab.interpolants -> recollapse rename -> cloud-family + SFR-norm STAB.
    Depending afterany (not afterok) means a timed-out phase doesn't cancel the chain — the
    gate simply resumes its leftover tasks. For a DTM sweep the interpolant is built per DTM.
    """
    dep = ":".join(j for j in after_jobs.values() if j and j != "DRYRUN") or None
    dtms = args.dust_to_metal or [1.0]
    resdir = Path(args.work_dir) / "results"
    self_path = Path(args.work_dir) / "campaign_stab.sh"
    cloudy_sh = Path(args.work_dir) / "campaign_cloudy.sh"
    phases = " ".join(after_jobs.keys())

    # Match the STAB grid axes AND stellar population to the evolution sims actually run
    # (axes from the .dat filenames, population from the evolution-dir path), via the
    # toddlers.stab.config env overrides. No hand-editing of config.py.
    axis_export = ""
    pop_export = ""
    if args.evolution_dir:
        zs, es, ns = _stab_axes_from_dats(args.evolution_dir)
        if zs and es and ns:
            axis_export = ("export "
                           f"TODDLERS_STAB_Z={','.join(map(str, zs))} "
                           f"TODDLERS_STAB_SFE={','.join(map(str, es))} "
                           f"TODDLERS_STAB_NCL={','.join(map(str, ns))}")
        template, imf, star_type = _stab_population_from_dir(args.evolution_dir)
        if template and imf and star_type:
            pop_export = ("export "
                          f"TODDLERS_STAB_TEMPLATE={template} "
                          f"TODDLERS_STAB_IMF={imf} "
                          f"TODDLERS_STAB_STARTYPE={star_type}")
    # The v2-DTM SEDs include the unattenuated diffuse nebular continuum in the noDust
    # variant (paper Appendix B); enable it when running a DTM sweep. Requires the
    # patched Cloudy (cloudy_patches/) so ".diffContUnatt" carries the DiffContUnatt column.
    neb_export = "export TODDLERS_STAB_NEBULAR_CONT=1" if args.dust_to_metal else ""
    # Put the (large, transient) interpolant cache on scratch when --cache-dir is given, so it
    # does not consume the code/home-filesystem quota on a big DTM sweep (TODDLERS_INTERP_CACHE
    # is read by interpolants.DataManager; the build's CACHE_DIR resolves the same way).
    cache_export = f"export TODDLERS_INTERP_CACHE={os.path.abspath(args.cache_dir)}" if args.cache_dir else ""
    stab_done = os.path.join(os.path.abspath(args.stab_dir), ".stab_build_complete")
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=campaign_stab",
        f"#SBATCH --account={args.account}",
        f"#SBATCH --partition={args.partition}",
        f"#SBATCH --nodes=1 --ntasks={args.ntasks}",
        f"#SBATCH --time={args.stab_walltime}",
        f"#SBATCH --output={resdir}/%x_%j.out",
        f"#SBATCH --error={resdir}/%x_%j.err",
        "set -euo pipefail",
        "source /etc/profile.d/modules.sh 2>/dev/null || true",
        f"module load {args.python_module}" if args.python_module else "",
        args.activate_env,
        f"export PYTHONPATH={args.toddlers_src}:${{PYTHONPATH:-}}",
        f"export CLOUDY_EXE={args.cloudy_exe} CLOUDY_DATA_DIR={args.cloudy_data}",
        # ---- build-completion guard + self-re-arm bookkeeping ----
        # STAB_DONE marks a fully finished build. The build self-re-arms (afterany) so it
        # survives the walltime: a timed-out build is continued by its successor
        # (interpolant_exists skips finished DTMs), independent of any client session. The
        # sentinel is cleared only on a FRESH launch (ROUND==0 and BUILD_ROUND==0) so a stale
        # sentinel from a previous campaign cannot short-circuit a new one.
        f'STAB_DONE={stab_done}',
        f'ROUND="${{ROUND:-0}}"; BUILD_ROUND="${{BUILD_ROUND:-0}}"; MAXROUND={args.max_resume_rounds}',
        'if [ "$ROUND" -eq 0 ] && [ "$BUILD_ROUND" -eq 0 ]; then rm -f "$STAB_DONE"; fi',
        'if [ -f "$STAB_DONE" ]; then echo "[stab] build already complete -> exit 0"; exit 0; fi',
        # ---- resume gate: ensure every Cloudy task is OK before building ----
        f"CORES={args.ntasks}; MAXNODES={args.max_nodes}",
        f'RESUME_DEP=""',
        f'for ph in {phases}; do',
        f'  python3 -m toddlers.hpc.check_status --task-file "{taskdir}/cloudy_${{ph}}.tasks" '
        f'--results "{resdir}/cloudy_${{ph}}_*.results.*" -o "{taskdir}/cloudy_${{ph}}.resume.tasks" || true',
        f'  if [ -s "{taskdir}/cloudy_${{ph}}.resume.tasks" ]; then',
        f'    NT=$(wc -l < "{taskdir}/cloudy_${{ph}}.resume.tasks")',
        '    NODES=$(( (NT + CORES - 1) / CORES )); [ "$NODES" -lt 1 ] && NODES=1; [ "$NODES" -gt "$MAXNODES" ] && NODES=$MAXNODES',
        f'    JID=$(sbatch --parsable --nodes=$NODES --ntasks=$((NODES*CORES)) '
        f'--export=ALL,PHASE=${{ph}},TASKFILE={taskdir}/cloudy_${{ph}}.resume.tasks "{cloudy_sh}" | cut -d";" -f1)',
        '    RESUME_DEP="${RESUME_DEP}:${JID}"; echo "[resume] $ph: $NT unfinished -> job $JID ($NODES node(s))"',
        '  fi',
        'done',
        'if [ -n "$RESUME_DEP" ] && [ "$ROUND" -lt "$MAXROUND" ]; then',
        '  echo "[resume] round $ROUND incomplete; re-arming gate afterany$RESUME_DEP"',
        f'  sbatch --dependency=afterany${{RESUME_DEP}} --export=ALL,ROUND=$((ROUND+1)) "{self_path}"',
        '  exit 0',
        'fi',
        '[ -n "$RESUME_DEP" ] && echo "[resume] WARNING: still incomplete after $MAXROUND rounds; building with available models"',
        # ---- self-re-arm the BUILD (survives walltime; successor no-ops once STAB_DONE) ----
        # Armed here (cwd still the job WorkDir, before the cd into stab_dir) so the relative
        # self_path resolves. The successor continues a timed-out build; on a finished build it
        # hits the STAB_DONE guard above and exits immediately.
        ('if [ "$BUILD_ROUND" -lt "$MAXROUND" ]; then '
         f'RSM=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID '
         f'--export=ALL,BUILD_ROUND=$((BUILD_ROUND+1)) "{self_path}") || RSM=arm-failed; '
         'echo "[stab] armed build successor $RSM afterany $SLURM_JOB_ID '
         '(build round $((BUILD_ROUND+1))/$MAXROUND)"; fi'),
        # ---- archive non-essential Cloudy output (inode relief on large grids) ----
        # The grid is complete here; pack each parameter dir's diagnostic dumps into one
        # tar and drop the originals, keeping .in/.out/.cont/.phy/.rad loose so the build
        # below still reads them. Reversible (archive_cloudy_output --untar). Housekeeping,
        # so `|| true`: a tar hiccup must not abort the STAB build.
        ('echo "=== archiving non-essential Cloudy output (inode relief) ==="\n'
         f"python3 -m toddlers.hpc.archive_cloudy_output "
         f"{os.path.join(args.toddlers_src, 'cloudy_output')} || true")
        if args.archive_cloudy else "",
        # ---- build (grid complete) ----
        axis_export,
        pop_export,
        neb_export,
        cache_export,
        f"cd {args.stab_dir}",
        'PREFIX="$(python3 -c \'from toddlers.stab import config; print(config.MODEL_PREFIX)\')"',
        "mkdir -p hdf5",
        # The selective interpolant cache lives in the PACKAGE dir (toddlers/stab/cache), not
        # the cwd. It is per (Z,eta,n,logM,DTM); each DTM's 64 entries are ~25 GB, so keeping
        # all 7 DTMs' caches at once would blow the data-partition quota. The cache is only
        # needed WHILE a given DTM's interpolants are being built -- once that DTM's pkls are
        # saved, a resume skips it via interpolant_exists (not the cache). So we clear the
        # cache BEFORE each DTM build (below), bounding it to one DTM's worth; clearing it
        # also drops any stale cache from an earlier Cloudy run that could mask fresh output.
        'CACHE_DIR="$(python3 -c \'import os; from pathlib import Path; import toddlers.stab.interpolants as m; print(os.environ.get("TODDLERS_INTERP_CACHE") or Path(m.__file__).parent / "cache")\')"',
        'echo "  interpolant cache dir: $CACHE_DIR"',
        # --keep-interp-cache: retain every DTM's cache (for debugging), but still drop any
        # cross-run stale cache once on a fresh build so the kept data is from THIS run.
        ('if [ "$BUILD_ROUND" -eq 0 ]; then echo "  [keep-interp-cache] clearing cross-run stale cache once"; '
         'rm -f "$CACHE_DIR"/*.pkl "$CACHE_DIR"/*.tmp 2>/dev/null || true; fi'
         if args.keep_interp_cache else ""),
    ]
    # One interpolant build per DTM (the .pkl carry a _dtm<val> suffix, so they coexist).
    # The interpolant stage runs after `cd {stab_dir}`, so the evolution dir must be absolute
    # (it is relative to the repo root where the campaign is invoked / the job's WorkDir).
    evo_dir_abs = os.path.abspath(args.evolution_dir)
    for dtm in dtms:
        block = [f'echo "=== interpolant for DTM={dtm} ==="']
        if not args.keep_interp_cache:
            # Bound the cache to one DTM: clear before each build (the previous DTM's pkls are
            # already saved, so its cache is dead weight; a resume skips done DTMs via
            # interpolant_exists). Keeps the data-partition footprint flat across a DTM sweep.
            block.append('rm -f "$CACHE_DIR"/*.pkl "$CACHE_DIR"/*.tmp 2>/dev/null || true')
        block.append(
            f"python3 -m toddlers.stab.interpolants --evolution-dir {evo_dir_abs} "
            f"--output-dir ${{PREFIX}}_interp_tables --dust-to-metal {dtm}")
        lines += block
    # Recollapse data is DTM-independent; SFR-norm reads hdf5/recollapse_data_<PREFIX>.hdf5.
    lines.append("cp ${PREFIX}_interp_tables/recollapse_data.h5 hdf5/recollapse_data_${PREFIX}.hdf5")
    if args.stab in ("both", "cloud"):
        lines.append('echo "=== cloud-family STAB ==="; python3 -m toddlers.stab.cloud_family_stab')
    if args.stab in ("both", "sfr"):
        # The SFR-normalized STAB reads resampled SED text files from the
        # <PREFIX>_sed_output_<Dust|noDust>_<lr|hr> folders, which are produced by the
        # paramspace resampler (one invocation per SED type x resolution). This stage must
        # run before toddlers.stab.sfr_normalized_stab, which otherwise finds no folders and
        # writes nothing. For a DTM sweep, all DTM values are passed so the SEDs carry _dtm suffixes.
        dtm_flag = (" --dust-to-metal " + " ".join(str(d) for d in args.dust_to_metal)
                    if args.dust_to_metal else "")
        lines.append('echo "=== resampling SFR-scaled SEDs (Dust/noDust x lr/hr) ==="')
        for st in ("Dust", "noDust"):
            for res in ("lr", "hr"):
                lines.append(
                    f"python3 -m toddlers.stab.sfr_scaled_seds "
                    f"--sed-type {st} --resolution {res}{dtm_flag}")
        lines.append('echo "=== SFR-normalised STAB ==="; python3 -m toddlers.stab.sfr_normalized_stab')
    lines.append('echo "campaign post-processing done"')
    # Mark the build complete so the armed successor (and any re-submit) no-ops via the guard.
    lines.append('touch "$STAB_DONE"; echo "[stab] completion sentinel written"')

    script = _write_script(args.work_dir, "campaign_stab.sh", "\n".join(lines) + "\n")
    jid = _sbatch(script, dependency=dep, dependency_type="afterany", dry_run=args.dry_run)
    print(f"stab resume-gate + build: job {jid} (afterany {dep})")
    return jid


def _evolution_leaf_from_grid(grid_path, toddlers_src):
    """Derive the evolution-output leaf directory from the grid JSON.

    Evolution writes to ``<toddlers_src>/evolution_output/template_*/imf_*/star_type_*/
    cluster_mode_*/profile_type_*/`` (see Evolution.get_output_paths), and the grid JSON
    fixes those five axes, so the Cloudy stage's input dir is known before evolution runs.
    Requires each path axis to be single-valued (true for a STAB grid)."""
    import json
    grid = json.load(open(grid_path))

    def one(key, default=None):
        val = grid.get(key, default)
        if isinstance(val, list):
            if len(val) != 1:
                raise SystemExit(
                    f"--grid auto-chain needs a single value for '{key}' (got {val}); the "
                    "evolution output would span multiple leaf dirs. Either fix that axis, "
                    "or run evolution first and pass --evolution-dir <leaf> for the Cloudy/STAB stage.")
            return val[0]
        return val

    return os.path.join(
        toddlers_src, "evolution_output",
        f"template_{one('template')}", f"imf_{one('imf')}", f"star_type_{one('star_type')}",
        f"cluster_mode_{one('cluster_formation_mode', 'burst')}",
        f"profile_type_{one('profile_type', 'uniform')}")


def _stage2_argv(args, leaf):
    """Reconstruct the from-existing campaign invocation for the deferred stage-2 job."""
    a = ["--evolution-dir", leaf, "--work-dir", args.work_dir, "--stab-dir", args.stab_dir,
         "--pattern", args.pattern, "--time-method", args.time_method, "--stab", args.stab,
         "--account", args.account, "--partition", args.partition,
         "--ntasks", str(args.ntasks), "--max-nodes", str(args.max_nodes),
         "--max-resume-rounds", str(args.max_resume_rounds),
         "--walltime", args.walltime, "--stab-walltime", args.stab_walltime,
         "--toddlers-src", args.toddlers_src, "--cloudy-exe", args.cloudy_exe,
         "--cloudy-data", args.cloudy_data]
    if args.python_module:
        a += ["--python-module", args.python_module]
    if args.activate_env:
        a += ["--activate-env", args.activate_env]
    if args.toddlers_data:
        a += ["--toddlers-data", args.toddlers_data]
    if args.dust_to_metal:
        a += ["--dust-to-metal"] + [str(d) for d in args.dust_to_metal]
    if args.add_dig:
        a += ["--add-dig"]
    if not args.continue_after_dissolution:
        a += ["--no-continue-after-dissolution"]
    if not args.archive_cloudy:
        a += ["--no-archive-cloudy"]
    if args.keep_interp_cache:
        a += ["--keep-interp-cache"]
    if args.cache_dir:
        a += ["--cache-dir", args.cache_dir]
    return a


def _submit_stage2(args, evo_job, leaf):
    """A small job that (afterok evolution) re-runs this driver in from-existing mode on
    the now-populated leaf dir: it enumerates Cloudy tasks and submits the chain + STAB.
    Requires the cluster to allow `sbatch` from within a job (true on VSC dodrio)."""
    inner = " ".join(["python3", "-m", "toddlers.hpc.campaign"] + _stage2_argv(args, leaf))
    lines = [
        "#!/bin/bash", "#SBATCH --job-name=campaign_stage2",
        f"#SBATCH --account={args.account}", f"#SBATCH --partition={args.partition}",
        "#SBATCH --nodes=1 --ntasks=1", "#SBATCH --time=00:20:00",
        f"#SBATCH --output={Path(args.work_dir)/'results'}/%x_%j.out",
        f"#SBATCH --error={Path(args.work_dir)/'results'}/%x_%j.err",
        "set -euo pipefail", "source /etc/profile.d/modules.sh 2>/dev/null || true",
        f"module load {args.python_module}" if args.python_module else "",
        args.activate_env, f"export PYTHONPATH={args.toddlers_src}:${{PYTHONPATH:-}}",
        f"export CLOUDY_EXE={args.cloudy_exe} CLOUDY_DATA_DIR={args.cloudy_data}",
        f'echo "stage-2: enumerating Cloudy from {leaf} and submitting the chain"',
        inner,
    ]
    script = _write_script(args.work_dir, "campaign_stage2.sh", "\n".join(l for l in lines if l) + "\n")
    jid = _sbatch(script, dependency=evo_job, dry_run=args.dry_run)
    print(f"stage-2 (cloudy+STAB, deferred): job {jid} (afterok {evo_job})")
    return jid


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m toddlers.hpc.campaign",
        description="End-to-end STAB campaign on SLURM (evolution -> Cloudy -> interpolant -> STAB).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--grid", help="evolution grid JSON (full run: generate + run evolution first)")
    src.add_argument("--evolution-dir", help="existing evolution-output dir (start-from-existing)")

    p.add_argument("--work-dir", default="runs", help="where tasks/ and results/ go")
    p.add_argument("--stab-dir", default="examples/stab", help="dir holding the STAB scripts")
    p.add_argument("--pattern", default="*.dat", help="glob for evolution .dat (cloudy stage)")

    # Cloudy options
    p.add_argument("--dust-to-metal", type=float, nargs="+", default=None,
                   help="DTM value(s). Omit = fiducial single run (no DTM axis); a list = DTM sweep (5D STAB).")
    p.add_argument("--add-dig", action="store_true", help="include DIG models (default off; paper variable-DTM has none)")
    p.add_argument("--time-method", default="toddlers_v1",
                   choices=["adaptive", "uniform", "toddlers_v1"],
                   help="Cloudy time grid (default toddlers_v1, required by the STAB interpolant)")
    p.add_argument("--continue-after-dissolution", action=argparse.BooleanOptionalAction, default=True,
                   help="run post-dissolution models (default on; the STAB grid spans the full time range)")
    p.add_argument("--stab", choices=["both", "cloud", "sfr", "none"], default="both",
                   help="which STAB families to build in post-processing")
    p.add_argument("--archive-cloudy", action=argparse.BooleanOptionalAction, default=True,
                   help="after the Cloudy grid is complete, tar each parameter dir's "
                        "non-essential diagnostic files (keeping .in/.out/.cont/.phy/.rad "
                        "loose) and remove the originals, to stay under the cluster's inode "
                        "quota on large grids. Reversible via "
                        "`python -m toddlers.hpc.archive_cloudy_output <dir> --untar`. "
                        "Default on; use --no-archive-cloudy to keep all files loose.")
    p.add_argument("--max-resume-rounds", type=int, default=3,
                   help="how many times the post-process resume-gate will resubmit unfinished "
                        "Cloudy tasks (failed or timed-out) before building anyway. Guards "
                        "against a persistently-failing model looping forever.")
    p.add_argument("--keep-interp-cache", action="store_true",
                   help="keep the per-DTM interpolant cache (toddlers/stab/cache; parsed Cloudy "
                        "data, size scales with the grid) instead of clearing it before each DTM "
                        "build. Off by default: a DTM sweep keeps only one DTM's cache at a time, "
                        "else the caches accumulate per DTM and can exceed the data-partition "
                        "quota. Turn on to inspect the cached parsed-Cloudy data when debugging "
                        "(needs the disk).")
    p.add_argument("--cache-dir", default="",
                   help="directory for the interpolant build cache (a large transient of parsed "
                        "Cloudy data whose size scales with the grid). Default: the package dir "
                        "(toddlers/stab/cache) on the code filesystem. Point it at SCRATCH for "
                        "large DTM sweeps so the cache does not consume the home/data-partition "
                        "quota (sets TODDLERS_INTERP_CACHE for the build).")

    # cluster parameters
    p.add_argument("--account", required=True)
    p.add_argument("--partition", required=True)
    p.add_argument("--ntasks", type=int, default=128,
                   help="cores per node (worker-pool size per node)")
    p.add_argument("--max-nodes", type=int, default=1,
                   help="max nodes to scale a phase across. Each phase is sized to its task "
                        "count: nodes=ceil(tasks/cores_per_node) capped here. Raise it for "
                        "large grids (e.g. a DTM sweep) for wall-clock speed without wasting "
                        "CPU-hours (same core-hours, fewer waves).")
    p.add_argument("--walltime", default="03:00:00", help="walltime per Cloudy/evolution phase")
    p.add_argument("--stab-walltime", default="06:00:00", help="walltime for the post-processing job")
    p.add_argument("--python-module", default="", help="module to load (e.g. SciPy-bundle/2024.05-gfbf-2024a)")
    p.add_argument("--activate-env", default="", help="shell command to activate the toddlers env (conda/venv)")
    p.add_argument("--toddlers-src", required=True, help="path to the package src/ (added to PYTHONPATH)")
    p.add_argument("--toddlers-data", default=None, help="TODDLERS_DATA dir (default: <src>/toddlers/database)")
    p.add_argument("--cloudy-exe", default="cloudy.exe")
    p.add_argument("--cloudy-data", default="", help="Cloudy data dir (CLOUDY_DATA_DIR)")

    p.add_argument("--dry-run", action="store_true", help="print the generate_tasks/sbatch commands, submit nothing")
    args = p.parse_args(argv)

    os.makedirs(Path(args.work_dir) / "results", exist_ok=True)

    if args.grid:
        # From scratch: Cloudy enumeration reads the evolution .dat, which don't exist
        # until evolution finishes, so the Cloudy+STAB stage is deferred to a dependent
        # stage-2 job. The grid JSON gives the evolution leaf dir up front.
        leaf = _evolution_leaf_from_grid(args.grid, args.toddlers_src)
        taskfile = _gen_evolution_tasks(args)
        evo_job = _submit_evolution(args, taskfile)
        _submit_stage2(args, evo_job, leaf)
        print(f"\nFrom-scratch submitted: evolution -> (stage-2) Cloudy chain + STAB on {leaf}.")
    else:
        # Start-from-existing: the .dat are present, so enumerate + chain now (one shot).
        taskdir = _gen_cloudy_tasks(args)
        cloudy_jobs = _submit_cloudy_chain(args, taskdir, after=None)
        if args.stab != "none":
            _submit_postprocess(args, taskdir, cloudy_jobs)

    print("\nSubmitted. Track with: squeue --me ; "
          "resume failures with: python -m toddlers.hpc.check_status --task-file <phase>.tasks "
          "--results 'results/<phase>_*.results.*' -o <phase>.resume.tasks")


if __name__ == "__main__":
    main()
