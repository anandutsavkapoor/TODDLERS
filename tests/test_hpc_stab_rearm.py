#!/usr/bin/env python3
"""The generated STAB gate script must make the build self-re-arming.

The interpolant + STAB build runs after the Cloudy resume gate and can exceed the job
walltime (one DTM-keyed interpolant cache is rebuilt per DTM). Without a self-re-arm the
build dies on TIMEOUT with no recovery (it sits past the Cloudy re-arm branch). This test
checks the generated ``campaign_stab.sh`` carries the build self-re-arm scaffolding:

  * a STAB_DONE completion sentinel, cleared only on a FRESH launch and guarded at the top
    so a finished build (or a stale sentinel's absence) is handled correctly;
  * a build-round successor armed via ``sbatch --dependency=afterany`` BEFORE the ``cd`` into
    the stab dir (so the relative script path resolves);
  * the interpolant cache cleared only on the first build entry (BUILD_ROUND==0), so a
    re-armed build resumes (interpolant_exists skips finished DTMs) instead of restarting;
  * the sentinel written after the build finishes.
"""
import sys
import types
from pathlib import Path

project_root = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(project_root))

from toddlers.hpc import campaign


def _gen(tmp_path, keep_interp_cache=False):
    evo = "/fake/evolution_output/template_BPASS/imf_chab100/star_type_bin/cluster_mode_burst/profile_type_uniform"
    args = types.SimpleNamespace(
        account="acc", partition="part", ntasks=128, stab_walltime="06:00:00",
        work_dir=str(tmp_path), toddlers_src="/src", cloudy_exe="/cl.exe", cloudy_data="/cldata",
        python_module="SciPy", activate_env="", max_resume_rounds=3, max_nodes=8,
        stab_dir=str(tmp_path / "stab"), dust_to_metal=[0.02, 0.1, 1.0], evolution_dir=evo,
        stab="both", archive_cloudy=True, keep_interp_cache=keep_interp_cache, dry_run=True)
    campaign._submit_postprocess(args, tmp_path / "tasks",
                                 {"shell": "1", "unified": "2", "dissolved": "3"})
    return (tmp_path / "campaign_stab.sh").read_text(), args


def test_build_selfrearm_scaffolding_present(tmp_path):
    txt, _ = _gen(tmp_path)
    assert "STAB_DONE=" in txt
    assert 'if [ "$ROUND" -eq 0 ] && [ "$BUILD_ROUND" -eq 0 ]; then rm -f "$STAB_DONE"; fi' in txt
    assert 'if [ -f "$STAB_DONE" ]; then' in txt and "exit 0" in txt
    assert "--dependency=afterany:$SLURM_JOB_ID" in txt
    assert "BUILD_ROUND=$((BUILD_ROUND+1))" in txt
    assert 'touch "$STAB_DONE"' in txt


def test_rearm_before_cd_and_cache_bounded_per_dtm(tmp_path):
    txt, args = _gen(tmp_path)
    # successor armed while cwd is still the WorkDir (relative self_path resolves)
    assert txt.index("armed build successor") < txt.index("cd " + args.stab_dir)
    # cache is bounded to one DTM: cleared once per DTM build (not all kept at once),
    # which is what keeps the data-partition footprint flat across a DTM sweep
    assert "CACHE_DIR=" in txt
    n_clears = txt.count('rm -f "$CACHE_DIR"/*.pkl')
    n_dtm = txt.count("interpolant for DTM")
    assert n_dtm == 3 and n_clears == n_dtm        # one clear before each of the 3 DTMs
    # the clear precedes the actual interpolant build (the python call), not just the label
    assert txt.index('rm -f "$CACHE_DIR"/*.pkl') < txt.index("python3 -m toddlers.stab.interpolants")
    # sentinel written only after the build completes
    assert txt.index('touch "$STAB_DONE"') > txt.index("campaign post-processing done")


def test_keep_interp_cache_retains_but_drops_stale_once(tmp_path):
    txt, _ = _gen(tmp_path, keep_interp_cache=True)
    # no per-DTM clears (caches kept for debugging)
    assert txt.count('rm -f "$CACHE_DIR"/*.pkl') == 1          # only the one-time fresh clear
    assert "[keep-interp-cache]" in txt
    # the single clear is the fresh-build (BUILD_ROUND==0) cross-run-stale drop, before the DTM loop
    assert 'if [ "$BUILD_ROUND" -eq 0 ]; then echo "  [keep-interp-cache]' in txt
    assert txt.index('rm -f "$CACHE_DIR"/*.pkl') < txt.index("python3 -m toddlers.stab.interpolants")
