#!/usr/bin/env python3
"""The per-DTM coordinator must build a correct single-DTM campaign command.

In particular it must forward --activate-env verbatim (the seam used to inject the extra
'module load' lines some clusters need, e.g. Tier-2 where matplotlib/h5py are separate
modules from SciPy-bundle), preserving embedded newlines, and omit the flag when empty.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toddlers.hpc import dtm_sweep as S


def _args(**over):
    base = dict(
        evolution_dir="/scratch/evo", work_root="/scratch/work", cloudy_output="/scratch/cloudy",
        stab_dir="/scratch/stab", cache_dir="/scratch/cache", prefix="BPASS_chab100_bin",
        small_to_large_ratio=0.40, account="gvo00053", partition="doduo", ntasks=96, max_nodes=8,
        walltime="3-00:00:00", stab_walltime="12:00:00", toddlers_src="/data/toddlers_public/src",
        cloudy_exe="/data/cloudy/source/cloudy.exe", cloudy_data="/data/cloudy/data",
        python_module="SciPy-bundle/2024.05-gfbf-2024a", max_resume_rounds=10, activate_env="",
    )
    base.update(over)
    return argparse.Namespace(**base)


def _pairs(cmd):
    return {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1)}


def test_single_dtm_campaign_shape():
    cmd = S._campaign_cmd(_args(), 0.1)
    p = _pairs(cmd)
    assert p["--dust-to-metal"] == "0.1"
    assert p["--stab"] == "none"
    assert p["--small-to-large-ratio"] == "0.4"
    assert p["--evolution-dir"] == "/scratch/evo"


def test_activate_env_forwarded_with_newlines():
    env = "module load matplotlib/3.9.2-gfbf-2024a\nmodule load h5py/3.12.1-foss-2024a"
    cmd = S._campaign_cmd(_args(activate_env=env), 1.0)
    assert "--activate-env" in cmd
    # exact value preserved, including the embedded newline that splits it into two module lines
    assert cmd[cmd.index("--activate-env") + 1] == env
    assert cmd[cmd.index("--activate-env") + 1].count("\n") == 1


def test_activate_env_omitted_when_empty():
    cmd = S._campaign_cmd(_args(activate_env=""), 1.0)
    assert "--activate-env" not in cmd


def test_work_dir_uses_suffix_label():
    # 1e-3 must not collapse; DTM=1.0 gets the literal '1.00' work-dir tag
    assert S._suffix(0.001) == "_dtm0.001"
    assert S._suffix(1.0) == ""
