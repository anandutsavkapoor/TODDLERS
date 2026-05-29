#!/usr/bin/env python3
"""`--stab none` must still build the per-DTM interpolant (just no STAB families).

Regression test: the start-from-existing path used to gate the whole post-processing job on
`args.stab != "none"`, so `--stab none` ran only the Cloudy chain and never built the SED
interpolant or wrote the .stab_build_complete sentinel. The per-DTM dtm_sweep coordinator drives
campaign with exactly `--stab none` and then polls that sentinel + consumes the interpolant .pkl,
so the gate left the coordinator with nothing to wait for / assemble. Post-processing must always
run; `--stab` only selects which STAB *families* (cloud / sfr) are layered on top.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toddlers.hpc import campaign


def _run_main(tmp_path, stab):
    evo = (tmp_path / "evolution_output/template_BPASS/imf_chab100/star_type_bin/"
           "cluster_mode_burst/profile_type_uniform")
    evo.mkdir(parents=True)
    (evo / "sim_Z0.020_eta0.050_n160.0_logM6.00.dat").write_text("{}")
    work = tmp_path / "work"
    campaign.main([
        "--evolution-dir", str(evo), "--work-dir", str(work), "--stab-dir", str(tmp_path / "stab"),
        "--dust-to-metal", "1.0", "--stab", stab,
        "--account", "acc", "--partition", "doduo",
        "--toddlers-src", "/src", "--cloudy-exe", "/cl.exe", "--cloudy-data", "/d",
        "--python-module", "SciPy", "--dry-run",
    ])
    return work / "campaign_stab.sh"


def test_stab_none_still_submits_interpolant_build(tmp_path):
    stab_sh = _run_main(tmp_path, "none")
    assert stab_sh.exists(), "post-processing job not generated for --stab none"
    txt = stab_sh.read_text()
    assert "toddlers.stab.interpolants" in txt          # interpolant is always built
    assert 'touch "$STAB_DONE"' in txt                  # ...and the completion sentinel written
    # but NO STAB families with --stab none
    assert "cloud_family_stab" not in txt
    assert "sfr_normalized_stab" not in txt
    assert "sfr_scaled_seds" not in txt


def test_stab_both_adds_families(tmp_path):
    txt = _run_main(tmp_path, "both").read_text()
    assert "toddlers.stab.interpolants" in txt
    assert "cloud_family_stab" in txt
    assert "sfr_normalized_stab" in txt
