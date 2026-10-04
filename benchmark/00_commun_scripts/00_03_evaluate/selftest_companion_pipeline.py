#!/usr/bin/env python3
"""End-to-end SELF-TEST of the EVAL-ONLY-COMPANION chain on synthetic data, in a throwaway tree (cannot touch real results). Run it for a new companion's naming
(or after touching roster_lib.sh / run_all_*_cross_common.sh / evaluate_companion_run_common.sh / write_companion_configs_from_roster.py / ladder_from_roster.run_companion).

Builds under --workdir: a fake SOURCE task (pins + run dirs shaped like nnU-Net output, so the trainer class and dataset id can be parsed) and a fake COMPANION (tiny NIfTI
ground truth `labelsTs_<item>`, fake 'predictions' = copies of GT in the companion layout). Then checks:
  1. predict roster loop (stub shim, no GPU): one call per (source contrast x pinned method) with the right RUN_ID / METHOD / CATEGORY / TRAINER / DATASET_ID parsed from the run dir;
  2. evaluate driver (REAL evaluate.py, inline): Dice 1.0, per-item layout, ladder rungs under ablations/, plus the guards: a missing prediction FAILS, a wrong CATEGORY FAILS;
  3. config generation -> aggregate -> combined -> companion ladder on SYNTHETIC metrics for the full roster (11 methods x contrasts x items x 3 folds).
CPU only; submit through run_job:
  run_job --name selftest_comp --gpus 0 --cpus 4 --mem 16G --time 00:30:00 --wait -- .venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/selftest_companion_pipeline.py \
      --source ispy2 --companion ispy1 --model-type ispy2_model --prefix ISPY2 --contrasts t1wce t2w --items t1wce precontrast
"""
from __future__ import annotations
import argparse, csv, os, subprocess, sys
from pathlib import Path
import numpy as np
import nibabel as nib

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from roster_runs import HEADLINE_METHODS, LADDER  # noqa: E402

CATEGORY = {"baseline": "nnUNet", "v26_6_2_train050_val100": "nnUNet"}
TRAINER = {"baseline": "Baseline", "auglabAug_v26_6_2_train050_val000": "AugLabDualVal", "auglabAug_v26_6_2_train050_val100": "AugLabDualVal", "v26_6_2_train050_val100": "AugLabValSynth"}
BASE = {"baseline": .40, "synthseg_noEM": .45, "synthseg_EM": .47, "auglab_default": .50, "srcsm": .52, "auglabAug_v26_6_2_train050_val000": .58,
        "auglabAug_v26_6_2_train050_val100": .57, "baseline_kmeans": .42, "baseline_kmeans_label_remap": .44, "baseline_kmeans_label_remap_voronoi": .46, "v26_6_2_train050_val100": .54}


def sh(cmd, env, check=True):
    r = subprocess.run(["bash", "-c", cmd], env=env, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2500:]); sys.exit(f"SELFTEST FAILED: {cmd[:120]}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True); ap.add_argument("--companion", required=True); ap.add_argument("--model-type", required=True)
    ap.add_argument("--prefix", required=True, help="source env-block prefix, e.g. ISPY2"); ap.add_argument("--contrasts", nargs="+", required=True)
    ap.add_argument("--items", nargs="+", required=True); ap.add_argument("--task", default="selftest_task"); ap.add_argument("--workdir")
    a = ap.parse_args()
    W = Path(a.workdir or Path(os.environ.get("SCRATCH", "/tmp")) / f"selftest_companion_{a.companion}").resolve()
    S = W / "benchmark/02_tasks" / a.task / a.source; C = W / "benchmark/02_tasks" / a.task / a.companion
    SR, CR = S / f"8_results_{a.source}", C / f"8_results_{a.companion}"
    methods = list(dict.fromkeys(HEADLINE_METHODS + [m for _, _, m, _ in LADDER]))
    ablation = {m for _, _, m, ab in LADDER if ab}
    rng = np.random.default_rng(0)
    # ---------------- source: pins + run dirs + dataset.json
    ids = {c: 100 + i for i, c in enumerate(a.contrasts)}
    for c in a.contrasts:
        pins = []
        for m in methods:
            cat = CATEGORY.get(m, "auglab"); rid = f"{a.source}_{c}_{m}_20990101_000000"; pins.append(f"{m}\t{cat}\t{rid}")
            td = SR / "01_predictions" / a.model_type / c / cat / rid / f"Dataset{ids[c]}_FAKE" / f"nnUNetTrainer{a.source.upper()}{TRAINER.get(m, 'AugLabDefault')}__nnUNetPlans__3d_fullres" / "fold_0"
            td.mkdir(parents=True)
        (SR / "01_predictions" / a.model_type / c / "roster_run_ids.tsv").write_text("\n".join(pins) + "\n")
    djp = S / f"2_nnUNet_{a.source}/raw/Dataset100_FAKE/dataset.json"; djp.parent.mkdir(parents=True, exist_ok=True)
    djp.write_text('{"labels": {"background": 0, "lesion": 1}, "channel_names": {"0": "x"}, "numTraining": 1, "file_ending": ".nii.gz"}')
    # ---------------- companion: tiny GT + (fake) predictions for a SUBSET roster (3 methods, 1 contrast) for the real evaluator
    cases = [f"{a.companion}_{i:03d}" for i in range(1, 6)]
    raw = C / f"2_nnUNet_{a.companion}/raw"
    for it in a.items:
        (raw / f"labelsTs_{it}").mkdir(parents=True)
        for cs in cases:
            m = np.zeros((14, 14, 14), np.uint8); o = int(rng.integers(2, 6)); m[o:o + 5, o:o + 5, o:o + 5] = 1
            nib.save(nib.Nifti1Image(m, np.eye(4)), str(raw / f"labelsTs_{it}" / f"{cs}.nii.gz"))
    c0 = a.contrasts[0]; sub = ["baseline", "baseline_kmeans", "auglabAug_v26_6_2_train050_val000"]
    for m in sub:
        cat = CATEGORY.get(m, "auglab"); rid = f"{a.source}_{c0}_{m}_20990101_000000"
        for f in (0,):
            for it in a.items:
                d = CR / "01_predictions" / a.model_type / c0 / cat / rid / f"fold{f}" / it; d.mkdir(parents=True)
                for g in (raw / f"labelsTs_{it}").glob("*.nii.gz"):
                    (d / g.name).symlink_to(g)

    base_env = dict(os.environ, PROJECT_ROOT=str(REPO), SCRATCH=os.environ.get("SCRATCH", "/tmp"))
    # env.sh stand-in for the companion (what the scaffolded env.sh exports)
    envsh = W / "env_companion.sh"
    envsh.write_text(f'''export PROJECT_ROOT="{REPO}" DATASET_NAME="{a.companion}"
export nnUNet_raw="{raw}" PREDICTIONS_ROOT="{CR}/01_predictions" METRICS_ROOT="{CR}/02_metrics"
export {a.prefix}_PREDICTIONS_ROOT="{SR}/01_predictions" {a.prefix}_MODEL_TYPE="{a.model_type}" {a.prefix}_DATASET_JSON="{djp}"
source "{REPO}/scripts/job_runner/run_job.sh"
''')
    # ---------------- 1. predict roster loop with a stub shim
    calls = W / "predict_calls.tsv"; stub = W / "stub_shim.sh"
    stub.write_text(f'echo -e "${{1}}\\t${{METHOD}}\\t${{CATEGORY}}\\t${{TRAINER}}\\t${{{a.prefix}_DATASET_ID}}\\t${{{a.prefix}_TRAINING_CONTRAST}}\\t${{2}}" >> {calls}\n')
    sh(f'source {envsh}; export SOURCE_PREFIX={a.prefix} SOURCE_CONTRASTS="{" ".join(a.contrasts)}" PREDICT_SHIM={stub}; '
       f'source {REPO}/benchmark/00_commun_scripts/00_02_predict/run_all_predict_cross_common.sh', base_env)
    rows = [l.split("\t") for l in calls.read_text().strip().splitlines()]
    assert len(rows) == len(methods) * len(a.contrasts), f"predict loop made {len(rows)} calls, expected {len(methods) * len(a.contrasts)}"
    for rid, m, cat, tr, dsid, c, fold in rows:
        assert rid == f"{a.source}_{c}_{m}_20990101_000000" and cat == CATEGORY.get(m, "auglab") and fold == "all", (rid, m, cat, fold)
        assert tr == f"nnUNetTrainer{a.source.upper()}{TRAINER.get(m, 'AugLabDefault')}" and dsid == str(ids[c]), (tr, dsid, m, c)
    print(f"1. predict roster loop OK ({len(rows)} calls, trainer + dataset id parsed from run dirs)")
    # ---------------- 2. evaluate driver (real evaluate.py), inline
    shim = W / "06_01_shim.sh"
    shim.write_text(f'''set -euo pipefail
source {envsh}
SOURCE_PREFIX={a.prefix}; EVAL_ITEMS="{" ".join(a.items)}"; EVAL_LABELS="lesion"; EVAL_JOB_PREFIX="selftest_eval"; EVAL_FOLDS="0"; export EVAL_INLINE=1
source {REPO}/benchmark/00_commun_scripts/00_03_evaluate/evaluate_companion_run_common.sh "$@"
''')
    sh(f'export SOURCE_PREFIX={a.prefix} SOURCE_CONTRASTS="{c0}" EVAL_SCRIPT={shim} ROSTER_ONLY="{" ".join(sub)}" ROSTER_SKIP_MISSING=0; source {envsh}; '
       f'source {REPO}/benchmark/00_commun_scripts/00_03_evaluate/run_all_evaluate_cross_common.sh', base_env)
    mroot = CR / "02_metrics" / a.model_type / c0
    for m in sub:
        cat = CATEGORY.get(m, "auglab"); rid = f"{a.source}_{c0}_{m}_20990101_000000"; abl = "ablations/" if m in ablation else ""
        for it in a.items:
            f = mroot / (abl) / it / f"{cat}_{rid}" / "fold0" / "eval_all.csv"
            assert f.exists(), f"missing {f}"
            rows_ = list(csv.DictReader(open(f)))
            assert len(rows_) == len(cases) and all(float(r["dice"]) == 1.0 for r in rows_), f"{f}: expected Dice 1.0 on {len(cases)} cases"
    print("2. evaluate driver OK (Dice 1.0, per-item layout, ladder rung under ablations/)")
    # guards: missing prediction, wrong category
    victim = next((CR / "01_predictions" / a.model_type / c0 / "nnUNet" / f"{a.source}_{c0}_baseline_20990101_000000" / "fold0" / a.items[0]).glob("*.nii.gz")); victim.unlink()
    r = sh(f'bash {shim} {a.source}_{c0}_baseline_20990101_000000 nnUNet {c0} 0', base_env, check=False)
    assert r.returncode != 0 and "predictions vs" in (r.stdout + r.stderr), "missing prediction did not fail"
    r = sh(f'bash {shim} {a.source}_{c0}_baseline_20990101_000000 auglab {c0} 0', base_env, check=False)
    assert r.returncode != 0 and "no predictions at" in (r.stdout + r.stderr), "wrong CATEGORY did not fail"
    print("   guards OK (missing prediction and wrong CATEGORY both fail)")
    # ---------------- 3. configs / aggregate / combined / ladder on synthetic metrics for the FULL roster
    for c in a.contrasts:
        for m in methods:
            cat = CATEGORY.get(m, "auglab"); rid = f"{a.source}_{c}_{m}_20990101_000000"; abl = "ablations" if m in ablation else ""
            for it in a.items:
                for f in range(3):
                    fd = CR / "02_metrics" / a.model_type / c / abl / it / f"{cat}_{rid}" / f"fold{f}"; fd.mkdir(parents=True, exist_ok=True)
                    with open(fd / "eval_all.csv", "w", newline="") as fh:
                        w = csv.writer(fh); w.writerow(["group", "case", "label", "dice", "hd95"])
                        for cs in cases:
                            w.writerow([it, cs, "lesion", float(np.clip(BASE[m] + rng.normal(0, .1), 0, 1)), float(abs(rng.normal(30 - 40 * BASE[m], 8)))])
    env = dict(os.environ, PROJECT_ROOT=str(W)); py = sys.executable
    sh(f'{py} {HERE}/write_companion_configs_from_roster.py --companion-root {C} --source-root {S} --model-type {a.model_type} --contrasts {" ".join(a.contrasts)} --items {" ".join(a.items)}', env)
    cfg = C / f"5_scripts_{a.companion}/06_evaluate/configs"
    for c in a.contrasts:
        for it in a.items:
            sh(f'{py} {HERE}/aggregate_from_config.py {cfg}/{a.companion}_{it}_{c}_01_results.yaml', env)
    sh(f'{py} {HERE}/combined_modality_summary.py {cfg}/{a.companion}_combined_01_results.yaml', env)
    for c in a.contrasts:
        for it in a.items:
            sh(f'{py} -c "import sys; sys.path.insert(0, {str(HERE)!r}); from ladder_from_roster import run_companion; '
               f'run_companion(companion_root={str(C)!r}, source_root={str(S)!r}, model_type={a.model_type!r}, contrast={c!r}, item={it!r}, task_name=\'selftest {c} {it}\')"', env)
    M = CR / "02_metrics" / a.model_type
    expect = [M / c / it / "01_results_summary.md" for c in a.contrasts for it in a.items] + [M / "combined_contrasts" / "01_results_summary.md"] \
        + [M / c / "ablations" / it / "ladder_summary.md" for c in a.contrasts for it in a.items]
    missing = [str(p) for p in expect if not p.exists()]
    if missing:
        sys.exit("SELFTEST FAILED, missing outputs:\n  " + "\n  ".join(missing))
    print(f"3. configs + aggregate + combined + ladders OK ({len(expect)} outputs)")
    print("SELFTEST PASSED")


if __name__ == "__main__":
    main()
