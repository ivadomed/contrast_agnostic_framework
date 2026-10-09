#!/usr/bin/env python3
"""Scaffold an EVAL-ONLY COMPANION dataset's scripts (a dataset used only for TESTING another task's trained models) on the shared drivers in
benchmark/00_commun_scripts. Sibling of create_dataset_structure.py (empty 9-subdir skeleton: run THAT first) and create_pipeline_scripts.py (training tasks).

Writes under <dataset>/5_scripts_<dataset>/ (refuses to overwrite unless --force):
  00_utils/env.sh                      companion env + the SOURCE block (<PREFIX>_* vars read by predict_common.sh cross mode and the evaluate driver)
  02_nnunet/02_01_convert_test.py/.sh  TEMPLATE converter (BIDS -> flat imagesTs_<item>/labelsTs_<item>): hooks for exclusions, LPS reorient, FOV crop, asserts, manifest
  02_nnunet/02_02_fov_audit.sh         shared fov_audit.py: test geometry vs the source's TRAINING geometry (decides whether/what to crop)
  05_predict/05_01_predict_<src>_common.sh, 05_02_run_all_predict.sh, 05_03_tamia_pack_predict.sh      roster-driven over the SOURCE's pinned runs (no per-method wrappers)
  06_evaluate/06_00..06_07 + 06_1X ladders (one per source contrast x item)
  and repo-level scripts/cluster/tamia_env_<dataset>.sh
Everything reads the SOURCE roster pins (<src>/8_results_<src>/01_predictions/<model_type>/<contrast>/roster_run_ids.tsv); for sources that predate the roster drivers create them once
with 00_00_utils/bootstrap_source_pins.py. Trainer class and dataset id come from the source run dirs (no mapping). Verified by 00_03_evaluate/selftest_companion_pipeline.py.

  .venv/bin/python benchmark/create_eval_companion_scripts.py --task <task> --dataset <slug> --source <src slug> --source-task <task> --source-prefix ISPY2 \
      --model-type ispy2_model --source-contrasts t1wce t2w --items t1wce precontrast --labels tumour [--source-dataset-json <path rel. to source root>]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def files(a):
    d, s, P = a.dataset, a.source, a.source_prefix
    cont = " ".join(a.source_contrasts); items = " ".join(a.items)
    F = {}
    F["00_utils/env.sh"] = f'''#!/usr/bin/env bash
# Source at the top of every {d} pipeline script.   source "$(dirname "$0")/../00_utils/env.sh"
# >>> SCAFFOLDED by benchmark/create_eval_companion_scripts.py: review the header, BIDS_SUBDIR and the resource lines.
# {d} = EVAL-ONLY companion of the {s} task: its test items ({items}) are predicted with the {s}-trained models (source contrasts: {cont}) and scored here.
# No training role, no Dataset<id>/ dir: flat 2_nnUNet_{d}/raw/{{imagesTs,labelsTs}}_<item>/ . License / citation: see 1_BIDS_{d}/<leaf>/README + LICENSE.
DATASET_ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../.." && pwd)"
export DATASET_NAME="{d}"
export MODEL_TYPE="{a.model_type}"                       # results are filed under the SOURCE model type (predict cross mode / evaluate driver)
export DATASET_ROLE="eval_only"
export TRAINING_CONTRAST="${{TRAINING_CONTRAST:-{a.source_contrasts[0]}}}"
export RUN_JOB_TIME_DEFAULT="${{RUN_JOB_TIME_DEFAULT:-04:00:00}}"
# set BEFORE common_env.sh (it sources run_job and freezes the defaults); adapt to the test volumes' size
export RUN_JOB_CPUS_PER_GPU="${{RUN_JOB_CPUS_PER_GPU:-8}}"
export RUN_JOB_MEM_PER_GPU="${{RUN_JOB_MEM_PER_GPU:-40G}}"
export BIDS_SUBDIR="{a.bids_leaf or 'TODO-bids-leaf'}"
CE_SUBDIRS="preprocessed splits"
source "${{DATASET_ROOT}}/../../../00_commun_scripts/00_00_utils/common_env.sh"
export METRICS_ROOT="${{METRICS_ROOT:-${{DATASET_ROOT}}/8_results_{d}/02_metrics}}"
export CHECKPOINTS_DIR="${{DATASET_ROOT}}/6_checkpoints_{d}"
export RESULTS_DIR="${{DATASET_ROOT}}/8_results_{d}"

# ── SOURCE block ({s}): every var guarded (${{VAR:-default}}) so cluster override files (tamia_env_{d}.sh) win ──
export {P}_DATASET_ROOT="${{{P}_DATASET_ROOT:-${{DATASET_ROOT}}/../../{a.source_task}/{s}}}"
export {P}_PREDICTIONS_ROOT="${{{P}_PREDICTIONS_ROOT:-${{{P}_DATASET_ROOT}}/8_results_{s}/01_predictions}}"
export {P}_NNUNET_RAW="${{{P}_NNUNET_RAW:-${{{P}_DATASET_ROOT}}/2_nnUNet_{s}/raw}}"
export {P}_NNUNET_PREPROCESSED="${{{P}_NNUNET_PREPROCESSED:-${{{P}_DATASET_ROOT}}/2_nnUNet_{s}/preprocessed}}"
export {P}_MODEL_TYPE="{a.model_type}"
export {P}_TRAINING_CONTRAST="${{{P}_TRAINING_CONTRAST:-{a.source_contrasts[0]}}}"   # the roster driver sets these two per source contrast / run
export {P}_DATASET_ID="${{{P}_DATASET_ID:-TODO}}"
# label numbering of the source (background 0, foreground 1 ...): the companion's GT masks MUST use the same numbering (else add --label_map in the evaluate step)
export {P}_DATASET_JSON="${{{P}_DATASET_JSON:-${{{P}_NNUNET_RAW}}/{a.source_dataset_json or 'Dataset<ID>_<Name>/dataset.json'}}}"
export PYTHONPATH="${{{P}_DATASET_ROOT}}/5_scripts_{s}:${{PYTHONPATH:-}}"      # the source's trainer package (nnUNetTrainer<NAME>*), needed by nnUNetv2_predict
'''
    F["02_nnunet/02_01_convert_test.py"] = f'''#!/usr/bin/env python3
"""TEMPLATE -- complete every TODO. BIDS (1_BIDS_{d}/<leaf>/) -> flat nnU-Net TEST set 2_nnUNet_{d}/raw/{{imagesTs,labelsTs}}_<item>/  for items: {items}.

Contract (each point cost this project a real debugging session):
  * case ids IDENTICAL across items and carry NO contrast suffix ("{d}_<id>"), so the patient-level significance merge pairs them;
  * pathology/stage/timepoint EXCLUSIONS (bilateral disease, implants, empty or garbage masks, label-side disagreement, wrong timepoint ...) are decided BEFORE any prediction,
    applied uniformly and recorded in the manifest with the reason per case;
  * orientation: reorient to LPS with shared orient.reorient_file (records the raw axcodes); look at a rendered PNG afterwards (aff2axcodes is blind to 180-degree flips);
  * GEOMETRY MATCH: run 02_02_fov_audit.sh first. If test FOV/spacing differ from the source's training data, crop/resample HERE (planned, documented, applied before
    prediction -- crop-before-predict, never mask-at-eval): shared helpers unilateral_crop.py (lesion_side_half, ap_skin_window), derive_ap_crop.py (breast), fov.py +
    00_02_predict/fov_crop_predict_evaluate.sh (CHAOS-style S-I slab), or a documented local rule. Name cropped items distinctly (e.g. <item>_crop) and re-audit;
  * assertions per case: image/label grids equal, labels binary/expected values, mask non-empty, and an EMPIRICAL channel/physics check (lesion mean vs brain in the right direction);
  * write {d}_test_manifest.json: per-case shape/spacing/lesion volume, exclusions + reasons, and the lesion-size distribution next to the SOURCE's labelsTr distribution.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import nibabel as nib
import numpy as np

DATASET_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = DATASET_ROOT.parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "00_commun_scripts" / "00_00_utils"))
from orient import TARGET_AXCODES, reorient_file  # noqa: E402,F401
# from unilateral_crop import crop_axis0, lesion_side_half, crop_axis1, ap_skin_window  # noqa: E402  (breast crop helpers)

BIDS = DATASET_ROOT / "1_BIDS_{d}" / "{a.bids_leaf or 'TODO-bids-leaf'}"
OUT = DATASET_ROOT / "2_nnUNet_{d}" / "raw"
ITEMS = {{{", ".join(f'"{i}": "TODO bids suffix/acq for {i}"' for i in a.items)}}}
SOURCE_LABELS_TR = Path("TODO: <source>/2_nnUNet_<src>/raw/Dataset<ID>_<Name>/labelsTr")   # for the size-distribution comparison


def iter_cases():
    """TODO: yield (case_id, {{item: image_path}}, mask_path, meta_dict) from BIDS. case_id = '{d}_<subject>' (no contrast suffix)."""
    raise NotImplementedError


def exclude(case_id, meta):
    """TODO: return a reason string to EXCLUDE this case (pathology/stage/timepoint/label rule fixed before predicting), else None."""
    return None


def main():
    manifest = {{"cases": {{}}, "excluded": {{}}}}
    for cid, imgs, mask, meta in iter_cases():
        why = exclude(cid, meta)
        if why:
            manifest["excluded"][cid] = why; continue
        # TODO: reorient each file to LPS (copy first: reorient_file works in place), crop if the FOV audit demanded it, assert grids/labels/non-empty/physics,
        #       then write OUT/imagesTs_<item>/<cid>_0000.nii.gz and OUT/labelsTs_<item>/<cid>.nii.gz (uint8 {{0,1}}) for every item.
        raise NotImplementedError("fill in the conversion")
    (OUT / "{d}_test_manifest.json").write_text(json.dumps(manifest, indent=1))
    print("cases:", len(manifest["cases"]), "excluded:", len(manifest["excluded"]))


if __name__ == "__main__":
    main()
'''
    F["02_nnunet/02_01_convert_test.sh"] = f'''#!/bin/bash
# BIDS -> flat nnU-Net test set via run_job (CPU).   bash 02_01_convert_test.sh
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REPO_ROOT="$(cd "${{HERE}}/../../../../../.." && pwd)"
LOG_DIR="${{REPO_ROOT}}/benchmark/02_tasks/{a.task}/{d}/2_nnUNet_{d}/logs"; mkdir -p "${{LOG_DIR}}"
source "${{REPO_ROOT}}/scripts/job_runner/run_job.sh"
run_job --name {d}_convert_test --gpus 0 --cpus 2 --mem 8G --time 01:00:00 --log "${{LOG_DIR}}/convert_test_$(date +%Y%m%d_%H%M%S).log" --wait -- \\
    "${{REPO_ROOT}}/.venv/bin/python" "${{HERE}}/02_01_convert_test.py"
'''
    F["02_nnunet/02_02_fov_audit.sh"] = f'''#!/usr/bin/env bash
# FOV/geometry audit: this companion's test items vs the SOURCE's TRAINING images (shared 00_00_utils/fov_audit.py), via run_job (CPU).
# Run BEFORE choosing a crop and again after cropping. MISMATCH lines mean: crop/resample to the training geometry before predicting.
#   bash 02_02_fov_audit.sh [SOURCE_CONTRAST={a.source_contrasts[0]}] [ITEM_DIR_SUFFIX=]        (ITEM_DIR_SUFFIX e.g. _crop to audit cropped items)
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
source "${{HERE}}/../00_utils/env.sh"; cd "${{PROJECT_ROOT}}"
SC="${{1:-{a.source_contrasts[0]}}}"; SUF="${{2:-}}"
TRAIN_DIR="$(ls -d "${{{P}_NNUNET_RAW}}"/Dataset*/imagesTr | head -1)"   # TODO if the source has one Dataset per contrast: pick the one trained on ${{SC}}
mkdir -p "${{RESULTS_DIR}}/_logs"
run_job --name {d}_fov_audit --gpus 0 --cpus 2 --mem 8G --time 00:20:00 --log "${{RESULTS_DIR}}/_logs/fov_audit_$(date +%Y%m%d_%H%M%S).log" --wait -- \\
    .venv/bin/python benchmark/00_commun_scripts/00_00_utils/fov_audit.py --train-dir "${{TRAIN_DIR}}" --test-dir {" ".join(f'"${{nnUNet_raw}}/imagesTs_{i}${{SUF}}"' for i in a.items)}
tail -25 "$(ls -t "${{RESULTS_DIR}}"/_logs/fov_audit_*.log | head -1)"
'''
    F["05_predict/05_01_predict_" + s + "_common.sh"] = f'''#!/usr/bin/env bash
# Cross-dataset predict shim: {s}'s trained models on {d}'s test items. Sourced by the roster driver (05_02) per source run; sets the cross-mode config and hands off to
# the shared benchmark/00_commun_scripts/00_02_predict/predict_common.sh. METHOD/TRAINER/CATEGORY and {P}_TRAINING_CONTRAST/{P}_DATASET_ID are set by the driver.
set -euo pipefail
source "$(dirname "${{BASH_SOURCE[0]}}")/../00_utils/env.sh"
cd "${{PROJECT_ROOT}}"
PREDICT_MODE="cross"
SOURCE_PREFIX="{P}"
PREDICT_JOB_PREFIX="{d}_predict"
PREDICT_LOG_PREFIX="{d}_predict"
PREDICT_ITEMS_DEFAULT="{items}"
PREDICT_FOLD_DEFAULT="all"
PREDICT_TIME="01:00:00"
PREDICT_EXTRA_FLAGS="-npp 4 -nps 2"
source "${{PROJECT_ROOT}}/benchmark/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
'''
    F["05_predict/05_02_run_all_predict.sh"] = f'''#!/usr/bin/env bash
# Predict the whole {s} roster (source contrasts: {cont}; every method pinned in the SOURCE's roster_run_ids.tsv, incl. ladder rungs) on {d}'s items. No timestamps here.
#   bash 05_02_run_all_predict.sh        env: ROSTER_ONLY="m1 m2", ROSTER_SKIP_MISSING=1, CHECKPOINT=checkpoint_final.pth
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
export SOURCE_PREFIX="{P}" SOURCE_CONTRASTS="{cont}"
export PREDICT_SHIM="$(cd "$(dirname "$0")" && pwd)/05_01_predict_{s}_common.sh"
source "${{PROJECT_ROOT}}/benchmark/00_commun_scripts/00_02_predict/run_all_predict_cross_common.sh"
'''
    F["05_predict/05_03_tamia_pack_predict.sh"] = f'''#!/usr/bin/env bash
# TamIA whole-node predict pack for {d} (all source contrasts in ONE pack: the source RUN_IDs name their contrast, so cmd files cannot collide; grep-verified).
# Run ON TamIA after:  source .../00_utils/env.sh ; source scripts/cluster/tamia_env_{d}.sh ; export RUN_JOB_CPUS_PER_GPU=12 RUN_JOB_MEM_PER_GPU=115G
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
export RUN_JOB_PACK_DIR="${{SCRATCH:?}}/{d}/_packruns/predict_$(date +%Y%m%d_%H%M%S)"; mkdir -p "${{RUN_JOB_PACK_DIR}}"
export SOURCE_PREFIX="{P}" SOURCE_CONTRASTS="{cont}" PREDICT_SHIM="${{HERE}}/05_01_predict_{s}_common.sh"
source "${{PROJECT_ROOT}}/benchmark/00_commun_scripts/00_02_predict/run_all_predict_cross_common.sh"
n=$(grep -c . "${{RUN_JOB_PACK_DIR}}/index.tsv"); echo "[pack] recorded ${{n}} fold-predict commands"
PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${{PACK_TIME:-02:00:00}}" PACK_CHAIN=1 PACK_JOB_NAME="{d}_predict" \\
    bash "${{PROJECT_ROOT}}/scripts/job_runner/run_job_pack_submit.sh" "${{RUN_JOB_PACK_DIR}}"
'''
    F[f"06_evaluate/06_00_evaluate_{d}.py"] = f'''#!/usr/bin/env python3
"""{d} evaluator: thin shim over the shared evaluator benchmark/00_commun_scripts/00_03_evaluate/evaluate.py (Dice / HD95 per case per label); same CLI.
Labels: the SOURCE's dataset.json label set; GT numbering must match (else pass --label_map)."""
import os, sys
from pathlib import Path
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise SystemExit("PROJECT_ROOT not set -- source 00_utils/env.sh first")
sys.path.insert(0, str(Path(PROJECT_ROOT) / "benchmark" / "00_commun_scripts" / "00_03_evaluate"))
from evaluate import main  # noqa: E402
if __name__ == "__main__":
    main()
'''
    F["06_evaluate/06_01_evaluate_run.sh"] = f'''#!/usr/bin/env bash
# Evaluate ONE run of the {s} roster on {d}'s items (every fold x item), via the shared 00_03_evaluate/evaluate_companion_run_common.sh.
#   bash 06_01_evaluate_run.sh <RUN_ID> <CATEGORY:nnUNet|auglab> <SOURCE_TRAINING_CONTRAST:{"|".join(a.source_contrasts)}> [FOLD=all]     env: LADDER=1 (rungs), CKPT_TAG, METRICS_SUBDIR n/a
set -euo pipefail
source "$(dirname "${{BASH_SOURCE[0]}}")/../00_utils/env.sh"
SOURCE_PREFIX="{P}"; EVAL_ITEMS="{items}"; EVAL_LABELS="{a.labels}"; EVAL_JOB_PREFIX="{d}_eval"
source "${{PROJECT_ROOT}}/benchmark/00_commun_scripts/00_03_evaluate/evaluate_companion_run_common.sh" "$@"
'''
    for n, tool, what in (("02_aggregate_from_config", "aggregate_from_config.py", "headline table"), ("03_significance_from_config", "significance_from_config.py", "paired significance report"),
                          ("04_combined_modality_summary", "combined_modality_summary.py", "combined (all source contrasts) table")):
        dflt = f"configs/{d}_combined_01_results.yaml" if n.startswith("04") else None
        F[f"06_evaluate/06_{n}.sh"] = f'''#!/usr/bin/env bash
# {d}: {what} via the shared {tool} (never hand-compute tables / p-values). Configs are GENERATED by 06_05_write_configs.sh.
#   bash 06_{n}.sh {'[' + dflt + ']' if dflt else 'configs/<config>.yaml'}
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
source "${{HERE}}/../00_utils/env.sh"; cd "${{PROJECT_ROOT}}"
CFG="${{1{':-' + dflt if dflt else ':?usage: $0 <config.yaml>'}}}"; [[ "$CFG" != /* ]] && CFG="${{HERE}}/${{CFG}}"
.venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/{tool} "${{CFG}}"
'''
    F["06_evaluate/06_05_write_configs.sh"] = f'''#!/usr/bin/env bash
# (Re)generate {d}'s aggregate + combined configs from the SOURCE roster pins (no timestamps) via the shared write_companion_configs_from_roster.py (CPU job).
#   bash 06_05_write_configs.sh [--skip-missing] [--force]
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
source "${{HERE}}/../00_utils/env.sh"; cd "${{PROJECT_ROOT}}"; mkdir -p "${{RESULTS_DIR}}/_logs"
run_job --name {d}_write_configs --gpus 0 --cpus 1 --mem 4G --time 00:10:00 --log "${{RESULTS_DIR}}/_logs/write_configs_$(date +%Y%m%d_%H%M%S).log" --wait -- \\
    .venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/write_companion_configs_from_roster.py --companion-root "${{DATASET_ROOT}}" \\
    --source-root "${{{P}_DATASET_ROOT}}" --model-type "${{MODEL_TYPE}}" --contrasts {cont} --items {items} "$@"
'''
    F["06_evaluate/06_06_run_all_eval.sh"] = f'''#!/usr/bin/env bash
# Evaluate the whole {s} roster on {d}'s items (headline -> <contrast>/<item>/, ladder rungs -> <contrast>/ablations/<item>/, automatic) via the shared
# run_all_evaluate_cross_common.sh. env: ROSTER_ONLY, ROSTER_SKIP_MISSING=1, CKPT_TAG, EVAL_INLINE=1 (inside a job).   bash 06_06_run_all_eval.sh
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
source "${{HERE}}/../00_utils/env.sh"
export SOURCE_PREFIX="{P}" SOURCE_CONTRASTS="{cont}" EVAL_SCRIPT="${{HERE}}/06_01_evaluate_run.sh"
source "${{PROJECT_ROOT}}/benchmark/00_commun_scripts/00_03_evaluate/run_all_evaluate_cross_common.sh"
'''
    lad = "".join(f"\n.venv/bin/python '${{HERE}}/{n}'" for n in ladder_names(a))
    F["06_evaluate/06_07_run_all_aggregation.sh"] = f'''#!/usr/bin/env bash
# Regenerate every {d} table + ladder in ONE CPU job (after 06_06 and 06_05). Never python on the login node.   bash 06_07_run_all_aggregation.sh
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
source "${{HERE}}/../00_utils/env.sh"; cd "${{PROJECT_ROOT}}"; mkdir -p "${{RESULTS_DIR}}/_logs"
run_job --name {d}_aggregation --gpus 0 --cpus 4 --mem 16G --time 01:00:00 --log "${{RESULTS_DIR}}/_logs/aggregation_$(date +%Y%m%d_%H%M%S).log" --wait -- bash -c "
set -euo pipefail
cd '${{PROJECT_ROOT}}'
for c in '${{HERE}}'/configs/{d}_*_01_results.yaml; do
    case \\"\\$c\\" in *combined*) continue;; esac
    bash '${{HERE}}/06_02_aggregate_from_config.sh' \\"\\$c\\"
done
bash '${{HERE}}/06_04_combined_modality_summary.sh'{lad}
"
'''
    for n, (c, it) in zip(ladder_numbers(a), [(c, it) for c in a.source_contrasts for it in a.items]):
        nm = f"06_{n}_ladder_summary_{s}cross_{it}_{c}.py"
        F["06_evaluate/" + nm] = f'''"""{d} cross-dataset causal-ablation ladder: item `{it}` scored with the {s} {c}-TRAINED models (rungs from the SOURCE roster pins, no timestamps).
An item in the SAME contrast as the source's training contrast ({c}) is cross-DATASET evidence only (supplementary; never pooled into an OOD bucket); a different contrast is genuine
held-out-contrast evidence. Pool into the source task's own ladder only by TRUE contrast (see the skill add-eval-companion-end-to-end). Usage: .venv/bin/python {nm}"""
import sys
from pathlib import Path
DATASET_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DATASET_ROOT.parents[2] / "00_commun_scripts" / "00_03_evaluate"))
from ladder_from_roster import run_companion  # noqa: E402
if __name__ == "__main__":
    run_companion(companion_root=DATASET_ROOT, source_root=DATASET_ROOT.parents[1] / "{a.source_task}" / "{s}",
                  model_type="{a.model_type}", contrast="{c}", item="{it}", task_name="{d} {it} ({s} {c.upper()}-trained)")
'''
    return F


def ladder_numbers(a):
    return [f"{10 + i}" for i in range(len(a.source_contrasts) * len(a.items))]


def ladder_names(a):
    return [f"06_{n}_ladder_summary_{a.source}cross_{it}_{c}.py" for n, (c, it) in zip(ladder_numbers(a), [(c, it) for c in a.source_contrasts for it in a.items])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--source", required=True); ap.add_argument("--source-task", required=True); ap.add_argument("--source-prefix", required=True)
    ap.add_argument("--model-type", required=True); ap.add_argument("--source-contrasts", nargs="+", required=True); ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--labels", required=True); ap.add_argument("--source-dataset-json"); ap.add_argument("--bids-leaf")
    ap.add_argument("--out-root"); ap.add_argument("--force", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_root).resolve() if a.out_root else PROJECT / "benchmark/02_tasks" / a.task / a.dataset
    sdir = out / f"5_scripts_{a.dataset}"
    n = 0
    for rel, text in files(a).items():
        p = sdir / rel
        if p.exists() and not a.force:
            print("EXISTS (skipped):", p); continue
        print(("would write " if a.dry_run else "write "), p)
        if not a.dry_run:
            p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
        n += 1
    # TamIA env override
    P = a.source_prefix; d = a.dataset; s = a.source
    tam = (PROJECT / "scripts/cluster" if not a.out_root else out / "scripts_cluster") / f"tamia_env_{d}.sh"
    if not tam.exists() or a.force:
        text = f'''# Source AFTER .../{d}/5_scripts_{d}/00_utils/env.sh on TamIA: points {d}'s test data / results at scratch and the SOURCE ({s}) models at ITS scratch copy.
# Override every path OUTRIGHT (a ${{VAR:-}} guard here is a no-op: env.sh already fired its guards). RUN_JOB_MEM_PER_GPU is in GiB (115G), never raw MB.
export SCRATCH="${{SCRATCH:-/scratch/p/paulh}}"
export nnUNet_raw="$SCRATCH/{d}/2_nnUNet/raw"
export PREDICTIONS_ROOT="$SCRATCH/{d}/8_results/01_predictions"
export METRICS_ROOT="$SCRATCH/{d}/8_results/02_metrics"
mkdir -p "$PREDICTIONS_ROOT" "$METRICS_ROOT"
export {P}_DATASET_ROOT="$SCRATCH/{s}"
export {P}_NNUNET_RAW="$SCRATCH/{s}/2_nnUNet/raw"
export {P}_NNUNET_PREPROCESSED="$SCRATCH/{s}/2_nnUNet/preprocessed"
export {P}_PREDICTIONS_ROOT="$SCRATCH/{s}/8_results/01_predictions"
export {P}_DATASET_JSON="${{{P}_NNUNET_RAW}}/{a.source_dataset_json or 'Dataset<ID>_<Name>/dataset.json'}"
export RUN_JOB_ACCOUNT="aip-jcohen"
export RUN_JOB_GPU_TYPE="h100"
'''
        print(("would write " if a.dry_run else "write "), tam)
        if not a.dry_run:
            tam.parent.mkdir(parents=True, exist_ok=True); tam.write_text(text)
        n += 1
    print(f"\n{n} files {'planned' if a.dry_run else 'written'}. NEXT: complete 02_01_convert_test.py, run 02_02_fov_audit.sh, then the skill's checklist.")


if __name__ == "__main__":
    main()
