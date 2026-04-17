#!/usr/bin/env bash
# run_evaluation_nnunet.sh
# ── Post-training evaluation for seg_B (nnUNet) experiments ──────────────
#
# Usage (MUST run under set_slot for GPU/RAM access):
#   set_slot <gpu> CUDA_VISIBLE_DEVICES=<gpu> \
#     bash scripts/run_evaluation_nnunet.sh [gpu] [fold] [run_name] [eval_contrast]
#
# run_name      : which trained model to use (determines dataset + trainer)
# eval_contrast : (optional) single contrast to evaluate; omit to evaluate all 4
#
# Examples — evaluate on all contrasts (default):
#   set_slot 0 CUDA_VISIBLE_DEVICES=0 bash scripts/run_evaluation_nnunet.sh 0 0 seg_B_gen_raw_t1w
#   set_slot 1 CUDA_VISIBLE_DEVICES=1 bash scripts/run_evaluation_nnunet.sh 1 0 seg_B_gen_raw_t2w
#   set_slot 0 CUDA_VISIBLE_DEVICES=0 bash scripts/run_evaluation_nnunet.sh 0 0 seg_B_gen_19_t1w
#   set_slot 1 CUDA_VISIBLE_DEVICES=1 bash scripts/run_evaluation_nnunet.sh 1 0 seg_B_gen_19_t2w
#
# Examples — single contrast:
#   set_slot 0 CUDA_VISIBLE_DEVICES=0 bash scripts/run_evaluation_nnunet.sh 0 0 seg_B_gen_raw_t1w t1w
#   set_slot 0 CUDA_VISIBLE_DEVICES=0 bash scripts/run_evaluation_nnunet.sh 0 0 seg_B_gen_raw_t1w flair
#
# Results land in:
#   gen_raw models → results/eval/seg_B_baseline/multiclass/
#   gen_19  models → results/eval/v19/seg_B/
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

GPU_ID="${1:-0}"
export FOLD="${2:-0}"
RUN_NAME="${3:-seg_B_gen_raw_t1w}"
EVAL_CONTRAST_ARG="${4:-}"   # optional override; defaults to model contrast

NNUNET_VENV="/home/ge.polymtl.ca/pahoa/nih_project/.venv"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export nnUNet_raw="${PROJECT_ROOT}/data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_ROOT}/data/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_ROOT}/results/nnUNet"

CONFIG="3d_fullres"

# ── Default: evaluate on all contrasts, then aggregate once ───────────────
if [[ -z "${EVAL_CONTRAST_ARG}" ]]; then
    for c in t1w t2w flair t1gd; do
        SKIP_AGGREGATE=1 bash "$0" "${GPU_ID}" "${FOLD}" "${RUN_NAME}" "${c}"
    done
    # Aggregate once after all contrasts are done so long/wide stay in sync
    NNUNET_VENV="/home/ge.polymtl.ca/pahoa/nih_project/.venv"
    PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
    export nnUNet_results="${PROJECT_ROOT}/results/nnUNet"
    if [[ "${RUN_NAME}" == *bookends* ]]; then
        if [[ "${RUN_NAME}" == *t2w* ]]; then _mc="t2w"; else _mc="t1w"; fi
        FAMILY="bookends"
        AGGREGATE_OUT="${PROJECT_ROOT}/results/eval/v19/multiclass/seg_B/finetuning/${_mc}"
    elif [[ "${RUN_NAME}" == *gen_19* ]]; then
        FAMILY="gen_19"; AGGREGATE_OUT="${PROJECT_ROOT}/results/eval/v19/multiclass/seg_B"
    else
        FAMILY="gen_raw"; AGGREGATE_OUT="${PROJECT_ROOT}/results/eval/seg_B_baseline/multiclass"
    fi
    echo "[eval] Updating ${AGGREGATE_OUT} ..."
    "${NNUNET_VENV}/bin/python" "${PROJECT_ROOT}/scripts/nnunet_scripts/aggregate_nnunet_results.py" \
        --eval-dir       "${PROJECT_ROOT}/results/eval" \
        --output-dir     "${AGGREGATE_OUT}" \
        --nnunet-results "${nnUNet_results}" \
        --family         "${FAMILY}"
    exit 0
fi

# ── Infer model dataset and trainer from run name ─────────────────────────
# (only t1w/t2w models exist; flair/t1gd are OOD targets only)
if [[ "${RUN_NAME}" == *t2w* ]]; then
    MODEL_CONTRAST="t2w"
    DATASET_ID="023"
    DATASET_NAME="Dataset023_BraTST2w_gen_raw"
else
    MODEL_CONTRAST="t1w"
    DATASET_ID="022"
    DATASET_NAME="Dataset022_BraTST1w_gen_raw"
fi

if [[ "${RUN_NAME}" == *bookends* ]]; then
    TRAINER="nnUNetTrainer_Bookends"
    FAMILY="bookends"
    AGGREGATE_OUT="${PROJECT_ROOT}/results/eval/v19/multiclass/seg_B/finetuning/${MODEL_CONTRAST}"
    # DATASET_NAME/DATASET_ID already set correctly above (022 for t1w, 023 for t2w)
elif [[ "${RUN_NAME}" == *gen_19* ]]; then
    TRAINER="nnUNetTrainerBraTSGen19Wandb"
    FAMILY="gen_19"
    AGGREGATE_OUT="${PROJECT_ROOT}/results/eval/v19/seg_B"
else
    TRAINER="nnUNetTrainerBraTSWandb"
    FAMILY="gen_raw"
    AGGREGATE_OUT="${PROJECT_ROOT}/results/eval/seg_B_baseline/multiclass"
fi

# ── Infer eval contrast (default = model contrast) ────────────────────────
EVAL_CONTRAST="${EVAL_CONTRAST_ARG:-${MODEL_CONTRAST}}"

# ── Infer eval dataset (images to predict on) ────────────────────────────
if [[ "${EVAL_CONTRAST}" == "t2w" ]]; then
    EVAL_DATASET_NAME="Dataset023_BraTST2w_gen_raw"
elif [[ "${EVAL_CONTRAST}" == "flair" ]]; then
    EVAL_DATASET_NAME="Dataset021_BraTSFlair_gen_raw"
elif [[ "${EVAL_CONTRAST}" == "t1gd" ]]; then
    EVAL_DATASET_NAME="Dataset024_BraTST1gd_gen_raw"
else
    EVAL_DATASET_NAME="Dataset022_BraTST1w_gen_raw"
fi

export DATASET_ID DATASET_NAME TRAINER CONFIG EVAL_CONTRAST

# Model paths (splits + labels always come from the MODEL dataset)
export IMAGES_TR="${nnUNet_raw}/${DATASET_NAME}/imagesTr"
export LABELS_TR="${nnUNet_raw}/${DATASET_NAME}/labelsTr"
export SPLITS_FILE="${nnUNet_preprocessed}/${DATASET_NAME}/splits_final.json"
DATASET_JSON="${nnUNet_raw}/${DATASET_NAME}/dataset.json"
PLANS_JSON="${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans.json"

# Eval images come from the eval contrast dataset
export EVAL_IMAGES_TR="${nnUNet_raw}/${EVAL_DATASET_NAME}/imagesTr"

MODEL_DIR="${nnUNet_results}/${DATASET_NAME}/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}"

# Output dirs include eval contrast so in-domain and OOD don't collide
PRED_DIR="${PROJECT_ROOT}/results/eval/${RUN_NAME}/predictions_${EVAL_CONTRAST}_fold${FOLD}"
export VAL_IMAGES_DIR="${PRED_DIR}/_val_images"
export VAL_LABELS_DIR="${PRED_DIR}/_val_labels"
export OUT_DIR="${PROJECT_ROOT}/results/eval/${RUN_NAME}"
export SUMMARY_SRC="${PRED_DIR}/summary.json"
export SUMMARY_DST="${OUT_DIR}/eval_summary_${EVAL_CONTRAST}_fold${FOLD}.json"

mkdir -p "${PRED_DIR}" "${VAL_IMAGES_DIR}" "${VAL_LABELS_DIR}" "${OUT_DIR}"

echo "============================================================"
echo "  seg_B nnUNet evaluation"
echo "  run           : ${RUN_NAME}"
echo "  model dataset : ${DATASET_NAME}"
echo "  eval contrast : ${EVAL_CONTRAST}  ($([ "${EVAL_CONTRAST}" = "${MODEL_CONTRAST}" ] && echo "in-domain" || echo "OOD"))"
echo "  trainer       : ${TRAINER}"
echo "  fold          : ${FOLD}   gpu: ${GPU_ID}"
echo "  model dir     : ${MODEL_DIR}"
echo "  output        : ${OUT_DIR}"
echo "============================================================"
echo ""

# ── Guard: require training checkpoint ────────────────────────────────────
BEST_CKPT="${MODEL_DIR}/checkpoint_best.pth"
if [ ! -f "${BEST_CKPT}" ]; then
    echo "ERROR: checkpoint_best.pth not found at ${BEST_CKPT}"
    echo "       Training must complete before evaluation."
    exit 1
fi

# ── Guard: ensure eval contrast images are converted ──────────────────────
if [ ! -d "${EVAL_IMAGES_TR}" ]; then
    echo "[eval] ${EVAL_CONTRAST} images not found — running convert_to_nnunet_format.py ..."
    "${NNUNET_VENV}/bin/python" "${PROJECT_ROOT}/scripts/nnunet_scripts/convert_to_nnunet_format.py" \
        --contrast "${EVAL_CONTRAST}"
    echo "[eval] Conversion complete."
    echo ""
fi

# ── Step 1: Extract val-fold case IDs and symlink into temp dirs ──────────
echo "[eval] Extracting fold ${FOLD} validation cases from splits_final.json ..."

"${NNUNET_VENV}/bin/python" - <<'PYEOF'
import json, os
from pathlib import Path

fold          = int(os.environ["FOLD"])
splits_file   = os.environ["SPLITS_FILE"]
eval_images   = os.environ["EVAL_IMAGES_TR"]   # may differ from model imagesTr (OOD)
labels_tr     = os.environ["LABELS_TR"]        # always from model dataset
val_img_dir   = os.environ["VAL_IMAGES_DIR"]
val_lbl_dir   = os.environ["VAL_LABELS_DIR"]

with open(splits_file) as f:
    splits = json.load(f)

val_cases = splits[fold]["val"]
print(f"  Fold {fold} val cases: {len(val_cases)}")

img_linked = lbl_linked = 0
for case_id in val_cases:
    # images (from eval contrast dataset — may be OOD)
    src = Path(eval_images) / f"{case_id}_0000.nii.gz"
    dst = Path(val_img_dir) / f"{case_id}_0000.nii.gz"
    if src.exists() and not dst.exists():
        os.symlink(src, dst)
        img_linked += 1
    # labels (always from model dataset — same BraTS GT across contrasts)
    src = Path(labels_tr) / f"{case_id}.nii.gz"
    dst = Path(val_lbl_dir) / f"{case_id}.nii.gz"
    if src.exists() and not dst.exists():
        os.symlink(src, dst)
        lbl_linked += 1

print(f"  Symlinked {img_linked} images → {val_img_dir}")
print(f"  Symlinked {lbl_linked} labels → {val_lbl_dir}")
PYEOF

echo ""

# ── Step 2: Predict on val-fold images ────────────────────────────────────
echo "[eval] Running nnUNetv2_predict on ${EVAL_CONTRAST} validation images ..."
echo "[eval] $(date)"

"${NNUNET_VENV}/bin/nnUNetv2_predict" \
    -i  "${VAL_IMAGES_DIR}" \
    -o  "${PRED_DIR}" \
    -d  "${DATASET_ID}" \
    -c  "${CONFIG}" \
    -tr "${TRAINER}" \
    -f  "${FOLD}" \
    -chk checkpoint_best.pth \
    --disable_tta

echo ""
echo "[eval] Prediction complete.  $(date)"
echo ""

# ── Step 3: Evaluate predictions against ground-truth labels ──────────────
echo "[eval] Running nnUNetv2_evaluate_folder ..."

"${NNUNET_VENV}/bin/nnUNetv2_evaluate_folder" \
    "${VAL_LABELS_DIR}" \
    "${PRED_DIR}" \
    -djfile "${DATASET_JSON}" \
    -pfile  "${PLANS_JSON}"

# ── Step 4: Pretty-print results ──────────────────────────────────────────
if [ -f "${SUMMARY_SRC}" ]; then
    cp "${SUMMARY_SRC}" "${SUMMARY_DST}"
    echo ""
    echo "[eval] Results saved to: ${SUMMARY_DST}"

    "${NNUNET_VENV}/bin/python" - <<'PYEOF'
import json, os
from collections import defaultdict

with open(os.environ["SUMMARY_DST"]) as f:
    data = json.load(f)

print("\n── Dice per class ({}, fold {}) ──────────────────".format(
    os.environ["EVAL_CONTRAST"], os.environ["FOLD"]))
sums, counts = defaultdict(float), defaultdict(int)
for entry in data.get("metric_per_case", []):
    for label, metrics in entry.get("metrics", {}).items():
        v = metrics.get("Dice")
        if v is not None:
            sums[label] += v
            counts[label] += 1

label_names = {"1": "NCR", "2": "ED", "3": "ET"}
for label in sorted(sums.keys(), key=int):
    mean_dice = sums[label] / counts[label]
    name = label_names.get(str(label), f"label_{label}")
    print(f"  {name} (label {label}): {mean_dice:.4f}")

all_fg = [sums[l] / counts[l] for l in sums if int(l) > 0]
if all_fg:
    print(f"  Mean foreground Dice : {sum(all_fg)/len(all_fg):.4f}")
PYEOF
fi

# ── Step 5: Update aggregated seg_B results table ─────────────────────────
# (skipped when called from the all-contrasts loop; parent call aggregates once)
if [[ "${SKIP_AGGREGATE:-0}" != "1" ]]; then
echo ""
echo "[eval] Updating ${AGGREGATE_OUT} ..."

"${NNUNET_VENV}/bin/python" "${PROJECT_ROOT}/scripts/nnunet_scripts/aggregate_nnunet_results.py" \
    --eval-dir       "${PROJECT_ROOT}/results/eval" \
    --output-dir     "${AGGREGATE_OUT}" \
    --nnunet-results "${nnUNet_results}" \
    --family         "${FAMILY}"
fi

echo ""
echo "============================================================"
echo "  Evaluation complete.  Results: ${OUT_DIR}"
echo "  Aggregate table : ${AGGREGATE_OUT}"
echo "============================================================"
