#!/usr/bin/env bash
# run_seg_B_gen_raw_t2w.sh
# ── seg_B_gen_raw T2w: pure nnUNet 3d_fullres baseline, fold 0 ───────────
#
# Usage (from project root — MUST run under set_slot for GPU/RAM access):
#   tmux new-session -d -s slot3 "set_slot 3 bash scripts/run_seg_B_gen_raw_t2w.sh 2>&1 | tee /tmp/slot3_seg_B_gen_raw_t2w.log"
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

NNUNET_VENV="/home/ge.polymtl.ca/pahoa/nih_project/.venv"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export nnUNet_raw="${PROJECT_ROOT}/data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_ROOT}/data/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_ROOT}/results/nnUNet"

DATASET_ID="023"
DATASET_NAME="Dataset023_BraTST2w_gen_raw"
TRAINER="nnUNetTrainerBraTSWandb"

export WANDB_PROJECT="${WANDB_PROJECT:-brats-segmenter}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-seg_B_gen_raw_t2w_fold0}"

mkdir -p "${nnUNet_preprocessed}" "${nnUNet_results}"

echo "[seg_B_gen_raw_t2w] nnUNet_raw         = ${nnUNet_raw}"
echo "[seg_B_gen_raw_t2w] nnUNet_preprocessed= ${nnUNet_preprocessed}"
echo "[seg_B_gen_raw_t2w] nnUNet_results     = ${nnUNet_results}"
echo "[seg_B_gen_raw_t2w] WandB project=${WANDB_PROJECT}  run=${WANDB_RUN_NAME}"
echo ""

# ── Step 1: Dataset conversion (skip if already done) ─────────────────────
DATASET_MARKER="${nnUNet_raw}/${DATASET_NAME}/dataset.json"
if [ -f "${DATASET_MARKER}" ]; then
    echo "[seg_B_gen_raw_t2w] Dataset artefacts found — skipping conversion."
else
    echo "[seg_B_gen_raw_t2w] Converting T2w dataset (Dataset${DATASET_ID}) ..."
    "${NNUNET_VENV}/bin/python" "${PROJECT_ROOT}/scripts/convert_to_nnunet_format.py" \
        --contrast t2w
    echo "[seg_B_gen_raw_t2w] Conversion COMPLETE."
fi

echo ""

# ── Step 2: plan_and_preprocess (skip if already done) ────────────────────
PREPROCESS_MARKER="${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans_3d_fullres"
if [ -d "${PREPROCESS_MARKER}" ]; then
    echo "[seg_B_gen_raw_t2w] Preprocessing artefacts found — skipping plan_and_preprocess."
else
    echo "[seg_B_gen_raw_t2w] Running nnUNetv2_plan_and_preprocess -d ${DATASET_ID} ..."
    "${NNUNET_VENV}/bin/nnUNetv2_plan_and_preprocess" \
        -d "${DATASET_ID}" \
        --verify_dataset_integrity
    echo "[seg_B_gen_raw_t2w] plan_and_preprocess COMPLETE."
fi

echo ""

# ── Step 3: Training fold 0 ────────────────────────────────────────────────
echo "[seg_B_gen_raw_t2w] Starting nnUNetv2_train ${DATASET_ID} 3d_fullres 0 -tr ${TRAINER} ..."
echo "[seg_B_gen_raw_t2w] $(date)"
echo ""

"${NNUNET_VENV}/bin/nnUNetv2_train" "${DATASET_ID}" 3d_fullres 0 -tr "${TRAINER}"

echo ""
echo "[seg_B_gen_raw_t2w] Training fold 0 COMPLETE.  $(date)"
