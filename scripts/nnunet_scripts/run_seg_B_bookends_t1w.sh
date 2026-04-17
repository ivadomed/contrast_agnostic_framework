#!/usr/bin/env bash
# run_seg_B_bookends_t1w.sh
# ── Bookends fine-tuning: T1w, fold 0, loaded from v19 pretrained weights ──
#
# Fine-tunes on Dataset022_BraTST1w_gen_raw (already preprocessed).
# nnUNet_Bookends__nnUNetPlans__3d_fullres is created as a NEW folder —
# the original nnUNetTrainerBraTSGen19Wandb checkpoint is NEVER overwritten.
# Training runs for 100 epochs (overridden in nnUNetTrainer_Bookends).
#
# Usage (MUST run under set_slot for GPU/RAM access):
#   tmux new-session -d -s bookends_t1w \
#     "set_slot 0 CUDA_VISIBLE_DEVICES=0 \
#      bash scripts/nnunet_scripts/run_seg_B_bookends_t1w.sh 2>&1 | \
#      tee /tmp/slot0_bookends_t1w.log"
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

NNUNET_VENV="/home/ge.polymtl.ca/pahoa/nih_project/.venv"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

export nnUNet_raw="${PROJECT_ROOT}/data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_ROOT}/data/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_ROOT}/results/nnUNet"

DATASET_ID="022"
DATASET_NAME="Dataset022_BraTST1w_gen_raw"
TRAINER="nnUNetTrainer_Bookends"
PRETRAINED_CKPT="${PROJECT_ROOT}/results/nnUNet/${DATASET_NAME}/nnUNetTrainerBraTSGen19Wandb__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"

export WANDB_PROJECT="${WANDB_PROJECT:-brats-segmenter}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-seg_B_bookends_t1w_fold0}"

mkdir -p "${nnUNet_results}"

echo "[bookends_t1w] ─────────────────────────────────────────────────────────"
echo "[bookends_t1w] Bookends fine-tuning — T1w, fold 0, 100 epochs"
echo "[bookends_t1w] nnUNet_raw          = ${nnUNet_raw}"
echo "[bookends_t1w] nnUNet_preprocessed = ${nnUNet_preprocessed}"
echo "[bookends_t1w] nnUNet_results      = ${nnUNet_results}"
echo "[bookends_t1w] Dataset             = ${DATASET_NAME} (ID=${DATASET_ID})"
echo "[bookends_t1w] Pretrained weights  = ${PRETRAINED_CKPT}"
echo "[bookends_t1w] Trainer             = ${TRAINER}"
echo "[bookends_t1w] ─────────────────────────────────────────────────────────"
echo ""

# ── Guard: pretrained checkpoint must exist ────────────────────────────────────
if [ ! -f "${PRETRAINED_CKPT}" ]; then
    echo "ERROR: Pretrained checkpoint not found:"
    echo "       ${PRETRAINED_CKPT}"
    echo "       Ensure the v19 T1w training (nnUNetTrainerBraTSGen19Wandb, Dataset022) has completed."
    exit 1
fi

# ── Guard: preprocessed data must exist (skip plan_and_preprocess) ────────────
PREPROCESS_MARKER="${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans_3d_fullres"
if [ ! -d "${PREPROCESS_MARKER}" ]; then
    echo "ERROR: Preprocessed data not found at ${PREPROCESS_MARKER}"
    echo "       Expected data/nnUNet_preprocessed/Dataset022_BraTST1w_gen_raw to be present."
    exit 1
fi
echo "[bookends_t1w] Preprocessed data found — skipping plan_and_preprocess."
echo ""

# ── Fine-tuning fold 0 with pretrained weights ────────────────────────────────
echo "[bookends_t1w] Starting Bookends fine-tuning: ${DATASET_ID} 3d_fullres fold 0"
echo "[bookends_t1w] $(date)"
echo ""

"${NNUNET_VENV}/bin/nnUNetv2_train" \
    "${DATASET_ID}" \
    3d_fullres \
    0 \
    -tr "${TRAINER}" \
    -pretrained_weights "${PRETRAINED_CKPT}"

echo ""
echo "[bookends_t1w] Fine-tuning fold 0 COMPLETE.  $(date)"
echo "[bookends_t1w] Output: ${nnUNet_results}/${DATASET_NAME}/${TRAINER}__nnUNetPlans__3d_fullres/fold_0/"
