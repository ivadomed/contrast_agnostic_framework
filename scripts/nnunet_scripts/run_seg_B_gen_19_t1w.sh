#!/usr/bin/env bash
# run_seg_B_gen_19_t1w.sh
# ── seg_B_gen_19 T1w: nnUNet 3d_fullres + v19 online generator, fold 0 ───
#
# Usage (MUST run under set_slot for GPU/RAM access):
#   tmux new-session -d -s slot0 "set_slot 0 CUDA_VISIBLE_DEVICES=0 bash scripts/run_seg_B_gen_19_t1w.sh 2>&1 | tee /tmp/slot0_seg_B_gen_19_t1w.log"
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

NNUNET_VENV="/home/ge.polymtl.ca/pahoa/nih_project/.venv"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export nnUNet_raw="${PROJECT_ROOT}/data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_ROOT}/data/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_ROOT}/results/nnUNet"

DATASET_ID="022"
TRAINER="nnUNetTrainerBraTSGen19Wandb"

export WANDB_PROJECT="${WANDB_PROJECT:-brats-segmenter}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-seg_B_gen_19_t1w_fold0}"
export GENERATOR_CKPT="${GENERATOR_CKPT:-${PROJECT_ROOT}/checkpoints/v19/generator/t1w/run1/last.ckpt}"

echo "[seg_B_gen_19_t1w] nnUNet_raw         = ${nnUNet_raw}"
echo "[seg_B_gen_19_t1w] nnUNet_preprocessed= ${nnUNet_preprocessed}"
echo "[seg_B_gen_19_t1w] nnUNet_results     = ${nnUNet_results}"
echo "[seg_B_gen_19_t1w] WandB project=${WANDB_PROJECT}  run=${WANDB_RUN_NAME}"
echo "[seg_B_gen_19_t1w] Generator ckpt=${GENERATOR_CKPT}"
echo ""

# ── Preprocessing (skip if 3d_fullres already done) ───────────────────────
PREPROCESS_MARKER="${nnUNet_preprocessed}/Dataset022_BraTST1w_gen_raw/nnUNetPlans_3d_fullres"
if [ -d "${PREPROCESS_MARKER}" ]; then
    echo "[seg_B_gen_19_t1w] Preprocessing artefacts found — skipping plan_and_preprocess."
else
    echo "[seg_B_gen_19_t1w] Running nnUNetv2_plan_and_preprocess -d ${DATASET_ID} ..."
    "${NNUNET_VENV}/bin/nnUNetv2_plan_and_preprocess" \
        -d "${DATASET_ID}" \
        --verify_dataset_integrity
    echo "[seg_B_gen_19_t1w] plan_and_preprocess COMPLETE."
fi

echo ""

# ── Training fold 0 ────────────────────────────────────────────────────────
echo "[seg_B_gen_19_t1w] Starting nnUNetv2_train ${DATASET_ID} 3d_fullres 0 -tr ${TRAINER} ..."
echo "[seg_B_gen_19_t1w] $(date)"
echo ""

"${NNUNET_VENV}/bin/nnUNetv2_train" "${DATASET_ID}" 3d_fullres 0 -tr "${TRAINER}"

echo ""
echo "[seg_B_gen_19_t1w] Training fold 0 COMPLETE.  $(date)"
