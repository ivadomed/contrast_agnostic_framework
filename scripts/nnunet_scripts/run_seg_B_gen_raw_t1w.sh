#!/usr/bin/env bash
# run_seg_B_gen_raw_t1w.sh
# ── seg_B_gen_raw T1w: pure nnUNet 3d_fullres baseline, fold 0 ────────────
#
# Complies with AI_COPILOT_GUIDELINES.md:
#   - Must be launched inside a tmux session (slot1)
#   - Monitor for >= 8 minutes after launch
#   - Verify plan_and_preprocess completes + Epoch 0 starts without CUDA OOM
#
# Usage (from project root — MUST run under set_slot for GPU/RAM access):
#   tmux new-session -d -s slot2 "set_slot 2 bash scripts/run_seg_B_gen_raw_t1w.sh 2>&1 | tee /tmp/slot2_seg_B_gen_raw_t1w.log"
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

NNUNET_VENV="/home/ge.polymtl.ca/pahoa/nih_project/.venv"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── nnUNet environment variables ───────────────────────────────────────────
export nnUNet_raw="${PROJECT_ROOT}/data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_ROOT}/data/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_ROOT}/results/nnUNet"

mkdir -p "${nnUNet_preprocessed}" "${nnUNet_results}"

echo "[seg_B_gen_raw] nnUNet_raw         = ${nnUNet_raw}"
echo "[seg_B_gen_raw] nnUNet_preprocessed= ${nnUNet_preprocessed}"
echo "[seg_B_gen_raw] nnUNet_results     = ${nnUNet_results}"
echo ""

# ── Step 1: plan_and_preprocess (skip if already done) ────────────────────
PREPROCESS_MARKER="${nnUNet_preprocessed}/Dataset022_BraTST1w_gen_raw/nnUNetPlans_3d_fullres"
if [ -d "${PREPROCESS_MARKER}" ]; then
    echo "[seg_B_gen_raw] Preprocessing artefacts found — skipping plan_and_preprocess."
else
    echo "[seg_B_gen_raw] Running nnUNetv2_plan_and_preprocess -d 022 --verify_dataset_integrity ..."
    "${NNUNET_VENV}/bin/nnUNetv2_plan_and_preprocess" \
        -d 022 \
        --verify_dataset_integrity
    echo "[seg_B_gen_raw] plan_and_preprocess COMPLETE."
fi

echo ""

# ── Step 2: Training fold 0 ───────────────────────────────────────────────
TRAINER="nnUNetTrainerBraTSWandb"
export WANDB_PROJECT="${WANDB_PROJECT:-brats-segmenter}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-seg_B_gen_raw_fold0}"

echo "[seg_B_gen_raw] Starting nnUNetv2_train 022 3d_fullres 0 -tr ${TRAINER} ..."
echo "[seg_B_gen_raw] WandB project=${WANDB_PROJECT}  run=${WANDB_RUN_NAME}"
echo "[seg_B_gen_raw] $(date)"
echo ""

"${NNUNET_VENV}/bin/nnUNetv2_train" 022 3d_fullres 0 -tr "${TRAINER}"

echo ""
echo "[seg_B_gen_raw] Training fold 0 COMPLETE.  $(date)"
