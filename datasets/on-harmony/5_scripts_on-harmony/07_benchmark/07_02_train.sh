#!/usr/bin/env bash
# =============================================================================
# Train a segmentation model on the benchmark.
# =============================================================================
# Usage:
#   bash scripts/benchmark/train.sh [OPTIONS]
#
# Options (env vars, all optional):
#   METHOD=v26_6          baseline | v26_6 | synthseg_a | synthseg_b
#   DATASET=on_harmony    dataset name
#   LABEL_SET=7class      7class | 31class
#   GPUS=0-3              GPU slots (0, 0-1, 0-3, 0,2)
#   DA_WORKERS=auto       DA worker count (auto = 0 for synth, 16 for baseline)
#   N_EPOCHS=500         training epochs
#   FOLDS="0 1 2 3"       which folds to run
#   RUN_ID=auto           resume existing run or create new timestamped one
#
# Examples:
#   bash scripts/benchmark/train.sh
#   METHOD=baseline GPUS=0-1 bash scripts/benchmark/train.sh
#   METHOD=synthseg_a RUN_ID=synthseg_a_20260601_154222 bash scripts/benchmark/train.sh
set -euo pipefail
source "$(dirname "$0")/07_01_config.sh"

cd "$PROJECT_ROOT"

# ── Resolve RUN_ID ────────────────────────────────────────────────────────────
if [ "$RUN_ID" = "auto" ]; then
    RUN_ID="${METHOD}_$(date +%Y%m%d_%H%M%S)"
fi
TRAINER_DIR="$DATASET_NAME/${TRAINER}__nnUNetPlans__3d_fullres"
echo "[train] RUN_ID=$RUN_ID"
mkdir -p "$LOG_DIR"

# ── Launch one fold per GPU slot (run_job --wait, background — wait on PIDs) ──
declare -A PIDS
SLOT=0
for FOLD in $FOLDS; do
    # Pick GPU slot round-robin over available slots
    GPU_ARR=($GPU_LIST)
    SLOT_IDX=$(( FOLD % ${#GPU_ARR[@]} ))
    SLOT="${GPU_ARR[$SLOT_IDX]}"

    CKPT="$NNUNET_RES/$RUN_ID/$TRAINER_DIR/fold_${FOLD}/checkpoint_latest.pth"
    CONTINUE_FLAG=""
    if [ -f "$CKPT" ]; then
        EPOCH=$($PY -c "
import torch
try:
    ck = torch.load('$CKPT', map_location='cpu', weights_only=False)
    print(ck.get('current_epoch', '?'))
except Exception:
    print('?')
" 2>/dev/null)
        CONTINUE_FLAG="--c"
        echo "[train] fold $FOLD → GPU $SLOT  resuming from epoch $EPOCH"
    else
        echo "[train] fold $FOLD → GPU $SLOT  fresh start"
    fi

    run_job --name "benchmark_train_${RUN_ID}_fold${FOLD}" \
        --gpus 1 --slot "${SLOT}" --wait \
        --log "$LOG_DIR/fold${FOLD}.log" -- \
        bash -c "
        export CUDA_VISIBLE_DEVICES='$SLOT'
        export nnUNet_raw='$NNUNET_RAW'
        export nnUNet_preprocessed='$NNUNET_PRE'
        export nnUNet_results='$NNUNET_RES/$RUN_ID'
        export NNUNET_PROJECT_ROOT='$PROJECT_ROOT'
        export NNUNET_LABEL_SET='$LABEL_SET'
        export nnUNet_n_proc_DA=$DA_WORKERS
        export nnUNet_wandb_enabled=1
        export nnUNet_wandb_project='mri_synthesis_seg'
        export nnUNet_wandb_run_name='${RUN_ID}_fold${FOLD}'
        export RUN_ID='$RUN_ID'
        export TF_USE_LEGACY_KERAS=1
        export OMP_NUM_THREADS=${DA_WORKERS:-8}
        export MKL_NUM_THREADS=${DA_WORKERS:-8}
        export OPENBLAS_NUM_THREADS=${DA_WORKERS:-8}
        export OMP_WAIT_POLICY=passive
        cd '$PROJECT_ROOT'
        .venv/bin/nnUNetv2_train $DATASET_ID 3d_fullres $FOLD $CONTINUE_FLAG \
            -tr $TRAINER \
            -p nnUNetPlans
    " &
    PIDS[$FOLD]=$!
done

wait "${PIDS[@]}"
echo "[train] All folds complete → $NNUNET_RES/$RUN_ID"
echo "[train] RUN_ID=$RUN_ID"
