#!/usr/bin/env bash
# One-off resume: v26_6_2 folds 0-3 from epoch 1600 → 2500 (+900 epochs),
# GPU-pinned (fold f → GPU f), continuing the ORIGINAL wandb runs by id.
#
# initial_lr=0.005 so the PolyLR restart LR at epoch 1600 is gentle:
#   0.005 * (1 - 1600/2500)^0.9  ≈  0.00199   (≤ 0.002, as requested)
#
# Usage: bash resume_v26_6_2_to_2500.sh
# Each fold becomes an independent sbatch job (fire-and-exit via run_job default).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

RUN_ID="brats2024-glioma_v26_6_2_train090_val000_20260608_003445"
TDIR="${nnUNet_results}/${RUN_ID}/Dataset051_BraTS2024GliomaT1n/nnUNetTrainerBraTS2024GliomaV26_6_2__nnUNetPlans__3d_fullres"
IDS=(rx5xtndf mymxmv1h uem9bokf i1evwzl4)
LOGD=/tmp/nnunet_v26_6_2_resume2500
mkdir -p "$LOGD"

for f in 0 1 2 3; do
    # resume exactly from the completed-1600 weights: checkpoint_final → checkpoint_latest
    cp -f "$TDIR/fold_$f/checkpoint_final.pth" "$TDIR/fold_$f/checkpoint_latest.pth"
    run_job --name "fold${f}_resume2500" --gpus 1 --slot "$f" \
        --log "$LOGD/fold${f}.log" -- \
        bash -c "
        export CUDA_VISIBLE_DEVICES='$f'
        export nnUNet_raw='${nnUNet_raw}'
        export nnUNet_preprocessed='${nnUNet_preprocessed}'
        export nnUNet_results='${nnUNet_results}/${RUN_ID}'
        export NNUNET_PROJECT_ROOT='${PROJECT_ROOT}'
        export PYTHONPATH='${PYTHONPATH}'
        export nnUNet_n_proc_DA=16
        export nnUNet_compile=1
        export NNUNET_NUM_EPOCHS=3000
        export NNUNET_INITIAL_LR=0.002
        export nnUNet_wandb_enabled=1
        export nnUNet_wandb_project=mri_synthesis_seg_brats2024-glioma
        export nnUNet_wandb_run_name='${RUN_ID}_fold${f}'
        export nnUNet_wandb_run_id='${IDS[$f]}'
        cd '${PROJECT_ROOT}'
        .venv/bin/nnUNetv2_train 051 3d_fullres $f --c \
            -tr nnUNetTrainerBraTS2024GliomaV26_6_2 -p nnUNetPlans -num_gpus 1
    "
    echo "fold$f → GPU $f  (wandb ${IDS[$f]})  log: $LOGD/fold${f}.log"
done
echo "all 4 folds launched (1600→2500, lr_start≈0.00199); logs: $LOGD/fold{0..3}.log"
