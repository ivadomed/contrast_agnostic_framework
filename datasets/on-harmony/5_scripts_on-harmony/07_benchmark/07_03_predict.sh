#!/usr/bin/env bash
# =============================================================================
# Run inference on the test set for all contrasts.
# =============================================================================
# Usage:
#   bash scripts/benchmark/predict.sh [OPTIONS]
#
# Options (env vars):
#   METHOD=v26_6                     method name
#   RUN_ID=v26_6_20260601_xxxxxx     REQUIRED: run ID to predict with
#   GPUS=0-3                         GPU slots
#   CONTRASTS="T1w T2w bold ..."     contrasts to predict (default = all 6)
#   FOLDS="0 1 2 3"                  folds to predict
#   FORCE=0                          set to 1 to re-run existing predictions
#
# Examples:
#   RUN_ID=v26_6_20260601_110150 bash scripts/benchmark/predict.sh
#   METHOD=synthseg_a RUN_ID=synthseg_a_20260601_154222 GPUS=0-1 bash scripts/benchmark/predict.sh
set -euo pipefail
source "$(dirname "$0")/07_01_config.sh"

if [ -z "${RUN_ID:-}" ] || [ "$RUN_ID" = "auto" ]; then
    echo "[predict] ERROR: RUN_ID must be set (not 'auto')" >&2; exit 1
fi

cd "$PROJECT_ROOT"
CONTRASTS="${CONTRASTS:-T1w T2w bold dwi_ap epi_ap gre_echo1_mag}"
FORCE="${FORCE:-0}"
GPU_ARR=($GPU_LIST)
N_GPUS=${#GPU_ARR[@]}

for CONTRAST in $CONTRASTS; do
    IMG_DIR="$EVAL_DIR/$RUN_ID/$CONTRAST/images_native"
    # images_native may be a symlink from baseline — find source
    if [ ! -d "$IMG_DIR" ]; then
        # Try symlinking from baseline if images not copied yet
        BASE_IMG="$EVAL_DIR/$(ls $EVAL_DIR | grep baseline | head -1)/$CONTRAST/images_native"
        if [ -d "$BASE_IMG" ]; then
            mkdir -p "$EVAL_DIR/$RUN_ID/$CONTRAST"
            ln -sf "$(realpath $BASE_IMG)" "$IMG_DIR"
            GT_SRC="$EVAL_DIR/$(ls $EVAL_DIR | grep baseline | head -1)/$CONTRAST/gt_native"
            [ -d "$GT_SRC" ] && ln -sf "$(realpath $GT_SRC)" "$EVAL_DIR/$RUN_ID/$CONTRAST/gt_native"
        else
            echo "[predict] $CONTRAST: no images found, skipping"; continue
        fi
    fi

    N=$(ls "$IMG_DIR" 2>/dev/null | wc -l)
    [ "$N" -eq 0 ] && echo "[predict] $CONTRAST: empty dir, skipping" && continue

    # Check if all fold predictions already exist
    ALL_DONE=1
    for FOLD in $FOLDS; do
        NC=$(ls "$EVAL_DIR/$RUN_ID/$CONTRAST/predictions_1mm/fold_${FOLD}/"*.nii.gz 2>/dev/null | wc -l)
        [ "$NC" -lt "$N" ] && ALL_DONE=0
    done
    if [ "$ALL_DONE" -eq 1 ] && [ "$FORCE" -eq 0 ]; then
        echo "[predict] $CONTRAST: already done"; continue
    fi

    echo "[predict] $CONTRAST: $N images across folds $FOLDS"
    declare -A PIDS
    FOLD_IDX=0
    for FOLD in $FOLDS; do
        PRED_DIR="$EVAL_DIR/$RUN_ID/$CONTRAST/predictions_1mm/fold_${FOLD}"
        mkdir -p "$PRED_DIR"
        SLOT="${GPU_ARR[$((FOLD_IDX % N_GPUS))]}"
        FOLD_IDX=$((FOLD_IDX + 1))

        run_job --name "benchmark_predict_${RUN_ID}_${CONTRAST}_fold${FOLD}" \
            --gpus 1 --slot "${SLOT}" --wait \
            --log "/tmp/pred_${RUN_ID}_${CONTRAST}_f${FOLD}.log" -- \
            bash -c "
            export CUDA_VISIBLE_DEVICES='$SLOT'
            export nnUNet_raw='$NNUNET_RAW'
            export nnUNet_preprocessed='$NNUNET_PRE'
            export nnUNet_results='$NNUNET_RES/$RUN_ID'
            export NNUNET_PROJECT_ROOT='$PROJECT_ROOT'
            export PYTHONPATH='$PROJECT_ROOT/src/nnunet'
            cd '$PROJECT_ROOT'
            .venv/bin/nnUNetv2_predict \
                -i '$IMG_DIR' \
                -o '$PRED_DIR' \
                -d $DATASET_ID -c 3d_fullres \
                -f $FOLD -tr $TRAINER -p nnUNetPlans
        " &
        PIDS[$FOLD]=$!
    done
    for FOLD in $FOLDS; do wait "${PIDS[$FOLD]}"; done
    echo "[predict] $CONTRAST: done"
done
echo "[predict] All contrasts done for $RUN_ID"
