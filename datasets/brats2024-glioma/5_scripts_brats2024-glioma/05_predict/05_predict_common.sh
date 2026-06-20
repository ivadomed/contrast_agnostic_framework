#!/usr/bin/env bash
# Shared predict template — sourced by 05_0X_predict_<method>.sh, NOT run directly.
#
# Runs inference for one experiment across one or more contrasts of the held-out BraTS
# test set, and saves predictions for downstream analysis.
#
# Set by the calling wrapper:
#   METHOD      label, e.g. v26_6_2
#   TRAINER     nnUNet trainer class, e.g. nnUNetTrainerBraTS2024GliomaV26_6_2
#   CATEGORY    prediction group: "nnUNet" (default) or "auglab"
#               Controls the output path: PREDICTIONS_ROOT/CATEGORY/RUN_ID/fold{k}/{contrast}/
# Provided as args by the wrapper (passed through from the user):
#   RUN_ID      $1  required — the training run dir under $nnUNet_results
#   FOLD        $2  optional — fold number 0-3, or "all" for all folds in parallel
#                   (default 0; "all" runs fold N on slot N / GPU N simultaneously)
#   CONTRASTS   $3… optional — space-separated subset of: t1n t1c t2w t2f (default all)
#
# Optional env overrides:
#   CHECKPOINT  checkpoint_best.pth (default) | checkpoint_final.pth
#   SLOT GPU    slot / CUDA device override (used only when FOLD is a single number)
#
# Test-input dirs (imagesTs_<contrast>/) must exist — build once with:
#   python 05_00_build_test_inputs.py
#
# Each prediction is launched through run_job() (scripts/job_runner/run_job.sh,
# sourced transitively via 00_utils/env.sh) with --gpus 1 --wait: modalities
# within a fold run sequentially on the same GPU, folds run in parallel.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

RUN_ID="${1:?RUN_ID required (training run dir name under nnUNet_results)}"
FOLD="${2:-0}"
shift $(( $# >= 2 ? 2 : $# ))
CONTRASTS=("$@"); [ ${#CONTRASTS[@]} -eq 0 ] && CONTRASTS=(t1n t1c t2w t2f)

DATASET_ID="${DATASET_ID:-051}"
CHECKPOINT="${CHECKPOINT:-checkpoint_best.pth}"
CATEGORY="${CATEGORY:-nnUNet}"
# Models live under PREDICTIONS_ROOT/{model_type}/{training_contrast}/{category}/ — derive here
# (after env.sh, which always resets nnUNet_results to the nnUNet root and would clobber a wrapper export).
export nnUNet_results="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}"
_DS_NAME="$(ls "${nnUNet_raw}" | grep "^Dataset${DATASET_ID}_" | head -1)"
RUN_DIR="${nnUNet_results}/${RUN_ID}"   # model location

[ -d "$RUN_DIR" ] || { echo "ERROR: run dir not found: $RUN_DIR" >&2; exit 1; }

predict_fold() {
    local F="$1" SLOT="$2" GPU="$3"
    echo "[$(date '+%H:%M:%S')] predict ${METHOD} | run=${RUN_ID} | fold=${F} | ckpt=${CHECKPOINT} | slot=${SLOT} gpu=${GPU}"
    for contrast in "${CONTRASTS[@]}"; do
        local INPUT_DIR="${nnUNet_raw}/${_DS_NAME}/imagesTs_${contrast}"
        local OUTPUT_DIR="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}/fold${F}/${contrast}"
        if [ ! -d "$INPUT_DIR" ] || [ -z "$(ls -A "$INPUT_DIR" 2>/dev/null)" ]; then
            echo "  ! fold${F} skip ${contrast}: input dir missing/empty ($INPUT_DIR) — run 05_00_build_test_inputs.py" >&2
            continue
        fi
        mkdir -p "$OUTPUT_DIR"
        echo "  → fold${F} ${contrast}: $(ls "$INPUT_DIR" | wc -l) cases → $OUTPUT_DIR"
        run_job --name "brats_predict_${METHOD}_fold${F}_${contrast}" \
            --gpus 1 --slot "${SLOT}" \
            --log "/tmp/predict_${METHOD}_${RUN_ID}_fold${F}_${contrast}.log" --wait -- \
            bash -c "
            export nnUNet_raw='${nnUNet_raw}'
            export nnUNet_preprocessed='${nnUNet_preprocessed}'
            export nnUNet_results='${RUN_DIR}'
            export NNUNET_PROJECT_ROOT='${PROJECT_ROOT}'
            export PYTHONPATH='${PYTHONPATH}'
            export CUDA_VISIBLE_DEVICES='${GPU}'
            export TF_USE_LEGACY_KERAS=1
            cd '${PROJECT_ROOT}'
            .venv/bin/nnUNetv2_predict \
                -i '${INPUT_DIR}' \
                -o '${OUTPUT_DIR}' \
                -d ${DATASET_ID} \
                -c 3d_fullres \
                -tr ${TRAINER} \
                -f ${F} \
                --disable_tta \
                -chk ${CHECKPOINT}
        "
        echo "  ✓ fold${F} ${contrast} done"
    done
    echo "[$(date '+%H:%M:%S')] fold${F} done → ${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}/fold${F}/"
}

if [ "$FOLD" = "all" ]; then
    echo "[$(date '+%H:%M:%S')] predict ${METHOD} | run=${RUN_ID} | ALL FOLDS (parallel, fold→slot) | ckpt=${CHECKPOINT}"
    echo "  contrasts: ${CONTRASTS[*]}"
    for F in 0 1 2 3; do
        predict_fold "$F" "$F" "$F" &
    done
    wait
    echo "[$(date '+%H:%M:%S')] all folds done → ${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}/"
else
    echo "  contrasts: ${CONTRASTS[*]}"
    predict_fold "${FOLD}" "${SLOT:-0}" "${GPU:-0}"
fi
