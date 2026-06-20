#!/usr/bin/env bash
# Shared predict template — sourced by 05_0X_predict_<method>.sh, NOT run directly.
# Ported from brats2024-glioma/05_predict_common.sh.
#
# Runs inference for one experiment across one or more modalities of the CHAOS
# internal test set (per-modality imagesTs_<mod>/ dirs built by 05_00_build_test_inputs.py).
#
# Set by the calling wrapper:
#   METHOD      label, e.g. baseline
#   TRAINER     nnUNet trainer class, e.g. nnUNetTrainerCHAOSBaseline
#   CATEGORY    prediction group (default "nnUNet") → PREDICTIONS_ROOT/CATEGORY/RUN_ID/...
# Args (from user):
#   RUN_ID      $1  required — training run dir under PREDICTIONS_ROOT/CATEGORY
#   FOLD        $2  optional — 0-3 or "all" (default 0)
#   MODALITIES  $3… optional — subset of: t1in t1out t2spir ct (default all)
#
# Output: PREDICTIONS_ROOT/CATEGORY/RUN_ID/fold{k}/{modality}/{case}.nii.gz
#
# Each prediction is launched through run_job() (scripts/job_runner/run_job.sh,
# sourced transitively via 00_utils/env.sh) with --gpus 1 --wait: modalities
# within a fold run sequentially on the same GPU, folds run in parallel.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

RUN_ID="${1:?RUN_ID required (training run dir name)}"
FOLD="${2:-0}"
shift $(( $# >= 2 ? 2 : $# ))
MODALITIES=("$@"); [ ${#MODALITIES[@]} -eq 0 ] && MODALITIES=(t1in t1out t2spir ct)

DATASET_ID="${DATASET_ID:-60}"
CHECKPOINT="${CHECKPOINT:-checkpoint_best.pth}"
CATEGORY="${CATEGORY:-nnUNet}"
export nnUNet_results="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}"
_DS_NAME="$(ls "${nnUNet_raw}" | grep "^Dataset0*${DATASET_ID}_" | head -1)"
RUN_DIR="${nnUNet_results}/${RUN_ID}"

[ -d "$RUN_DIR" ] || { echo "ERROR: run dir not found: $RUN_DIR" >&2; exit 1; }

predict_fold() {
    local F="$1" SLOT="$2" GPU="$3"
    # On Slurm each fold prediction is its own isolated sbatch job — the allocated
    # GPU is always device 0 within the job's cgroup namespace. The physical GPU
    # index (used by set_slot, where all 4 GPUs are visible) is meaningless here.
    local _CUDA_DEV="${GPU}"
    if [ "${RUN_JOB_BACKEND:-}" = "slurm" ]; then
        _CUDA_DEV="0"
    fi
    echo "[$(date '+%H:%M:%S')] predict ${METHOD} | run=${RUN_ID} | fold=${F} | ckpt=${CHECKPOINT} | slot=${SLOT} gpu=${GPU}"
    for mod in "${MODALITIES[@]}"; do
        local INPUT_DIR="${nnUNet_raw}/${_DS_NAME}/imagesTs_${mod}"
        local OUTPUT_DIR="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}/fold${F}/${mod}"
        if [ ! -d "$INPUT_DIR" ] || [ -z "$(ls -A "$INPUT_DIR" 2>/dev/null)" ]; then
            echo "  ! fold${F} skip ${mod}: input dir missing/empty ($INPUT_DIR) — run 05_00_build_test_inputs.py" >&2
            continue
        fi
        mkdir -p "$OUTPUT_DIR"
        echo "  → fold${F} ${mod}: $(ls "$INPUT_DIR" | wc -l) cases → $OUTPUT_DIR"
        run_job --name "chaos_predict_${METHOD}_fold${F}_${mod}" \
            --gpus 1 --slot "${SLOT}" \
            --log "/tmp/predict_${METHOD}_${RUN_ID}_fold${F}_${mod}.log" --wait -- \
            bash -c "
            export nnUNet_raw='${nnUNet_raw}'
            export nnUNet_preprocessed='${nnUNet_preprocessed}'
            export nnUNet_results='${RUN_DIR}'
            export NNUNET_PROJECT_ROOT='${PROJECT_ROOT}'
            export PYTHONPATH='${PYTHONPATH}'
            export CUDA_VISIBLE_DEVICES='${_CUDA_DEV}'
            export TF_USE_LEGACY_KERAS=1
            cd '${PROJECT_ROOT}'
            .venv/bin/nnUNetv2_predict \
                -i '${INPUT_DIR}' -o '${OUTPUT_DIR}' \
                -d ${DATASET_ID} -c 3d_fullres -tr ${TRAINER} -f ${F} \
                --disable_tta -chk ${CHECKPOINT}
        "
        echo "  ✓ fold${F} ${mod} done"
    done
    echo "[$(date '+%H:%M:%S')] fold${F} done → ${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}/fold${F}/"
}

if [ "$FOLD" = "all" ]; then
    echo "[$(date '+%H:%M:%S')] predict ${METHOD} | ALL FOLDS (parallel, fold→slot) | modalities: ${MODALITIES[*]}"
    for F in 0 1 2 3; do predict_fold "$F" "$F" "$F" & done
    wait
    echo "[$(date '+%H:%M:%S')] all folds done → ${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}/"
else
    echo "  modalities: ${MODALITIES[*]}"
    predict_fold "${FOLD}" "${SLOT:-0}" "${GPU:-0}"
fi
