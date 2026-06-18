#!/usr/bin/env bash
# ============================================================================
#  SLIVER07 prediction — USES MODELS TRAINED ON ANOTHER DATASET (chaos).
#  ----------------------------------------------------------------------------
#  SLIVER07 is EVALUATION-ONLY (see datasets/sliver07/README.md). It has NO
#  trained models of its own; these scripts load chaos-trained checkpoints
#  (MR T1-DUAL in-phase) and run them over SLIVER07's CT volumes to measure
#  MR→CT generalization. That is why:
#    - nnUNet_results / nnUNet_raw / -tr / -d all point at CHAOS (CHAOS_* env vars);
#    - the trainer classes are nnUNetTrainerCHAOS* (chaos), not sliver07 classes;
#    - outputs are segregated under PREDICTIONS_ROOT/${MODEL_SOURCE}_models/ so that
#      IF SLIVER07 is ever trained on natively, those predictions live elsewhere
#      (e.g. sliver07_models/) and never collide with these cross-dataset ones.
#  To predict with sliver07-native models one day: add a sibling common script
#  (MODEL_SOURCE=sliver07) pointing nnUNet_results at 6_checkpoints_sliver07.
# ============================================================================
# Shared predict template — sourced by 05_0X_predict_chaos_<method>.sh, NOT run directly.
# Adapted from chaos/05_01_predict_common.sh.
#
# Set by the calling wrapper:
#   METHOD      label, e.g. baseline
#   TRAINER     chaos trainer class, e.g. nnUNetTrainerCHAOSBaseline
#   CATEGORY    chaos prediction group: nnUNet | auglab
#   RUN_ID      chaos training run dir name (wrapper provides a sensible default)
# Args (from user):
#   RUN_ID      $1  optional — overrides the wrapper default
#   FOLD        $2  optional — 0-3 or "all" (default all)
#   MODALITIES  $3… optional — only "ct" exists in SLIVER07 (default ct)
#
# Output: PREDICTIONS_ROOT/${MODEL_SOURCE}_models/CATEGORY/RUN_ID/fold{k}/{modality}/{case}.nii.gz

set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"

# Where the weights come from. "chaos" = cross-dataset (the only mode today).
MODEL_SOURCE="${MODEL_SOURCE:-chaos}"

RUN_ID="${1:-${RUN_ID:?RUN_ID required (chaos training run dir name)}}"
FOLD="${2:-all}"
shift $(( $# >= 2 ? 2 : $# )) || true
MODALITIES=("$@"); [ ${#MODALITIES[@]} -eq 0 ] && MODALITIES=(ct)

DATASET_ID="${CHAOS_DATASET_ID:-60}"
CHECKPOINT="${CHECKPOINT:-checkpoint_best.pth}"   # chaos runs only ship checkpoint_best.pth
CATEGORY="${CATEGORY:-nnUNet}"

# Model source = chaos run dir (contains Dataset0XX_.../TRAINER__nnUNetPlans__3d_fullres/fold_k).
CHAOS_RUN_DIR="${CHAOS_PREDICTIONS_ROOT}/${CATEGORY}/${RUN_ID}"
[ -d "$CHAOS_RUN_DIR" ] || { echo "ERROR: chaos run dir not found: $CHAOS_RUN_DIR" >&2; exit 1; }
echo "[$(date '+%H:%M:%S')] ⚠ MODEL SOURCE = ${MODEL_SOURCE} (cross-dataset): ${CHAOS_RUN_DIR}"

predict_fold() {
    local F="$1" SLOT="$2" GPU="$3"
    echo "[$(date '+%H:%M:%S')] predict ${METHOD} | chaos run=${RUN_ID} | fold=${F} | ckpt=${CHECKPOINT} | slot=${SLOT} gpu=${GPU}"
    for mod in "${MODALITIES[@]}"; do
        local INPUT_DIR="${nnUNet_raw}/imagesTs_${mod}"
        local OUTPUT_DIR="${PREDICTIONS_ROOT}/${MODEL_SOURCE}_models/${CATEGORY}/${RUN_ID}/fold${F}/${mod}"
        if [ ! -d "$INPUT_DIR" ] || [ -z "$(ls -A "$INPUT_DIR" 2>/dev/null)" ]; then
            echo "  ! fold${F} skip ${mod}: input dir missing/empty ($INPUT_DIR) — run 05_00_build_test_inputs.py" >&2
            continue
        fi
        mkdir -p "$OUTPUT_DIR"
        echo "  → fold${F} ${mod}: $(ls "$INPUT_DIR" | wc -l) inputs → $OUTPUT_DIR"
        set_slot ${SLOT} bash -c "
            export nnUNet_raw='${CHAOS_NNUNET_RAW}'
            export nnUNet_preprocessed='${CHAOS_DATASET_ROOT}/2_nnUNet_chaos/preprocessed'
            export nnUNet_results='${CHAOS_RUN_DIR}'
            export NNUNET_PROJECT_ROOT='$(pwd)'
            export PYTHONPATH='${PYTHONPATH}'
            export CUDA_VISIBLE_DEVICES='${GPU}'
            export TF_USE_LEGACY_KERAS=1
            cd '$(pwd)'
            .venv/bin/nnUNetv2_predict \
                -i '${INPUT_DIR}' -o '${OUTPUT_DIR}' \
                -d ${DATASET_ID} -c 3d_fullres -tr ${TRAINER} -f ${F} \
                --disable_tta -chk ${CHECKPOINT} \
                -npp 12 -nps 6
        " 2>&1 | tee "/tmp/sliver07_predict_${METHOD}_${RUN_ID}_fold${F}_${mod}.log"
        echo "  ✓ fold${F} ${mod} done"
    done
    echo "[$(date '+%H:%M:%S')] fold${F} done → ${PREDICTIONS_ROOT}/${MODEL_SOURCE}_models/${CATEGORY}/${RUN_ID}/fold${F}/"
}

if [ "$FOLD" = "all" ]; then
    echo "[$(date '+%H:%M:%S')] predict ${METHOD} | ALL FOLDS (parallel, fold→slot/gpu) | modalities: ${MODALITIES[*]}"
    for F in 0 1 2 3; do predict_fold "$F" "$F" "$F" & done
    wait
    echo "[$(date '+%H:%M:%S')] all folds done → ${PREDICTIONS_ROOT}/${MODEL_SOURCE}_models/${CATEGORY}/${RUN_ID}/"
else
    echo "  modalities: ${MODALITIES[*]}"
    predict_fold "${FOLD}" "${SLOT:-0}" "${GPU:-0}"
fi
