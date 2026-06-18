#!/usr/bin/env bash
# ============================================================================
#  AMOS prediction — USES MODELS TRAINED ON ANOTHER DATASET (chaos).
#  ----------------------------------------------------------------------------
#  AMOS currently has NO trained models of its own (training pipeline pending —
#  see datasets/amos/README.md). These scripts load chaos-trained checkpoints
#  (MR T1-DUAL in-phase, Dataset060_CHAOS_MR_T1in) and run them over AMOS CT
#  and MRI volumes as a cross-dataset generalization probe.
#
#  Key mechanics (same pattern as datasets/sliver07/5_scripts_sliver07/05_predict/):
#    - nnUNet_results / nnUNet_raw / -tr / -d  all point at CHAOS via CHAOS_* env vars
#    - trainer classes are nnUNetTrainerCHAOS* (defined in chaos's scripts dir,
#      added to PYTHONPATH by env.sh)
#    - outputs segregated under PREDICTIONS_ROOT/chaos_models/  so that
#      future AMOS-native model predictions (MODEL_SOURCE=amos) never collide
#
#  If AMOS is eventually trained natively, add a sibling common script
#  pointing nnUNet_results at 6_checkpoints_amos or 8_results_amos/nnUNet.
# ============================================================================
# Shared predict template — sourced by 05_0X_predict_chaos_<method>.sh.
# NOT invoked directly.
#
# Set by the calling wrapper:
#   METHOD      label, e.g. v26_6_2
#   TRAINER     chaos trainer class, e.g. nnUNetTrainerCHAOSV26_6_2
#   CATEGORY    chaos prediction group: nnUNet | auglab
#   RUN_ID      chaos training run dir name (wrapper provides default)
# Args (from user):
#   $1  RUN_ID  optional — overrides wrapper default
#   $2  FOLD    optional — 0-3 or "all" (default all)
#   $3… MODALITIES  optional subset of: ct mri (default: ct mri)
#
# Output:
#   PREDICTIONS_ROOT/chaos_models/{CATEGORY}/{RUN_ID}/fold{k}/{modality}/{case}.nii.gz

set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"

MODEL_SOURCE="${MODEL_SOURCE:-chaos}"

RUN_ID="${1:-${RUN_ID:?RUN_ID required (chaos training run dir name)}}"
FOLD="${2:-all}"
shift $(( $# >= 2 ? 2 : $# )) || true
MODALITIES=("$@"); [ ${#MODALITIES[@]} -eq 0 ] && MODALITIES=(ct mri)

DATASET_ID="${CHAOS_DATASET_ID:-60}"
CHECKPOINT="${CHECKPOINT:-checkpoint_best.pth}"
CATEGORY="${CATEGORY:-nnUNet}"

CHAOS_RUN_DIR="${CHAOS_PREDICTIONS_ROOT}/${CATEGORY}/${RUN_ID}"
[ -d "$CHAOS_RUN_DIR" ] || {
    echo "ERROR: chaos run dir not found: $CHAOS_RUN_DIR" >&2; exit 1
}
echo "=================================================================="
echo " AMOS ← CHAOS models (cross-dataset inference)"
echo " MODEL_SOURCE : ${MODEL_SOURCE}"
echo " METHOD       : ${METHOD}"
echo " TRAINER      : ${TRAINER}"
echo " CHAOS RUN    : ${CHAOS_RUN_DIR}"
echo " MODALITIES   : ${MODALITIES[*]}"
echo "=================================================================="

predict_fold() {
    local F="$1" SLOT="$2" GPU="$3"
    echo "[$(date '+%H:%M:%S')] predict ${METHOD} | chaos run=${RUN_ID} | fold=${F} | slot=${SLOT} gpu=${GPU}"
    for mod in "${MODALITIES[@]}"; do
        local INPUT_DIR="${nnUNet_raw}/imagesTs_${mod}"
        local OUTPUT_DIR="${PREDICTIONS_ROOT}/${MODEL_SOURCE}_models/${CATEGORY}/${RUN_ID}/fold${F}/${mod}"
        if [ ! -d "$INPUT_DIR" ] || [ -z "$(ls -A "$INPUT_DIR" 2>/dev/null)" ]; then
            echo "  ! fold${F} skip ${mod}: $INPUT_DIR missing/empty — run 05_00_build_test_inputs.py" >&2
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
        " 2>&1 | tee "/tmp/amos_predict_${METHOD}_${RUN_ID}_fold${F}_${mod}.log"
        echo "  ✓ fold${F} ${mod} done"
    done
    echo "[$(date '+%H:%M:%S')] fold${F} all modalities done"
}

if [ "$FOLD" = "all" ]; then
    echo "[$(date '+%H:%M:%S')] ALL FOLDS (parallel, fold→slot/gpu) | modalities: ${MODALITIES[*]}"
    for F in 0 1 2 3; do predict_fold "$F" "$F" "$F" & done
    wait
    echo "[$(date '+%H:%M:%S')] all folds done → ${PREDICTIONS_ROOT}/${MODEL_SOURCE}_models/${CATEGORY}/${RUN_ID}/"
else
    predict_fold "${FOLD}" "${SLOT:-0}" "${GPU:-0}"
fi
