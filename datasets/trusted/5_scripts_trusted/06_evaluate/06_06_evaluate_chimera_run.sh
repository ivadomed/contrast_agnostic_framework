#!/usr/bin/env bash
# Evaluate one chaos-model run on the CT⊕US CHIMERAS — kidney Dice/HD95 measured
# STRICTLY on the pasted US region (via roiTs_chimera), with a center-prior control.
# See 06_00_evaluate_trusted_chimera.py and 05_predict/05_20_build_chimera_inputs.py.
#
# Predictions: PREDICTIONS_ROOT/{CHAOS_MODEL_TYPE}/{CHAOS_TRAINING_CONTRAST}/{CATEGORY}/{RUN_ID}/fold{k}/chimera/
# GT  : 2_nnUNet_trusted/raw/labelsTs_chimera/   ROI : 2_nnUNet_trusted/raw/roiTs_chimera/
# Metrics (ISOLATED from ct/us so the experiment never clobbers the standard results):
#   METRICS_ROOT/{CHAOS_MODEL_TYPE}/{CHAOS_TRAINING_CONTRAST}/chimera/{CATEGORY}_{RUN_ID}/fold{k}/chimera_metrics.csv
#
# Usage:  bash 06_06_evaluate_chimera_run.sh <RUN_ID> [FOLD]
#   FOLD: 0-3 or "all" (default all, folds parallel). CATEGORY: env override or auto-detect.
#   For t2spir: pre-export CHAOS_TRAINING_CONTRAST=t2spir (+ DATASET_ID/DS_NAME) first.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

RUN_ID="${1:?RUN_ID required}"
FOLD="${2:-all}"

if [ -z "${CATEGORY:-}" ]; then
    _matches=()
    for _c in "${PREDICTIONS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}"/*/; do
        [ -d "${_c}${RUN_ID}" ] && _matches+=("$(basename "$_c")")
    done
    case "${#_matches[@]}" in
        1) CATEGORY="${_matches[0]}";;
        0) echo "ERROR: RUN_ID '${RUN_ID}' not found under ${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}/<cat>/" >&2; exit 1;;
        *) echo "ERROR: RUN_ID in multiple categories: ${_matches[*]}. Set CATEGORY=." >&2; exit 1;;
    esac
fi

EVALUATE_PY="${PROJECT_ROOT}/datasets/trusted/5_scripts_trusted/06_evaluate/06_00_evaluate_trusted_chimera.py"
PRED_BASE="${PREDICTIONS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"
METRICS_BASE="${METRICS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}/chimera/${CATEGORY}_${RUN_ID}"
GT_DIR="${nnUNet_raw}/labelsTs_chimera"
ROI_DIR="${nnUNet_raw}/roiTs_chimera"
[ -d "$PRED_BASE" ] || { echo "ERROR: no predictions at $PRED_BASE" >&2; exit 1; }

eval_fold() {
    local F="$1" SLOT="$2"
    local PRED_DIR="${PRED_BASE}/fold${F}/chimera"
    local EVAL_DIR="${METRICS_BASE}/fold${F}"
    if [ ! -d "$PRED_DIR" ] || [ -z "$(ls -A "$PRED_DIR"/*.nii.gz 2>/dev/null)" ]; then
        echo "  ! fold${F}: no chimera predictions at $PRED_DIR — skipping" >&2; return; fi
    mkdir -p "$EVAL_DIR"
    echo "[$(date '+%H:%M:%S')] eval chimera ${CATEGORY}/${RUN_ID} fold${F}"
    run_job --name "trusted_evalchim_${RUN_ID}_fold${F}" --gpus 0 --slot "${SLOT}" --mem 96G --time 02:00:00 --wait -- \
        .venv/bin/python "$EVALUATE_PY" \
        --pred_dir "$PRED_DIR" --gt_dir "$GT_DIR" --roi_dir "$ROI_DIR" \
        --name chimera --out_csv "${EVAL_DIR}/chimera_metrics.csv" --workers 4
    echo "[$(date '+%H:%M:%S')] fold${F} done → ${EVAL_DIR}/"
}

if [ "$FOLD" = "all" ]; then
    for F in 0 1 2 3; do eval_fold "$F" "$F" & done; wait
    echo "[$(date '+%H:%M:%S')] all folds done → ${METRICS_BASE}/"
else
    eval_fold "${FOLD}" "${SLOT:-0}"
fi
