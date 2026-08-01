#!/usr/bin/env bash
# Evaluate one chaos-model run on KIDNEY-T2W: Dice + HD95 for the kidney label
# only (chaos right_kidney(2)+left_kidney(3) merged vs this GT's binary kidney(1),
# via the dedicated 06_00_evaluate_kidney_t2w.py shim -- mirrors TRUSTED exactly).
#
# Usage: bash 06_01_evaluate_run.sh <RUN_ID> [FOLD]
#   FOLD: 0-2 or "all" (default: all, project fold policy)
#   CATEGORY (nnUNet|auglab): env override; otherwise auto-detected from RUN_ID.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

RUN_ID="${1:?RUN_ID required (chaos training run dir name)}"
FOLD="${2:-all}"

if [ -z "${CATEGORY:-}" ]; then
    _matches=()
    for _c in "${PREDICTIONS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}"/*/; do
        [ -d "${_c}${RUN_ID}" ] && _matches+=("$(basename "$_c")")
    done
    case "${#_matches[@]}" in
        1) CATEGORY="${_matches[0]}";;
        0) echo "ERROR: RUN_ID '${RUN_ID}' not found under any ${PREDICTIONS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}/<category>/" >&2; exit 1;;
        *) echo "ERROR: RUN_ID '${RUN_ID}' in multiple categories: ${_matches[*]}. Set CATEGORY=<one>." >&2; exit 1;;
    esac
    echo "[$(date '+%H:%M:%S')] auto-detected CATEGORY=${CATEGORY} for ${RUN_ID}"
fi

EVALUATE_PY="$(dirname "${BASH_SOURCE[0]}")/06_00_evaluate_kidney_t2w.py"
GT_DIR="${nnUNet_raw}/labelsTs_t2"
PRED_BASE="${PREDICTIONS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"
METRICS_BASE="${METRICS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}/${CATEGORY}_${RUN_ID}"

[ -d "$PRED_BASE" ] || { echo "ERROR: no predictions at $PRED_BASE" >&2; exit 1; }
[ -d "$GT_DIR" ]    || { echo "ERROR: GT dir missing: $GT_DIR — run 05_00_build_test_inputs.py" >&2; exit 1; }

# CHAOS FOV restriction, anchored on KIDNEY (this dataset's own GT id 1 -- both
# kidneys merged into one label, so the anchor bbox spans both).
FOV="${FOV:-1}"
FOV_JSON="${CHAOS_DATASET_ROOT}/5_scripts_chaos/06_evaluate/chaos_fov_margins.json"
fov_flags() {
    [ "$FOV" != "1" ] && return 0
    [ -f "$FOV_JSON" ] || { echo "ERROR: FOV margins JSON missing: $FOV_JSON" >&2; exit 1; }
    local mm
    mm=$(.venv/bin/python -c "import json;d=json.load(open('$FOV_JSON'))['${CHAOS_TRAINING_CONTRAST}']['kidney'];print(d['sup_mm'],d['inf_mm'])") \
        || { echo "ERROR: no FOV margins for contrast=${CHAOS_TRAINING_CONTRAST} anchor=kidney" >&2; exit 1; }
    echo "--fov_anchor_gt_ids 1 --fov_sup_mm ${mm% *} --fov_inf_mm ${mm#* }"
}

eval_fold() {
    local F="$1" SLOT="$2"
    local PRED_DIR="${PRED_BASE}/fold${F}/t2"
    local EVAL_DIR="${METRICS_BASE}/fold${F}"

    if [ ! -d "$PRED_DIR" ] || [ -z "$(ls -A "$PRED_DIR"/*.nii.gz 2>/dev/null)" ]; then
        echo "  ! fold${F}: no predictions at $PRED_DIR — skipping" >&2
        return
    fi
    mkdir -p "$EVAL_DIR"
    echo "[$(date '+%H:%M:%S')] evaluate ${CATEGORY}/${RUN_ID} fold${F}"

    .venv/bin/python "$EVALUATE_PY" \
        --pred_dir "$PRED_DIR" --gt_dir "$GT_DIR" \
        --name t2 --out_csv "${EVAL_DIR}/t2_metrics.csv" \
        --workers 4 $(fov_flags)

    .venv/bin/python "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/summarize_fold.py" \
        "${EVAL_DIR}" "${RUN_ID}" "${F}" --group-col modality --groups-word Modalities \
        --title-suffix " | KIDNEY-T2W MRI | kidney only" \
        --groups t2
    echo "[$(date '+%H:%M:%S')] fold${F} done → ${EVAL_DIR}/"
}

if [ "$FOLD" = "all" ]; then
    for F in 0 1 2; do eval_fold "$F" "$F" & done
    wait
else
    eval_fold "${FOLD}" "${SLOT:-0}"
fi
