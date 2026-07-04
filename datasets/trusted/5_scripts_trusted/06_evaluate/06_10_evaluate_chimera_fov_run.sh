#!/usr/bin/env bash
# Evaluate one chaos-model run on the CT⊕US CHIMERAS with FOV-restricted, side-aware
# kidney Dice/HD95 (see 06_00_evaluate_trusted_chimera_fov.py). Unlike the ROI-only
# control (06_06), this scores over the whole CHAOS-equivalent slab, so kidney
# predictions elsewhere in the FOV count as false positives.
#
# Predictions: PREDICTIONS_ROOT/{CHAOS_MODEL_TYPE}/{CHAOS_TRAINING_CONTRAST}/{CATEGORY}/{RUN_ID}/fold{k}/chimera/
# GT   : 2_nnUNet_trusted/raw/labelsTs_chimera/     (placed US kidney)
# CTgt : 1_BIDS_trusted/trusted-kidney/derivatives/manual_masks/  (real kidneys → FOV anchor)
# Manifest: 2_nnUNet_trusted/raw/chimera_manifest.csv (inserted side per patient)
# Metrics (ISOLATED namespace, parallel to the ROI 'chimera/' one):
#   METRICS_ROOT/{CHAOS_MODEL_TYPE}/{CHAOS_TRAINING_CONTRAST}/chimera_fov/{CATEGORY}_{RUN_ID}/fold{k}/chimera_metrics.csv
#
# Usage:  bash 06_10_evaluate_chimera_fov_run.sh <RUN_ID> [FOLD]
#   FOLD: 0-3 or "all" (default all). CATEGORY: env override or auto-detect.
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

EVALUATE_PY="${PROJECT_ROOT}/datasets/trusted/5_scripts_trusted/06_evaluate/06_00_evaluate_trusted_chimera_fov.py"
PRED_BASE="${PREDICTIONS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"
METRICS_BASE="${METRICS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}/chimera_fov/${CATEGORY}_${RUN_ID}"
GT_DIR="${nnUNet_raw}/labelsTs_chimera"
MANIFEST="${nnUNet_raw}/chimera_manifest.csv"
CT_GT_DIR="${BIDS_ROOT}/derivatives/manual_masks"
[ -d "$PRED_BASE" ] || { echo "ERROR: no predictions at $PRED_BASE" >&2; exit 1; }
[ -f "$MANIFEST" ]  || { echo "ERROR: chimera manifest missing: $MANIFEST" >&2; exit 1; }
[ -d "$CT_GT_DIR" ] || { echo "ERROR: CT dseg dir missing: $CT_GT_DIR" >&2; exit 1; }

# CHAOS kidney FOV margins (median mm) for this training contrast.
FOV_JSON="${CHAOS_DATASET_ROOT}/5_scripts_chaos/06_evaluate/chaos_fov_margins.json"
[ -f "$FOV_JSON" ] || { echo "ERROR: FOV margins JSON missing: $FOV_JSON — run chaos 06_30_measure_chaos_fov.sh" >&2; exit 1; }
read -r SUP_MM INF_MM < <(.venv/bin/python -c "import json;d=json.load(open('$FOV_JSON'))['${CHAOS_TRAINING_CONTRAST}']['kidney'];print(d['sup_mm'],d['inf_mm'])")

eval_fold() {
    local F="$1" SLOT="$2"
    local PRED_DIR="${PRED_BASE}/fold${F}/chimera"
    local EVAL_DIR="${METRICS_BASE}/fold${F}"
    if [ ! -d "$PRED_DIR" ] || [ -z "$(ls -A "$PRED_DIR"/*.nii.gz 2>/dev/null)" ]; then
        echo "  ! fold${F}: no chimera predictions at $PRED_DIR — skipping" >&2; return; fi
    mkdir -p "$EVAL_DIR"
    echo "[$(date '+%H:%M:%S')] eval chimera-fov ${CATEGORY}/${RUN_ID} fold${F} (slab +${SUP_MM}/${INF_MM}mm)"
    run_job --name "trusted_evalchimfov_${RUN_ID}_fold${F}" --gpus 0 --slot "${SLOT}" --mem 64G --time 02:00:00 --wait -- \
        .venv/bin/python "$EVALUATE_PY" \
        --pred_dir "$PRED_DIR" --gt_dir "$GT_DIR" --manifest "$MANIFEST" --ct_gt_dir "$CT_GT_DIR" \
        --name chimera --out_csv "${EVAL_DIR}/chimera_metrics.csv" \
        --fov_sup_mm "$SUP_MM" --fov_inf_mm "$INF_MM" --workers 4
    echo "[$(date '+%H:%M:%S')] fold${F} done → ${EVAL_DIR}/"
}

if [ "$FOLD" = "all" ]; then
    for F in 0 1 2 3; do eval_fold "$F" "$F" & done; wait
    echo "[$(date '+%H:%M:%S')] all folds done → ${METRICS_BASE}/"
else
    eval_fold "${FOLD}" "${SLOT:-0}"
fi
