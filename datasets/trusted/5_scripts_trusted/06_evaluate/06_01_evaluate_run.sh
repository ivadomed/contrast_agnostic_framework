#!/usr/bin/env bash
# Evaluate one chaos-model run on TRUSTED CT+US: Dice + HD95 for the kidney.
#
# IMPORTANT — cross-dataset, cross-label-space evaluation:
#   Predictions are from chaos-trained models (chaos label IDs: right_kidney 2,
#   left_kidney 3). TRUSTED GT is a single binary kidney label (1). The evaluator
#   06_00_evaluate_trusted.py MERGES chaos {2,3} → "kidney" — see its header.
#   Scoreable organ: kidney (only).
#
# Predictions: PREDICTIONS_ROOT/{CHAOS_MODEL_TYPE}/{CHAOS_TRAINING_CONTRAST}/{CATEGORY}/{RUN_ID}/fold{k}/{item}/
# GT:          2_nnUNet_trusted/raw/labelsTs_{item}/        (item ∈ {ct,us})
# Metrics:     METRICS_ROOT/{CHAOS_MODEL_TYPE}/{CHAOS_TRAINING_CONTRAST}/{CATEGORY}_{RUN_ID}/fold{k}/{item}_metrics.csv
#                                                                                                  eval_all.csv
#                                                                                                  eval_summary.md
#
# Usage (same CLI as amos/sliver07 06_01_evaluate_run.sh):
#   bash 06_01_evaluate_run.sh <RUN_ID> [FOLD] [ITEMS...]
#   FOLD: 0-3 or "all" (default: all)
#   ITEMS: ct us or subset (default: ct us)
#   CATEGORY (nnUNet|auglab): env override; otherwise auto-detected from RUN_ID.
#
# Examples:
#   bash 06_01_evaluate_run.sh chaos_t1in_v26_6_2_train050_val100_20260615_213615
#   CATEGORY=auglab bash 06_01_evaluate_run.sh chaos_t1in_synthseg_EM_train100_val000_20260611_120000 all us
#
# Evaluation is CPU-only — each item is launched through run_job() --gpus 0 --wait,
# since the per-fold summary script below needs every item's CSV to exist first.
# For T2spir runs, pre-export the CHAOS_* vars (or source env_t2spir.sh) before
# calling — same mechanism as the predict wrappers.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

RUN_ID="${1:?RUN_ID required (chaos training run dir name)}"
FOLD="${2:-all}"
shift $(( $# >= 2 ? 2 : $# )) || true
ITEMS=("$@"); [ ${#ITEMS[@]} -eq 0 ] && ITEMS=(ct us)

# CATEGORY: env override if given, else auto-detect which
# PREDICTIONS_ROOT/{CHAOS_MODEL_TYPE}/{CHAOS_TRAINING_CONTRAST}/<category>/ holds this RUN_ID.
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

EVALUATE_PY="${PROJECT_ROOT}/datasets/trusted/5_scripts_trusted/06_evaluate/06_00_evaluate_trusted.py"
PRED_BASE="${PREDICTIONS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"
METRICS_BASE="${METRICS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}/${CATEGORY}_${RUN_ID}"

[ -d "$PRED_BASE" ] || { echo "ERROR: no predictions at $PRED_BASE" >&2; exit 1; }

eval_fold() {
    local F="$1" SLOT="$2"
    local EVAL_DIR="${METRICS_BASE}/fold${F}"
    mkdir -p "$EVAL_DIR"
    echo "[$(date '+%H:%M:%S')] evaluate ${CATEGORY}/${RUN_ID} fold${F} | items: ${ITEMS[*]}"

    local pids=() item_ok=()
    for item in "${ITEMS[@]}"; do
        local PRED_DIR="${PRED_BASE}/fold${F}/${item}"
        local GT_DIR="${nnUNet_raw}/labelsTs_${item}"
        if [ ! -d "$PRED_DIR" ] || [ -z "$(ls -A "$PRED_DIR"/*.nii.gz 2>/dev/null)" ]; then
            echo "  ! fold${F} ${item}: no predictions at $PRED_DIR — skipping" >&2
            continue
        fi
        if [ ! -d "$GT_DIR" ]; then
            echo "  ! fold${F} ${item}: no GT dir $GT_DIR — run 05_00_build_test_inputs.py" >&2
            continue
        fi
        item_ok+=("$item")
        # 96G: TRUSTED CT is large (up to ~244 M voxels) and monai's HD95 surface-distance
        # is memory-heavy on it (measured ~50 G at 208 M → OOM at the old 48 G). US (1.5 mm,
        # ~5 M voxels) needs far less but shares this request — harmless, packs ~5/node.
        run_job --name "trusted_eval_${RUN_ID}_fold${F}_${item}" --gpus 0 --slot "${SLOT}" --mem 96G --time 02:00:00 --wait -- \
            .venv/bin/python "$EVALUATE_PY" \
            --pred_dir "$PRED_DIR" \
            --gt_dir   "$GT_DIR" \
            --name     "$item" \
            --out_csv  "${EVAL_DIR}/${item}_metrics.csv" \
            --workers  1 &
        pids+=($!)
    done
    [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
    [ ${#item_ok[@]} -eq 0 ] && { echo "  ! fold${F}: nothing to evaluate"; return; }

    # Merge per-item CSVs → eval_all.csv + per-fold summary (shared)
    .venv/bin/python "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/summarize_fold.py" \
        "$EVAL_DIR" "$RUN_ID" "$F" --group-col modality --groups-word Modalities --label-word Organs \
        --title-suffix " | TRUSTED CT+US | kidney only | chaos-trained models" \
        --note "chaos right_kidney(2)+left_kidney(3) merged → kidney vs TRUSTED binary GT(1)." \
        --groups "${item_ok[@]}"
    echo "[$(date '+%H:%M:%S')] fold${F} done → ${EVAL_DIR}/"
}

if [ "$FOLD" = "all" ]; then
    echo "[$(date '+%H:%M:%S')] evaluate ${CATEGORY}/${RUN_ID} | ALL FOLDS (parallel)"
    for F in 0 1 2 3; do eval_fold "$F" "$F" & done
    wait
    echo "[$(date '+%H:%M:%S')] all folds done → ${METRICS_BASE}/"
else
    eval_fold "${FOLD}" "${SLOT:-0}"
fi
