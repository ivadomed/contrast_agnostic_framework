#!/usr/bin/env bash
# Evaluate one chaos-model run on SLIVER07 CT: Dice + HD95 for the liver label only.
#
# SLIVER07 GT annotates the liver alone (label 1). Chaos models emit 4 labels but
# we restrict scoring to liver via --labels liver on the chaos dataset.json.
# The chaos evaluate.py is method-agnostic and reused directly (no copy needed).
#
# Predictions are expected under:
#   PREDICTIONS_ROOT/{CHAOS_MODEL_TYPE}/{CHAOS_TRAINING_CONTRAST}/{CATEGORY}/{RUN_ID}/fold{k}/ct/
# GT is under:
#   2_nnUNet_sliver07/raw/labelsTs_ct/
# Metrics written to:
#   METRICS_ROOT/{CHAOS_MODEL_TYPE}/{CHAOS_TRAINING_CONTRAST}/{CATEGORY}_{RUN_ID}/fold{k}/ct_metrics.csv
#                                                                                          eval_all.csv
#                                                                                          eval_summary.md
#
# Usage (same CLI as brats/chaos 06_01_evaluate_run.sh):
#   bash 06_01_evaluate_run.sh <RUN_ID> [FOLD]
#   FOLD: 0-3 or "all" (default: all, 4 folds parallel)
#   CATEGORY (nnUNet|auglab): env override; otherwise auto-detected from RUN_ID.
#
# Examples:
#   bash 06_01_evaluate_run.sh chaos_t1in_baseline_20260614_153230
#   CATEGORY=auglab bash 06_01_evaluate_run.sh chaos_t1in_synthseg_EM_train100_val000_20260611_120000 all
#   bash 06_01_evaluate_run.sh chaos_t1in_v26_6_2_train090_val000_20260614_205937 2
#
# Evaluation is CPU-only (no GPU needed) — launched through run_job()
# (scripts/job_runner/run_job.sh, sourced transitively via 00_utils/env.sh)
# with --gpus 0 --wait, since the per-fold summary right after needs the CSV
# to already exist.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

# CLI matches brats/chaos 06_01: positional <RUN_ID> [FOLD]; CATEGORY is an env override
# (CATEGORY=auglab bash 06_01 ...) or auto-detected below.
RUN_ID="${1:?RUN_ID required (chaos training run dir name)}"
FOLD="${2:-all}"

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

EVALUATE_PY="${CHAOS_DATASET_ROOT}/5_scripts_chaos/06_evaluate/06_00_evaluate.py"
GT_DIR="${nnUNet_raw}/labelsTs_ct"
PRED_BASE="${PREDICTIONS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"
METRICS_BASE="${METRICS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}/${CATEGORY}_${RUN_ID}"

[ -d "$PRED_BASE" ] || { echo "ERROR: no predictions at $PRED_BASE" >&2; exit 1; }
[ -d "$GT_DIR" ]    || { echo "ERROR: GT dir missing: $GT_DIR — run 05_00_build_test_inputs.py" >&2; exit 1; }
[ -f "$EVALUATE_PY" ] || { echo "ERROR: evaluate script not found: $EVALUATE_PY" >&2; exit 1; }

eval_fold() {
    local F="$1" SLOT="$2"
    local PRED_DIR="${PRED_BASE}/fold${F}/ct"
    local EVAL_DIR="${METRICS_BASE}/fold${F}"

    if [ ! -d "$PRED_DIR" ] || [ -z "$(ls -A "$PRED_DIR"/*.nii.gz 2>/dev/null)" ]; then
        echo "  ! fold${F}: no predictions at $PRED_DIR — skipping" >&2
        return
    fi
    mkdir -p "$EVAL_DIR"
    echo "[$(date '+%H:%M:%S')] evaluate ${CATEGORY}/${RUN_ID} fold${F}"

    run_job --name "sliver07_eval_${RUN_ID}_fold${F}" --gpus 0 --slot "${SLOT}" --mem 48G --time 01:00:00 --wait -- \
        .venv/bin/python "$EVALUATE_PY" \
        --pred_dir  "$PRED_DIR" \
        --gt_dir    "$GT_DIR" \
        --dataset_json "$CHAOS_DATASET_JSON" \
        --labels liver \
        --name ct \
        --out_csv "${EVAL_DIR}/ct_metrics.csv" \
        --workers 4

    # Merge ct CSV → eval_all.csv (consumed by 06_03_aggregate) + summary (shared)
    .venv/bin/python "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/summarize_fold.py" \
        "${EVAL_DIR}" "${RUN_ID}" "${F}" --group-col modality --groups-word Modalities \
        --title-suffix " | SLIVER07 CT | liver only" \
        --groups ct
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
