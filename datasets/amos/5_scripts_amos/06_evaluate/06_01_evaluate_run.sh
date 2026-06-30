#!/usr/bin/env bash
# Evaluate one chaos-model run on AMOS CT+MRI: Dice + HD95 for 4 organs.
#
# IMPORTANT — cross-dataset evaluation:
#   Predictions are from chaos-trained models (chaos label IDs).
#   AMOS GT uses different label IDs for the same organs.
#   06_00_evaluate_amos.py handles the remapping explicitly — see its header.
#   Scoreable organs: liver, right_kidney, left_kidney, spleen.
#
# Predictions: PREDICTIONS_ROOT/{CHAOS_MODEL_TYPE}/{CHAOS_TRAINING_CONTRAST}/{CATEGORY}/{RUN_ID}/fold{k}/{mod}/
# GT:          2_nnUNet_amos/raw/labelsTs_{mod}/
# Metrics:     METRICS_ROOT/{CHAOS_MODEL_TYPE}/{CHAOS_TRAINING_CONTRAST}/{CATEGORY}_{RUN_ID}/fold{k}/{mod}_metrics.csv
#                                                                                                  eval_all.csv
#                                                                                                  eval_summary.md
#
# Usage (same CLI as brats/chaos 06_01_evaluate_run.sh):
#   bash 06_01_evaluate_run.sh <RUN_ID> [FOLD] [MODALITIES...]
#   FOLD: 0-3 or "all" (default: all)
#   MODALITIES: ct mri or subset (default: ct mri)
#   CATEGORY (nnUNet|auglab): env override; otherwise auto-detected from RUN_ID.
#
# Examples:
#   bash 06_01_evaluate_run.sh chaos_t1in_v26_6_2_train090_val000_20260614_205937
#   CATEGORY=auglab bash 06_01_evaluate_run.sh chaos_t1in_synthseg_EM_train100_val000_20260611_120000 all ct
#
# Evaluation is CPU-only (no GPU needed) — each modality is launched through
# run_job() (scripts/job_runner/run_job.sh, sourced transitively via
# 00_utils/env.sh) with --gpus 0 --wait, since the per-fold summary script
# below needs every modality's CSV to actually exist before it runs.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

# CLI matches brats/chaos 06_01: positional <RUN_ID> [FOLD] [MODALITIES...]; CATEGORY is
# an env override (CATEGORY=auglab bash 06_01 ...) or auto-detected below.
RUN_ID="${1:?RUN_ID required (chaos training run dir name)}"
FOLD="${2:-all}"
shift $(( $# >= 2 ? 2 : $# )) || true
MODALITIES=("$@"); [ ${#MODALITIES[@]} -eq 0 ] && MODALITIES=(ct mri)

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

EVALUATE_PY="${PROJECT_ROOT}/datasets/amos/5_scripts_amos/06_evaluate/06_00_evaluate_amos.py"
PRED_BASE="${PREDICTIONS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"
METRICS_BASE="${METRICS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}/${CATEGORY}_${RUN_ID}"

[ -d "$PRED_BASE" ] || { echo "ERROR: no predictions at $PRED_BASE" >&2; exit 1; }

eval_fold() {
    local F="$1" SLOT="$2"
    local EVAL_DIR="${METRICS_BASE}/fold${F}"
    mkdir -p "$EVAL_DIR"
    echo "[$(date '+%H:%M:%S')] evaluate ${CATEGORY}/${RUN_ID} fold${F} | modalities: ${MODALITIES[*]}"

    local pids=() mod_ok=()
    for mod in "${MODALITIES[@]}"; do
        local PRED_DIR="${PRED_BASE}/fold${F}/${mod}"
        local GT_DIR="${nnUNet_raw}/labelsTs_${mod}"
        if [ ! -d "$PRED_DIR" ] || [ -z "$(ls -A "$PRED_DIR"/*.nii.gz 2>/dev/null)" ]; then
            echo "  ! fold${F} ${mod}: no predictions at $PRED_DIR — skipping" >&2
            continue
        fi
        if [ ! -d "$GT_DIR" ]; then
            echo "  ! fold${F} ${mod}: no GT dir $GT_DIR — run 05_00_build_test_inputs.py" >&2
            continue
        fi
        mod_ok+=("$mod")
        run_job --name "amos_eval_${RUN_ID}_fold${F}_${mod}" --gpus 0 --slot "${SLOT}" --mem 48G --time 02:00:00 --wait -- \
            .venv/bin/python "$EVALUATE_PY" \
            --pred_dir "$PRED_DIR" \
            --gt_dir   "$GT_DIR" \
            --name     "$mod" \
            --out_csv  "${EVAL_DIR}/${mod}_metrics.csv" \
            --workers  1 &
        pids+=($!)
    done
    [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
    [ ${#mod_ok[@]} -eq 0 ] && { echo "  ! fold${F}: nothing to evaluate"; return; }

    # Merge per-modality CSVs → eval_all.csv + per-fold summary (shared)
    .venv/bin/python "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/summarize_fold.py" \
        "$EVAL_DIR" "$RUN_ID" "$F" --group-col modality --groups-word Modalities --label-word Organs \
        --title-suffix " | AMOS CT+MRI | chaos-trained models" \
        --note "Predictions from chaos-trained models (chaos label IDs remapped → AMOS IDs)." \
        --groups "${mod_ok[@]}"
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
