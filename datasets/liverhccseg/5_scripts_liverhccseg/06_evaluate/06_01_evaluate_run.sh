#!/usr/bin/env bash
# Evaluate one atlas-liver-hcc-model run on LiverHccSeg (4 CE-T1w phases: pre/art/pv/
# del): Dice + HD95 for the tumour label only, per item present in the prediction dir.
#
# LiverHccSeg GT annotates tumour as GT label 1 (rater1, merged across tumor1/2/3
# instances where present — see 00_utils/00_01_bidsify.py). atlas-liver-hcc models emit
# tumour as label 2, so this uses --label_map (cross-label-space, same mechanism as
# lld-mmri-hcc / msd-spleen). Only 14/17 patients are in the test-input dirs (tumour-
# annotated only); liver is deliberately NOT scored here even though this dataset has
# real liver masks, so every OOD/cross-dataset column measures the same construct.
#
# Predictions are expected under:
#   PREDICTIONS_ROOT/{ATLAS_MODEL_TYPE}/{ATLAS_TRAINING_CONTRAST}/{CATEGORY}/{RUN_ID}/fold{k}/{ce-pre_T1w,ce-art_T1w,ce-pv_T1w,ce-del_T1w}/
# GT is under:
#   2_nnUNet_liverhccseg/raw/labelsTs_{item}/
# Metrics written to:
#   METRICS_ROOT/{ATLAS_MODEL_TYPE}/{ATLAS_TRAINING_CONTRAST}/{CATEGORY}_{RUN_ID}/fold{k}/{item}_metrics.csv
#                                                                                             eval_all.csv
#                                                                                             eval_summary.md
#
# Usage:
#   bash 06_01_evaluate_run.sh <RUN_ID> [FOLD]
#   FOLD: 0-2 or "all" (default: all, project fold policy — folds 0 1 2 only)
#   CATEGORY (nnUNet|auglab): env override; otherwise auto-detected from RUN_ID.
#
# Evaluation is CPU-only (no GPU needed) — launched through run_job()
# (scripts/job_runner/run_job.sh, sourced transitively via 00_utils/env.sh)
# with --gpus 0 --wait.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

RUN_ID="${1:?RUN_ID required (atlas-liver-hcc training run dir name)}"
FOLD="${2:-all}"

if [ -z "${CATEGORY:-}" ]; then
    _matches=()
    for _c in "${PREDICTIONS_ROOT}/${ATLAS_MODEL_TYPE}/${ATLAS_TRAINING_CONTRAST}"/*/; do
        [ -d "${_c}${RUN_ID}" ] && _matches+=("$(basename "$_c")")
    done
    case "${#_matches[@]}" in
        1) CATEGORY="${_matches[0]}";;
        0) echo "ERROR: RUN_ID '${RUN_ID}' not found under any ${PREDICTIONS_ROOT}/${ATLAS_MODEL_TYPE}/${ATLAS_TRAINING_CONTRAST}/<category>/" >&2; exit 1;;
        *) echo "ERROR: RUN_ID '${RUN_ID}' in multiple categories: ${_matches[*]}. Set CATEGORY=<one>." >&2; exit 1;;
    esac
    echo "[$(date '+%H:%M:%S')] auto-detected CATEGORY=${CATEGORY} for ${RUN_ID}"
fi

EVALUATE_PY="${ATLAS_DATASET_ROOT}/5_scripts_atlas-liver-hcc/06_evaluate/06_00_evaluate.py"
PRED_BASE="${PREDICTIONS_ROOT}/${ATLAS_MODEL_TYPE}/${ATLAS_TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"
# METRICS_SUBDIR (optional): unset (default) writes to the normal flat METRICS_ROOT/.../
# <contrast>/ layout. Set to e.g. "ablations" to write to METRICS_ROOT/.../<contrast>/
# ablations/ instead — for non-headline result sets (CLAUDE.md's "Within 02_metrics/
# <model>/<contrast>/, a non-headline result set gets its own dedicated subdir"
# convention; mirrors atlas-liver-hcc's/chaos's/brats2024-glioma's/on-harmony's
# 06_01_evaluate_run.sh).
METRICS_SUBDIR="${METRICS_SUBDIR:-}"
METRICS_BASE="${METRICS_ROOT}/${ATLAS_MODEL_TYPE}/${ATLAS_TRAINING_CONTRAST}${METRICS_SUBDIR:+/${METRICS_SUBDIR}}/${CATEGORY}_${RUN_ID}"
# atlas-liver-hcc tumour id (2) -> this dataset's GT tumour id (1). See header note.
LABEL_MAP='{"tumour": [2, 1]}'

[ -d "$PRED_BASE" ] || { echo "ERROR: no predictions at $PRED_BASE" >&2; exit 1; }
[ -f "$EVALUATE_PY" ] || { echo "ERROR: evaluate script not found: $EVALUATE_PY" >&2; exit 1; }

eval_fold() {
    local F="$1" SLOT="$2"
    local PRED_ROOT="${PRED_BASE}/fold${F}"
    local EVAL_DIR="${METRICS_BASE}/fold${F}"

    if [ ! -d "$PRED_ROOT" ]; then
        echo "  ! fold${F}: no predictions dir at $PRED_ROOT — skipping" >&2
        return
    fi
    mkdir -p "$EVAL_DIR"
    echo "[$(date '+%H:%M:%S')] evaluate ${CATEGORY}/${RUN_ID} fold${F}"

    local items=() pids=()
    for d in "$PRED_ROOT"/*/; do
        local it; it="$(basename "$d")"
        [ -n "$(ls -A "$d"/*.nii.gz 2>/dev/null)" ] || continue
        local GT_DIR="${nnUNet_raw}/labelsTs_${it}"
        if [ ! -d "$GT_DIR" ]; then
            echo "  ! fold${F} ${it}: no GT dir ($GT_DIR) — skipping" >&2
            continue
        fi
        items+=("$it")
        run_job --name "liverhccseg_eval_${RUN_ID}_fold${F}_${it}" --gpus 0 --mem 32G --time 00:45:00 --slot "${SLOT}" --wait -- \
            .venv/bin/python "$EVALUATE_PY" \
            --pred_dir "$d" --gt_dir "$GT_DIR" --label_map "$LABEL_MAP" \
            --name "$it" --out_csv "${EVAL_DIR}/${it}_metrics.csv" \
            --workers 2 &
        pids+=($!)
    done
    [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"

    if [ ${#items[@]} -eq 0 ]; then
        echo "  ! fold${F}: no item predictions found — skipping summary" >&2
        return
    fi

    .venv/bin/python "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/summarize_fold.py" \
        "${EVAL_DIR}" "${RUN_ID}" "${F}" --group-col contrast --groups-word Contrasts \
        --title-suffix " | LiverHccSeg MRI | tumour only" \
        --groups "${items[@]}"
    echo "[$(date '+%H:%M:%S')] fold${F} done → ${EVAL_DIR}/"
}

if [ "$FOLD" = "all" ]; then
    echo "[$(date '+%H:%M:%S')] evaluate ${CATEGORY}/${RUN_ID} | FOLDS 0-2 (parallel)"
    for F in 0 1 2; do eval_fold "$F" "$F" & done
    wait
    echo "[$(date '+%H:%M:%S')] all folds done → ${METRICS_BASE}/"
else
    eval_fold "${FOLD}" "${SLOT:-0}"
fi
