#!/usr/bin/env bash
# Evaluate one chaos-model run on CIRRMRI-LIVER T1w+T2w: Dice + HD95 for the liver
# label only, per modality present in the prediction dir.
#
# CIRRMRI-LIVER GT annotates the liver alone, as GT label 1 — the SAME numbering
# chaos itself uses (background 0, liver 1, right_kidney 2, left_kidney 3, spleen 4),
# so this uses the simpler --dataset_json --labels liver mode (like sliver07),
# NOT msd-spleen's --label_map cross-space remap (which was needed there because
# chaos's spleen id (4) didn't match msd-spleen's own GT numbering (1)).
#
# Predictions are expected under:
#   PREDICTIONS_ROOT/{CHAOS_MODEL_TYPE}/{CHAOS_TRAINING_CONTRAST}/{CATEGORY}/{RUN_ID}/fold{k}/{t1,t2}/
# GT is under:
#   2_nnUNet_cirrmri-liver/raw/labelsTs_{t1,t2}/
# Metrics written to:
#   METRICS_ROOT/{CHAOS_MODEL_TYPE}/{CHAOS_TRAINING_CONTRAST}/{CATEGORY}_{RUN_ID}/fold{k}/{t1,t2}_metrics.csv
#                                                                                            eval_all.csv
#                                                                                            eval_summary.md
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

EVALUATE_PY="${CHAOS_DATASET_ROOT}/5_scripts_chaos/06_evaluate/06_00_evaluate.py"
PRED_BASE="${PREDICTIONS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"
METRICS_BASE="${METRICS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}/${CATEGORY}_${RUN_ID}"

[ -d "$PRED_BASE" ] || { echo "ERROR: no predictions at $PRED_BASE" >&2; exit 1; }
[ -f "$EVALUATE_PY" ] || { echo "ERROR: evaluate script not found: $EVALUATE_PY" >&2; exit 1; }

# ── CHAOS FOV restriction ────────────────────────────────────────────────────
# Full-extent MRI (these volumes cover more S-I range than CHAOS's restricted
# training FOV — same rationale as the CT cross-eval sets). CIRRMRI-LIVER has no
# kidneys/spleen (liver only), so anchor on the LIVER (GT id 1), reusing chaos's
# existing liver FOV margins (no new anchor needed — same organ as sliver07).
FOV="${FOV:-1}"
FOV_JSON="${CHAOS_DATASET_ROOT}/5_scripts_chaos/06_evaluate/chaos_fov_margins.json"
fov_flags() {
    [ "$FOV" != "1" ] && return 0
    [ -f "$FOV_JSON" ] || { echo "ERROR: FOV margins JSON missing: $FOV_JSON" >&2; exit 1; }
    local mm
    mm=$(.venv/bin/python -c "import json;d=json.load(open('$FOV_JSON'))['${CHAOS_TRAINING_CONTRAST}']['liver'];print(d['sup_mm'],d['inf_mm'])") \
        || { echo "ERROR: no FOV margins for contrast=${CHAOS_TRAINING_CONTRAST} anchor=liver" >&2; exit 1; }
    echo "--fov_anchor_gt_ids 1 --fov_sup_mm ${mm% *} --fov_inf_mm ${mm#* }"
}

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

    local mods=() pids=()
    for d in "$PRED_ROOT"/*/; do
        local m; m="$(basename "$d")"
        [ -n "$(ls -A "$d"/*.nii.gz 2>/dev/null)" ] || continue
        local GT_DIR="${nnUNet_raw}/labelsTs_${m}"
        if [ ! -d "$GT_DIR" ]; then
            echo "  ! fold${F} ${m}: no GT dir ($GT_DIR) — skipping" >&2
            continue
        fi
        mods+=("$m")
        run_job --name "cirrmri_liver_eval_${RUN_ID}_fold${F}_${m}" --gpus 0 --mem 32G --time 00:45:00 --slot "${SLOT}" --wait -- \
            .venv/bin/python "$EVALUATE_PY" \
            --pred_dir "$d" --gt_dir "$GT_DIR" --dataset_json "$CHAOS_DATASET_JSON" \
            --labels liver --name "$m" --out_csv "${EVAL_DIR}/${m}_metrics.csv" \
            --workers 2 $(fov_flags) &
        pids+=($!)
    done
    [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"

    if [ ${#mods[@]} -eq 0 ]; then
        echo "  ! fold${F}: no modality predictions found — skipping summary" >&2
        return
    fi

    .venv/bin/python "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/summarize_fold.py" \
        "${EVAL_DIR}" "${RUN_ID}" "${F}" --group-col modality --groups-word Modalities \
        --title-suffix " | CIRRMRI-LIVER MRI | liver only" \
        --groups "${mods[@]}"
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
