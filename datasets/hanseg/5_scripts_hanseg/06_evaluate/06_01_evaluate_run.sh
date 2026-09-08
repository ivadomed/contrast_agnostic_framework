#!/usr/bin/env bash
# Evaluate one toothfairy2-model prediction run on the FOV-matched HaN-Seg CT test
# set: every fold, mandible only.
#
# Two-step, and the first step is NOT optional: predictions are 3-class
# (mandible/lower_teeth/pharynx) while HaN-Seg's GT is a single Bone_Mandible that
# INCLUDES the lower dentition. 05_20_merge_mandible_union.py collapses the
# prediction to that union first; scoring `mandible` alone would count every lower
# tooth as a false negative and report a large artefactual deficit.
#
# Writes ct_metrics.csv + eval_all.csv + eval_summary.md per fold under
#   8_results_hanseg/02_metrics/toothfairy2_model/cbct/<CATEGORY>_<RUN_ID>/fold{F}/
#
# Usage: bash 06_01_evaluate_run.sh <RUN_ID> <CATEGORY:nnUNet|auglab> [FOLD(default all)]
# Optional env: METRICS_SUBDIR (e.g. ablations), CKPT_TAG (default best)
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

RUN_ID="${1:?need RUN_ID}"
CATEGORY="${2:?need CATEGORY (nnUNet|auglab)}"
FOLD_ARG="${3:-all}"
CKPT_TAG="${CKPT_TAG:-best}"
ITEM="ct"

PRED_BASE="${PREDICTIONS_ROOT}/${TF2_MODEL_TYPE}/${TF2_TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"
_PRED_SUBDIR=""; [ "${CKPT_TAG}" != "best" ] && _PRED_SUBDIR="${CKPT_TAG}/"
_OUT_SUFFIX=""; [ "${CKPT_TAG}" != "best" ] && _OUT_SUFFIX="_${CKPT_TAG}"
METRICS_SUBDIR="${METRICS_SUBDIR:-}"
OUT_BASE="${METRICS_ROOT}/${TF2_MODEL_TYPE}/${TF2_TRAINING_CONTRAST}${METRICS_SUBDIR:+/${METRICS_SUBDIR}}/${CATEGORY}_${RUN_ID}${_OUT_SUFFIX}"
GT_DIR="${nnUNet_raw}/labelsTs_${ITEM}"

if [ "${FOLD_ARG}" = "all" ]; then FOLDS="0 1 2"; else FOLDS="${FOLD_ARG}"; fi
[ -d "$PRED_BASE" ] || { echo "ERROR: no predictions at $PRED_BASE" >&2; exit 1; }
[ -d "$GT_DIR" ]    || { echo "ERROR: no GT at $GT_DIR — run 01_prepare first" >&2; exit 1; }

mkdir -p "${OUT_BASE}/_logs"
run_job --name "hanseg_eval_${CATEGORY}_${RUN_ID}${_OUT_SUFFIX}" --gpus 0 --cpus 8 \
    --mem "${EVAL_MEM:-32G}" --time "${EVAL_TIME:-1:30:00}" \
    --log "${OUT_BASE}/_logs/eval_${RUN_ID}${_OUT_SUFFIX}.log" --wait -- bash -c "
set -euo pipefail
cd '${PROJECT_ROOT}'
for F in ${FOLDS}; do
    PRED_DIR='${PRED_BASE}'/fold\${F}/${_PRED_SUBDIR}${ITEM}
    [ -d \"\${PRED_DIR}\" ] || { echo \"  skip fold\${F}: no preds (\${PRED_DIR})\"; continue; }
    MERGED=\"\${PRED_DIR}_mandible_union\"
    .venv/bin/python datasets/hanseg/5_scripts_hanseg/05_predict/05_20_merge_mandible_union.py \
        --pred_dir \"\${PRED_DIR}\" --out_dir \"\${MERGED}\"
    OUT_CSV='${OUT_BASE}'/fold\${F}/${ITEM}_metrics.csv
    mkdir -p \"\$(dirname \"\${OUT_CSV}\")\"
    echo \"[\$(date '+%H:%M:%S')] eval ${RUN_ID} fold\${F} ${ITEM}\"
    .venv/bin/python datasets/hanseg/5_scripts_hanseg/06_evaluate/06_00_evaluate.py \
        --pred_dir \"\${MERGED}\" --gt_dir '${GT_DIR}' \
        --label_map '{\"mandible\": [1, 1]}' \
        --name '${ITEM}' --out_csv \"\${OUT_CSV}\" --workers 8
    OUT_FOLD='${OUT_BASE}'/fold\${F}
    if ls \"\${OUT_FOLD}\"/*_metrics.csv >/dev/null 2>&1; then
        .venv/bin/python datasets/00_commun_scripts/00_03_evaluate/summarize_fold.py \
            \"\${OUT_FOLD}\" '${RUN_ID}' \"\${F}\" \
            --groups ${ITEM} --group-col contrast --groups-word Contrasts
    fi
done
echo '→ ${OUT_BASE}'
"
