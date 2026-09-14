#!/usr/bin/env bash
# Evaluate one ispy2 OWN-model prediction run (headline ispy2-internal
# cross-contrast eval): every fold x every held-out test item (t1wce/t2w),
# scoring the `tumour` label against the per-item held-out GT
# (nnUNet_raw/Dataset<id>_.../labelsTs_<item>). Modeled directly on
# open-ms's 06_01_evaluate_run.sh (single-label, cross-contrast pattern).
# Writes <item>_metrics.csv + eval_all.csv + eval_summary.md per fold under
#   8_results_ispy2/02_metrics/ispy2_model/<TRAINING_CONTRAST>/<CATEGORY>_<RUN_ID>/fold{F}/
#
# Usage: bash 06_01_evaluate_own_run.sh <RUN_ID> <CATEGORY:nnUNet|auglab> <DATASET_ID:100|101> [FOLD(default all)]
# Optional env: CKPT_TAG (default "best") -- checkpoint_best is the project
#   default (see CLAUDE.md); "final" reads fold{F}/<tag>/<item> predictions
#   instead (predict with CHECKPOINT=checkpoint_final.pth first).
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

RUN_ID="${1:?need RUN_ID}"
CATEGORY="${2:?need CATEGORY (nnUNet|auglab)}"
DATASET_ID="${3:?need DATASET_ID (100=t1wce-trained, 101=t2w-trained)}"
FOLD_ARG="${4:-all}"
CKPT_TAG="${CKPT_TAG:-best}"

# TRAINING_CONTRAST must match the DATASET_ID passed (own-model results are
# stored under PREDICTIONS_ROOT/<MODEL_TYPE>/<TRAINING_CONTRAST>/<CATEGORY>/) --
# derive it here rather than trusting env.sh's t1wce default, since this script
# is called for both training contrasts (CLAUDE.md's "DATASET_ID/CATEGORY must
# be passed explicitly for non-primary contrast" gotcha).
case "${DATASET_ID}" in
    100) export TRAINING_CONTRAST="t1wce" ;;
    101) export TRAINING_CONTRAST="t2w" ;;
    *) echo "ERROR: unknown DATASET_ID ${DATASET_ID} (expect 100 or 101)" >&2; exit 1 ;;
esac

ITEMS=(t1wce t2w)
_DS_NAME="$(ls "${nnUNet_raw}" | grep "^Dataset0*${DATASET_ID}_" | head -1)"
DJ="${nnUNet_raw}/${_DS_NAME}/dataset.json"
PRED_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"
_PRED_SUBDIR=""; [ "${CKPT_TAG}" != "best" ] && _PRED_SUBDIR="${CKPT_TAG}/"
_OUT_SUFFIX=""; [ "${CKPT_TAG}" != "best" ] && _OUT_SUFFIX="_${CKPT_TAG}"
METRICS_SUBDIR="${METRICS_SUBDIR:-}"
OUT_BASE="${METRICS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}${METRICS_SUBDIR:+/${METRICS_SUBDIR}}/${CATEGORY}_${RUN_ID}${_OUT_SUFFIX}"

if [ "${FOLD_ARG}" = "all" ]; then FOLDS="0 1 2"; else FOLDS="${FOLD_ARG}"; fi

[ -d "$PRED_BASE" ] || { echo "ERROR: no predictions at $PRED_BASE" >&2; exit 1; }

# CPU-only metric computation -- too slow for the login node, route through
# run_job as one CPU-only compute-node job (never request a GPU for this).
mkdir -p "${OUT_BASE}/_logs"
run_job --name "ispy2_own_eval_${CATEGORY}_${RUN_ID}${_OUT_SUFFIX}" --gpus 0 --cpus 8 --mem "${EVAL_MEM:-32G}" --time "${EVAL_TIME:-1:00:00}" \
    --log "${OUT_BASE}/_logs/eval_${RUN_ID}${_OUT_SUFFIX}.log" --wait -- bash -c "
set -euo pipefail
cd '${PROJECT_ROOT}'
for F in ${FOLDS}; do
    for item in ${ITEMS[*]}; do
        PRED_DIR='${PRED_BASE}'/fold\${F}/${_PRED_SUBDIR}\${item}
        GT_DIR='${nnUNet_raw}/${_DS_NAME}'/labelsTs_\${item}
        [ -d \"\${PRED_DIR}\" ] || { echo \"  skip fold\${F}/\${item}: no preds (\${PRED_DIR})\"; continue; }
        OUT_CSV='${OUT_BASE}'/fold\${F}/\${item}_metrics.csv
        mkdir -p \"\$(dirname \"\${OUT_CSV}\")\"
        echo \"[\$(date '+%H:%M:%S')] eval ${RUN_ID} fold\${F} \${item}\"
        .venv/bin/python datasets/ispy2/5_scripts_ispy2/06_evaluate/06_00_evaluate.py \
            --pred_dir \"\${PRED_DIR}\" --gt_dir \"\${GT_DIR}\" --dataset_json '${DJ}' \
            --labels tumour --name \"\${item}\" --out_csv \"\${OUT_CSV}\" --workers 8
    done
    OUT_FOLD='${OUT_BASE}'/fold\${F}
    if ls \"\${OUT_FOLD}\"/*_metrics.csv >/dev/null 2>&1; then
        .venv/bin/python datasets/00_commun_scripts/00_03_evaluate/summarize_fold.py \
            \"\${OUT_FOLD}\" '${RUN_ID}' \"\${F}\" \
            --groups ${ITEMS[*]} --group-col contrast --groups-word Contrasts
    fi
done
echo '→ ${OUT_BASE}'
"
