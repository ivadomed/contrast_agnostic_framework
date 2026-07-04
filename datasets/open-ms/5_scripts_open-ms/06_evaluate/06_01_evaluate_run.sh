#!/usr/bin/env bash
# Evaluate one open-ms prediction run: every fold × every contrast (flair/t2w/t1w),
# scoring the `lesion` label against the per-contrast held-out GT (labelsTs_<contrast>).
# Writes <contrast>_metrics.csv + eval_all.csv per fold under
#   8_results_open-ms/02_metrics/<MODEL_TYPE>/<TRAINING_CONTRAST>/<CATEGORY>_<RUN_ID>/fold{F}/
#
# Usage: bash 06_01_evaluate_run.sh <RUN_ID> <CATEGORY:nnUNet|auglab> [FOLD(default all)]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

RUN_ID="${1:?need RUN_ID}"
CATEGORY="${2:?need CATEGORY (nnUNet|auglab)}"
FOLD_ARG="${3:-all}"

ITEMS=(flair t2w t1w)
_DS_NAME="$(ls "${nnUNet_raw}" | grep '^Dataset0*70_' | head -1)"
DJ="${nnUNet_raw}/${_DS_NAME}/dataset.json"
PRED_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"
OUT_BASE="${METRICS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}_${RUN_ID}"

if [ "${FOLD_ARG}" = "all" ]; then FOLDS=(0 1 2 3); else FOLDS=("${FOLD_ARG}"); fi

for F in "${FOLDS[@]}"; do
    for item in "${ITEMS[@]}"; do
        PRED_DIR="${PRED_BASE}/fold${F}/${item}"
        GT_DIR="${nnUNet_raw}/${_DS_NAME}/labelsTs_${item}"
        [ -d "${PRED_DIR}" ] || { echo "  skip fold${F}/${item}: no preds (${PRED_DIR})"; continue; }
        OUT_CSV="${OUT_BASE}/fold${F}/${item}_metrics.csv"
        echo "[$(date '+%H:%M:%S')] eval ${RUN_ID} fold${F} ${item}"
        .venv/bin/python datasets/open-ms/5_scripts_open-ms/06_evaluate/06_00_evaluate.py \
            --pred_dir "${PRED_DIR}" --gt_dir "${GT_DIR}" --dataset_json "${DJ}" \
            --labels lesion --name "${item}" --out_csv "${OUT_CSV}" --workers 8
    done
    # concat per-contrast CSVs into eval_all.csv (header from the first present file)
    OUT_FOLD="${OUT_BASE}/fold${F}"
    if ls "${OUT_FOLD}"/*_metrics.csv >/dev/null 2>&1; then
        { head -1 "$(ls "${OUT_FOLD}"/*_metrics.csv | head -1)";
          for f in "${OUT_FOLD}"/*_metrics.csv; do tail -n +2 "$f"; done; } > "${OUT_FOLD}/eval_all.csv"
    fi
done
echo "→ ${OUT_BASE}"
