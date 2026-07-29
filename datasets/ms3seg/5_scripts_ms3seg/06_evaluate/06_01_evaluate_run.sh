#!/usr/bin/env bash
# Evaluate one open-ms-trained model's predictions on MS3SEG: every fold x every
# contrast (flair/t1w/t2w), scoring MS-lesion only via a cross-label-space
# --label_map (pred_id=1 open-ms lesion -> gt_id=255 MS3SEG abWMH/MS-lesion; see
# 06_00_evaluate_ms3seg.py). GT: labelsTs_<contrast> in 2_nnUNet_ms3seg/raw.
#
# Usage: bash 06_01_evaluate_run.sh <RUN_ID> <CATEGORY:nnUNet|auglab> [FOLD(default all)]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

RUN_ID="${1:?need RUN_ID}"
CATEGORY="${2:?need CATEGORY (nnUNet|auglab)}"
FOLD_ARG="${3:-all}"

ITEMS=(flair t1w t2w)
LABEL_MAP='{"lesion": [1, 255]}'
PRED_BASE="${PREDICTIONS_ROOT}/${OPENMS_MODEL_TYPE}/${OPENMS_TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"
OUT_BASE="${METRICS_ROOT}/${OPENMS_MODEL_TYPE}/${OPENMS_TRAINING_CONTRAST}/${CATEGORY}_${RUN_ID}"

if [ "${FOLD_ARG}" = "all" ]; then FOLDS="0 1 2"; else FOLDS="${FOLD_ARG}"; fi

[ -d "$PRED_BASE" ] || { echo "ERROR: no predictions at $PRED_BASE" >&2; exit 1; }

mkdir -p "${OUT_BASE}/_logs"
run_job --name "ms3seg_eval_${CATEGORY}_${RUN_ID}" --gpus 0 --cpus 8 --mem 16G --time "${EVAL_TIME:-1:00:00}" \
    --log "${OUT_BASE}/_logs/eval_${RUN_ID}.log" --wait -- bash -c "
set -euo pipefail
cd '${PROJECT_ROOT}'
for F in ${FOLDS}; do
    for item in ${ITEMS[*]}; do
        PRED_DIR='${PRED_BASE}'/fold\${F}/\${item}
        GT_DIR='${nnUNet_raw}'/labelsTs_\${item}
        [ -d \"\${PRED_DIR}\" ] || { echo \"  skip fold\${F}/\${item}: no preds (\${PRED_DIR})\"; continue; }
        OUT_CSV='${OUT_BASE}'/fold\${F}/\${item}_metrics.csv
        echo \"[\$(date '+%H:%M:%S')] eval ${RUN_ID} fold\${F} \${item}\"
        .venv/bin/python datasets/ms3seg/5_scripts_ms3seg/06_evaluate/06_00_evaluate_ms3seg.py \
            --pred_dir \"\${PRED_DIR}\" --gt_dir \"\${GT_DIR}\" --label_map '${LABEL_MAP}' \
            --name \"\${item}\" --out_csv \"\${OUT_CSV}\" --workers 8
    done
    OUT_FOLD='${OUT_BASE}'/fold\${F}
    if ls \"\${OUT_FOLD}\"/*_metrics.csv >/dev/null 2>&1; then
        .venv/bin/python datasets/00_commun_scripts/00_03_evaluate/summarize_fold.py \
            \"\${OUT_FOLD}\" '${RUN_ID}' \"\${F}\" \
            --groups ${ITEMS[*]} --group-col contrast --groups-word Contrasts
    fi
done
echo '-> ${OUT_BASE}'
"
