#!/usr/bin/env bash
# Evaluate one open-ms-trained model's predictions on MSLesSeg: every fold x every
# contrast (flair/t1w/t2w), scoring the `lesion` label against MSLesSeg's own GT
# (labelsTs_<contrast> in 2_nnUNet_mslesseg/raw — built by 05_00_build_test_inputs.py).
# Same label space as open-ms (lesion=1 both sides) — no cross-label-space merge needed,
# unlike trusted's kidney case.
#
# Writes <contrast>_metrics.csv + eval_all.csv + eval_summary.md per fold under
#   8_results_mslesseg/02_metrics/<OPENMS_MODEL_TYPE>/<OPENMS_TRAINING_CONTRAST>/<CATEGORY>_<RUN_ID>/fold{F}/
#
# Usage: bash 06_01_evaluate_run.sh <RUN_ID> <CATEGORY:nnUNet|auglab> [FOLD(default all)]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

RUN_ID="${1:?need RUN_ID}"
CATEGORY="${2:?need CATEGORY (nnUNet|auglab)}"
FOLD_ARG="${3:-all}"

ITEMS=(flair t1w t2w)
DJ="${OPENMS_DATASET_JSON}"
PRED_BASE="${PREDICTIONS_ROOT}/${OPENMS_MODEL_TYPE}/${OPENMS_TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"
OUT_BASE="${METRICS_ROOT}/${OPENMS_MODEL_TYPE}/${OPENMS_TRAINING_CONTRAST}/${CATEGORY}_${RUN_ID}"

# "0 1 2" — 3-fold policy (see CLAUDE.md FOLD POLICY).
if [ "${FOLD_ARG}" = "all" ]; then FOLDS="0 1 2"; else FOLDS="${FOLD_ARG}"; fi

[ -d "$PRED_BASE" ] || { echo "ERROR: no predictions at $PRED_BASE" >&2; exit 1; }

mkdir -p "${OUT_BASE}/_logs"
run_job --name "mslesseg_eval_${CATEGORY}_${RUN_ID}" --gpus 0 --cpus 8 --mem 16G --time "${EVAL_TIME:-1:00:00}" \
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
        .venv/bin/python datasets/mslesseg/5_scripts_mslesseg/06_evaluate/06_00_evaluate_mslesseg.py \
            --pred_dir \"\${PRED_DIR}\" --gt_dir \"\${GT_DIR}\" --dataset_json '${DJ}' \
            --labels lesion --name \"\${item}\" --out_csv \"\${OUT_CSV}\" --workers 8
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
