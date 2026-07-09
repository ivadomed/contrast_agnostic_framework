#!/usr/bin/env bash
# Evaluate one open-ms prediction run: every fold × every contrast (flair/t2w/t1w),
# scoring the `lesion` label against the per-contrast held-out GT (labelsTs_<contrast>).
# Writes <contrast>_metrics.csv + eval_all.csv + eval_summary.md per fold under
#   8_results_open-ms/02_metrics/<MODEL_TYPE>/<TRAINING_CONTRAST>/<CATEGORY>_<RUN_ID>/fold{F}/
# via the shared per-fold summariser (00_commun_scripts/00_03_evaluate/summarize_fold.py —
# same one chaos/amos use), not a bespoke concatenation.
#
# Usage: bash 06_01_evaluate_run.sh <RUN_ID> <CATEGORY:nnUNet|auglab> [FOLD(default all)]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

RUN_ID="${1:?need RUN_ID}"
CATEGORY="${2:?need CATEGORY (nnUNet|auglab)}"
FOLD_ARG="${3:-all}"

ITEMS=(flair t2w t1w)
# DATASET_ID selects the nnUNet dataset providing the per-contrast GT (labelsTs_<contrast>)
# + dataset.json. Defaults to 70 (FLAIR-trained); the T1w wrapper (06_13) pre-exports 71.
# GT masks are byte-identical across the FLAIR/T1w datasets (same co-registered consensus
# masks), so this only affects provenance/paths, not the scores.
DATASET_ID="${DATASET_ID:-70}"
_DS_NAME="$(ls "${nnUNet_raw}" | grep "^Dataset0*${DATASET_ID}_" | head -1)"
DJ="${nnUNet_raw}/${_DS_NAME}/dataset.json"
PRED_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"
OUT_BASE="${METRICS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}_${RUN_ID}"

# "0 1 2": matches EVAL_FOLDS in datasets/00_commun_scripts/00_00_utils/eval_folds.py
# (the single source of truth the aggregators cap to) — kept in sync by hand since bash
# and python don't share one file. Older runs may have fold3 predictions; they're simply
# never evaluated here, consistent with every run being compared on the same 3 folds.
if [ "${FOLD_ARG}" = "all" ]; then FOLDS="0 1 2"; else FOLDS="${FOLD_ARG}"; fi

# Metric computation (MONAI Dice+HD95, ~60s per fold×contrast) is real CPU work — too
# slow for the login node. Route through run_job as one CPU-only compute-node job per
# invocation, matching the "never run substantial work on a login node" rule (see
# CLAUDE.md). --log goes under OUT_BASE (shared storage), not /tmp, for the same reason
# the training/predict logs were moved there.
mkdir -p "${OUT_BASE}/_logs"
run_job --name "openms_eval_${CATEGORY}_${RUN_ID}" --gpus 0 --cpus 8 --mem 16G --time 1:00:00 \
    --log "${OUT_BASE}/_logs/eval_${RUN_ID}.log" --wait -- bash -c "
set -euo pipefail
cd '${PROJECT_ROOT}'
for F in ${FOLDS}; do
    for item in ${ITEMS[*]}; do
        PRED_DIR='${PRED_BASE}'/fold\${F}/\${item}
        GT_DIR='${nnUNet_raw}/${_DS_NAME}'/labelsTs_\${item}
        [ -d \"\${PRED_DIR}\" ] || { echo \"  skip fold\${F}/\${item}: no preds (\${PRED_DIR})\"; continue; }
        OUT_CSV='${OUT_BASE}'/fold\${F}/\${item}_metrics.csv
        echo \"[\$(date '+%H:%M:%S')] eval ${RUN_ID} fold\${F} \${item}\"
        .venv/bin/python datasets/open-ms/5_scripts_open-ms/06_evaluate/06_00_evaluate.py \
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
