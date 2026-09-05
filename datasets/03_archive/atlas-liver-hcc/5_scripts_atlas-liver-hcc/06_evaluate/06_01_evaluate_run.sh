#!/usr/bin/env bash
# Evaluate one atlas-liver-hcc prediction run: every fold x the single T1w test item,
# scoring `liver` + `tumour` against the held-out GT (labelsTs_t1w). Writes
# t1w_metrics.csv + eval_all.csv + eval_summary.md per fold under
#   8_results_atlas-liver-hcc/02_metrics/<MODEL_TYPE>/t1w/<CATEGORY>_<RUN_ID>/fold{F}/
# via the shared per-fold summariser (00_commun_scripts/00_03_evaluate/summarize_fold.py
# — same one chaos/brats/open-ms use), not a bespoke concatenation.
#
# SINGLE-MODALITY DATASET: there is only ONE test item ("t1w") — no cross-contrast axis
# within this dataset (unlike chaos/brats/open-ms/on-harmony, which each score 2+ test
# contrasts). The held-out 12-patient split (stratified by tumour burden, see
# 01_create_splits/01_01_create_splits.py) IS the generalization test here — do not
# read a missing cross-contrast breakdown as a bug.
#
# Usage: bash 06_01_evaluate_run.sh <RUN_ID> <CATEGORY:nnUNet|auglab> [FOLD(default all)]
# Optional env: CKPT_TAG (default "best") — set to "final" to evaluate a
#   checkpoint_final prediction run (predict with CHECKPOINT=checkpoint_final.pth
#   first; see 05_predict/05_01_predict_common.sh). "best" reads/writes the
#   original flat paths unchanged; any other tag reads predictions from
#   fold{F}/<tag>/<item> (see predict_common.sh's CKPT_SUBDIR) and writes metrics
#   to a sibling <CATEGORY>_<RUN_ID>_<tag> dir so it never collides with, or
#   overwrites, the checkpoint_best metrics for the same RUN_ID.
# Optional env: EVAL_TIME (default 1:00:00) — run_job --time override.
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

RUN_ID="${1:?need RUN_ID}"
CATEGORY="${2:?need CATEGORY (nnUNet|auglab)}"
FOLD_ARG="${3:-all}"
CKPT_TAG="${CKPT_TAG:-best}"

ITEMS=(t1w)
DATASET_ID="${DATASET_ID:-80}"
_DS_NAME="$(ls "${nnUNet_raw}" | grep "^Dataset0*${DATASET_ID}_" | head -1)"
DJ="${nnUNet_raw}/${_DS_NAME}/dataset.json"
PRED_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"
_PRED_SUBDIR=""; [ "${CKPT_TAG}" != "best" ] && _PRED_SUBDIR="${CKPT_TAG}/"
_OUT_SUFFIX=""; [ "${CKPT_TAG}" != "best" ] && _OUT_SUFFIX="_${CKPT_TAG}"
OUT_BASE="${METRICS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}_${RUN_ID}${_OUT_SUFFIX}"

# "0 1 2": matches EVAL_FOLDS in datasets/00_commun_scripts/00_00_utils/eval_folds.py
# (the single source of truth the aggregators cap to) — kept in sync by hand since bash
# and python don't share one file.
if [ "${FOLD_ARG}" = "all" ]; then FOLDS="0 1 2"; else FOLDS="${FOLD_ARG}"; fi

# Metric computation (MONAI Dice+HD95) is real CPU work — too slow for the login node.
# Route through run_job as one CPU-only compute-node job per invocation, matching the
# "never run substantial work on a login node" rule (see CLAUDE.md).
mkdir -p "${OUT_BASE}/_logs"
run_job --name "atlashcc_eval_${CATEGORY}_${RUN_ID}${_OUT_SUFFIX}" --gpus 0 --cpus 8 --mem 16G --time "${EVAL_TIME:-1:00:00}" \
    --log "${OUT_BASE}/_logs/eval_${RUN_ID}${_OUT_SUFFIX}.log" --wait -- bash -c "
set -euo pipefail
cd '${PROJECT_ROOT}'
for F in ${FOLDS}; do
    for item in ${ITEMS[*]}; do
        PRED_DIR='${PRED_BASE}'/fold\${F}/${_PRED_SUBDIR}\${item}
        GT_DIR='${nnUNet_raw}/${_DS_NAME}'/labelsTs_\${item}
        [ -d \"\${PRED_DIR}\" ] || { echo \"  skip fold\${F}/\${item}: no preds (\${PRED_DIR})\"; continue; }
        OUT_CSV='${OUT_BASE}'/fold\${F}/\${item}_metrics.csv
        echo \"[\$(date '+%H:%M:%S')] eval ${RUN_ID} fold\${F} \${item}\"
        .venv/bin/python datasets/atlas-liver-hcc/5_scripts_atlas-liver-hcc/06_evaluate/06_00_evaluate.py \
            --pred_dir \"\${PRED_DIR}\" --gt_dir \"\${GT_DIR}\" --dataset_json '${DJ}' \
            --labels liver tumour --name \"\${item}\" --out_csv \"\${OUT_CSV}\" --workers 8
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
