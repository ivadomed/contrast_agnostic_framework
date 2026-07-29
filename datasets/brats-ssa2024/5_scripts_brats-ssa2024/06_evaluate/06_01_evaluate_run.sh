#!/usr/bin/env bash
# Evaluate one brats2024-glioma-trained model's predictions on BraTS-SSA 2024: every
# fold x every contrast (t1n/t1c/t2w/t2f), scoring the 3 labels BraTS-SSA actually has
# (NCR/SNFH/ET -- RC excluded, pre-treatment cohort has none) against BraTS-SSA's own GT
# (labelsTs_<contrast> in 2_nnUNet_brats-ssa2024/raw -- built by 05_00_build_test_inputs.py).
#
# Writes <contrast>_metrics.csv + eval_all.csv + eval_summary.md per fold under
#   8_results_brats-ssa2024/02_metrics/<BRATS_MODEL_TYPE>/<BRATS_TRAINING_CONTRAST>/<CATEGORY>_<RUN_ID>/fold{F}/
#
# Usage: bash 06_01_evaluate_run.sh <RUN_ID> <CATEGORY:nnUNet|auglab> [FOLD(default all)]
# For T2w-trained runs, pre-export BRATS_TRAINING_CONTRAST=t2w/BRATS_DATASET_ID=052/
# BRATS_DS_NAME=Dataset052_BraTS2024GliomaT2w (or source env_t2w.sh) before calling --
# see 06_03_evaluate_t2w_all.sh.
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

RUN_ID="${1:?need RUN_ID}"
CATEGORY="${2:?need CATEGORY (nnUNet|auglab)}"
FOLD_ARG="${3:-all}"

ITEMS=(t1n t1c t2w t2f)
# Derive DJ fresh from BRATS_NNUNET_RAW + BRATS_DS_NAME (NOT the pre-exported
# BRATS_DATASET_JSON) -- a cluster override computes BRATS_DATASET_JSON eagerly at
# source time from whatever BRATS_DS_NAME was current then; if a later per-contrast
# wrapper (env_t2w.sh / inline exports) changes BRATS_DS_NAME afterwards, that cached
# value goes stale. Mirrors mslesseg's 06_01_evaluate_run.sh fix (2026-07-29).
DJ="${BRATS_NNUNET_RAW}/${BRATS_DS_NAME}/dataset.json"
PRED_BASE="${PREDICTIONS_ROOT}/${BRATS_MODEL_TYPE}/${BRATS_TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"
OUT_BASE="${METRICS_ROOT}/${BRATS_MODEL_TYPE}/${BRATS_TRAINING_CONTRAST}/${CATEGORY}_${RUN_ID}"

# "0 1 2" -- 3-fold policy (see CLAUDE.md FOLD POLICY).
if [ "${FOLD_ARG}" = "all" ]; then FOLDS="0 1 2"; else FOLDS="${FOLD_ARG}"; fi

[ -d "$PRED_BASE" ] || { echo "ERROR: no predictions at $PRED_BASE" >&2; exit 1; }

mkdir -p "${OUT_BASE}/_logs"
# --mem 48G (not the 16G default): BraTS's 240x240x155 volumes with 8 parallel HD95
# surface-distance workers OOM-killed at 16G (confirmed 2026-07-29 on tamia). 48G is
# ample headroom for 8.9M-voxel volumes -- nowhere near TRUSTED's 96G US case.
run_job --name "brats_ssa_eval_${CATEGORY}_${RUN_ID}" --gpus 0 --cpus 8 --mem 48G --time "${EVAL_TIME:-1:00:00}" \
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
        .venv/bin/python datasets/brats-ssa2024/5_scripts_brats-ssa2024/06_evaluate/06_00_evaluate_brats_ssa.py \
            --pred_dir \"\${PRED_DIR}\" --gt_dir \"\${GT_DIR}\" --dataset_json '${DJ}' \
            --labels NCR SNFH ET --name \"\${item}\" --out_csv \"\${OUT_CSV}\" --workers 8
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
