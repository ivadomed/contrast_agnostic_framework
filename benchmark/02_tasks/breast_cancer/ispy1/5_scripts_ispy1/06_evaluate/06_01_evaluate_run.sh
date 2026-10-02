#!/usr/bin/env bash
# Evaluate one ISPY2-TRAINED model's cross-dataset prediction run on ispy1's
# 167-case MAMA-MIA-expert-masked I-SPY1 test set: every fold x BOTH test items
# (t1wce, precontrast), `tumour` label, against ispy1's own GT
# (2_nnUNet_ispy1/raw/labelsTs_<item>). Same evaluator/summariser as
# duke-breast-mri's 06_01_evaluate_ispy2_run.sh; each item gets its own metrics
# subdir (the layout duke uses for its non-default items), so headline and
# ablation-ladder runs read uniformly as:
#   8_results_ispy1/02_metrics/ispy2_model/<TRAINING_CONTRAST>/<item>/<CATEGORY>_<RUN_ID>/fold{F}/
#   8_results_ispy1/02_metrics/ispy2_model/<TRAINING_CONTRAST>/ablations/<item>/<CATEGORY>_<RUN_ID>/fold{F}/  (LADDER=1)
#
# Usage: bash 06_01_evaluate_run.sh <RUN_ID> <CATEGORY:nnUNet|auglab> <TRAINING_CONTRAST:t1wce|t2w> [FOLD(default all)]
# Optional env: LADDER=1 (ablation rung -> ablations/<item>), ITEMS_OVERRIDE="t1wce", CKPT_TAG (default best).
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

RUN_ID="${1:?need RUN_ID}"
CATEGORY="${2:?need CATEGORY (nnUNet|auglab)}"
TRAINING_CONTRAST="${3:?need TRAINING_CONTRAST (t1wce|t2w)}"
FOLD_ARG="${4:-all}"
CKPT_TAG="${CKPT_TAG:-best}"
read -ra ITEMS <<< "${ITEMS_OVERRIDE:-t1wce precontrast}"
DJ="${ISPY2_DATASET_JSON}"
PRED_BASE="${PREDICTIONS_ROOT}/${ISPY2_MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"
_PRED_SUBDIR=""; [ "${CKPT_TAG}" != "best" ] && _PRED_SUBDIR="${CKPT_TAG}/"
_OUT_SUFFIX=""; [ "${CKPT_TAG}" != "best" ] && _OUT_SUFFIX="_${CKPT_TAG}"
_ABL=""; [ "${LADDER:-0}" = "1" ] && _ABL="ablations/"
METRICS_BASE="${METRICS_ROOT}/${ISPY2_MODEL_TYPE}/${TRAINING_CONTRAST}/${_ABL}"
if [ "${FOLD_ARG}" = "all" ]; then FOLDS="0 1 2"; else FOLDS="${FOLD_ARG}"; fi
[ -d "$PRED_BASE" ] || { echo "ERROR: no predictions at $PRED_BASE" >&2; exit 1; }

LOG_DIR="${METRICS_BASE}${ITEMS[0]}/${CATEGORY}_${RUN_ID}${_OUT_SUFFIX}/_logs"
mkdir -p "${LOG_DIR}"
run_job --name "ispy1_eval_${CATEGORY}_${RUN_ID}${_OUT_SUFFIX}" --gpus 0 --cpus 8 --mem "${EVAL_MEM:-16G}" --time "${EVAL_TIME:-1:00:00}" \
    --log "${LOG_DIR}/eval_${RUN_ID}${_OUT_SUFFIX}.log" --wait -- bash -c "
set -euo pipefail
cd '${PROJECT_ROOT}'
for item in ${ITEMS[*]}; do
    OUT_BASE='${METRICS_BASE}'\${item}/'${CATEGORY}_${RUN_ID}${_OUT_SUFFIX}'
    for F in ${FOLDS}; do
        PRED_DIR='${PRED_BASE}'/fold\${F}/${_PRED_SUBDIR}\${item}
        GT_DIR='${nnUNet_raw}'/labelsTs_\${item}
        [ -d \"\${PRED_DIR}\" ] || { echo \"  skip fold\${F}/\${item}: no preds (\${PRED_DIR})\"; continue; }
        n_pred=\$(ls \"\${PRED_DIR}\"/*.nii.gz 2>/dev/null | wc -l); n_gt=\$(ls \"\${GT_DIR}\"/*.nii.gz | wc -l)
        [ \"\${n_pred}\" = \"\${n_gt}\" ] || { echo \"ERROR fold\${F}/\${item}: \${n_pred} preds vs \${n_gt} GT\" >&2; exit 1; }
        OUT_CSV=\"\${OUT_BASE}/fold\${F}/\${item}_metrics.csv\"
        mkdir -p \"\$(dirname \"\${OUT_CSV}\")\"
        echo \"[\$(date '+%H:%M:%S')] eval ${RUN_ID} fold\${F} \${item}\"
        .venv/bin/python benchmark/02_tasks/breast_cancer/ispy1/5_scripts_ispy1/06_evaluate/06_00_evaluate_ispy1.py \
            --pred_dir \"\${PRED_DIR}\" --gt_dir \"\${GT_DIR}\" --dataset_json '${DJ}' \
            --labels tumour --name \"\${item}\" --out_csv \"\${OUT_CSV}\" --workers 8
        .venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/summarize_fold.py \
            \"\${OUT_BASE}/fold\${F}\" '${RUN_ID}' \"\${F}\" \
            --groups \${item} --group-col contrast --groups-word Contrasts
    done
    echo \"→ \${OUT_BASE}\"
done
"
