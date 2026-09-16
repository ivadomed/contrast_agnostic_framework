#!/usr/bin/env bash
# Evaluate one totalseg-pelvic OWN-model prediction run: every fold x TWO items (ct, mri
# — own-contrast and cross-contrast, both in one pass since every item's ground truth is
# consolidated under Dataset130's raw tree, see 05_predict's build_test_inputs script),
# scoring the 10 pelvic/hip labels against nnUNet_raw/Dataset130_.../labelsTs_<item>.
# Writes <item>_metrics.csv + eval_all.csv + eval_summary.md per fold under
#   8_results_totalseg-pelvic/02_metrics/totalseg_pelvic_model/<contrast>/<CATEGORY>_<RUN_ID>/fold{F}/
#
# Usage: bash 06_01_evaluate_run.sh <RUN_ID> <CATEGORY:nnUNet|auglab> [FOLD(default all)]
#
# ⚠️ CATEGORY must be passed explicitly and correctly — a wrong CATEGORY does NOT error,
# it silently writes an empty, _logs-only metrics dir (the open-ms gotcha in CLAUDE.md).
# Check eval_all.csv actually exists before trusting a run finished.
#
# ⚠️ TRAINING_CONTRAST must match the model being evaluated (source env.sh for a
# CT-trained RUN_ID, env_mri.sh for a MRI-trained one) — this determines where
# 8_results_totalseg-pelvic/02_metrics/totalseg_pelvic_model/<contrast>/... is written,
# NOT which items get scored (both items are always scored).
#
# Optional env:
#   CKPT_TAG        default "best"; "final" reads fold{F}/<tag>/<item> predictions
#   METRICS_SUBDIR  route output into a subdir (e.g. METRICS_SUBDIR=ablations)
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

RUN_ID="${1:?need RUN_ID}"
CATEGORY="${2:?need CATEGORY (nnUNet|auglab)}"
FOLD_ARG="${3:-all}"
CKPT_TAG="${CKPT_TAG:-best}"
DATASET_ID="130"   # fixed — imagesTs_*/labelsTs_* consolidated under Dataset130

ITEMS=(ct mri)
LABELS="hip_left hip_right sacrum gluteus_maximus_left gluteus_maximus_right gluteus_medius_left gluteus_medius_right gluteus_minimus_left gluteus_minimus_right iliopsoas_left iliopsoas_right"
_DS_NAME="$(ls "${nnUNet_raw}" | grep "^Dataset0*${DATASET_ID}_" | head -1)"
DJ="${nnUNet_raw}/${_DS_NAME}/dataset.json"
PRED_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"
_PRED_SUBDIR=""; [ "${CKPT_TAG}" != "best" ] && _PRED_SUBDIR="${CKPT_TAG}/"
_OUT_SUFFIX=""; [ "${CKPT_TAG}" != "best" ] && _OUT_SUFFIX="_${CKPT_TAG}"
METRICS_SUBDIR="${METRICS_SUBDIR:-}"
OUT_BASE="${METRICS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}${METRICS_SUBDIR:+/${METRICS_SUBDIR}}/${CATEGORY}_${RUN_ID}${_OUT_SUFFIX}"

if [ "${FOLD_ARG}" = "all" ]; then FOLDS="0 1 2"; else FOLDS="${FOLD_ARG}"; fi
[ -d "$PRED_BASE" ] || { echo "ERROR: no predictions at $PRED_BASE" >&2; exit 1; }

mkdir -p "${OUT_BASE}/_logs"
run_job --name "totalseg_pelvic_eval_${CATEGORY}_${RUN_ID}${_OUT_SUFFIX}" --gpus 0 --cpus 8 \
    --mem "${EVAL_MEM:-32G}" --time "${EVAL_TIME:-2:00:00}" \
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
        .venv/bin/python datasets/totalseg-pelvic/5_scripts_totalseg-pelvic/06_evaluate/06_00_evaluate.py \
            --pred_dir \"\${PRED_DIR}\" --gt_dir \"\${GT_DIR}\" --dataset_json '${DJ}' \
            --labels ${LABELS} --name \"\${item}\" --out_csv \"\${OUT_CSV}\" --workers 8
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
