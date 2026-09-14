#!/usr/bin/env bash
# Evaluate one ISPY2-TRAINED model's cross-dataset prediction run on
# duke-breast-mri's 291-case MAMA-MIA-expert-masked test set: every fold x the
# SINGLE t1wce test item (Duke has no t2w acquisition -- see 00_utils/env.sh),
# scoring the `tumour` label against duke's own held-out GT
# (2_nnUNet_duke-breast-mri/raw/labelsTs_t1wce -- flat layout, see
# 02_nnunet/02_01_convert_test_t1wce.py). Modeled on
# ambl/5_scripts_ambl/06_evaluate/06_20_evaluate_ispy2_run.sh, collapsed to one
# item since Duke only has t1wce-family imaging.
#
# Both the ispy2 t1wce-trained model (same-modality-family) AND the t2w-trained
# model (cross-contrast) are evaluated this way, per the project's "train on
# one modality -> predict/eval on every available modality of the target
# dataset" rule -- TRAINING_CONTRAST selects which ispy2 model this run came
# from, not which Duke item is scored (there is only one).
# Writes <item>_metrics.csv + eval_all.csv + eval_summary.md per fold under
#   8_results_duke-breast-mri/02_metrics/ispy2_model/<TRAINING_CONTRAST>/<CATEGORY>_<RUN_ID>/fold{F}/
#
# Usage: bash 06_20_evaluate_ispy2_run.sh <RUN_ID> <CATEGORY:nnUNet|auglab> <TRAINING_CONTRAST:t1wce|t2w> [FOLD(default all)]
# Optional env: CKPT_TAG (default "best").
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

RUN_ID="${1:?need RUN_ID}"
CATEGORY="${2:?need CATEGORY (nnUNet|auglab)}"
TRAINING_CONTRAST="${3:?need TRAINING_CONTRAST (t1wce|t2w, i.e. which ispy2 model trained this run)}"
FOLD_ARG="${4:-all}"
CKPT_TAG="${CKPT_TAG:-best}"

ITEMS=("${DUKE_ITEM:-t1wce}")   # override via DUKE_ITEM env var (e.g. "precontrast")
DJ="${ISPY2_DATASET_JSON}"   # background=0, tumour=1 -- identical numbering to duke's own manifest
PRED_BASE="${PREDICTIONS_ROOT}/${ISPY2_MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"
_PRED_SUBDIR=""; [ "${CKPT_TAG}" != "best" ] && _PRED_SUBDIR="${CKPT_TAG}/"
_OUT_SUFFIX=""; [ "${CKPT_TAG}" != "best" ] && _OUT_SUFFIX="_${CKPT_TAG}"
# non-default items get their own dedicated subdir so they never collide with
# the real headline t1wce output (same convention as METRICS_SUBDIR=ablations)
if [ "${DUKE_ITEM:-t1wce}" != "t1wce" ]; then
    METRICS_SUBDIR="${METRICS_SUBDIR:-${DUKE_ITEM}}"
else
    METRICS_SUBDIR="${METRICS_SUBDIR:-}"
fi
OUT_BASE="${METRICS_ROOT}/${ISPY2_MODEL_TYPE}/${TRAINING_CONTRAST}${METRICS_SUBDIR:+/${METRICS_SUBDIR}}/${CATEGORY}_${RUN_ID}${_OUT_SUFFIX}"

if [ "${FOLD_ARG}" = "all" ]; then FOLDS="0 1 2"; else FOLDS="${FOLD_ARG}"; fi

[ -d "$PRED_BASE" ] || { echo "ERROR: no predictions at $PRED_BASE" >&2; exit 1; }

mkdir -p "${OUT_BASE}/_logs"
run_job --name "duke_ispy2_eval_${CATEGORY}_${RUN_ID}${_OUT_SUFFIX}" --gpus 0 --cpus 8 --mem "${EVAL_MEM:-32G}" --time "${EVAL_TIME:-1:00:00}" \
    --log "${OUT_BASE}/_logs/eval_${RUN_ID}${_OUT_SUFFIX}.log" --wait -- bash -c "
set -euo pipefail
cd '${PROJECT_ROOT}'
for F in ${FOLDS}; do
    for item in ${ITEMS[*]}; do
        PRED_DIR='${PRED_BASE}'/fold\${F}/${_PRED_SUBDIR}\${item}
        GT_DIR='${nnUNet_raw}'/labelsTs_\${item}
        [ -d \"\${PRED_DIR}\" ] || { echo \"  skip fold\${F}/\${item}: no preds (\${PRED_DIR})\"; continue; }
        OUT_CSV='${OUT_BASE}'/fold\${F}/\${item}_metrics.csv
        mkdir -p \"\$(dirname \"\${OUT_CSV}\")\"
        echo \"[\$(date '+%H:%M:%S')] eval ${RUN_ID} fold\${F} \${item}\"
        .venv/bin/python datasets/duke-breast-mri/5_scripts_duke-breast-mri/06_evaluate/06_00_evaluate.py \
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
