#!/usr/bin/env bash
# Shared per-run evaluate driver for EVAL-ONLY COMPANIONS (source task's model, companion's test items). Companion layout = per-ITEM run dirs:
#   <METRICS_ROOT>/<src model type>/<src training contrast>/[ablations/]<item>/<CATEGORY>_<RUN_ID>/fold{F}/eval_all.csv
# (the own-model driver evaluate_run_common.sh puts all items in ONE run dir; do not mix the two layouts). NOT invoked directly: the companion's
# 06_01_evaluate_run.sh sources env.sh, sets the vars below, then sources this with its own "$@".
#   bash 06_01_evaluate_run.sh <RUN_ID> <CATEGORY:nnUNet|auglab> <SOURCE_TRAINING_CONTRAST> [FOLD=all]       (LADDER=1 -> .../ablations/<item>/...)
# Shim must set: SOURCE_PREFIX (e.g. ISPY2: needs ${SOURCE_PREFIX}_MODEL_TYPE and _DATASET_JSON), EVAL_ITEMS, EVAL_LABELS, EVAL_JOB_PREFIX.
# Optional env: CKPT_TAG (best|final), EVAL_INLINE=1, EVAL_TIME, EVAL_MEM, EVAL_FOLDS (default "0 1 2").
# Same safety nets as evaluate_run_common.sh: missing prediction dir = ERROR, #preds != #GT = ERROR, stale item CSV deleted first, eval_all.csv rows asserted.
set -euo pipefail
cd "${PROJECT_ROOT:?source env.sh first}"
RUN_ID="${1:?need RUN_ID}"; CATEGORY="${2:?need CATEGORY (nnUNet|auglab)}"; TC="${3:?need SOURCE_TRAINING_CONTRAST}"; FOLD_ARG="${4:-all}"
: "${SOURCE_PREFIX:?}" "${EVAL_ITEMS:?}" "${EVAL_LABELS:?}" "${EVAL_JOB_PREFIX:?}"
case "${CATEGORY}" in nnUNet|auglab) ;; *) echo "ERROR: CATEGORY must be nnUNet|auglab, got '${CATEGORY}'" >&2; exit 1;; esac
_src() { local n="${SOURCE_PREFIX}_$1"; echo "${!n}"; }
SRC_MT="$(_src MODEL_TYPE)"; DJ="$(_src DATASET_JSON)"; [ -f "${DJ}" ] || { echo "ERROR: source dataset.json not found: ${DJ}" >&2; exit 1; }
CKPT_TAG="${CKPT_TAG:-best}"; _PRED_SUBDIR=""; _OUT_SUFFIX=""; [ "${CKPT_TAG}" != "best" ] && { _PRED_SUBDIR="${CKPT_TAG}/"; _OUT_SUFFIX="_${CKPT_TAG}"; }
_ABL=""; [ "${LADDER:-0}" = "1" ] && _ABL="ablations/"
PRED_BASE="${PREDICTIONS_ROOT}/${SRC_MT}/${TC}/${CATEGORY}/${RUN_ID}"
METRICS_BASE="${METRICS_ROOT}/${SRC_MT}/${TC}/${_ABL}"
[ -d "${PRED_BASE}" ] || { echo "ERROR: no predictions at ${PRED_BASE} (wrong CATEGORY or contrast?)" >&2; exit 1; }
if [ "${FOLD_ARG}" = "all" ]; then FOLDS="${EVAL_FOLDS:-0 1 2}"; else FOLDS="${FOLD_ARG}"; fi
FIRST_ITEM="${EVAL_ITEMS%% *}"; LOG_DIR="${METRICS_BASE}${FIRST_ITEM}/${CATEGORY}_${RUN_ID}${_OUT_SUFFIX}/_logs"; mkdir -p "${LOG_DIR}"
_EVAL_BODY="
set -euo pipefail
cd '${PROJECT_ROOT}'
for item in ${EVAL_ITEMS}; do
    OUT_BASE='${METRICS_BASE}'\${item}/'${CATEGORY}_${RUN_ID}${_OUT_SUFFIX}'
    for F in ${FOLDS}; do
        PRED_DIR='${PRED_BASE}'/fold\${F}/${_PRED_SUBDIR}\${item}
        GT_DIR='${nnUNet_raw}'/labelsTs_\${item}
        [ -d \"\${PRED_DIR}\" ] || { echo \"ERROR: no preds fold\${F}/\${item} (\${PRED_DIR})\" >&2; exit 1; }
        NP=\$(ls \"\${PRED_DIR}\"/*.nii.gz 2>/dev/null | wc -l); NG=\$(ls \"\${GT_DIR}\"/*.nii.gz | wc -l)
        [ \"\${NP}\" = \"\${NG}\" ] || { echo \"ERROR: fold\${F}/\${item}: \${NP} predictions vs \${NG} GT cases\" >&2; exit 1; }
        OUT_CSV=\"\${OUT_BASE}/fold\${F}/\${item}_metrics.csv\"
        mkdir -p \"\$(dirname \"\${OUT_CSV}\")\"; rm -f \"\${OUT_CSV}\"
        echo \"[\$(date '+%H:%M:%S')] eval ${RUN_ID} fold\${F} \${item} (\${NG} cases)\"
        .venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/evaluate.py --pred_dir \"\${PRED_DIR}\" --gt_dir \"\${GT_DIR}\" \
            --dataset_json '${DJ}' --labels ${EVAL_LABELS} --name \"\${item}\" --out_csv \"\${OUT_CSV}\" --workers 8
        .venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/summarize_fold.py \"\${OUT_BASE}/fold\${F}\" '${RUN_ID}' \"\${F}\" \
            --groups \${item} --group-col contrast --groups-word Contrasts
        NR=\$(( \$(wc -l < \"\${OUT_BASE}/fold\${F}/eval_all.csv\") - 1 )); EXP=\$(( NG * \$(echo ${EVAL_LABELS} | wc -w) ))
        [ \"\${NR}\" = \"\${EXP}\" ] || { echo \"ERROR: \${OUT_BASE}/fold\${F}/eval_all.csv has \${NR} rows, expected \${EXP}\" >&2; exit 1; }
    done
    echo \"→ \${OUT_BASE}\"
done
"
if [ "${EVAL_INLINE:-0}" = "1" ]; then
    bash -c "${_EVAL_BODY}" 2>&1 | tee "${LOG_DIR}/eval_${RUN_ID}${_OUT_SUFFIX}.log"; exit "${PIPESTATUS[0]}"
fi
run_job --name "${EVAL_JOB_PREFIX}_${CATEGORY}_${RUN_ID}${_OUT_SUFFIX}" --gpus 0 --cpus 8 --mem "${EVAL_MEM:-16G}" --time "${EVAL_TIME:-1:00:00}" \
    --log "${LOG_DIR}/eval_${RUN_ID}${_OUT_SUFFIX}.log" --wait -- bash -c "${_EVAL_BODY}"
