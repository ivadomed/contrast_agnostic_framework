#!/usr/bin/env bash
# Shared per-run evaluate driver (replaces the copy-pasted 06_01_evaluate_run.sh bodies). NOT invoked directly: a dataset's
# 06_01 shim sources env.sh, sets the vars below, then sources this with its own "$@".
#
# Usage of the shim:  bash 06_01_evaluate_run.sh <RUN_ID> <CATEGORY:nnUNet|auglab> [FOLD=all]
#
# Shim must set:
#   EVAL_ITEMS            space list of test items, one imagesTs_<item>/labelsTs_<item> pair each (e.g. "dwi adc flair")
#   EVAL_LABELS           label name(s) from dataset.json to score (e.g. "lesion")
#   EVAL_JOB_PREFIX       slurm job-name prefix
#   EVAL_DATASET_ID       nnU-Net Dataset id that provides labelsTs_* + dataset.json (default: $DATASET_ID)
# Optional env: EVAL_INLINE=1 (run in the current shell instead of a run_job CPU job), CKPT_TAG (best|final; non-best reads fold{F}/<tag>/<item> and writes a sibling <CATEGORY>_<RUN_ID>_<tag> dir),
#   METRICS_SUBDIR (e.g. ablations -> .../<contrast>/ablations/...), EVAL_TIME, EVAL_MEM, EVAL_FOLDS (default "0 1 2").
#
# Safety nets this adds over the old per-dataset copies (each one bit this project before):
#   - a missing/empty prediction dir is an ERROR (silent skips produced short tables), not a "skip";
#   - #predictions must equal #ground-truth cases per fold x item, else the job FAILS before writing metrics;
#   - the item's old metrics CSV is deleted first (evaluate.py silently reuses stale CSVs);
#   - after the fold summary, eval_all.csv must exist, with 1 + (#cases x #items x #labels) rows.
# Metrics via the shared commun evaluate.py (Dice + HD95) and summarize_fold.py; CPU-only via run_job.
set -euo pipefail
cd "${PROJECT_ROOT:?source env.sh first}"
RUN_ID="${1:?need RUN_ID}"; CATEGORY="${2:?need CATEGORY (nnUNet|auglab)}"; FOLD_ARG="${3:-all}"
: "${EVAL_ITEMS:?}" "${EVAL_LABELS:?}" "${EVAL_JOB_PREFIX:?}"
case "${CATEGORY}" in nnUNet|auglab) ;; *) echo "ERROR: CATEGORY must be nnUNet|auglab, got '${CATEGORY}'" >&2; exit 1;; esac
DATASET_ID="${EVAL_DATASET_ID:-${DATASET_ID:?set EVAL_DATASET_ID}}"
CKPT_TAG="${CKPT_TAG:-best}"
_DS_NAME="$(ls "${nnUNet_raw}" | grep "^Dataset0*${DATASET_ID}_" | head -1)"
[ -n "${_DS_NAME}" ] || { echo "ERROR: no Dataset${DATASET_ID}_* in ${nnUNet_raw}" >&2; exit 1; }
DJ="${nnUNet_raw}/${_DS_NAME}/dataset.json"
PRED_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"
[ -d "${PRED_BASE}" ] || { echo "ERROR: no predictions at ${PRED_BASE} (wrong CATEGORY?)" >&2; exit 1; }
_PRED_SUBDIR=""; [ "${CKPT_TAG}" != "best" ] && _PRED_SUBDIR="${CKPT_TAG}/"
_OUT_SUFFIX=""; [ "${CKPT_TAG}" != "best" ] && _OUT_SUFFIX="_${CKPT_TAG}"
METRICS_SUBDIR="${METRICS_SUBDIR:-}"
OUT_BASE="${METRICS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}${METRICS_SUBDIR:+/${METRICS_SUBDIR}}/${CATEGORY}_${RUN_ID}${_OUT_SUFFIX}"
# 3-fold policy; must match EVAL_FOLDS in 00_00_utils/eval_folds.py
if [ "${FOLD_ARG}" = "all" ]; then FOLDS="${EVAL_FOLDS:-0 1 2}"; else FOLDS="${FOLD_ARG}"; fi
EVAL_PY="benchmark/00_commun_scripts/00_03_evaluate/evaluate.py"

mkdir -p "${OUT_BASE}/_logs"
_EVAL_BODY="
set -euo pipefail
cd '${PROJECT_ROOT}'
for F in ${FOLDS}; do
    N_ALL=0
    for item in ${EVAL_ITEMS}; do
        PRED_DIR='${PRED_BASE}'/fold\${F}/${_PRED_SUBDIR}\${item}
        GT_DIR='${nnUNet_raw}/${_DS_NAME}'/labelsTs_\${item}
        [ -d \"\${PRED_DIR}\" ] || { echo \"ERROR: no preds fold\${F}/\${item} (\${PRED_DIR})\" >&2; exit 1; }
        NP=\$(ls \"\${PRED_DIR}\"/*.nii.gz 2>/dev/null | wc -l); NG=\$(ls \"\${GT_DIR}\"/*.nii.gz | wc -l)
        [ \"\${NP}\" = \"\${NG}\" ] || { echo \"ERROR: fold\${F}/\${item}: \${NP} predictions vs \${NG} GT cases\" >&2; exit 1; }
        N_ALL=\$((N_ALL + NG))
        OUT_CSV='${OUT_BASE}'/fold\${F}/\${item}_metrics.csv
        mkdir -p \"\$(dirname \"\${OUT_CSV}\")\"; rm -f \"\${OUT_CSV}\"
        echo \"[\$(date '+%H:%M:%S')] eval ${RUN_ID} fold\${F} \${item} (\${NG} cases)\"
        .venv/bin/python ${EVAL_PY} --pred_dir \"\${PRED_DIR}\" --gt_dir \"\${GT_DIR}\" --dataset_json '${DJ}' \
            --labels ${EVAL_LABELS} --name \"\${item}\" --out_csv \"\${OUT_CSV}\" --workers 8
    done
    OUT_FOLD='${OUT_BASE}'/fold\${F}
    .venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/summarize_fold.py \
        \"\${OUT_FOLD}\" '${RUN_ID}' \"\${F}\" --groups ${EVAL_ITEMS} --group-col contrast --groups-word Contrasts
    [ -f \"\${OUT_FOLD}/eval_all.csv\" ] || { echo \"ERROR: no eval_all.csv in \${OUT_FOLD}\" >&2; exit 1; }
    NR=\$(( \$(wc -l < \"\${OUT_FOLD}/eval_all.csv\") - 1 ))
    EXP=\$(( N_ALL * \$(echo ${EVAL_LABELS} | wc -w) ))
    [ \"\${NR}\" = \"\${EXP}\" ] || { echo \"ERROR: \${OUT_FOLD}/eval_all.csv has \${NR} rows, expected \${EXP}\" >&2; exit 1; }
done
echo '→ ${OUT_BASE}'
"
# EVAL_INLINE=1: run in THIS shell (use when already inside a compute allocation, e.g. a TamIA post-training eval job: no nested run_job).
if [ "${EVAL_INLINE:-0}" = "1" ]; then
    bash -c "${_EVAL_BODY}" 2>&1 | tee "${OUT_BASE}/_logs/eval_${RUN_ID}${_OUT_SUFFIX}.log"; exit "${PIPESTATUS[0]}"
fi
run_job --name "${EVAL_JOB_PREFIX}_${CATEGORY}_${RUN_ID}${_OUT_SUFFIX}" --gpus 0 --cpus 8 --mem "${EVAL_MEM:-16G}" --time "${EVAL_TIME:-1:00:00}" \
    --log "${OUT_BASE}/_logs/eval_${RUN_ID}${_OUT_SUFFIX}.log" --wait -- bash -c "${_EVAL_BODY}"
