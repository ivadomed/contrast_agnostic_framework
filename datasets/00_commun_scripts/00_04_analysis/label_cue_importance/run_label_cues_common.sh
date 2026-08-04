#!/usr/bin/env bash
# Shared driver for the label boundary-cue analysis. Each dataset's
# 7_analysis_<ds>/label_cue_importance_lvl_1/scripts/run_*.sh sets a few variables and sources
# this; the real work is in ../../../00_commun_scripts/00_04_analysis/label_cue_importance/.
#
# Per the project rule, this dispatches through run_job -- never a bare python call, never a raw
# sbatch. One job per (dataset, modality) branch.
#
# Variables the caller must set before sourcing:
#   DS            dataset dir name, e.g. brats2024-glioma
#   BRANCHES      newline-separated "<modality>|<imagesDir>|<labelsDir>|<dataset.json>" records
# Optional:
#   N_SUBJECTS    default 40
#   UNION_IDS     comma-separated label ids to ALSO score merged (BraTS whole tumour), default ""
#   UNION_NAME    name for that merged region, default "union"
#   JOB_TIME      default 02:00:00
#   JOB_GPUS      default 1
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SHARED="${REPO}/datasets/00_commun_scripts/00_04_analysis/label_cue_importance"
LVL1="${REPO}/datasets/${DS}/7_analysis_${DS}/label_cue_importance_lvl_1"
DATA="${LVL1}/outputs/data"
LOGS="${LVL1}/outputs/logs"
PY="${REPO}/.venv/bin/python"
: "${N_SUBJECTS:=40}"; : "${UNION_IDS:=}"; : "${UNION_NAME:=union}"
: "${JOB_TIME:=02:00:00}"; : "${JOB_GPUS:=1}"
mkdir -p "${DATA}" "${LOGS}"

source "${REPO}/scripts/job_runner/run_job.sh"

# The phantom self-test is a precondition, not a formality: it is what establishes that the four
# cues are separable at all. Run it before every submission so a silent edit to cue_metrics.py
# cannot quietly invalidate a batch of results (it is seconds of CPU, well inside the login-node
# allowance).
"${PY}" "${SHARED}/cue_metrics.py" --sanity

while IFS='|' read -r MOD IMAGES LABELS DSJSON; do
    [ -z "${MOD:-}" ] && continue
    run_job --name "cues_${DS}_${MOD}" --gpus "${JOB_GPUS}" --time "${JOB_TIME}" \
            --log "${LOGS}/${MOD}.log" -- \
        "${PY}" "${SHARED}/compute_label_cues.py" \
            --images "${IMAGES}" --labels "${LABELS}" --labels-json "${DSJSON}" \
            --out-dir "${DATA}" --dataset "${DS}" --modality "${MOD}" \
            --n-subjects "${N_SUBJECTS}" \
            ${UNION_IDS:+--extra-union "${UNION_IDS}" --union-name "${UNION_NAME}"}
    echo "submitted ${DS}/${MOD}"
done <<< "${BRANCHES}"
