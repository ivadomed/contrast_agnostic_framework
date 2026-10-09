#!/usr/bin/env bash
# Aggregate the label boundary-cue results into the summary table.
#
# Goes through run_job rather than running inline: the grouped-CV model fitting is a few CPU
# minutes across all groups, which belongs in a job, not on a login node (an earlier inline run
# held ~6 login-node cores for 9+ minutes -- past the Alliance allowance -- before being killed).
#
#   IN_DIR=<dir of .npz> OUT_DIR=<dir> bash run_aggregate.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../../.." && pwd)"
: "${IN_DIR:?set IN_DIR}"; : "${OUT_DIR:?set OUT_DIR}"
mkdir -p "${OUT_DIR}"
source "${REPO}/scripts/job_runner/run_job.sh"
run_job --name cue_aggregate --gpus 0 --cpus 8 --mem 32G --time 00:30:00 \
        --log "${OUT_DIR}/aggregate.log" --wait -- \
    "${REPO}/.venv/bin/python" "${HERE}/aggregate_cue_importance.py" \
        --in-dir "${IN_DIR}" --out-dir "${OUT_DIR}"
