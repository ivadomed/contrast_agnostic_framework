#!/bin/bash
# Large labelled renders of selected subjects for eyeballing mask placement vs anatomy, via run_job (CPU).  bash 02_11_look_closely.sh [case ...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
mkdir -p "${HERE}/logs"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"
LOG="${HERE}/logs/look_closely_$(date +%Y%m%d_%H%M%S).log"
run_job --name pansegdata_look --gpus 0 --cpus 2 --mem 12G --time 00:20:00 --log "${LOG}" --wait -- \
    "${REPO_ROOT}/.venv/bin/python" "${HERE}/02_11_look_closely.py" "$@"
tail -6 "${LOG}"
