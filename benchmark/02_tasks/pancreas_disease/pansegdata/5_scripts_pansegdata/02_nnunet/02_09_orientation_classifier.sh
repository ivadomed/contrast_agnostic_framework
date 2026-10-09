#!/bin/bash
# Learn + validate each scan's true in-plane orientation (CNN trained on verified NYU/AHN scans), via run_job (CPU).  bash 02_09_orientation_classifier.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
mkdir -p "${HERE}/logs"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"
LOG="${HERE}/logs/orientation_classifier_$(date +%Y%m%d_%H%M%S).log"
run_job --name pansegdata_orient_cnn --gpus 0 --cpus 4 --mem 24G --time 02:00:00 --log "${LOG}" --wait -- \
    "${REPO_ROOT}/.venv/bin/python" "${HERE}/02_09_orientation_classifier.py"
tail -40 "${LOG}"
