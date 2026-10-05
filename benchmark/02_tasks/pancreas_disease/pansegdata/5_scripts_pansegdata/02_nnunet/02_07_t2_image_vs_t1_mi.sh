#!/bin/bash
# Mutual-information test: is the T2 IMAGE (not only its mask) rotated relative to T1? Run after 02_06.  bash 02_07_t2_image_vs_t1_mi.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
mkdir -p "${HERE}/logs"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"
LOG="${HERE}/logs/t2_vs_t1_mi_$(date +%Y%m%d_%H%M%S).log"
run_job --name pansegdata_t2t1_mi --gpus 0 --cpus 4 --mem 24G --time 01:00:00 --log "${LOG}" --wait -- \
    "${REPO_ROOT}/.venv/bin/python" "${HERE}/02_07_t2_image_vs_t1_mi.py"
tail -10 "${LOG}"
