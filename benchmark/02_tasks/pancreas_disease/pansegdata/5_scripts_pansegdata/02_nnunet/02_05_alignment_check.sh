#!/bin/bash
# Quantitative mask-image alignment check of all converted scans (flip test, posterior-mask test, T1/T2 pair agreement) via run_job (CPU).  bash 02_05_alignment_check.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
mkdir -p "${HERE}/logs"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"
LOG="${HERE}/logs/alignment_check_$(date +%Y%m%d_%H%M%S).log"
run_job --name pansegdata_align_check --gpus 0 --cpus 4 --mem 24G --time 01:00:00 --log "${LOG}" --wait -- \
    "${REPO_ROOT}/.venv/bin/python" "${HERE}/02_05_alignment_check.py"
tail -20 "${LOG}"
