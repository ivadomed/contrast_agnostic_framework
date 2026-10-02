#!/bin/bash
# Build ispy1's eval-only nnU-Net test set (t1wce + precontrast) from the BIDS
# tree, via run_job (CPU-only). See 02_01_convert_test.py.
#   bash 02_01_convert_test.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/benchmark/02_tasks/breast_cancer/ispy1/2_nnUNet_ispy1/logs"
mkdir -p "${LOG_DIR}"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"
run_job --name ispy1_convert_test --gpus 0 --cpus 2 --mem 8G --time 01:00:00 \
    --log "${LOG_DIR}/convert_test_$(date +%Y%m%d_%H%M%S).log" --wait -- \
    "${REPO_ROOT}/.venv/bin/python" "${HERE}/02_01_convert_test.py"
