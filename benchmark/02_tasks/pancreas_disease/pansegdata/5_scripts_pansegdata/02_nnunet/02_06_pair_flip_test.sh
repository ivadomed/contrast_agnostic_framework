#!/bin/bash
# T1-mask vs T2-mask flip test on same-shape pairs + polarity-correct T2 contrast test via run_job (CPU).  bash 02_06_pair_flip_test.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
mkdir -p "${HERE}/logs"
source "${REPO_ROOT}/scripts/job_runner/run_job.sh"
LOG="${HERE}/logs/pair_flip_test_$(date +%Y%m%d_%H%M%S).log"
run_job --name pansegdata_pair_flip --gpus 0 --cpus 4 --mem 24G --time 01:00:00 --log "${LOG}" --wait -- \
    "${REPO_ROOT}/.venv/bin/python" "${HERE}/02_06_pair_flip_test.py"
tail -14 "${LOG}"
