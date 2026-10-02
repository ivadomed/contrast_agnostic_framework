#!/usr/bin/env bash
# Boundary-PV self-tests (equivalence vs git HEAD, correctness, GPU timing). Usage: bash run_pv_selftest.sh
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${D}/../../../../../.." && pwd)"
cd "${REPO}"
source "${REPO}/benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh"
mkdir -p "${D}/outputs"
run_job --name pv_selftest --gpus 1 --cpus 4 --mem 32G --time 00:30:00 --wait \
    --log "${D}/outputs/pv_selftest.log" -- "${REPO}/.venv/bin/python" "${D}/pv_selftest.py"
cat "${D}/outputs/pv_selftest.log"
