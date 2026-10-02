#!/usr/bin/env bash
# Boundary-PV preview on BraTS t1n (CPU, Vulcan via run_job). Usage: bash run_pv_preview.sh
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${D}/../../../../../.." && pwd)"
cd "${REPO}"
source "${REPO}/benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh"
mkdir -p "${D}/outputs"
run_job --name pv_preview --gpus 0 --cpus 4 --mem 16G --time 00:30:00 --wait \
    --log "${D}/outputs/pv_preview_final.log" -- "${REPO}/.venv/bin/python" "${D}/pv_preview.py"
cat "${D}/outputs/pv_preview_final.log"
