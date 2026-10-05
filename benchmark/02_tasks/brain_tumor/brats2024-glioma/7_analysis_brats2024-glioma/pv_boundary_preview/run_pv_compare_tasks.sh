#!/usr/bin/env bash
# BraTS half of the ToothFairy2-vs-BraTS PV comparison (CPU, Vulcan via run_job; the ToothFairy2 half runs on TamIA:
#   sbatch --wrap=".venv/bin/python <this dir>/pv_compare_tasks.py <out> toothfairy2"). Usage: bash run_pv_compare_tasks.sh
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${D}/../../../../../.." && pwd)"
cd "${REPO}"
source "${REPO}/benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh"
run_job --name pv_compare_brats --gpus 0 --cpus 4 --mem 16G --time 00:30:00 --wait \
    --log "${D}/outputs/pv_compare_brats.log" -- "${REPO}/.venv/bin/python" "${D}/pv_compare_tasks.py" "${D}/outputs" brats
cat "${D}/outputs/pv_compare_brats.log" | grep -v -i "warning\|custom_fwd"
