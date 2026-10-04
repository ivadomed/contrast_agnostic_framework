#!/usr/bin/env bash
# Run both totalseg-pelvic causal-ablation ladders (CT- and MRI-trained) as one small CPU job.
# Usage: bash 06_12_run_ladders.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
LOG="${METRICS_ROOT}/${MODEL_TYPE}/_logs/ladders_$(date +%Y%m%d_%H%M%S).log"; mkdir -p "$(dirname "${LOG}")"
run_job --name totalseg_pelvic_ladders --gpus 0 --cpus 2 --mem 8G --time 00:30:00 --wait --log "${LOG}" -- \
    bash -c "'${PROJECT_ROOT}/.venv/bin/python' '${HERE}/06_10_ladder_summary_ct.py' && '${PROJECT_ROOT}/.venv/bin/python' '${HERE}/06_11_ladder_summary_mri.py'"
cat "${LOG}"
