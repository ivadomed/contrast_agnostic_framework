#!/usr/bin/env bash
# Run both open-ms causal-ablation ladders (FLAIR 06_19, T1w 06_18) as one small CPU job via run_job
# (no python on the login node). Usage: bash 06_20_run_ladders.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
LOG="${METRICS_ROOT}/${MODEL_TYPE}/_logs/ladders_$(date +%Y%m%d_%H%M%S).log"; mkdir -p "$(dirname "${LOG}")"
run_job --name openms_ladders --gpus 0 --cpus 2 --mem 8G --time 00:30:00 --wait --log "${LOG}" -- \
    bash -c ".venv/bin/python '${HERE}/06_19_ladder_summary_ood.py' && .venv/bin/python '${HERE}/06_18_ladder_summary_t1w.py'"
cat "${LOG}"
