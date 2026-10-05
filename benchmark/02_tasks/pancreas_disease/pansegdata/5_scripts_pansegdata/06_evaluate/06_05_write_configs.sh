#!/usr/bin/env bash
# (Re)generate pansegdata's aggregate / significance / combined YAML configs from the roster pin files (no hardcoded
# timestamps) via the shared 00_03_evaluate/write_configs_from_roster.py, as a CPU job.
#   bash 06_05_write_configs.sh [--skip-missing] [--force]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
mkdir -p "${RESULTS_DIR}/_logs"
run_job --name pansegdata_write_configs --gpus 0 --cpus 1 --mem 4G --time 00:10:00 \
    --log "${RESULTS_DIR}/_logs/write_configs_$(date +%Y%m%d_%H%M%S).log" --wait -- \
    .venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/write_configs_from_roster.py \
    --dataset-root "${DATASET_ROOT}" --model-type "${MODEL_TYPE}" --contrasts t1wce t2w --items t1wce t2w "$@"
