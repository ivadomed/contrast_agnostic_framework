#!/usr/bin/env bash
# Aggregate evaluation results from a YAML config file.
#
# Usage:
#   bash 06_10_aggregate_from_config.sh <config.yaml>
#   bash 06_10_aggregate_from_config.sh configs/chaos_t1in_03_results.yaml
#
# The config specifies which runs to include, the metrics_dir, and the output
# prefix. See configs/ for available configs and scripts/evaluate/aggregate_from_config.py
# for the full config format.
#
# To run all T1in configs in one shot:
#   for cfg in configs/chaos_t1in_*.yaml; do bash 06_10_aggregate_from_config.sh "$cfg"; done
# For T2spir configs, source env_t2spir.sh first (sets TRAINING_CONTRAST; METRICS_ROOT is identical):
#   source "$(dirname "$0")/../00_utils/env_t2spir.sh"
#   bash 06_10_aggregate_from_config.sh configs/chaos_t2spir_01_results.yaml

set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

HERE="$(cd "$(dirname "$0")" && pwd)"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <config.yaml>" >&2
    exit 1
fi

CONFIG="$1"
# Resolve relative paths against the script directory
if [[ "$CONFIG" != /* ]]; then
    CONFIG="${HERE}/${CONFIG}"
fi

echo "[$(date '+%H:%M:%S')] aggregating from ${CONFIG}"
.venv/bin/python "${PROJECT_ROOT}/scripts/evaluate/aggregate_from_config.py" "${CONFIG}"
echo "[$(date '+%H:%M:%S')] done"
