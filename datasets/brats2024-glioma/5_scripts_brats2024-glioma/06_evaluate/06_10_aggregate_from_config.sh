#!/usr/bin/env bash
# Aggregate evaluation results from a YAML config file.
#
# Usage:
#   bash 06_10_aggregate_from_config.sh <config.yaml>
#   bash 06_10_aggregate_from_config.sh configs/brats_t1n_01_results.yaml
#
# The config specifies which runs to include, the metrics_dir, and the output
# prefix. See configs/ for available configs and scripts/evaluate/aggregate_from_config.py
# for the full config format.
#
# To run all configs in one shot:
#   for cfg in configs/brats_*.yaml; do bash 06_10_aggregate_from_config.sh "$cfg"; done

set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

HERE="$(cd "$(dirname "$0")" && pwd)"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <config.yaml>" >&2
    exit 1
fi

CONFIG="$1"
if [[ "$CONFIG" != /* ]]; then
    CONFIG="${HERE}/${CONFIG}"
fi

echo "[$(date '+%H:%M:%S')] aggregating from ${CONFIG}"
.venv/bin/python "${PROJECT_ROOT}/scripts/evaluate/aggregate_from_config.py" "${CONFIG}"
echo "[$(date '+%H:%M:%S')] done"
