#!/usr/bin/env bash
# Aggregate evaluation results from a YAML config file.
#
# Usage:
#   bash 06_29_aggregate_from_config.sh <config.yaml>
#   bash 06_29_aggregate_from_config.sh configs/duke_t1wce_uni_t1wce_01_results.yaml
#
# The config specifies which runs to include, the metrics_dir, and the output
# prefix. See configs/ for available configs and
# benchmark/00_commun_scripts/00_03_evaluate/aggregate_from_config.py for the
# full config format.

set -euo pipefail
# HERE must be resolved BEFORE the cd below and via BASH_SOURCE (not $0) -- see
# chaos's 06_10_aggregate_from_config.sh header for why (bit msd-spleen onboarding).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

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
.venv/bin/python "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_03_evaluate/aggregate_from_config.py" "${CONFIG}"
echo "[$(date '+%H:%M:%S')] done"
