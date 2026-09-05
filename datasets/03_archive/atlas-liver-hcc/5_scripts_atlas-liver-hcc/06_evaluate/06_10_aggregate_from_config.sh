#!/usr/bin/env bash
# Aggregate evaluation results from a YAML config file.
#
# Usage:
#   bash 06_10_aggregate_from_config.sh <config.yaml>
#   bash 06_10_aggregate_from_config.sh configs/atlas-liver-hcc_t1w_01_results.yaml
#
# The config specifies which runs to include, the metrics_dir, and the output
# prefix. See configs/ for available configs and
# datasets/00_commun_scripts/00_03_evaluate/aggregate_from_config.py for the full config
# format.
set -euo pipefail
# HERE must be resolved BEFORE the cd below and via BASH_SOURCE (not $0): once cwd
# changes to PROJECT_ROOT, `dirname "$0"` for a script invoked with a bare/relative
# name collapses to "." relative to the NEW cwd, silently resolving HERE to
# PROJECT_ROOT instead of this script's directory and breaking every relative
# CONFIG path. Same bug class as chaos's 06_10/06_11 (see those scripts'
# comments) -- hit here directly 2026-08-29 while wiring atlas-liver-hcc into
# the paper's meta-task results.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <config.yaml>" >&2
    exit 1
fi

CONFIG="$1"
if [[ "$CONFIG" != /* ]]; then
    CONFIG="${HERE}/${CONFIG}"
fi

echo "[$(date '+%H:%M:%S')] aggregating from ${CONFIG}"
.venv/bin/python "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/aggregate_from_config.py" "${CONFIG}"
echo "[$(date '+%H:%M:%S')] done"
