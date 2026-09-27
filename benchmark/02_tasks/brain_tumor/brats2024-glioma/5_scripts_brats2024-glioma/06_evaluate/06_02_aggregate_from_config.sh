#!/usr/bin/env bash
# Aggregate evaluation results from a YAML config file.
#
# Usage:
#   bash 06_02_aggregate_from_config.sh <config.yaml>
#   bash 06_02_aggregate_from_config.sh configs/brats_t1n_01_results.yaml
#
# The config specifies which runs to include, the metrics_dir, and the output
# prefix. See configs/ for available configs and benchmark/00_commun_scripts/00_03_evaluate/aggregate_from_config.py
# for the full config format.
#
# To run all configs in one shot:
#   for cfg in configs/brats_*.yaml; do bash 06_02_aggregate_from_config.sh "$cfg"; done

set -euo pipefail
# HERE must be resolved BEFORE the cd below and via BASH_SOURCE (not $0): once cwd
# changes to PROJECT_ROOT, `dirname "$0"` for a script invoked with a bare/relative
# name collapses to "." relative to the NEW cwd, silently resolving HERE to
# PROJECT_ROOT instead of this script's directory and breaking every relative
# CONFIG path. Same bug class already fixed in chaos's 06_02_aggregate_from_config.sh.
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
.venv/bin/python "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_03_evaluate/aggregate_from_config.py" "${CONFIG}"
echo "[$(date '+%H:%M:%S')] done"
