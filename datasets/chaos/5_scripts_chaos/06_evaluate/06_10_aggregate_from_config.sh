#!/usr/bin/env bash
# Aggregate evaluation results from a YAML config file.
#
# Usage:
#   bash 06_10_aggregate_from_config.sh <config.yaml>
#   bash 06_10_aggregate_from_config.sh configs/chaos_t1in_03_results.yaml
#
# The config specifies which runs to include, the metrics_dir, and the output
# prefix. See configs/ for available configs and datasets/00_commun_scripts/00_03_evaluate/aggregate_from_config.py
# for the full config format.
#
# To run all T1in configs in one shot:
#   for cfg in configs/chaos_t1in_*.yaml; do bash 06_10_aggregate_from_config.sh "$cfg"; done
# For T2spir configs, source env_t2spir.sh first (sets TRAINING_CONTRAST; METRICS_ROOT is identical):
#   source "$(dirname "$0")/../00_utils/env_t2spir.sh"
#   bash 06_10_aggregate_from_config.sh configs/chaos_t2spir_01_results.yaml

set -euo pipefail
# HERE must be resolved BEFORE the cd below and via BASH_SOURCE (not $0): once cwd
# changes to PROJECT_ROOT, `dirname "$0"` for a script invoked with a bare/relative
# name (exactly this file's own documented usage, e.g. `bash 06_10_....sh configs/x`
# from inside 06_evaluate/) collapses to "." relative to the NEW cwd, silently
# resolving HERE to PROJECT_ROOT instead of this script's directory and breaking
# every relative CONFIG path. Hit this onboarding msd-spleen (2026-07-31).
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
.venv/bin/python "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/aggregate_from_config.py" "${CONFIG}"
echo "[$(date '+%H:%M:%S')] done"
