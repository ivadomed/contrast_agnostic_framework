#!/usr/bin/env bash
# Aggregate evaluation results from a YAML config file.
#   bash 06_10_aggregate_from_config.sh configs/brats-ssa_t1n_01_results.yaml
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ $# -ne 1 ]; then echo "Usage: $0 <config.yaml>" >&2; exit 1; fi
CONFIG="$1"
if [[ "$CONFIG" != /* ]]; then CONFIG="${HERE}/${CONFIG}"; fi
echo "[$(date '+%H:%M:%S')] aggregating from ${CONFIG}"
.venv/bin/python "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/aggregate_from_config.py" "${CONFIG}"
echo "[$(date '+%H:%M:%S')] done"
