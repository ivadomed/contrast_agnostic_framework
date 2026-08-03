#!/usr/bin/env bash
# Aggregate evaluation results from a YAML config file.
#
# Usage:
#   bash 06_10_aggregate_from_config.sh <config.yaml>
#   bash 06_10_aggregate_from_config.sh configs/picai-prostate_flair_01_results.yaml
#
# The config specifies which runs to include, the metrics_dir, and the output
# prefix. See configs/ for available configs and datasets/00_commun_scripts/00_03_evaluate/aggregate_from_config.py
# for the full config format.
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
[ -f "${PROJECT_ROOT}/scripts/cluster/tamia_env_picai.sh" ] && [ -d /scratch/p/paulh ] && \
    source "${PROJECT_ROOT}/scripts/cluster/tamia_env_picai.sh"
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
.venv/bin/python "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/aggregate_from_config.py" "${CONFIG}"
echo "[$(date '+%H:%M:%S')] done"
