#!/usr/bin/env bash
# Aggregate evaluation results from a YAML config file -- per-organ report shape
# (see 00_commun_scripts/00_03_evaluate/aggregate_per_organ_from_config.py's own
# docstring; for SLIVER07 this collapses to a single "liver" column since its GT
# is liver-only, but stays on the same shared per-organ aggregator amos uses).
#
# Usage:
#   bash 06_02_aggregate_per_organ_from_config.sh <config.yaml>
#   bash 06_02_aggregate_per_organ_from_config.sh configs/sliver07_t1in_00_comparison.yaml
#   bash 06_02_aggregate_per_organ_from_config.sh configs/sliver07_t2spir_00_comparison.yaml

set -euo pipefail
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

echo "[$(date '+%H:%M:%S')] aggregating (per-organ) from ${CONFIG}"
.venv/bin/python "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_03_evaluate/aggregate_per_organ_from_config.py" "${CONFIG}"
echo "[$(date '+%H:%M:%S')] done"
