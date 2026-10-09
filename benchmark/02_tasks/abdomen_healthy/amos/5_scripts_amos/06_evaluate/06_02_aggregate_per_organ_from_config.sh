#!/usr/bin/env bash
# Aggregate evaluation results from a YAML config file -- per-organ report shape
# (this dataset's own predictions are organ-labeled: liver/right_kidney/
# left_kidney/spleen -- see 00_commun_scripts/00_03_evaluate/aggregate_per_organ_from_config.py's
# own docstring for why AMOS needs this sibling script instead of the canonical
# aggregate_from_config.py used elsewhere).
#
# Usage:
#   bash 06_02_aggregate_per_organ_from_config.sh <config.yaml>
#   bash 06_02_aggregate_per_organ_from_config.sh configs/amos_t1in_00_comparison.yaml
#   bash 06_02_aggregate_per_organ_from_config.sh configs/amos_t2spir_00_comparison.yaml

set -euo pipefail
# HERE must be resolved BEFORE the cd below and via BASH_SOURCE (not $0): once cwd
# changes to PROJECT_ROOT, `dirname "$0"` for a bare/relative invocation collapses
# to "." relative to the NEW cwd. Same bug class as chaos's/brats2024-glioma's
# 06_02_aggregate_from_config.sh (see those files' header comments).
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
