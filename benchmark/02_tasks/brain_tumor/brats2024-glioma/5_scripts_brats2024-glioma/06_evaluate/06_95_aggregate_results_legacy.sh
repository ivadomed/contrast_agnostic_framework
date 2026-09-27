#!/usr/bin/env bash
# Aggregate evaluation results across all experiments and folds.
#
# Reads METRICS_ROOT/{category}_{run_id}/fold{k}/eval_all.csv for every run
# (or a specific subset), computes cross-fold mean±std Dice and HD95 per
# contrast per label, and writes:
#   METRICS_ROOT/02_00_aggregated_metrics.md
#
# Usage:
#   bash 06_95_aggregate_results_legacy.sh                              # all with eval data
#   bash 06_95_aggregate_results_legacy.sh <KEY> [KEY ...]             # specific {cat}_{run_id} keys
#
# Prerequisites: run 06_01_evaluate_run.sh for each experiment first.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

_AGG_DIR="${METRICS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}"
echo "[$(date '+%H:%M:%S')] aggregating metrics from ${_AGG_DIR}/"

if [ $# -gt 0 ]; then
    .venv/bin/python "${HERE}/06_95_aggregate_results_legacy.py" \
        --metrics_dir "${_AGG_DIR}" \
        --run_keys "$@"
else
    .venv/bin/python "${HERE}/06_95_aggregate_results_legacy.py" \
        --metrics_dir "${_AGG_DIR}"
fi

echo "[$(date '+%H:%M:%S')] done → ${_AGG_DIR}/02_00_aggregated_metrics.md"
