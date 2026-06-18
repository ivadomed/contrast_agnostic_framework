#!/usr/bin/env bash
# Aggregate evaluation results across all experiments and folds.
#
# Reads METRICS_ROOT/{category}_{run_id}/fold{k}/eval_all.csv for every run
# (or a specific subset), computes cross-fold mean±std Dice and HD95 per
# contrast per label, and writes:
#   METRICS_ROOT/02_00_aggregated_metrics.md
#
# Usage:
#   bash 06_aggregate_results.sh                              # all with eval data
#   bash 06_aggregate_results.sh <KEY> [KEY ...]             # specific {cat}_{run_id} keys
#
# Prerequisites: run 06_evaluate_run.sh for each experiment first.

set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project
source "$(dirname "$0")/../00_utils/env.sh"

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "[$(date '+%H:%M:%S')] aggregating metrics from ${METRICS_ROOT}/"

if [ $# -gt 0 ]; then
    .venv/bin/python "${HERE}/06_02_aggregate_results.py" \
        --metrics_dir "${METRICS_ROOT}" \
        --run_keys "$@"
else
    .venv/bin/python "${HERE}/06_02_aggregate_results.py" \
        --metrics_dir "${METRICS_ROOT}"
fi

echo "[$(date '+%H:%M:%S')] done → ${METRICS_ROOT}/02_00_aggregated_metrics.md"
