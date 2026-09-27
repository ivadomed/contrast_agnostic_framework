#!/usr/bin/env bash
# Aggregate CHAOS T2spir evaluation results across all experiments and folds.
# Reads METRICS_ROOT/chaos_model/t2spir/{category}_{run_id}/fold{k}/eval_all.csv.
# Writes:
#   METRICS_ROOT/chaos_model/t2spir/02_00_aggregated_metrics.md
#   METRICS_ROOT/chaos_model/t2spir/02_01_summary_by_modality.md
#   METRICS_ROOT/chaos_model/t2spir/02_01_heatmap_dice.png
#   METRICS_ROOT/chaos_model/t2spir/02_01_heatmap_hd95.png
#
# Usage:
#   bash 06_08_aggregate_t2spir.sh                      # all runs with eval data
#   bash 06_08_aggregate_t2spir.sh <KEY> [KEY ...]      # specific {cat}_{run_id} keys
#
# Prerequisite: run 06_07_evaluate_t2spir.sh for each experiment first.

set -euo pipefail
source "$(dirname "$0")/../00_utils/env_t2spir.sh"
cd "${PROJECT_ROOT}"
HERE="$(cd "$(dirname "$0")" && pwd)"

_AGG_DIR="${METRICS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}"
echo "[$(date '+%H:%M:%S')] aggregating T2spir metrics from ${_AGG_DIR}/"

if [ $# -gt 0 ]; then
    .venv/bin/python "${HERE}/06_02_aggregate_results.py" \
        --metrics_dir "${_AGG_DIR}" \
        --run_keys "$@"
else
    .venv/bin/python "${HERE}/06_02_aggregate_results.py" \
        --metrics_dir "${_AGG_DIR}"
fi

echo "[$(date '+%H:%M:%S')] done → ${_AGG_DIR}/02_00_aggregated_metrics.md"
