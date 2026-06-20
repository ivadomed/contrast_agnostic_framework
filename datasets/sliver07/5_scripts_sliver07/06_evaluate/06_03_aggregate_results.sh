#!/usr/bin/env bash
# Aggregate all SLIVER07 chaos-model evaluation results into a comparison table.
# Run after 06_02_evaluate_all_chaos.sh has completed.
set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project
source "$(dirname "$0")/../00_utils/env.sh"
_AGG_DIR="${METRICS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}"
.venv/bin/python "$(dirname "$0")/06_03_aggregate_results.py" \
    --metrics_root "${_AGG_DIR}" "$@"
