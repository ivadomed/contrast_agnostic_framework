#!/usr/bin/env bash
# Aggregate all CIRRMRI-LIVER chaos-model evaluation results into a comparison table.
# Run after 06_02_evaluate_all_t1in.sh / 06_03_evaluate_all_t2spir.sh have completed.
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
_AGG_DIR="${METRICS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}"
.venv/bin/python "$(dirname "$0")/06_04_aggregate_results.py" \
    --metrics_root "${_AGG_DIR}" "$@"
