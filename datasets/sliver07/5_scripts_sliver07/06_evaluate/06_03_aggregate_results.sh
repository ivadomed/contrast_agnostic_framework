#!/usr/bin/env bash
# Aggregate all SLIVER07 chaos-model evaluation results into a comparison table.
# Run after 06_02_evaluate_all_chaos.sh has completed.
set -euo pipefail
# No hardcoded absolute path (would break on any host but the original workstation —
# checklist §10). env.sh derives PROJECT_ROOT from its own location.
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
_AGG_DIR="${METRICS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}"
.venv/bin/python "$(dirname "$0")/06_03_aggregate_results.py" \
    --metrics_root "${_AGG_DIR}" "$@"
