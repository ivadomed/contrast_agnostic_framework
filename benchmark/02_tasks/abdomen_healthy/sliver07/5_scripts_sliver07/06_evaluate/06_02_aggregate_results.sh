#!/usr/bin/env bash
# Aggregate all SLIVER07 chaos-model evaluation results into a comparison table.
# Run after 06_03_evaluate_all_chaos.sh has completed.
set -euo pipefail
# HERE must be resolved BEFORE the cd below and via BASH_SOURCE (not $0): once cwd
# changes to PROJECT_ROOT, `dirname "$0"` for a bare/relative invocation collapses
# to "." relative to the NEW cwd. Same bug class as chaos's
# 06_02_aggregate_from_config.sh (see that file's header comment).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
_AGG_DIR="${METRICS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}"
.venv/bin/python "${HERE}/06_02_aggregate_results.py" \
    --metrics_root "${_AGG_DIR}" "$@"
