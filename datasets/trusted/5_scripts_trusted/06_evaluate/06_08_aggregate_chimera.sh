#!/usr/bin/env bash
# Aggregate the chimera experiment into a comparison table (kidney vs center-prior).
# Run after 06_07_evaluate_chimera_all.sh.  bash 06_08_aggregate_chimera.sh
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"; cd "${PROJECT_ROOT}"
.venv/bin/python "$(dirname "$0")/06_08_aggregate_chimera.py" \
    --metrics_root "${METRICS_ROOT}/${CHAOS_MODEL_TYPE}"
