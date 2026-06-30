#!/usr/bin/env bash
# Low-data benchmark — build the learning curves from the per-run metrics. CPU-only,
# fast → runs locally (well under the login-node budget). Scores the IN-DOMAIN contrast
# only (= training contrast). Writes to the contrast-specific
# 8_results_chaos/02_metrics/chaos_model/<contrast>/05_01_low_data/.
#
# Usage: bash 06_22_lowdata_curve.sh
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

METRICS_BASE="${METRICS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}"

.venv/bin/python "$(dirname "$0")/06_21_lowdata_curve.py" \
    --metrics_base "${METRICS_BASE}" --in_domain "${TRAINING_CONTRAST}"
