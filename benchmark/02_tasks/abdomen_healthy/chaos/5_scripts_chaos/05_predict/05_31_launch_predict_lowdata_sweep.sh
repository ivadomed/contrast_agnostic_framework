#!/usr/bin/env bash
# Low-data benchmark — predict ALL trained low-data runs (discovered on disk), in
# parallel. Safe to re-run: only dispatches runs that exist.
#
# Usage: bash 05_31_launch_predict_lowdata_sweep.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$(dirname "$0")/../00_utils/env.sh"

_base="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}"
n=0
for _cat in nnUNet auglab; do
    for runroot in "${_base}/${_cat}"/*_lowdata_n*; do
        [ -d "${runroot}" ] || continue
        RUN_ID="$(basename "${runroot}")"
        echo "[$(date '+%H:%M:%S')] predict ${RUN_ID}"
        bash "${HERE}/05_30_predict_lowdata.sh" "${RUN_ID}" &
        n=$((n + 1)); sleep 3
    done
done
wait
echo "[$(date '+%H:%M:%S')] === ${n} low-data prediction runs dispatched ==="
