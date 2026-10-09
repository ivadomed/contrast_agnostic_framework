#!/usr/bin/env bash
# Low-data benchmark — evaluate ALL low-data runs (discovered on disk) with the
# standard per-run evaluator (06_01_evaluate_run.sh → Dice/HD95 per modality & label,
# writes the canonical fold*/eval_all.csv). Run AFTER predictions complete.
#
# Usage: bash 06_20_evaluate_lowdata_sweep.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$(dirname "$0")/../00_utils/env.sh"

_base="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}"
n=0
for _cat in nnUNet auglab; do
    for runroot in "${_base}/${_cat}"/*_lowdata_n*; do
        [ -d "${runroot}" ] || continue
        RUN_ID="$(basename "${runroot}")"
        echo "[$(date '+%H:%M:%S')] evaluate ${RUN_ID} (CATEGORY=${_cat})"
        CATEGORY="${_cat}" bash "${HERE}/06_01_evaluate_run.sh" "${RUN_ID}" &
        n=$((n + 1)); sleep 3
    done
done
wait
echo "[$(date '+%H:%M:%S')] === ${n} low-data runs evaluated → now run 06_22_lowdata_curve.sh ==="
