#!/usr/bin/env bash
# Evaluate all 6 open-ms T1w experiments (folds 0 1 2, contrasts FLAIR/T2W/T1W).
# Each call to 06_13 launches an INDEPENDENT CPU-only --wait job; they have no
# inter-dependencies, so fan them out in PARALLEL (like 05_17_predict_all_t1w.sh) rather
# than serially — 6 concurrent compute-node jobs finish in ~1/6 the wall-clock.
# Run IDs hardcoded to the canonical T1w training runs (2026-07-08).
#
# Prerequisite: 05_17_predict_all_t1w.sh must have produced predictions.
#
# Usage: bash 06_14_evaluate_all_t1w.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# "RUN_ID CATEGORY" — baseline & v26-alone are nnUNet-category; the rest are auglab.
RUNS=(
    "open-ms_t1w_baseline_20260708_083441 nnUNet"
    "open-ms_t1w_auglab_default_20260708_083511 auglab"
    "open-ms_t1w_synthseg_noEM_train100_val000_20260708_083541 auglab"
    "open-ms_t1w_synthseg_EM_train100_val000_20260708_083611 auglab"
    "open-ms_t1w_v26_6_2_train050_val100_20260708_083641 nnUNet"
    "open-ms_t1w_auglabAug_v26_6_2_train025_val100_20260708_083711 auglab"
)

echo "[$(date '+%H:%M:%S')] === Evaluating all 6 open-ms T1w experiments (parallel) ==="
for entry in "${RUNS[@]}"; do
    set -- $entry
    echo "[$(date '+%H:%M:%S')] launching eval $1 ($2)"
    bash "${HERE}/06_13_evaluate_t1w.sh" "$1" "$2" all &
done
wait
echo "[$(date '+%H:%M:%S')] === All 6 T1w evaluations done ==="
