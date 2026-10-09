#!/usr/bin/env bash
# Predict all 6 open-ms T1w experiments (folds 0 1 2, all test contrasts FLAIR/T2W/T1W).
# Each per-method wrapper fans out its folds; the 6 methods are launched in parallel here.
# Run IDs are hardcoded to the canonical T1w training runs (2026-07-08).
#
# Usage: bash 05_17_predict_all_t1w.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# ── Run IDs (canonical T1w training runs, all trained folds 0 1 2, epoch 2000) ──
RUN_BASELINE="open-ms_t1w_baseline_20260708_083441"
RUN_AUGLAB_DEFAULT="open-ms_t1w_auglab_default_20260708_083511"
RUN_SYNTHSEG_NOEM="open-ms_t1w_synthseg_noEM_train100_val000_20260708_083541"
RUN_SYNTHSEG_EM="open-ms_t1w_synthseg_EM_train100_val000_20260708_083611"
RUN_V26_6_2="open-ms_t1w_v26_6_2_train050_val100_20260708_083641"
RUN_AUGLAB_AUG="open-ms_t1w_auglabAug_v26_6_2_train025_val100_20260708_083711"

echo "[$(date '+%H:%M:%S')] === Predicting all 6 open-ms T1w experiments (folds 0 1 2) ==="

bash "${HERE}/05_11_predict_t1w_baseline.sh"                              "${RUN_BASELINE}"       all &
bash "${HERE}/05_12_predict_t1w_auglab_default.sh"                       "${RUN_AUGLAB_DEFAULT}" all &
bash "${HERE}/05_13_predict_t1w_synthseg_noEM.sh"                        "${RUN_SYNTHSEG_NOEM}"  all &
bash "${HERE}/05_14_predict_t1w_synthseg_EM.sh"                          "${RUN_SYNTHSEG_EM}"    all &
bash "${HERE}/05_15_predict_t1w_v26_6_2_train050_val100.sh"              "${RUN_V26_6_2}"        all &
bash "${HERE}/05_16_predict_t1w_auglabAug_v26_6_2_train025_val100.sh"    "${RUN_AUGLAB_AUG}"     all &
wait

echo "[$(date '+%H:%M:%S')] === All 6 T1w prediction suites launched ==="
