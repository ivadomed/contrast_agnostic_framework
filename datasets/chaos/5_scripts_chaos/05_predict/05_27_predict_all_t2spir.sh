#!/usr/bin/env bash
# Predict all 6 CHAOS T2spir experiments (all 4 folds each, all modalities).
# Each per-experiment script runs its 4 folds in parallel; the 6 experiments
# themselves are launched in parallel here.
#
# Run IDs are hardcoded to the canonical T2spir training runs.
#
# Usage: bash 05_27_predict_all_t2spir.sh

set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Run IDs (canonical T2spir training runs) ────────────────────────────────
RUN_BASELINE="chaos_t2spir_baseline_20260620_111146"
RUN_V26_6_2="chaos_t2spir_v26_6_2_train050_val100_20260620_112122"
RUN_AUGLAB_DEFAULT="chaos_t2spir_auglab_default_20260620_112240"
RUN_SYNTHSEG_EM="chaos_t2spir_synthseg_EM_20260620_112357"
RUN_SYNTHSEG_NOEM="chaos_t2spir_synthseg_noEM_20260620_112515"
RUN_AUGLAB_AUG="chaos_t2spir_auglabAug_v26_6_2_train025_val100_20260620_112633"

echo "[$(date '+%H:%M:%S')] === Predicting all 6 CHAOS T2spir experiments (all folds) ==="
echo ""

bash "${SCRIPTS_DIR}/05_21_predict_t2spir_baseline.sh"                           "${RUN_BASELINE}"      all &
bash "${SCRIPTS_DIR}/05_22_predict_t2spir_v26_6_2_train050_val100.sh"            "${RUN_V26_6_2}"       all &
bash "${SCRIPTS_DIR}/05_23_predict_t2spir_auglab_default.sh"                     "${RUN_AUGLAB_DEFAULT}" all &
bash "${SCRIPTS_DIR}/05_24_predict_t2spir_synthseg_EM.sh"                        "${RUN_SYNTHSEG_EM}"   all &
bash "${SCRIPTS_DIR}/05_25_predict_t2spir_synthseg_noEM.sh"                      "${RUN_SYNTHSEG_NOEM}" all &
bash "${SCRIPTS_DIR}/05_26_predict_t2spir_auglabAug_v26_6_2_train025_val100.sh"  "${RUN_AUGLAB_AUG}"    all &
wait

echo ""
echo "[$(date '+%H:%M:%S')] === All T2spir predictions done ==="
