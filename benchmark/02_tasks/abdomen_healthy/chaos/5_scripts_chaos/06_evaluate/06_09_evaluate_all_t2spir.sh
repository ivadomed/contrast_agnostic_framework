#!/usr/bin/env bash
# Evaluate all 6 CHAOS T2spir experiments (all folds) then aggregate.
# Evaluations run in parallel; aggregation waits for all to complete.
#
# Run IDs are hardcoded to the canonical T2spir training runs.
#
# Usage: bash 06_09_evaluate_all_t2spir.sh

set -euo pipefail
EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Run IDs (canonical T2spir training runs) ────────────────────────────────
RUN_BASELINE="chaos_t2spir_baseline_20260620_111146"
RUN_V26_6_2="chaos_t2spir_v26_6_2_train050_val100_20260620_112122"
RUN_AUGLAB_DEFAULT="chaos_t2spir_auglab_default_20260620_112240"
RUN_SYNTHSEG_EM="chaos_t2spir_synthseg_EM_20260620_112357"
RUN_SYNTHSEG_NOEM="chaos_t2spir_synthseg_noEM_20260620_112515"
RUN_AUGLAB_AUG="chaos_t2spir_auglabAug_v26_6_2_train025_val100_20260620_112633"

echo "[$(date '+%H:%M:%S')] === Evaluating all 6 CHAOS T2spir experiments (all folds, parallel) ==="
echo ""

bash "${EVAL_DIR}/06_07_evaluate_t2spir.sh" "${RUN_BASELINE}"       all &
bash "${EVAL_DIR}/06_07_evaluate_t2spir.sh" "${RUN_V26_6_2}"        all &
bash "${EVAL_DIR}/06_07_evaluate_t2spir.sh" "${RUN_AUGLAB_DEFAULT}"  all &
bash "${EVAL_DIR}/06_07_evaluate_t2spir.sh" "${RUN_SYNTHSEG_EM}"    all &
bash "${EVAL_DIR}/06_07_evaluate_t2spir.sh" "${RUN_SYNTHSEG_NOEM}"  all &
bash "${EVAL_DIR}/06_07_evaluate_t2spir.sh" "${RUN_AUGLAB_AUG}"     all &
wait

echo ""
echo "[$(date '+%H:%M:%S')] === All evaluations done — aggregating T2spir results ==="
bash "${EVAL_DIR}/06_08_aggregate_t2spir.sh"

echo ""
echo "[$(date '+%H:%M:%S')] === Done ==="
