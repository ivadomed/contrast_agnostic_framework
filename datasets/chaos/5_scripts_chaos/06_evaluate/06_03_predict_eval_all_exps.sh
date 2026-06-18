#!/usr/bin/env bash
# Predict, evaluate, and aggregate all 4 experiment runs (fold 0 each).
#
# Usage:
#   bash 06_03_predict_eval_all_exps.sh \
#       <RUN_v26_6_2> <RUN_v26_6_2_50_100> <RUN_auglabAug_synthseg_EM> <RUN_auglabAug_v26_6_2_train050_val000>
#
# Example:
#   bash 06_03_predict_eval_all_exps.sh \
#       chaos_v26_6_2_train090_val100_20260615_213615 \
#       chaos_v26_6_2_train050_val100_20260615_213615 \
#       chaos_auglabAug_synthseg_EM_train100_val000_20260615_213615 \
#       chaos_auglabAug_v26_6_2_train050_val000_20260615_213615

set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project
source datasets/chaos/5_scripts_chaos/00_utils/env.sh

RUN_V26_6_2="${1:?RUN_v26_6_2 required}"
RUN_V26_6_2_50="${2:?RUN_v26_6_2_50_100 required}"
RUN_AUGLAB_EM="${3:?RUN_auglabAug_synthseg_EM required}"
RUN_AUGLAB_V26="${4:?RUN_auglabAug_v26_6_2_train050_val000 required}"

PREDICT_DIR="$(cd datasets/chaos/5_scripts_chaos/05_predict && pwd)"
EVAL_DIR="$(cd datasets/chaos/5_scripts_chaos/06_evaluate && pwd)"

# --- Predict all 4 in parallel (fold 0, all modalities) ---
echo "[$(date '+%H:%M:%S')] Predicting all 4 runs on slots 0-3..."

SLOT=0 GPU=0 bash "${PREDICT_DIR}/05_03_predict_v26_6_2.sh"               "${RUN_V26_6_2}"    0 &  P0=$!
SLOT=1 GPU=1 bash "${PREDICT_DIR}/05_07_predict_v26_6_2_50_100.sh"        "${RUN_V26_6_2_50}" 0 &  P1=$!
SLOT=2 GPU=2 bash "${PREDICT_DIR}/05_08_predict_auglabAug_synthseg_EM.sh" "${RUN_AUGLAB_EM}"  0 &  P2=$!
SLOT=3 GPU=3 bash "${PREDICT_DIR}/05_09_predict_auglabAug_v26_6_2_train050_val000.sh" "${RUN_AUGLAB_V26}" 0 &  P3=$!
wait $P0 $P1 $P2 $P3
echo "[$(date '+%H:%M:%S')] All predictions done."

# --- Evaluate all 4 in parallel (fold 0 each) ---
echo "[$(date '+%H:%M:%S')] Evaluating all 4 runs..."
bash "${EVAL_DIR}/06_01_evaluate_run.sh" "${RUN_V26_6_2}"    0 &  E0=$!
bash "${EVAL_DIR}/06_01_evaluate_run.sh" "${RUN_V26_6_2_50}" 0 &  E1=$!
bash "${EVAL_DIR}/06_01_evaluate_run.sh" "${RUN_AUGLAB_EM}"  0 &  E2=$!
bash "${EVAL_DIR}/06_01_evaluate_run.sh" "${RUN_AUGLAB_V26}" 0 &  E3=$!
wait $E0 $E1 $E2 $E3
echo "[$(date '+%H:%M:%S')] All evaluations done."

# --- Aggregate ---
echo "[$(date '+%H:%M:%S')] Aggregating results..."
bash "${EVAL_DIR}/06_02_aggregate_results.sh"
echo "[$(date '+%H:%M:%S')] Done. See ${METRICS_ROOT}/02_00_aggregated_metrics.md"
