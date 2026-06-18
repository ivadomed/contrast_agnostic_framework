#!/usr/bin/env bash
# Predict folds 2,3 for the two now-4-fold runs, evaluate all folds, aggregate.
#
#   v26_6_2_train050_val000  (train 50% / val 0%)   folds 2,3  → slots 0,1
#   chaos_v26_6_2_train050_val100  (train 50% / val 100%) folds 2,3  → slots 2,3
#
# Folds 0,1 already predicted/evaluated; "all" re-evals them idempotently so the
# aggregate reflects the full 4-fold runs. This is the clean val-synth comparison.
#
# Usage:
#   bash 06_05_predict_eval_batch3.sh [RUN_train050_val000] [RUN_v26_6_2_50_100]

set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project
source datasets/chaos/5_scripts_chaos/00_utils/env.sh

RUN_T050="${1:-chaos_v26_6_2_train050_val000_20260616_010628}"
RUN_50_100="${2:-chaos_v26_6_2_train050_val100_20260615_213615}"

PD="$(cd datasets/chaos/5_scripts_chaos/05_predict && pwd)"
ED="$(cd datasets/chaos/5_scripts_chaos/06_evaluate && pwd)"

echo "[$(date '+%H:%M:%S')] === PREDICT folds 2,3 (4 GPUs) ==="
SLOT=0 GPU=0 bash "${PD}/05_10_predict_v26_6_2_train050_val000.sh" "${RUN_T050}"   2 &  P0=$!
SLOT=1 GPU=1 bash "${PD}/05_10_predict_v26_6_2_train050_val000.sh" "${RUN_T050}"   3 &  P1=$!
SLOT=2 GPU=2 bash "${PD}/05_07_predict_v26_6_2_50_100.sh"          "${RUN_50_100}" 2 &  P2=$!
SLOT=3 GPU=3 bash "${PD}/05_07_predict_v26_6_2_50_100.sh"          "${RUN_50_100}" 3 &  P3=$!
wait $P0 $P1 $P2 $P3
echo "[$(date '+%H:%M:%S')] predictions done."

echo "[$(date '+%H:%M:%S')] === EVALUATE (all folds) ==="
bash "${ED}/06_01_evaluate_run.sh" "${RUN_T050}"   all &  E0=$!
bash "${ED}/06_01_evaluate_run.sh" "${RUN_50_100}" all &  E1=$!
wait $E0 $E1
echo "[$(date '+%H:%M:%S')] evaluations done."

echo "[$(date '+%H:%M:%S')] === AGGREGATE ==="
bash "${ED}/06_02_aggregate_results.sh"
echo "[$(date '+%H:%M:%S')] done → ${METRICS_ROOT}/02_00_aggregated_metrics.md"
