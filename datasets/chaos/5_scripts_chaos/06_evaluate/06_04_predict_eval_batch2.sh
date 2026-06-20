#!/usr/bin/env bash
# Predict, evaluate, and aggregate the batch-2 runs.
#
#   NEW runs (predict folds 0 AND 1):
#     v26_6_2_train050_val000_<TS>            (nnUNet)  slot 0
#     v26_6_2_train025_val000_<TS>            (nnUNet)  slot 1
#     auglabAug_v26_6_2_train025_val000_<TS>  (auglab)  slot 2
#   EXISTING runs (predict fold 1 only — fold 0 already predicted):
#     chaos_t1in_auglabAug_synthseg_EM_train100_val000_20260615_213615   (auglab)  slot 3
#     chaos_t1in_v26_6_2_train050_val100_20260615_213615                 (nnUNet)  slot 3
#
# Evaluation uses "all" (skips absent folds 2,3; re-evals existing fold 0 idempotently),
# then a single global aggregate.
#
# Usage:
#   bash 06_04_predict_eval_batch2.sh <RUN_train050> <RUN_train025> <RUN_auglab_v26_train025> \
#        [EXISTING_auglabAug_synthseg_EM_train100_val000] [EXISTING_v26_6_2_train050_val100]

set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project
source datasets/chaos/5_scripts_chaos/00_utils/env.sh

RUN_T050="${1:?RUN_train050_val000 required}"
RUN_T025="${2:?RUN_train025_val000 required}"
RUN_AUG_T025="${3:?RUN_auglabAug_v26_6_2_train025_val000 required}"
EXIST_EM="${4:-chaos_t1in_auglabAug_synthseg_EM_train100_val000_20260615_213615}"
EXIST_50="${5:-chaos_t1in_v26_6_2_train050_val100_20260615_213615}"

PD="$(cd datasets/chaos/5_scripts_chaos/05_predict && pwd)"
ED="$(cd datasets/chaos/5_scripts_chaos/06_evaluate && pwd)"

echo "[$(date '+%H:%M:%S')] === PREDICT (4 GPUs) ==="
# slot 0 — train050 folds 0,1
( SLOT=0 GPU=0 bash "${PD}/05_10_predict_v26_6_2_train050_val000.sh" "${RUN_T050}" 0
  SLOT=0 GPU=0 bash "${PD}/05_10_predict_v26_6_2_train050_val000.sh" "${RUN_T050}" 1 ) &  J0=$!
# slot 1 — train025 folds 0,1
( SLOT=1 GPU=1 bash "${PD}/05_11_predict_v26_6_2_train025_val000.sh" "${RUN_T025}" 0
  SLOT=1 GPU=1 bash "${PD}/05_11_predict_v26_6_2_train025_val000.sh" "${RUN_T025}" 1 ) &  J1=$!
# slot 2 — auglab_v26_train025 folds 0,1
( SLOT=2 GPU=2 bash "${PD}/05_12_predict_auglabAug_v26_6_2_train025_val000.sh" "${RUN_AUG_T025}" 0
  SLOT=2 GPU=2 bash "${PD}/05_12_predict_auglabAug_v26_6_2_train025_val000.sh" "${RUN_AUG_T025}" 1 ) &  J2=$!
# slot 3 — fold-1 additions to existing runs
( SLOT=3 GPU=3 bash "${PD}/05_08_predict_auglabAug_synthseg_EM.sh" "${EXIST_EM}" 1
  SLOT=3 GPU=3 bash "${PD}/05_07_predict_v26_6_2_50_100.sh"        "${EXIST_50}" 1 ) &  J3=$!
wait $J0 $J1 $J2 $J3
echo "[$(date '+%H:%M:%S')] predictions done."

echo "[$(date '+%H:%M:%S')] === EVALUATE (all present folds) ==="
bash "${ED}/06_01_evaluate_run.sh" "${RUN_T050}"     all &  E0=$!
bash "${ED}/06_01_evaluate_run.sh" "${RUN_T025}"     all &  E1=$!
bash "${ED}/06_01_evaluate_run.sh" "${RUN_AUG_T025}" all &  E2=$!
wait $E0 $E1 $E2
bash "${ED}/06_01_evaluate_run.sh" "${EXIST_EM}" all
bash "${ED}/06_01_evaluate_run.sh" "${EXIST_50}" all
echo "[$(date '+%H:%M:%S')] evaluations done."

echo "[$(date '+%H:%M:%S')] === AGGREGATE ==="
bash "${ED}/06_02_aggregate_results.sh"
echo "[$(date '+%H:%M:%S')] done → ${METRICS_ROOT}/02_00_aggregated_metrics.md"
