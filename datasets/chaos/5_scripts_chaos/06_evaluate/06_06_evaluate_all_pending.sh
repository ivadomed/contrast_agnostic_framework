#!/usr/bin/env bash
# Evaluate all CHAOS runs that have predictions but missing/incomplete metrics.
# Runs 6 new experiments + re-evaluates 5 runs that gained new folds.
# All eval jobs run in parallel (CPU-only, no GPU needed).
#
# Fully-evaluated runs (4/4 folds) are skipped to save time:
#   baseline, v26_6_2_050_val000, v26_6_2_050_val100, v26_6_2_090_val000,
#   auglab_default, synthseg_EM_val000, synthseg_noEM
#
# After all evals complete, re-runs 06_02_aggregate_results.sh.

set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project
EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[$(date '+%H:%M:%S')] Evaluating all pending CHAOS runs"

# ── New runs (no metrics yet) — evaluate all 4 folds ────────────────────────
bash "${EVAL_DIR}/06_01_evaluate_run.sh" \
    "chaos_t1in_v26_6_2_train025_val100_20260616_200514" all &

bash "${EVAL_DIR}/06_01_evaluate_run.sh" \
    "chaos_t1in_v26_6_2_train090_val100_20260615_213615" all &

bash "${EVAL_DIR}/06_01_evaluate_run.sh" \
    "chaos_t1in_auglabAug_v26_6_2_train025_val100_20260616_200514" all &

bash "${EVAL_DIR}/06_01_evaluate_run.sh" \
    "chaos_t1in_auglabAug_v26_6_2_train050_val100_20260616_112420" all &

bash "${EVAL_DIR}/06_01_evaluate_run.sh" \
    "chaos_t1in_auglabAug_v26_6_2_train090_val000_20260616_200514" all &

bash "${EVAL_DIR}/06_01_evaluate_run.sh" \
    "chaos_t1in_auglabAug_v26_6_2_train090_val100_20260616_200514" all &

bash "${EVAL_DIR}/06_01_evaluate_run.sh" \
    "chaos_t1in_synthseg_EM_train100_val100_20260616_112420" all &

# ── Partial runs — only evaluate the new folds ──────────────────────────────
# v26_6_2_025_val000: had folds 0,1 → now adding 2,3
SLOT=2 bash "${EVAL_DIR}/06_01_evaluate_run.sh" \
    "chaos_t1in_v26_6_2_train025_val000_20260616_010628" 2 &
SLOT=3 bash "${EVAL_DIR}/06_01_evaluate_run.sh" \
    "chaos_t1in_v26_6_2_train025_val000_20260616_010628" 3 &

# auglabAug_synthseg_EM: had folds 0,1 → now adding 2,3
SLOT=2 bash "${EVAL_DIR}/06_01_evaluate_run.sh" \
    "chaos_t1in_auglabAug_synthseg_EM_train100_val000_20260615_213615" 2 &
SLOT=3 bash "${EVAL_DIR}/06_01_evaluate_run.sh" \
    "chaos_t1in_auglabAug_synthseg_EM_train100_val000_20260615_213615" 3 &

# auglabAug_025_val000: had folds 0,1 → now adding 2,3
SLOT=2 bash "${EVAL_DIR}/06_01_evaluate_run.sh" \
    "chaos_t1in_auglabAug_v26_6_2_train025_val000_20260616_010628" 2 &
SLOT=3 bash "${EVAL_DIR}/06_01_evaluate_run.sh" \
    "chaos_t1in_auglabAug_v26_6_2_train025_val000_20260616_010628" 3 &

# auglabAug_050_val000: had fold 0 → now adding 1,2,3
SLOT=1 bash "${EVAL_DIR}/06_01_evaluate_run.sh" \
    "chaos_t1in_auglabAug_v26_6_2_train050_val000_20260615_213615" 1 &
SLOT=2 bash "${EVAL_DIR}/06_01_evaluate_run.sh" \
    "chaos_t1in_auglabAug_v26_6_2_train050_val000_20260615_213615" 2 &
SLOT=3 bash "${EVAL_DIR}/06_01_evaluate_run.sh" \
    "chaos_t1in_auglabAug_v26_6_2_train050_val000_20260615_213615" 3 &

wait
echo "[$(date '+%H:%M:%S')] All evaluations done — running aggregation"

bash "${EVAL_DIR}/06_02_aggregate_results.sh"
echo "[$(date '+%H:%M:%S')] Aggregation complete"
