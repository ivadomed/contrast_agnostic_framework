#!/usr/bin/env bash
# Evaluate ALL chaos-trained models on SLIVER07 CT (all methods, all folds).
# Each method runs sequentially; within each method all 4 folds run in parallel.
# See 06_01_evaluate_run.sh for single-run usage.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=================================================================="
echo " SLIVER07 ← CHAOS models | evaluate 5 methods × 4 folds | liver"
echo "=================================================================="

bash "${HERE}/06_01_evaluate_run.sh" nnUNet chaos_t1in_baseline_20260614_153230
bash "${HERE}/06_01_evaluate_run.sh" nnUNet chaos_t1in_v26_6_2_train090_val000_20260614_205937
bash "${HERE}/06_01_evaluate_run.sh" auglab chaos_t1in_synthseg_EM_train100_val000_20260611_120000
bash "${HERE}/06_01_evaluate_run.sh" auglab chaos_t1in_auglab_default_20260611_120000
bash "${HERE}/06_01_evaluate_run.sh" auglab chaos_t1in_synthseg_noEM_train100_val000_20260611_120000

echo ""
echo "[$(date '+%H:%M:%S')] All evaluations done."
