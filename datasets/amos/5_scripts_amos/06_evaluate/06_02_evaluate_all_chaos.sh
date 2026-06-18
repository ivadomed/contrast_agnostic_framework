#!/usr/bin/env bash
# Evaluate ALL chaos-trained models on AMOS CT+MRI (5 methods × 4 folds × 4 organs).
# Each method runs sequentially; within each method all folds run in parallel.
# AMOS has no models of its own — all predictions are cross-dataset (chaos-trained).
# See 06_00_evaluate_amos.py for the chaos→AMOS label remap.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=================================================================="
echo " AMOS ← CHAOS models | evaluate 5 methods × 4 folds × CT+MRI"
echo " Organs: liver, right_kidney, left_kidney, spleen (chaos label remap)"
echo "=================================================================="

bash "${HERE}/06_01_evaluate_run.sh" nnUNet chaos_baseline_20260614_153230
bash "${HERE}/06_01_evaluate_run.sh" nnUNet chaos_v26_6_2_train090_val000_20260614_205937
bash "${HERE}/06_01_evaluate_run.sh" auglab chaos_synthseg_EM_train100_val000_20260611_120000
bash "${HERE}/06_01_evaluate_run.sh" auglab chaos_auglab_default_20260611_120000
bash "${HERE}/06_01_evaluate_run.sh" auglab chaos_synthseg_noEM_train100_val000_20260611_120000

echo ""
echo "[$(date '+%H:%M:%S')] All evaluations done."
