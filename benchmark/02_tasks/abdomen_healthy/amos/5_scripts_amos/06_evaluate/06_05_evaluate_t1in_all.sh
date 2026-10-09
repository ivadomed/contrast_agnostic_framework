#!/usr/bin/env bash
# Evaluate ALL 6 chaos T1in-trained models on AMOS CT+MRI (4 folds parallel per run).
# Organs: liver, right_kidney, left_kidney, spleen (chaos→AMOS label remap applied).
# See 06_01_evaluate_run.sh for single-run usage.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=================================================================="
echo " AMOS <- CHAOS T1in models | evaluate 6 methods x 4 folds x CT+MRI"
echo " Organs: liver, right_kidney, left_kidney, spleen"
echo "=================================================================="

CATEGORY=nnUNet bash "${HERE}/06_01_evaluate_run.sh" chaos_t1in_baseline_20260614_153230 &
CATEGORY=nnUNet bash "${HERE}/06_01_evaluate_run.sh" chaos_t1in_v26_6_2_train050_val100_20260615_213615 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t1in_synthseg_EM_train100_val000_20260611_120000 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t1in_synthseg_noEM_train100_val000_20260611_120000 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t1in_auglab_default_20260611_120000 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t1in_auglabAug_v26_6_2_train025_val100_20260616_200514 &
wait

echo ""
echo "[$(date '+%H:%M:%S')] All T1in evaluations done."
