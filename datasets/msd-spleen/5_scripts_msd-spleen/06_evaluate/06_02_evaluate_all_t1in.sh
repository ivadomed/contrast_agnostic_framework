#!/usr/bin/env bash
# Evaluate ALL 8 chaos T1in-trained runs on MSD-SPLEEN CT (folds 0-2 parallel per run).
# Spleen only (MSD-SPLEEN GT annotates spleen alone). See 06_01_evaluate_run.sh for
# single-run usage.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=================================================================="
echo " MSD-SPLEEN <- CHAOS T1in models | evaluate 8 runs x folds 0-2 | spleen"
echo "=================================================================="

CATEGORY=nnUNet bash "${HERE}/06_01_evaluate_run.sh" chaos_t1in_baseline_20260614_153230 &
CATEGORY=nnUNet bash "${HERE}/06_01_evaluate_run.sh" chaos_t1in_v26_6_2_train050_val100_20260615_213615 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t1in_synthseg_EM_train100_val000_20260611_120000 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t1in_auglab_default_20260611_120000 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t1in_synthseg_noEM_train100_val000_20260611_120000 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t1in_srcsm_20260710_011817 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t1in_auglabAug_v26_6_2_train050_val000_20260615_213615 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t1in_auglabAug_v26_6_2_train050_val100_20260616_112420 &
wait

echo ""
echo "[$(date '+%H:%M:%S')] All T1in evaluations done."
