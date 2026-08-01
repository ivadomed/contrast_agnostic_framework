#!/usr/bin/env bash
# Evaluate ALL 7 chaos T2spir-trained runs on KIDNEY-T2W MRI (folds 0-2 parallel per run).
# Kidney only (this GT annotates kidney alone, both sides merged). See
# 06_01_evaluate_run.sh for single-run usage.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=================================================================="
echo " KIDNEY-T2W <- CHAOS T2spir models | evaluate 7 runs x folds 0-2 | kidney"
echo "=================================================================="

CATEGORY=nnUNet bash "${HERE}/06_01_evaluate_run.sh" chaos_t2spir_baseline_20260620_111146 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t2spir_auglab_default_20260620_112240 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t2spir_synthseg_noEM_20260620_112515 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t2spir_synthseg_EM_20260620_112357 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t2spir_srcsm_20260709_121945 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t2spir_auglabAug_v26_6_2_train050_val000_20260710_054202 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t2spir_auglabAug_v26_6_2_train050_val100_20260723_194053 &
wait

echo ""
echo "[$(date '+%H:%M:%S')] All T2spir evaluations done."
