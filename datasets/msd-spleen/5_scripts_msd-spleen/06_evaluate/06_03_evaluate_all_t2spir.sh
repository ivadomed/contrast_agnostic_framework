#!/usr/bin/env bash
# Evaluate ALL 8 chaos T2spir-trained runs on MSD-SPLEEN CT (folds 0-2 parallel per run).
# Spleen only (MSD-SPLEEN GT annotates spleen alone). See 06_01_evaluate_run.sh for
# single-run usage.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=================================================================="
echo " MSD-SPLEEN <- CHAOS T2spir models | evaluate 8 runs x folds 0-2 | spleen"
echo "=================================================================="

CATEGORY=nnUNet bash "${HERE}/06_01_evaluate_run.sh" chaos_t2spir_baseline_20260620_111146 &
CATEGORY=nnUNet bash "${HERE}/06_01_evaluate_run.sh" chaos_t2spir_v26_6_2_train050_val100_20260620_112122 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t2spir_synthseg_EM_20260620_112357 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t2spir_auglab_default_20260620_112240 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t2spir_synthseg_noEM_20260620_112515 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t2spir_srcsm_20260709_121945 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t2spir_auglabAug_v26_6_2_train050_val000_20260710_054202 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t2spir_auglabAug_v26_6_2_train050_val100_20260723_194053 &
wait

echo ""
echo "[$(date '+%H:%M:%S')] All T2spir evaluations done."
