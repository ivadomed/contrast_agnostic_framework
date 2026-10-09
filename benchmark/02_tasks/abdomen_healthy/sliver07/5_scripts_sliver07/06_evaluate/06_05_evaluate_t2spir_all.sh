#!/usr/bin/env bash
# Evaluate ALL 6 chaos T2spir-trained models on SLIVER07 CT (4 folds parallel per run).
# Pre-exports CHAOS_TRAINING_CONTRAST=t2spir so env.sh routes metrics to chaos_model/t2spir/.
# Liver only (SLIVER07 GT annotates liver alone).
# See 06_01_evaluate_run.sh for single-run usage.
set -euo pipefail
export CHAOS_TRAINING_CONTRAST="t2spir"
export CHAOS_DATASET_ID="61"
export CHAOS_DS_NAME="Dataset061_CHAOS_MR_T2spir"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=================================================================="
echo " SLIVER07 <- CHAOS T2spir models | evaluate 6 methods x 4 folds | liver"
echo "=================================================================="

CATEGORY=nnUNet bash "${HERE}/06_01_evaluate_run.sh" chaos_t2spir_baseline_20260620_111146 &
CATEGORY=nnUNet bash "${HERE}/06_01_evaluate_run.sh" chaos_t2spir_v26_6_2_train050_val100_20260620_112122 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t2spir_synthseg_EM_20260620_112357 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t2spir_synthseg_noEM_20260620_112515 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t2spir_auglab_default_20260620_112240 &
CATEGORY=auglab bash "${HERE}/06_01_evaluate_run.sh" chaos_t2spir_auglabAug_v26_6_2_train025_val100_20260620_112633 &
wait

echo ""
echo "[$(date '+%H:%M:%S')] All T2spir evaluations done."
