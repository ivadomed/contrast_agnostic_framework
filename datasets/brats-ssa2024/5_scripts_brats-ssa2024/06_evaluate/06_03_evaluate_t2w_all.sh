#!/usr/bin/env bash
# Evaluate ALL 6 brats2024-glioma T2w-TRAINED models on BraTS-SSA 2024 -- all folds,
# all 4 contrasts. Pre-exports BRATS_TRAINING_CONTRAST=t2w so env.sh routes metrics to
# brats2024_glioma_model/t2w/ (guarded ${:-} exports in env.sh preserve this through the
# re-source 06_01_evaluate_run.sh's own env.sh sourcing does).
set -euo pipefail
export BRATS_TRAINING_CONTRAST="t2w"
export BRATS_DATASET_ID="052"
export BRATS_DS_NAME="Dataset052_BraTS2024GliomaT2w"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=================================================================="
echo " BraTS-SSA 2024 <- brats2024-glioma T2w-TRAINED models | evaluate 6-method suite"
echo "=================================================================="

declare -a RUNS_CATS=(
    "brats2024-glioma_t2w_baseline_20260620_125115 nnUNet"
    "brats2024-glioma_t2w_auglab_default_20260620_125306 auglab"
    "brats2024-glioma_t2w_synthseg_noEM_20260620_125442 auglab"
    "brats2024-glioma_t2w_synthseg_EM_20260620_125354 auglab"
    "brats2024-glioma_t2w_auglabAug_v26_6_2_train050_val000_20260725_113540 auglab"
    "brats2024-glioma_t2w_auglabAug_v26_6_2_train050_val100_20260725_113540 auglab"
)

for entry in "${RUNS_CATS[@]}"; do
    read -r run_id category <<< "$entry"
    echo ""
    echo ">>> [$(date '+%H:%M:%S')] ${category}/${run_id}"
    bash "${HERE}/06_01_evaluate_run.sh" "$run_id" "$category" all
done

echo ""
echo "[$(date '+%H:%M:%S')] All 6 T2w-trained methods evaluated."
