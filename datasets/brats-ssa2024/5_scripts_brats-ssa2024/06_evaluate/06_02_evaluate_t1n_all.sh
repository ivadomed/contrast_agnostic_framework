#!/usr/bin/env bash
# Evaluate ALL 6 brats2024-glioma T1n-TRAINED models on BraTS-SSA 2024 -- all folds,
# all 4 contrasts.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=================================================================="
echo " BraTS-SSA 2024 <- brats2024-glioma T1n-TRAINED models | evaluate 6-method suite"
echo "=================================================================="

declare -a RUNS_CATS=(
    "brats2024-glioma_t1n_baseline_20260622_044535 nnUNet"
    "brats2024-glioma_t1n_auglab_default_20260622_044535 auglab"
    "brats2024-glioma_t1n_synthseg_noEM_20260622_044535 auglab"
    "brats2024-glioma_t1n_synthseg_EM_20260622_044535 auglab"
    "brats2024-glioma_t1n_auglabAug_v26_6_2_train050_val000_20260710_040303 auglab"
    "brats2024-glioma_t1n_auglabAug_v26_6_2_train050_val100_20260725_113540 auglab"
)

for entry in "${RUNS_CATS[@]}"; do
    read -r run_id category <<< "$entry"
    echo ""
    echo ">>> [$(date '+%H:%M:%S')] ${category}/${run_id}"
    bash "${HERE}/06_01_evaluate_run.sh" "$run_id" "$category" all
done

echo ""
echo "[$(date '+%H:%M:%S')] All 6 T1n-trained methods evaluated."
