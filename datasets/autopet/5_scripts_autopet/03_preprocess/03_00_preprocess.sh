#!/usr/bin/env bash
# nnU-Net plan-and-preprocess for both autopet per-modality datasets (CT=120, PET=121).
# GPU-adjacent CPU work (resampling, fingerprinting) — run via run_job, not the login
# node (whole-body PET/CT volumes are large; do not assume this is login-node-light
# without checking).
#
# Usage: bash 03_00_preprocess.sh
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

for DATASET_ID in 120 121; do
    run_job --name "autopet_preprocess_${DATASET_ID}" --gpus 0 --cpus 16 --mem 64G \
        --time 08:00:00 --log "${RESULTS_DIR}/_logs/preprocess_${DATASET_ID}.log" -- \
        .venv/bin/nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -c 3d_fullres --verify_dataset_integrity
done
