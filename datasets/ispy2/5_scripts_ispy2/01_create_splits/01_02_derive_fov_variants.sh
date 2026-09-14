#!/usr/bin/env bash
# Derive the bilateral/unilateral FOV variants for every I-SPY2 patient.
# ~1250 volumes are decompressed, cropped and re-gzipped — far past the login-node
# ~10 CPU-min allowance, so it goes through run_job (CPU-only) like everything else.
#
#   bash 01_create_splits/01_02_derive_fov_variants.sh
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"

LOG_DIR="${DATASET_ROOT}/4_splits_ispy2/_logs"
mkdir -p "${LOG_DIR}"

run_job --name ispy2_fov_variants --gpus 0 --cpus 16 --mem 64G --time 03:00:00 \
        --log "${LOG_DIR}/01_02_derive_fov_variants.log" --wait -- \
    env ISPY2_FOV_WORKERS=16 \
        "${PROJECT_ROOT}/.venv/bin/python" \
        "$(dirname "$(readlink -f "$0")")/01_02_derive_fov_variants.py"

echo "--- log tail ---"
tail -n 40 "${LOG_DIR}/01_02_derive_fov_variants.log"
