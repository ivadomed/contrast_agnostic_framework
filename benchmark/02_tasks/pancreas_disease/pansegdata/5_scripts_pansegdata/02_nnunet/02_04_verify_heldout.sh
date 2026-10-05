#!/bin/bash
# Prove the pansegdata test set is completely held out across BOTH contrast datasets (shared 00_00_utils/verify_heldout.py), via run_job (CPU).
# Run after every conversion / preprocessing / re-split.   bash 02_04_verify_heldout.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
mkdir -p "${HERE}/logs"
run_job --name pansegdata_verify_heldout --gpus 0 --cpus 1 --mem 4G --time 00:10:00 \
    --log "${HERE}/logs/verify_heldout_$(date +%Y%m%d_%H%M%S).log" --wait -- \
    .venv/bin/python benchmark/00_commun_scripts/00_00_utils/verify_heldout.py \
    --raw "${nnUNet_raw}" --preprocessed "${nnUNet_preprocessed}" --splits-dir "${SPLITS_DIR}" \
    --participants "${BIDS_ROOT}/participants.tsv"
