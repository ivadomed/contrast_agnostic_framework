#!/usr/bin/env bash
# BIDSify the ATLAS challenge raw archive (0_raw already extracted). Pure
# directory-layout consistency -- nothing in the pipeline reads from 1_BIDS_atlas-liver-hcc,
# 02_00_convert.py reads directly from 0_raw. See 00_01_bidsify.py docstring.
#   bash 00_01_bidsify.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${PROJECT_ROOT}"
run_job --name atlashcc_bidsify --gpus 0 --slot 0 --mem 8G --time 00:30:00 \
    --log "${PROJECT_ROOT}/datasets/atlas-liver-hcc/0_raw_atlas-liver-hcc/_bidsify.log" --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/00_01_bidsify.py"
