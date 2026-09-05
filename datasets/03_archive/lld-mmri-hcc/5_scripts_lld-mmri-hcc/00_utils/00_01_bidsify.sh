#!/usr/bin/env bash
# BIDSify the LLD-MMRI-HCC raw archive (0_raw already downloaded+filtered to 157 HCC
# patients). See 00_01_bidsify.py docstring.
#   bash 00_01_bidsify.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${PROJECT_ROOT}"
run_job --name lldmmrihcc_bidsify --gpus 0 --slot 0 --mem 8G --time 00:20:00 \
    --log "${PROJECT_ROOT}/datasets/lld-mmri-hcc/0_raw_lld-mmri-hcc/_bidsify.log" --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/00_01_bidsify.py"
