#!/usr/bin/env bash
# BIDSify the LiverHccSeg raw archive (0_raw already downloaded+extracted). See
# 00_01_bidsify.py docstring.
#   bash 00_01_bidsify.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${PROJECT_ROOT}"
run_job --name liverhccseg_bidsify --gpus 0 --slot 0 --mem 8G --time 00:20:00 \
    --log "${PROJECT_ROOT}/datasets/liverhccseg/0_raw_liverhccseg/_bidsify.log" --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/00_01_bidsify.py"
