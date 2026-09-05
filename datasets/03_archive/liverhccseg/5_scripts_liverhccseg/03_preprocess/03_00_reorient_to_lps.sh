#!/usr/bin/env bash
# Reorient LiverHccSeg BIDS + nnUNet trees to LPS (lossless permute/flip). See
# 03_00_reorient_to_lps.py docstring — 13/14 cases were (R,A,S), 1/14 (L,A,S).
# 0_raw_liverhccseg is left untouched. Small (14 cases x 4 items) — CPU-only.
#   bash 03_00_reorient_to_lps.sh                # fix
#   bash 03_00_reorient_to_lps.sh --dry-run       # report only
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
run_job --name liverhccseg_reorient --gpus 0 --slot 0 --mem 8G --time 00:20:00 --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/03_00_reorient_to_lps.py" "$@"
