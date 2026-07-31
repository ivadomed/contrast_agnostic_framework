#!/usr/bin/env bash
# Reorient MSD-SPLEEN BIDS + nnUNet trees from RAS -> LPS (lossless permute/flip).
# 0_raw_msd-spleen is left untouched. Small (41 CT volumes) — CPU-only.
#   bash 03_00_reorient_to_lps.sh                # fix
#   bash 03_00_reorient_to_lps.sh --dry-run       # report only
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
run_job --name msd_spleen_reorient --gpus 0 --slot 0 --mem 8G --time 00:20:00 --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/03_00_reorient_to_lps.py" "$@"
