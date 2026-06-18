#!/usr/bin/env bash
# BIDSify AMOS22: reorganise 0_raw_amos/amos22/ into 1_BIDS_amos/amos-abdominal/.
# Run AFTER 00_00_download_and_extract.py (raw data must already be present).
#
# Usage:
#   bash 00_01_bidsify.sh           # BIDSify all labeled cases
#   bash 00_01_bidsify.sh --skip-bids
set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project
source "$(dirname "$0")/env.sh"
set_slot 0 .venv/bin/python "$(dirname "$0")/00_01_bidsify.py" "$@"
