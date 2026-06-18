#!/usr/bin/env bash
# Download SLIVER07 (labeled half) from Zenodo and BIDSify the .mhd/.raw volumes.
# SLIVER07 is evaluation-only (see datasets/sliver07/README.md).
#   bash 00_00_download_and_bidsify.sh                  # download + BIDSify
#   bash 00_00_download_and_bidsify.sh --skip-download  # 0_raw already populated
set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project
source "$(dirname "$0")/env.sh"
set_slot 0 .venv/bin/python "$(dirname "$0")/00_00_download_and_bidsify.py" "$@"
