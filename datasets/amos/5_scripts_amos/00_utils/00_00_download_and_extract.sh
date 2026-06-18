#!/usr/bin/env bash
# Download AMOS22 labeled data (500CT+100MRI, 24 GB) from Zenodo and extract.
# Unlabeled splits are NOT downloaded. See 00_00_download_and_extract.py for details.
#   bash 00_00_download_and_extract.sh
#   bash 00_00_download_and_extract.sh --skip-download  # 0_raw already populated
set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project
source "$(dirname "$0")/env.sh"
set_slot 0 .venv/bin/python "$(dirname "$0")/00_00_download_and_extract.py" "$@"
