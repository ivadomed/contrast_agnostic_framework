#!/usr/bin/env bash
# Download CHAOS, copy raw (train only), and DICOM→NIfTI BIDSify.
# Usage: bash 00_00_download_and_bidsify.sh [--skip-copy] [--skip-bids]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
REPO_ROOT="$(cd "${DATASET_ROOT}/../.." && pwd)"
SCRIPT="$(dirname "${BASH_SOURCE[0]}")/00_00_download_and_bidsify.py"

set_slot 0 "${REPO_ROOT}/.venv/bin/python" "${SCRIPT}" "$@"
