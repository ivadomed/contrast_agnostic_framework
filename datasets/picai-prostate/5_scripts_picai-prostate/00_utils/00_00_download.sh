#!/usr/bin/env bash
# Download the PI-CAI public training/development data (images from Zenodo record
# 6624726, labels from github.com/DIAGNijmegen/picai_labels) into 0_raw_picai-prostate/.
#
# RUNS ON THE LOGIN NODE, NOT THROUGH run_job — deliberate. tamia's compute nodes have no
# outbound network, so a submitted job cannot download anything; and this step is pure
# network I/O (no CPU, <1 GB RAM), which is inside CLAUDE.md's login-node exception.
# (amos/cirrmri-liver do wrap their downloads in run_job because those clusters' compute
# nodes can reach the internet — do not "fix" this one to match them.)
#
# Usage:
#   bash 00_00_download.sh                 # labels + all 5 image folds (~27 GB)
#   bash 00_00_download.sh --labels-only   # refresh just picai_labels
#   bash 00_00_download.sh --skip-extract  # download zips but do not unzip
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
[ -f "${PROJECT_ROOT}/scripts/cluster/tamia_env_picai.sh" ] && [ -d /scratch/p/paulh ] && \
    source "${PROJECT_ROOT}/scripts/cluster/tamia_env_picai.sh"
cd "${PROJECT_ROOT}"

echo "[download] RAW_ROOT=${RAW_ROOT}"
exec .venv/bin/python "${SCRIPT_DIR}/00_00_download.py" "$@"
