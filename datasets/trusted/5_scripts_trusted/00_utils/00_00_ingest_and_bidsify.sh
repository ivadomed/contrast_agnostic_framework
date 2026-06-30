#!/usr/bin/env bash
# Ingest the TRUSTED kidney archive (CT + US) and BIDSify it. TRUSTED is
# evaluation-only (see datasets/trusted/README.md). The archive (~15 GB zip) is
# expected to already be staged on $SCRATCH; override its path with --zip.
#   bash 00_00_ingest_and_bidsify.sh                       # extract + BIDSify
#   bash 00_00_ingest_and_bidsify.sh --skip-extract        # 0_raw already populated
#   bash 00_00_ingest_and_bidsify.sh --zip /path/to/TRUSTED_dataset_for_nsd.zip
# Heavy I/O (≈18 GB extract + SimpleITK mask binarization) → run on a compute node.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${PROJECT_ROOT}"
run_job --name trusted_ingest_bidsify --gpus 0 --cpus 8 --mem 32G --time 02:00:00 \
    --log /tmp/trusted_ingest_bidsify.log --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/00_00_ingest_and_bidsify.py" "$@"
