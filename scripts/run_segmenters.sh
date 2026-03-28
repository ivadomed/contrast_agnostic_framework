#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash scripts/run_segmenters.sh [slot] [contrast] [version] [batch_size_segmenter]
# Example:
#   bash scripts/run_segmenters.sh 1 t1w v13 4

SLOT_ID="${1:-3}"
VERSION="${2:-v13}"
CONTRAST="${3:-t2w}"

BATCH_SIZE_SEGMENTER="${4:-4}"

cd "$(dirname "$0")/.."

set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
    task=segmenter \
    data.source_contrast="${CONTRAST}" \
    model.generator.gen_version="${VERSION}" \
    data.batch_size_segmenter="${BATCH_SIZE_SEGMENTER}" \
    training.limit_val_batches=1.0 \
    training.segmenter.enable_train_image_logging=false \
    training.segmenter.val_image_log_every=1

