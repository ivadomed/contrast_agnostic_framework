#!/usr/bin/env bash

set -euo pipefail

# Ensure we're using the optimized TF32 precision under the hood via the training script.
SLOT_ID="${1:-3}"
VERSION="${2:-v5}"
CONTRAST="${3:-t2w}"

# If no specific weight provided, default to the newest generator last.ckpt.
# Prefer new run layout, then fall back to legacy versioned folders.
DEFAULT_CKPT=$(ls -t \
    checkpoints/generator/${CONTRAST}/run*/last.ckpt \
    checkpoints/${VERSION}/generator/${CONTRAST}/last.ckpt \
    checkpoints/${VERSION}/generator/${CONTRAST}/*.ckpt \
    2>/dev/null | head -n 1 || echo "")
CKPT_PATH="${4:-$DEFAULT_CKPT}"

# Example Baseline segmenter run
# set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train_segmenter.py \
#     version="${VERSION}" \
#     data.source_contrast="${CONTRAST}" \
#     model.segmenter.use_generator=false \
#     training.resume=false \
#     training.devices=1 \
#     training.precision=16-mixed

# Example Generator-augmented segmenter run
set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train_segmenter.py \
    version="${VERSION}" \
    data.source_contrast="${CONTRAST}" \
    model.segmenter.use_generator=true \
    model.segmenter.gen_version="${VERSION}" \
    model.segmenter.gen_weights="${CKPT_PATH}" \
    model.segmenter.fully_artificial=true \
    training.resume=false \
    training.devices=1 \
    training.precision=16-mixed

