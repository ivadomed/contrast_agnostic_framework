#!/usr/bin/env bash

set -euo pipefail

# Ensure we're using the optimized TF32 precision under the hood via the training script.
SLOT_ID="${1:-3}"
VERSION="${2:-v5}"
CONTRAST="${3:-t2w}"

# Example Generator-augmented segmenter run
set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train_segmenter.py \
    version="${VERSION}" \
    data.source_contrast="${CONTRAST}" \
    model.segmenter.use_generator=true \
    model.segmenter.gen_version="v4" \
    model.segmenter.gen_weights="checkpoints/v4/mri_generator_t2w_epoch_30.pth" \
    model.segmenter.fully_artificial=true \
    training.resume=false \
    training.devices=1 \
    training.precision=16-mixed