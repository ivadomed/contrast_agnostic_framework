#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash scripts/run_finetuning.sh <SLOT_ID> <VERSION> <TARGET_CONTRAST> <PRETRAINED_CKPT_PATH> <FREEZE_ENCODER>
#
# Example:
#   bash scripts/run_finetuning.sh 3 v15 t1w /path/to/checkpoint.ckpt true

SLOT_ID="${1:-3}"
VERSION="${2:-v15}"
TARGET_CONTRAST="${3:-t2w}"
PRETRAINED_CKPT_PATH="${4:-"/home/ge.polymtl.ca/pahoa/mri_synthesis_project/checkpoints/v15/segmenter/generator/t1w/run2/last.ckpt"}"
FREEZE_ENCODER="${5:-true}"

BATCH_SIZE_SEGMENTER="${6:-4}"

cd "$(dirname "$0")/.."

# Validate freeze_encoder is a boolean
if [[ "${FREEZE_ENCODER}" != "true" && "${FREEZE_ENCODER}" != "false" ]]; then
    echo "ERROR: FREEZE_ENCODER must be 'true' or 'false', got '${FREEZE_ENCODER}'"
    exit 1
fi

set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train_segmenter.py \
    task=segmenter \
    version="${VERSION}" \
    data.source_contrast="${TARGET_CONTRAST}" \
    model.segmenter.use_generator=false \
    model.segmenter.pretrained_ckpt_path="${PRETRAINED_CKPT_PATH}" \
    model.segmenter.freeze_encoder="${FREEZE_ENCODER}" \
    model.segmenter.gen_version=null \
    data.batch_size_segmenter="${BATCH_SIZE_SEGMENTER}" \
    training.limit_val_batches=1.0 \
    training.segmenter.enable_train_image_logging=false \
    training.segmenter.val_image_log_every=1 \
    training.max_epochs.segmenter=100 \
    training.lr.segmenter=5e-5
