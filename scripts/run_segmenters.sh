#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash scripts/run_segmenters.sh [slot] [contrast] [version] [batch_size_segmenter]
# Example:
#   bash scripts/run_segmenters.sh 1 t1w v13 4

SLOT_ID="${1:-3}"
VERSION="${2:-v13}"
CONTRAST="${3:-t2w}"
SEGMENTER_USE_GENERATOR="${SEGMENTER_USE_GENERATOR:-true}"

BATCH_SIZE_SEGMENTER="${4:-4}"
if [[ "${VERSION}" == "v16_bigaug" && "${4:-}" == "" ]]; then
    BATCH_SIZE_SEGMENTER="8"
fi

cd "$(dirname "$0")/.."

if [[ "${VERSION}" == "v16_bigaug" ]]; then
    set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
        task=segmenter \
        version="${VERSION}" \
        model=bigaug \
        data.source_contrast="${CONTRAST}" \
        data.batch_size_segmenter="${BATCH_SIZE_SEGMENTER}" \
        training.limit_val_batches=1.0 \
        training.segmenter.enable_train_image_logging=false \
        training.segmenter.val_image_log_every=1
elif [[ "${VERSION}" == "v17_lpci" ]]; then
    set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
        task=segmenter \
        version="${VERSION}" \
        model=v17_lpci \
        data.source_contrast="${CONTRAST}" \
        data.batch_size_segmenter="${BATCH_SIZE_SEGMENTER}" \
        training.limit_val_batches=1.0 \
        training.segmenter.enable_train_image_logging=false \
        training.segmenter.val_image_log_every=1
elif [[ "${VERSION}" == "v17_micro_anchor" ]]; then
    set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
        task=segmenter \
        version="${VERSION}" \
        model=v17 \
        data.source_contrast="${CONTRAST}" \
        data.batch_size_segmenter="${BATCH_SIZE_SEGMENTER}" \
        training.limit_val_batches=1.0 \
        training.segmenter.enable_train_image_logging=false \
        training.segmenter.val_image_log_every=1
elif [[ "${VERSION}" == "v18" ]]; then
    set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
        task=segmenter \
        version="${VERSION}" \
        model=v18 \
        data.source_contrast="${CONTRAST}" \
        data.batch_size_segmenter="${BATCH_SIZE_SEGMENTER}" \
        training.limit_val_batches=1.0 \
        training.segmenter.enable_train_image_logging=false \
        training.segmenter.val_image_log_every=1
elif [[ "${VERSION}" == "v18_1" ]]; then
    set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
        task=segmenter \
        version="${VERSION}" \
        model=v18_1 \
        data.source_contrast="${CONTRAST}" \
        data.batch_size_segmenter="${BATCH_SIZE_SEGMENTER}" \
        training.limit_val_batches=1.0 \
        training.segmenter.enable_train_image_logging=false \
        training.segmenter.val_image_log_every=1
elif [[ "${VERSION}" == "v18_2" ]]; then
    set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
        task=segmenter \
        version="${VERSION}" \
        model=v18_2 \
        data.source_contrast="${CONTRAST}" \
        data.batch_size_segmenter="${BATCH_SIZE_SEGMENTER}" \
        training.limit_val_batches=1.0 \
        training.segmenter.enable_train_image_logging=false \
        training.segmenter.val_image_log_every=1
elif [[ "${VERSION}" == "v18_3" ]]; then
    set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
        task=segmenter \
        version="${VERSION}" \
        model=v18_3 \
        data.source_contrast="${CONTRAST}" \
        data.batch_size_segmenter="${BATCH_SIZE_SEGMENTER}" \
        training.limit_val_batches=1.0 \
        training.segmenter.enable_train_image_logging=false \
        training.segmenter.val_image_log_every=1
elif [[ "${VERSION}" == "v18_4" ]]; then
    set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
        task=segmenter \
        version="${VERSION}" \
        model=v18_4 \
        data.source_contrast="${CONTRAST}" \
        data.batch_size_segmenter="${BATCH_SIZE_SEGMENTER}" \
        training.limit_val_batches=1.0 \
        training.segmenter.enable_train_image_logging=false \
        training.segmenter.val_image_log_every=1
elif [[ "${VERSION}" == "v18_5" ]]; then
    set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
        task=segmenter \
        version="${VERSION}" \
        model=v18_5 \
        data.source_contrast="${CONTRAST}" \
        data.batch_size_segmenter="${BATCH_SIZE_SEGMENTER}" \
        training.limit_val_batches=1.0 \
        training.segmenter.enable_train_image_logging=false \
        training.segmenter.val_image_log_every=1
elif [[ "${VERSION}" == "v18_6" ]]; then
    set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
        task=segmenter \
        version="${VERSION}" \
        model=v18_6 \
        data.source_contrast="${CONTRAST}" \
        data.batch_size_segmenter="${BATCH_SIZE_SEGMENTER}" \
        training.limit_val_batches=1.0 \
        training.segmenter.enable_train_image_logging=false \
        training.segmenter.val_image_log_every=1
elif [[ "${VERSION}" == "v18_7" ]]; then
    set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
        task=segmenter \
        version="${VERSION}" \
        model=v18_7 \
        data.source_contrast="${CONTRAST}" \
        data.batch_size_segmenter="${BATCH_SIZE_SEGMENTER}" \
        training.limit_val_batches=1.0 \
        training.segmenter.enable_train_image_logging=false \
        training.segmenter.val_image_log_every=1
else
    if [[ "${SEGMENTER_USE_GENERATOR}" == "true" ]]; then
        set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
            task=segmenter \
            version="${VERSION}" \
            data.source_contrast="${CONTRAST}" \
            model.segmenter.use_generator=true \
            model.segmenter.gen_version="${VERSION}" \
            model.generator.gen_version="${VERSION}" \
            data.batch_size_segmenter="${BATCH_SIZE_SEGMENTER}" \
            training.limit_val_batches=1.0 \
            training.segmenter.enable_train_image_logging=false \
            training.segmenter.val_image_log_every=1
    else
        set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
            task=segmenter \
            version="${VERSION}" \
            data.source_contrast="${CONTRAST}" \
            model.segmenter.use_generator=false \
            model.segmenter.gen_version=null \
            data.batch_size_segmenter="${BATCH_SIZE_SEGMENTER}" \
            training.limit_val_batches=1.0 \
            training.segmenter.enable_train_image_logging=false \
            training.segmenter.val_image_log_every=1
    fi
fi

