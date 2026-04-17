#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash scripts/run_segmenters.sh [slot] [version] [contrast] [batch_size_segmenter]
# Example:
#   bash scripts/run_segmenters.sh 1 v13 t1w 4

SLOT_ID="${1:-3}"
VERSION="${2:-v13}"
CONTRAST="${3:-t2w}"
SEGMENTER_USE_GENERATOR="${4:-true}"
if [[ "${4:-}" == "" && "${VERSION}" == "gen_raw" ]]; then
    SEGMENTER_USE_GENERATOR="false"
fi
DATASET_GROUP="brats"
if [[ "${CONTRAST}" == "t1_sag" || "${CONTRAST}" == "t2_sag" || "${CONTRAST}" == "t2_space" ]]; then
    DATASET_GROUP="spider_spine"
fi

GENERATOR_CFG="gen_raw"
if [[ "${VERSION}" == "v19" ]]; then
    GENERATOR_CFG="gen_19"
elif [[ "${VERSION}" == "v20" || "${VERSION}" == "v20_1" ]]; then
    GENERATOR_CFG="gen_20_1"
elif [[ "${VERSION}" == "v21" ]]; then
    GENERATOR_CFG="gen_21"
fi

SPIDER_DATA_ARGS=()
if [[ "${DATASET_GROUP}" == "spider_spine" ]]; then
    TASK_NAME="Dataset102_SpiderSpine_t1"
    if [[ "${CONTRAST}" == "t2_sag" ]]; then
        TASK_NAME="Dataset102_SpiderSpine_t2"
    elif [[ "${CONTRAST}" == "t2_space" ]]; then
        TASK_NAME="Dataset102_SpiderSpine_t2space"
    fi
    SPIDER_DATA_ARGS+=(data.data_dir=data/nnUNet_raw)
    SPIDER_DATA_ARGS+=(data.task_name="${TASK_NAME}")
    SPIDER_DATA_ARGS+=("data.contrasts=[${CONTRAST}]")
    if [[ "${CONTRAST}" == "t2_sag" ]]; then
        SPIDER_DATA_ARGS+=(data.num_workers=0)
        SPIDER_DATA_ARGS+=(data.val_batch_size=1)
    fi
fi

BATCH_SIZE_SEGMENTER="${5:-4}"
if [[ "${VERSION}" == "v16_bigaug" && "${5:-}" == "" ]]; then
    BATCH_SIZE_SEGMENTER="8"
fi

SEG_BATCH_OVERRIDE="${BATCH_SIZE_SEGMENTER}"
if [[ "${DATASET_GROUP}" == "spider_spine" && "${CONTRAST}" == "t2_sag" ]]; then
    # Keep Spider T2 runs robust to heterogeneous spatial sizes.
    SEG_BATCH_OVERRIDE="1"
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
        training.segmenter.enable_train_image_logging=true \
        training.segmenter.train_image_log_every=1 \
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
elif [[ "${VERSION}" == "synthseg_baseline" ]]; then
        set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train_segmenter.py \
                version="${VERSION}" \
                model=synthseg_baseline \
                data.source_contrast="${CONTRAST}" \
                training.resume=false \
                training.log_every_n_steps=50 \
                training.enable_model_summary=false \
                training.devices=1 \
                training.precision=16-mixed
elif [[ "${VERSION}" == "v19" ]]; then
    if [[ "${SEGMENTER_USE_GENERATOR}" == "true" ]]; then
    set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train_segmenter.py \
        version="${VERSION}" \
        data="${DATASET_GROUP}" \
        generator="${GENERATOR_CFG}" \
        segmenter=seg_B \
        data.source_contrast="${CONTRAST}" \
        "${SPIDER_DATA_ARGS[@]}" \
            data.batch_size_segmenter="${SEG_BATCH_OVERRIDE}" \
        training.resume=false \
        training.log_every_n_steps=50 \
        training.enable_model_summary=false \
        training.devices=1 \
        training.precision=16-mixed
    else
    set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train_segmenter.py \
        version="${VERSION}" \
        data="${DATASET_GROUP}" \
        segmenter=seg_B \
        generator=gen_raw \
        data.source_contrast="${CONTRAST}" \
        "${SPIDER_DATA_ARGS[@]}" \
        model.segmenter.use_generator=false \
        model.segmenter.gen_version=null \
        data.batch_size_segmenter="${SEG_BATCH_OVERRIDE}" \
        training.resume=false \
        training.log_every_n_steps=50 \
        training.enable_model_summary=false \
        training.devices=1 \
        training.precision=16-mixed
    fi
elif [[ "${VERSION}" == "v20" || "${VERSION}" == "v20_1" ]]; then
    set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train_segmenter.py \
        version="${VERSION}" \
        model=v20_1 \
        data.source_contrast="${CONTRAST}" \
        training.resume=false \
        training.log_every_n_steps=50 \
        training.enable_model_summary=false \
        training.devices=1 \
        training.precision=16-mixed
elif [[ "${VERSION}" == "v21" ]]; then
    set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train_segmenter.py \
        version="${VERSION}" \
        model=v21 \
        data.source_contrast="${CONTRAST}" \
        training.resume=false \
        training.log_every_n_steps=50 \
        training.enable_model_summary=false \
        training.devices=1 \
        training.precision=16-mixed \
        training.segmenter.enable_train_image_logging=true \
        training.segmenter.train_image_log_every=1
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
            data="${DATASET_GROUP}" \
            segmenter=seg_B \
            generator="${GENERATOR_CFG}" \
            data.source_contrast="${CONTRAST}" \
            "${SPIDER_DATA_ARGS[@]}" \
            data.num_workers=0 \
            model.segmenter.use_generator=true \
            model.segmenter.gen_version="${VERSION}" \
            model.generator.gen_version="${VERSION}" \
                data.batch_size_segmenter="${SEG_BATCH_OVERRIDE}" \
            training.limit_val_batches=1.0 \
            training.segmenter.enable_train_image_logging=false \
            training.segmenter.val_image_log_every=1
    else
        set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
            task=segmenter \
            version="${VERSION}" \
            data="${DATASET_GROUP}" \
            segmenter=seg_B \
            generator=gen_raw \
            data.source_contrast="${CONTRAST}" \
            "${SPIDER_DATA_ARGS[@]}" \
            data.num_workers=0 \
            model.segmenter.use_generator=false \
            model.segmenter.gen_version=null \
            data.batch_size_segmenter="${SEG_BATCH_OVERRIDE}" \
            training.limit_val_batches=1.0 \
            training.segmenter.enable_train_image_logging=false \
            training.segmenter.val_image_log_every=1
    fi
fi

