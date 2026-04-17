#!/usr/bin/env bash

set -euo pipefail

SLOT_ID="${1:-3}"
VERSION="${2:-v5}"
CONTRAST="${3:-t1w}"
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
		# T2 volumes can be highly heterogeneous after conversion; use batch size 1
		# to avoid collate-time shape mismatch while we keep training progressing.
		SPIDER_DATA_ARGS+=(data.batch_size_generator=1)
	fi
fi

if [[ "${VERSION}" == "v17_lpci" ]]; then
	set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
		version="${VERSION}" \
		model=v17_lpci \
		data.source_contrast="${CONTRAST}" \
		training.resume=false \
		training.generator.enable_image_logging=true \
		training.generator.log_aux_every_n_steps=100 \
		training.log_every_n_steps=50 \
		training.enable_model_summary=false \
		training.devices=1 \
		training.precision=16-mixed
elif [[ "${VERSION}" == "v17_micro_anchor" ]]; then
	set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
		version="${VERSION}" \
		model=v17 \
		data.source_contrast="${CONTRAST}" \
		training.resume=false \
		training.generator.enable_image_logging=true \
		training.generator.log_aux_every_n_steps=100 \
		training.log_every_n_steps=50 \
		training.enable_model_summary=false \
		training.devices=1 \
		training.precision=16-mixed
elif [[ "${VERSION}" == "v18" ]]; then
	set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
		version="${VERSION}" \
		model=v18 \
		data.source_contrast="${CONTRAST}" \
		training.resume=false \
		training.generator.enable_image_logging=true \
		training.generator.log_aux_every_n_steps=100 \
		training.log_every_n_steps=50 \
		training.enable_model_summary=false \
		training.devices=1 \
		training.precision=16-mixed
elif [[ "${VERSION}" == "v18_1" ]]; then
	set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
		version="${VERSION}" \
		model=v18_1 \
		data.source_contrast="${CONTRAST}" \
		training.resume=false \
		training.generator.enable_image_logging=true \
		training.generator.log_aux_every_n_steps=100 \
		training.log_every_n_steps=50 \
		training.enable_model_summary=false \
		training.devices=1 \
		training.precision=16-mixed
elif [[ "${VERSION}" == "v18_2" ]]; then
	set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
		version="${VERSION}" \
		model=v18_2 \
		data.source_contrast="${CONTRAST}" \
		training.resume=false \
		training.generator.enable_image_logging=true \
		training.generator.log_aux_every_n_steps=100 \
		training.log_every_n_steps=50 \
		training.enable_model_summary=false \
		training.devices=1 \
		training.precision=16-mixed
elif [[ "${VERSION}" == "v18_3" ]]; then
	set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
		version="${VERSION}" \
		model=v18_3 \
		data.source_contrast="${CONTRAST}" \
		training.resume=false \
		training.generator.enable_image_logging=true \
		training.generator.log_aux_every_n_steps=100 \
		training.log_every_n_steps=50 \
		training.enable_model_summary=false \
		training.devices=1 \
		training.precision=16-mixed
elif [[ "${VERSION}" == "v18_4" ]]; then
	set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
		version="${VERSION}" \
		model=v18_4 \
		data.source_contrast="${CONTRAST}" \
		training.resume=false \
		training.generator.enable_image_logging=true \
		training.generator.log_aux_every_n_steps=100 \
		training.log_every_n_steps=50 \
		training.enable_model_summary=false \
		training.devices=1 \
		training.precision=16-mixed
elif [[ "${VERSION}" == "v18_5" ]]; then
	set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
		version="${VERSION}" \
		model=v18_5 \
		data.source_contrast="${CONTRAST}" \
		training.resume=false \
		training.generator.enable_image_logging=true \
		training.generator.log_aux_every_n_steps=100 \
		training.log_every_n_steps=50 \
		training.enable_model_summary=false \
		training.devices=1 \
		training.precision=16-mixed
elif [[ "${VERSION}" == "v18_6" ]]; then
	set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
		version="${VERSION}" \
		model=v18_6 \
		data.source_contrast="${CONTRAST}" \
		training.resume=false \
		training.generator.enable_image_logging=true \
		training.generator.log_aux_every_n_steps=100 \
		training.log_every_n_steps=50 \
		training.enable_model_summary=false \
		training.devices=1 \
		training.precision=16-mixed
elif [[ "${VERSION}" == "synthseg_baseline" ]]; then
        set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
                version="${VERSION}" \
                model=synthseg_baseline \
                data.source_contrast="${CONTRAST}" \
                training.resume=false \
                training.generator.enable_image_logging=true \
                training.generator.log_aux_every_n_steps=100 \
                training.log_every_n_steps=50 \
                training.enable_model_summary=false \
                training.devices=1 \
                training.precision=16-mixed
elif [[ "${VERSION}" == "v19" ]]; then
        set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
                version="${VERSION}" \
		data="${DATASET_GROUP}" \
		generator="${GENERATOR_CFG}" \
		segmenter=seg_B \
                data.source_contrast="${CONTRAST}" \
		"${SPIDER_DATA_ARGS[@]}" \
                training.resume=false \
                training.generator.enable_image_logging=true \
                training.generator.log_aux_every_n_steps=100 \
                training.log_every_n_steps=50 \
                training.enable_model_summary=false \
                training.devices=1 \
                training.precision=16-mixed
elif [[ "${VERSION}" == "v18_7" ]]; then
	set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
		version="${VERSION}" \
		model=v18_7 \
		data.source_contrast="${CONTRAST}" \
		training.resume=false \
		training.generator.enable_image_logging=true \
		training.generator.log_aux_every_n_steps=100 \
		training.log_every_n_steps=50 \
		training.enable_model_summary=false \
		training.devices=1 \
		training.precision=16-mixed
else
	set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/train.py \
		version="${VERSION}" \
		data="${DATASET_GROUP}" \
		generator="${GENERATOR_CFG}" \
		segmenter=seg_B \
		data.source_contrast="${CONTRAST}" \
		"${SPIDER_DATA_ARGS[@]}" \
		data.num_workers=0 \
		training.resume=false \
		training.generator.enable_image_logging=true \
		training.generator.log_aux_every_n_steps=100 \
		training.log_every_n_steps=50 \
		training.enable_model_summary=false \
		training.devices=1 \
		training.precision=16-mixed
fi