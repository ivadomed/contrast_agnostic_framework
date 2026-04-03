#!/usr/bin/env bash

set -euo pipefail

SLOT_ID="${1:-3}"
VERSION="${2:-v5}"
CONTRAST="${3:-t1w}"

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
		data.source_contrast="${CONTRAST}" \
		training.resume=false \
		training.generator.enable_image_logging=true \
		training.generator.log_aux_every_n_steps=100 \
		training.log_every_n_steps=50 \
		training.enable_model_summary=false \
		training.devices=1 \
		training.precision=16-mixed
fi