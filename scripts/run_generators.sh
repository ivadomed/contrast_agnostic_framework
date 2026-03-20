#!/usr/bin/env bash

set -euo pipefail

SLOT_ID="${1:-3}"
VERSION="${2:-v5}"
CONTRAST="${3:-t1w}"

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