#!/usr/bin/env bash

set -euo pipefail

SLOT_ID=1
set_slot "${SLOT_ID}" CUDA_VISIBLE_DEVICES="${SLOT_ID}" .venv/bin/python scripts/generate_visualizations.py \
    --checkpoint "checkpoints/v5/generator/t2w/run6/last.ckpt"