#!/usr/bin/env bash

set -euo pipefail


GPU_ID=3
set_slot $GPU_ID CUDA_VISIBLE_DEVICES=$GPU_ID .venv/bin/python scripts/evaluate.py \
	--discover-checkpoints checkpoints/v5 \
	--output-dir results/eval/v5 \
	--num-workers 8 \
	--batch-size 8 \
	--sw-batch-size 24
