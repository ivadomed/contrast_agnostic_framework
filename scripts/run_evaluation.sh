#!/usr/bin/env bash

set -euo pipefail


GPU_ID=0
set_slot $GPU_ID CUDA_VISIBLE_DEVICES=$GPU_ID .venv/bin/python scripts/evaluate.py \
	--discover-checkpoints checkpoints/v6 \
	--skip-baseline-auto \
	--output-dir results/eval/v6 \
	--num-workers 12 \
	--batch-size 8 \
	--sw-batch-size 24 \
	--num-ensemble 4
