#!/usr/bin/env bash

set -euo pipefail

GPU_ID=1
VERSION="v18_2"


BASE_OUTPUT_DIR="results/eval/${VERSION}"

ENS=1
OUT_DIR="${BASE_OUTPUT_DIR}"
echo "Running evaluation with num_ensemble=${ENS} -> ${OUT_DIR}"

set_slot $GPU_ID CUDA_VISIBLE_DEVICES=$GPU_ID .venv/bin/python scripts/evaluate.py \
	--discover-checkpoints checkpoints/${VERSION} \
	--skip-baseline-auto \
	--output-dir "$OUT_DIR" \
	--task-mode auto \
	--num-workers 12 \
	--batch-size 8 \
	--sw-batch-size 24 \
	--num-ensemble "$ENS"

echo "Done. Saved evaluation outputs under ${BASE_OUTPUT_DIR}"
