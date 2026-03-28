#!/usr/bin/env bash

set -euo pipefail

GPU_ID=3
VERSION="v15"


BASE_OUTPUT_DIR="results/eval/${VERSION}"

for ENS in 1; do
	OUT_DIR="${BASE_OUTPUT_DIR}/ens${ENS}"
	echo "Running evaluation with num_ensemble=${ENS} -> ${OUT_DIR}"

	set_slot $GPU_ID CUDA_VISIBLE_DEVICES=$GPU_ID .venv/bin/python scripts/evaluate.py \
		--discover-checkpoints checkpoints/${VERSION} \
		--skip-baseline-auto \
		--output-dir "$OUT_DIR" \
		--num-workers 12 \
		--batch-size 8 \
		--sw-batch-size 24 \
		--num-ensemble "$ENS"
done

echo "Done. Saved ensemble sweep outputs under ${BASE_OUTPUT_DIR}/ens1..ens5"
