#!/usr/bin/env bash

set -euo pipefail

GPU_ID=1
VERSION="v18_6"

BASE_OUTPUT_DIR="results/eval/${VERSION}"

ENS=1
TTA_SAMPLES=0
EXCLUDE_ORIGINAL=0
BATCH_SIZE=${BATCH_SIZE:-2}
SW_BATCH_SIZE=${SW_BATCH_SIZE:-8}
MIN_SW_BATCH_SIZE=${MIN_SW_BATCH_SIZE:-2}
DISABLE_PIN_MEMORY=${DISABLE_PIN_MEMORY:-1}
DISABLE_PERSISTENT_WORKERS=${DISABLE_PERSISTENT_WORKERS:-1}

OUT_DIR="${BASE_OUTPUT_DIR}"
echo "Running evaluation with num_ensemble=${ENS}, tta_samples=${TTA_SAMPLES}, exclude_original=${EXCLUDE_ORIGINAL} -> ${OUT_DIR}"

CMD=(
set_slot "$GPU_ID" CUDA_VISIBLE_DEVICES="$GPU_ID" .venv/bin/python scripts/evaluate.py
--discover-checkpoints "checkpoints/${VERSION}"
--skip-baseline-auto
--output-dir "$OUT_DIR"
--task-mode auto
--num-workers 12
--batch-size "$BATCH_SIZE"
--sw-batch-size "$SW_BATCH_SIZE"
--min-sw-batch-size "$MIN_SW_BATCH_SIZE"
--num-ensemble "$ENS"
--tta-samples "$TTA_SAMPLES"
)

if [[ "$EXCLUDE_ORIGINAL" == "1" ]]; then
CMD+=(--exclude-original)
fi

CMD+=("$@")

if [[ "$DISABLE_PIN_MEMORY" == "1" ]]; then
CMD+=(--disable-pin-memory)
fi

if [[ "$DISABLE_PERSISTENT_WORKERS" == "1" ]]; then
CMD+=(--disable-persistent-workers)
fi

"${CMD[@]}"

echo "Done. Saved evaluation outputs under ${BASE_OUTPUT_DIR}"
