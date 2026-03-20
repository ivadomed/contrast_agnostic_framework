#!/usr/bin/env bash

set -euo pipefail


GPU_ID=1
set_slot $GPU_ID CUDA_VISIBLE_DEVICES=$GPU_ID .venv/bin/python scripts/evaluate.py --discover-checkpoints checkpoints/v4 --output-dir results/eval/v4
