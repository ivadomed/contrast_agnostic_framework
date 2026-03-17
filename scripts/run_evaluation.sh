#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=2 python scripts/evaluate.py --discover-checkpoints checkpoints/v3 --output-dir results/eval/v3
