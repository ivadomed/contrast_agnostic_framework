#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=3 python scripts/evaluate.py --discover-checkpoints checkpoints/v4 --output-dir results/eval/v4
