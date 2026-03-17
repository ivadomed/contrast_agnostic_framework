#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=3 python scripts/train.py --source-contrast t2w --epochs 30 --version v3
# CUDA_VISIBLE_DEVICES=3 python scripts/train.py --source-contrast t1gd --epochs 30 --version v2