#!/usr/bin/env bash

set -euo pipefail

# Baseline segmenters (saved to checkpoints/baseline/)
# CUDA_VISIBLE_DEVICES=1 python scripts/train_segmenter.py --baseline-contrast t1w
# CUDA_VISIBLE_DEVICES=1 python scripts/train_segmenter_bigaug.py --baseline-contrast t1w

# Generator-augmented segmenters (saved to checkpoints/<version>/)
# CUDA_VISIBLE_DEVICES=2 python scripts/train_segmenter.py --use-generator --baseline-contrast t1w --gen-version v2 --version v2 --gen-weights checkpoints/v2/mri_generator_t1w_epoch_30.pth
CUDA_VISIBLE_DEVICES=2 python scripts/train_segmenter.py --use-generator --baseline-contrast t1w --gen-version v3 --version v3 --gen-weights checkpoints/v3/mri_generator_t1w_epoch_30.pth