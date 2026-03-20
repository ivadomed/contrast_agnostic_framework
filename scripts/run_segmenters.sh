#!/usr/bin/env bash

set -euo pipefail

# Baseline segmenters (saved to checkpoints/baseline/)
# CUDA_VISIBLE_DEVICES=3 python scripts/train_segmenter.py --baseline-contrast t1gd
# CUDA_VISIBLE_DEVICES=1 python scripts/train_segmenter_bigaug.py --baseline-contrast t1w

# Generator-augmented segmenters (saved to checkpoints/<version>/)
# CUDA_VISIBLE_DEVICES=2 python scripts/train_segmenter.py --use-generator --baseline-contrast t1w --gen-version v2 --version v2 --gen-weights checkpoints/v2/mri_generator_t1w_epoch_30.pth
# CUDA_VISIBLE_DEVICES=2 python scripts/train_segmenter.py --use-generator --baseline-contrast t1w --gen-version v3 --version v3 --gen-weights checkpoints/v3/mri_generator_t1w_epoch_30.pth --aug-prob 1.0
CUDA_VISIBLE_DEVICES=2 python scripts/train_segmenter.py --use-generator --baseline-contrast t2w --gen-version v4 --version v4 --gen-weights checkpoints/v4/mri_generator_t2w_epoch_30.pth --fully-artificial