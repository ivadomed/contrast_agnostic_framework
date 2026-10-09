#!/usr/bin/env bash
# Minimal env.sh for healthy-spine-tum. Unlike every other dataset, this one has NO
# raw->BIDS->nnUNet->train->predict pipeline (results were imported externally from TUM,
# see ../../README.md) — so this file exists purely to give the eval-layer wrapper scripts
# below (06_XX_aggregate_from_config.sh / 06_XX_significance_from_config.sh) the same
# PROJECT_ROOT/METRICS_ROOT entry point every other dataset's env.sh provides, rather than
# invoking aggregate_from_config.py/significance_from_config.py directly. There is no
# 0_raw/1_BIDS/2_nnUNet/4_splits/6_checkpoints dir to derive paths from, so only the two
# vars needed by these wrappers are set.

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="healthy-spine-tum"
source "${DATASET_ROOT}/../../../00_commun_scripts/00_00_utils/common_env.sh"
