#!/usr/bin/env bash
# Predict with the CHAOS baseline across modalities (cross-modality generalization).
# Usage: bash 05_02_predict_baseline.sh <RUN_ID> [FOLD] [MODALITY ...]
# Example: bash 05_02_predict_baseline.sh baseline_20260614_160000 all
set -euo pipefail
METHOD="baseline"
TRAINER="nnUNetTrainerCHAOSBaseline"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
