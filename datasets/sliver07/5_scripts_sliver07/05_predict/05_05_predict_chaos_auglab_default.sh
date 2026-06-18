#!/usr/bin/env bash
# Predict on SLIVER07 CT with the CHAOS-TRAINED AugLab-default model.
# SLIVER07 has no model of its own — see 05_01_predict_common.sh.
# Usage: bash 05_05_predict_chaos_auglab_default.sh [CHAOS_RUN_ID] [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="auglab_default"
TRAINER="nnUNetTrainerCHAOSAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:-chaos_auglab_default_20260611_120000}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
