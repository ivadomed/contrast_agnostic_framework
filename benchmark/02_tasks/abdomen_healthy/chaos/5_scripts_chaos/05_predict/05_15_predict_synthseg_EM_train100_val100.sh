#!/usr/bin/env bash
# Predict with SynthSeg+EM (train_synth_prob=1.0, val_synth_prob=1.0) on the CHAOS test set.
# Usage: bash 05_15_predict_synthseg_EM_train100_val100.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
METHOD="synthseg_EM_train100_val100"
TRAINER="nnUNetTrainerCHAOSAugLabValSynth"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
