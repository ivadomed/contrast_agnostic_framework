#!/usr/bin/env bash
# Predict the T1n val100 run (AugLab + V26_6_2 @50% train / 100% val synth). This is a
# REGULAR single-checkpoint AugLabValSynth run (04_30) — NOT dual-val — because T1n's
# val000 sibling was already trained separately, so predict uses the standard
# TRAINER=...AugLabValSynth (val100 mirror of the deployed-config ladder).
#
# Usage:
#   bash 05_24_predict_t1n_auglabAug_v26_6_2_train050_val100.sh <RUN_ID> [FOLD] [CONTRAST ...]
# Example:
#   bash 05_24_predict_t1n_auglabAug_v26_6_2_train050_val100.sh brats2024-glioma_t1n_auglabAug_v26_6_2_train050_val100_<TS> all

set -euo pipefail
METHOD="auglabAug_v26_6_2_train050_val100"
TRAINER="nnUNetTrainerBraTS2024GliomaAugLabValSynth"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
