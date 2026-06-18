#!/usr/bin/env bash
# Fourth batch — two REAL val-synth runs, folds 0,1 each, 1 fold/GPU, 200 epochs.
#
#   GPU0,1 slot0,1: auglabAug_v26_6_2_train050_val100  folds 0,1
#   GPU2,3 slot2,3: synthseg_EM_train100_val100        folds 0,1
#
# Both use trainer nnUNetTrainerCHAOSAugLabValSynth (real synth-only validation_step).
#
# Usage:
#   bash 04_18_launch_batch4_valsynth.sh

set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"

echo "[${TS}] Launching batch 4 — 2 val-synth runs, folds 0,1 ×2, 1 fold/GPU, 200 epochs"

bash "${SCRIPTS_DIR}/04_16_train_auglabAug_v26_6_2_train050_val100.sh" "auglabAug_v26_6_2_train050_val100_${TS}"
bash "${SCRIPTS_DIR}/04_17_train_synthseg_EM_train100_val100.sh"       "synthseg_EM_train100_val100_${TS}"

echo "[$(date '+%H:%M:%S')] Batch 4 launched (4 processes)."
echo "  Logs: /tmp/nnunet_chaos_auglabAug_v26_6_2_train050_val100/fold{0,1}.log"
echo "        /tmp/nnunet_chaos_synthseg_EM_train100_val100/fold{0,1}.log"
