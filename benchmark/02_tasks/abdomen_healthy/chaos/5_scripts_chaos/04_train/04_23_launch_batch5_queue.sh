#!/usr/bin/env bash
# Batch 5 — queue 4 val-probe runs sequentially, 4 folds each, 1 fold/GPU, 200 epochs.
# Each run blocks until all 4 folds finish (LAUNCH_WAIT=1) before the next starts.
# All use the same timestamp so run names sort together.
#
#   Run 1: chaos_v26_6_2_train025_val100            (nnUNet, pure V26)
#   Run 2: chaos_auglabAug_v26_6_2_train025_val100  (AugLab+V26, val synth)
#   Run 3: chaos_auglabAug_v26_6_2_train090_val000  (AugLab+V26, clean val)
#   Run 4: chaos_auglabAug_v26_6_2_train090_val100  (AugLab+V26, val synth)
#
# Usage:
#   bash 04_23_launch_batch5_queue.sh

set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"

echo "[${TS}] Batch 5 — 4 queued runs, 4 folds/run, 1 fold/GPU, 200 epochs each"

LAUNCH_WAIT=1 bash "${SCRIPTS_DIR}/04_19_train_v26_6_2_train025_val100.sh"          "chaos_v26_6_2_train025_val100_${TS}"
LAUNCH_WAIT=1 bash "${SCRIPTS_DIR}/04_20_train_auglabAug_v26_6_2_train025_val100.sh" "chaos_auglabAug_v26_6_2_train025_val100_${TS}"
LAUNCH_WAIT=1 bash "${SCRIPTS_DIR}/04_21_train_auglabAug_v26_6_2_train090_val000.sh" "chaos_auglabAug_v26_6_2_train090_val000_${TS}"
LAUNCH_WAIT=1 bash "${SCRIPTS_DIR}/04_22_train_auglabAug_v26_6_2_train090_val100.sh" "chaos_auglabAug_v26_6_2_train090_val100_${TS}"

echo "[$(date '+%H:%M:%S')] Batch 5 complete."
