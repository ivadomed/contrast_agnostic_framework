#!/usr/bin/env bash
# Launch all 4 experiments simultaneously on slots 0-3 (one fold each, 300 epochs).
#
#   exp0 slot0: V26_6_2 default 90/100
#   exp1 slot1: V26_6_2 p50 50/100
#   exp2 slot2: AugLab + SynthSeg_EM (auglab augs enabled)
#   exp3 slot3: AugLab + V26_6_2 GPU transform (auglab augs enabled, val aug on)
#
# Each wrapper uses FOLD_SLOT_GPU / SINGLE_FOLD from 04_00_common.sh, so they
# detach immediately (fire-and-exit). Run this foreground in the Bash tool.
#
# Usage:
#   bash 04_10_launch_all_exps.sh

set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

TS="$(date +%Y%m%d_%H%M%S)"

echo "[${TS}] Launching exp0..exp3 on slots 0-3"

bash "${SCRIPTS_DIR}/04_06_train_exp0_v26_6_2_90_100.sh"      "chaos_v26_6_2_train090_val100_${TS}"
bash "${SCRIPTS_DIR}/04_07_train_exp1_v26_6_2_50_100.sh"      "chaos_v26_6_2_train050_val100_${TS}"
bash "${SCRIPTS_DIR}/04_08_train_exp2_auglabAug_synthseg_EM.sh" "chaos_auglabAug_synthseg_EM_train100_val000_${TS}"
bash "${SCRIPTS_DIR}/04_09_train_auglabAug_v26_6_2_train050_val000.sh" "chaos_auglabAug_v26_6_2_train050_val000_${TS}"

echo "[$(date '+%H:%M:%S')] All 4 experiments launched."
echo "  Logs: /tmp/nnunet_chaos_exp{0..3}_*/"
