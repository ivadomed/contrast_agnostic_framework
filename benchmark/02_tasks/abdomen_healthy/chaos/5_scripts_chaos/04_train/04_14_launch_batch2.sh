#!/usr/bin/env bash
# Second experiment batch — 8 training processes, 2 per GPU, 200 epochs each.
#
#   GPU0 slot0: v26_6_2_train050_val000            folds 0,1  (NEW run)
#   GPU1 slot1: v26_6_2_train025_val000            folds 0,1  (NEW run)
#   GPU2 slot2: auglabAug_synthseg_EM_train100_val000  fold 1     (into EXISTING run)
#   GPU2 slot2: v26_6_2_train050_val100                fold 1     (into EXISTING run)
#   GPU3 slot3: auglabAug_v26_6_2_train025_val000  folds 0,1  (NEW run)
#
# The two fold-1 additions reuse the existing fold-0 run IDs so they aggregate as
# 2-fold runs. Pass them as $1/$2 to override the defaults below.
#
# Usage:
#   bash 04_14_launch_batch2.sh [EXISTING_SYNTHSEG_EM_RUN] [EXISTING_V26_6_2_50_100_RUN]

set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

TS="$(date +%Y%m%d_%H%M%S)"

EXISTING_SYNTHSEG_EM="${1:-chaos_auglabAug_synthseg_EM_train100_val000_20260615_213615}"
EXISTING_V26_50_100="${2:-chaos_v26_6_2_train050_val100_20260615_213615}"

echo "[${TS}] Launching batch 2 — 8 processes, 2 per GPU, 200 epochs"
echo "  fold-1 add → ${EXISTING_SYNTHSEG_EM} (auglab)"
echo "  fold-1 add → ${EXISTING_V26_50_100} (nnUNet)"

# ── NEW runs (fresh timestamped RUN_IDs) ──────────────────────────────────────
bash "${SCRIPTS_DIR}/04_11_train_v26_6_2_train050_val000.sh"           "chaos_v26_6_2_train050_val000_${TS}"
bash "${SCRIPTS_DIR}/04_12_train_v26_6_2_train025_val000.sh"           "chaos_v26_6_2_train025_val000_${TS}"
bash "${SCRIPTS_DIR}/04_13_train_auglabAug_v26_6_2_train025_val000.sh" "chaos_auglabAug_v26_6_2_train025_val000_${TS}"

# ── Fold-1 additions into EXISTING runs (GPU 2, packed) ───────────────────────
SINGLE_FOLD=1 SINGLE_SLOT=2 SINGLE_GPU=2 NNUNET_NUM_EPOCHS=200 \
    bash "${SCRIPTS_DIR}/04_08_train_exp2_auglabAug_synthseg_EM.sh" "${EXISTING_SYNTHSEG_EM}"
SINGLE_FOLD=1 SINGLE_SLOT=2 SINGLE_GPU=2 NNUNET_NUM_EPOCHS=200 \
    bash "${SCRIPTS_DIR}/04_07_train_exp1_v26_6_2_50_100.sh"        "${EXISTING_V26_50_100}"

echo "[$(date '+%H:%M:%S')] Batch 2 launched (8 processes)."
echo "  Logs: /tmp/nnunet_chaos_v26_6_2_train050_val000/  /tmp/nnunet_chaos_v26_6_2_train025_val000/"
echo "        /tmp/nnunet_chaos_auglabAug_v26_6_2_train025_val000/"
echo "        /tmp/nnunet_chaos_auglabAug_synthseg_EM/  /tmp/nnunet_chaos_v26_6_2_train050_val100/"
