#!/usr/bin/env bash
# Third batch — add folds 2 and 3 to two existing runs, one fold per GPU, 200 epochs.
#
#   GPU0 slot0: v26_6_2_train050_val000  fold 2   (into existing run)
#   GPU1 slot1: v26_6_2_train050_val000  fold 3   (into existing run)
#   GPU2 slot2: v26_6_2_train050_val100  fold 2   (into existing run)
#   GPU3 slot3: v26_6_2_train050_val100  fold 3   (into existing run)
#
# Reuses the existing fold-0/1 run IDs so the runs become 4-fold. Pass them as
# $1/$2 to override the defaults below.
#
# Usage:
#   bash 04_15_launch_batch3_folds23.sh [RUN_train050_val000] [RUN_v26_6_2_50_100]

set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

RUN_T050="${1:-chaos_v26_6_2_train050_val000_20260616_010628}"
RUN_50_100="${2:-chaos_v26_6_2_train050_val100_20260615_213615}"

echo "[$(date '+%H:%M:%S')] Launching batch 3 — folds 2,3 ×2 runs, 1 fold/GPU, 200 epochs"
echo "  ${RUN_T050}  → folds 2 (GPU0), 3 (GPU1)"
echo "  ${RUN_50_100} → folds 2 (GPU2), 3 (GPU3)"

# v26_6_2_train050_val000 — fold 2 on GPU0, fold 3 on GPU1
NNUNET_NUM_EPOCHS=200 DA_WORKERS=16 FOLD_SLOT_GPU="2,0,0" \
    bash "${SCRIPTS_DIR}/04_11_train_v26_6_2_train050_val000.sh" "${RUN_T050}"
NNUNET_NUM_EPOCHS=200 DA_WORKERS=16 FOLD_SLOT_GPU="3,1,1" \
    bash "${SCRIPTS_DIR}/04_11_train_v26_6_2_train050_val000.sh" "${RUN_T050}"

# v26_6_2_train050_val100 — fold 2 on GPU2, fold 3 on GPU3
NNUNET_NUM_EPOCHS=200 DA_WORKERS=16 SINGLE_FOLD=2 SINGLE_SLOT=2 SINGLE_GPU=2 \
    bash "${SCRIPTS_DIR}/04_07_train_exp1_v26_6_2_50_100.sh" "${RUN_50_100}"
NNUNET_NUM_EPOCHS=200 DA_WORKERS=16 SINGLE_FOLD=3 SINGLE_SLOT=3 SINGLE_GPU=3 \
    bash "${SCRIPTS_DIR}/04_07_train_exp1_v26_6_2_50_100.sh" "${RUN_50_100}"

echo "[$(date '+%H:%M:%S')] Batch 3 launched (4 processes, 1/GPU)."
echo "  Logs: /tmp/nnunet_chaos_v26_6_2_train050_val000/fold{2,3}.log"
echo "        /tmp/nnunet_chaos_v26_6_2_train050_val100/fold{2,3}.log"
