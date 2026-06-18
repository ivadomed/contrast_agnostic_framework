#!/usr/bin/env bash
# BraTS 2024 Glioma — Batch 1: 2 new experiments, 4 folds each, 1500 epochs.
#
#   A = v26_6_2_train050_val100       (nnUNetTrainerBraTS2024GliomaV26_6_2_train050_val100)
#   B = auglabAug_v26_6_2_train025_val100 (nnUNetTrainerBraTS2024GliomaAugLabValSynth, train025)
#
# 8 folds distributed across slots 1, 2, 3 (slot 0 left free):
#   Slot 1 (GPU 1): A fold 0 → A fold 1 → B fold 0   [3 folds]
#   Slot 2 (GPU 2): A fold 2 → A fold 3 → B fold 1   [3 folds]
#   Slot 3 (GPU 3): B fold 2 → B fold 3               [2 folds]
#
# Fire-and-exit: run this script foreground, it returns immediately after
# backgrounding the 3 subshells. The subshells become orphan processes and
# continue sequentially via LAUNCH_WAIT=1. Monitor with:
#   tail -f /tmp/brats_gpu{1,2,3}_queue.log
#
# Usage:
#   bash datasets/brats2024-glioma/5_scripts_brats2024-glioma/04_train/04_12_launch_batch1.sh

cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project
S="$(cd "$(dirname "$0")" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"

RUN_A="brats2024-glioma_v26_6_2_train050_val100_${TS}"
RUN_B="brats2024-glioma_auglabAug_v26_6_2_train025_val100_${TS}"

echo "[${TS}] Launching BraTS batch1 — slots 1,2,3 (8 folds, 1500 ep each)"
echo "  RUN_A: ${RUN_A}"
echo "  RUN_B: ${RUN_B}"

# ── Slot 1 (GPU 1): A fold 0, A fold 1, B fold 0 ─────────────────────────────
(
set -euo pipefail
echo "[GPU1 $(date '+%H:%M:%S')] queue start"

echo "[GPU1 1/3] v26_6_2_train050_val100 fold 0 (1500ep)"
FOLD_SLOT_GPU="0,1,1" LAUNCH_WAIT=1 \
    bash "${S}/04_10_train_v26_6_2_train050_val100.sh" "${RUN_A}"

echo "[GPU1 2/3] v26_6_2_train050_val100 fold 1 (1500ep)"
FOLD_SLOT_GPU="1,1,1" LAUNCH_WAIT=1 \
    bash "${S}/04_10_train_v26_6_2_train050_val100.sh" "${RUN_A}"

echo "[GPU1 3/3] auglabAug_v26_6_2_train025_val100 fold 0 (1500ep)"
FOLD_SLOT_GPU="0,1,1" LAUNCH_WAIT=1 \
    bash "${S}/04_11_train_auglabAug_v26_6_2_train025_val100.sh" "${RUN_B}"

echo "[GPU1 $(date '+%H:%M:%S')] all 3 folds done"
) >> /tmp/brats_gpu1_queue.log 2>&1 &

# ── Slot 2 (GPU 2): A fold 2, A fold 3, B fold 1 ─────────────────────────────
(
set -euo pipefail
echo "[GPU2 $(date '+%H:%M:%S')] queue start"

echo "[GPU2 1/3] v26_6_2_train050_val100 fold 2 (1500ep)"
FOLD_SLOT_GPU="2,2,2" LAUNCH_WAIT=1 \
    bash "${S}/04_10_train_v26_6_2_train050_val100.sh" "${RUN_A}"

echo "[GPU2 2/3] v26_6_2_train050_val100 fold 3 (1500ep)"
FOLD_SLOT_GPU="3,2,2" LAUNCH_WAIT=1 \
    bash "${S}/04_10_train_v26_6_2_train050_val100.sh" "${RUN_A}"

echo "[GPU2 3/3] auglabAug_v26_6_2_train025_val100 fold 1 (1500ep)"
FOLD_SLOT_GPU="1,2,2" LAUNCH_WAIT=1 \
    bash "${S}/04_11_train_auglabAug_v26_6_2_train025_val100.sh" "${RUN_B}"

echo "[GPU2 $(date '+%H:%M:%S')] all 3 folds done"
) >> /tmp/brats_gpu2_queue.log 2>&1 &

# ── Slot 3 (GPU 3): B fold 2, B fold 3 ───────────────────────────────────────
(
set -euo pipefail
echo "[GPU3 $(date '+%H:%M:%S')] queue start"

echo "[GPU3 1/2] auglabAug_v26_6_2_train025_val100 fold 2 (1500ep)"
FOLD_SLOT_GPU="2,3,3" LAUNCH_WAIT=1 \
    bash "${S}/04_11_train_auglabAug_v26_6_2_train025_val100.sh" "${RUN_B}"

echo "[GPU3 2/2] auglabAug_v26_6_2_train025_val100 fold 3 (1500ep)"
FOLD_SLOT_GPU="3,3,3" LAUNCH_WAIT=1 \
    bash "${S}/04_11_train_auglabAug_v26_6_2_train025_val100.sh" "${RUN_B}"

echo "[GPU3 $(date '+%H:%M:%S')] all 2 folds done"
) >> /tmp/brats_gpu3_queue.log 2>&1 &

echo "[$(date '+%H:%M:%S')] All 3 queues launched"
echo "  Monitor: tail -f /tmp/brats_gpu{1,2,3}_queue.log"
