#!/usr/bin/env bash
# 4 parallel per-GPU queues — 28 folds total, 7 per GPU, balanced ~1600 epoch-units/GPU.
#
# Generates shared RUN_IDs for the 4 new batch5 experiments so every GPU trains
# a different fold of the same run ID. FOLD_SLOT_GPU overrides each script's
# default fold/slot/GPU assignment — takes priority over SINGLE_FOLD in 04_00_common.sh.
#
# Each GPU queue runs as a backgrounded subshell (fire-and-exit). Folds within
# each subshell are sequential via LAUNCH_WAIT=1. Subshells become orphan processes
# after this script exits and continue independently.
#
# GPU 0 (slot 0, 1600 ep-units):
#   v26_6_2_090_val100 f1 (300), auglabAug_synthseg_EM f2 (300),
#   v26_6_2_025_val000 f2 (200), new_025_val100 f0, new_aug025_val100 f0,
#   new_aug090_val000 f0, new_aug090_val100 f0
#
# GPU 1 (slot 1, 1600 ep-units):
#   v26_6_2_090_val100 f2 (300), auglabAug_synthseg_EM f3 (300),
#   v26_6_2_025_val000 f3 (200), new_025_val100 f1, new_aug025_val100 f1,
#   new_aug090_val000 f1, new_aug090_val100 f1
#
# GPU 2 (slot 2, 1600 ep-units):
#   v26_6_2_090_val100 f3 (300), auglabAug_050_val000 f1 (300),
#   auglabAug_025_val000 f2 (200), new_025_val100 f2, new_aug025_val100 f2,
#   new_aug090_val000 f2, new_aug090_val100 f2
#
# GPU 3 (slot 3, 1600 ep-units):
#   auglabAug_050_val000 f2 (300), auglabAug_050_val000 f3 (300),
#   auglabAug_025_val000 f3 (200), new_025_val100 f3, new_aug025_val100 f3,
#   new_aug090_val000 f3, new_aug090_val100 f3
#
# Usage:
#   bash datasets/chaos/5_scripts_chaos/04_train/04_26_launch_gpu_queues.sh

cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"

# Shared RUN_IDs for new batch5 runs (all 4 GPUs train different folds of each)
RUN_025_100="chaos_v26_6_2_train025_val100_${TS}"
RUN_AUG_025_100="chaos_auglabAug_v26_6_2_train025_val100_${TS}"
RUN_AUG_090_000="chaos_auglabAug_v26_6_2_train090_val000_${TS}"
RUN_AUG_090_100="chaos_auglabAug_v26_6_2_train090_val100_${TS}"

echo "[${TS}] Launching 4 parallel GPU queues (28 folds, 7/GPU, ~1600 epoch-units/GPU)"
echo "  New run IDs:"
echo "    ${RUN_025_100}"
echo "    ${RUN_AUG_025_100}"
echo "    ${RUN_AUG_090_000}"
echo "    ${RUN_AUG_090_100}"

# ── GPU 0 (slot 0) ─────────────────────────────────────────────────────────
(
set -euo pipefail
echo "[GPU0 $(date '+%H:%M:%S')] queue start"

echo "[GPU0 1/7] v26_6_2_train090_val100 fold 1 (200ep, resume)"
FOLD_SLOT_GPU="1,0,0" NNUNET_NUM_EPOCHS=200 LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_06_train_exp0_v26_6_2_90_100.sh" \
    "chaos_v26_6_2_train090_val100_20260615_213615"

echo "[GPU0 2/7] auglabAug_synthseg_EM fold 2 (200ep, resume)"
FOLD_SLOT_GPU="2,0,0" NNUNET_NUM_EPOCHS=200 LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_08_train_exp2_auglabAug_synthseg_EM.sh" \
    "chaos_auglabAug_synthseg_EM_train100_val000_20260615_213615"

echo "[GPU0 3/7] v26_6_2_train025_val000 fold 2 (200ep, resume)"
FOLD_SLOT_GPU="2,0,0" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_12_train_v26_6_2_train025_val000.sh" \
    "chaos_v26_6_2_train025_val000_20260616_010628"

echo "[GPU0 4/7] NEW v26_6_2_train025_val100 fold 0 (200ep)"
FOLD_SLOT_GPU="0,0,0" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_19_train_v26_6_2_train025_val100.sh" \
    "${RUN_025_100}"

echo "[GPU0 5/7] NEW auglabAug_v26_6_2_train025_val100 fold 0 (200ep)"
FOLD_SLOT_GPU="0,0,0" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_20_train_auglabAug_v26_6_2_train025_val100.sh" \
    "${RUN_AUG_025_100}"

echo "[GPU0 6/7] NEW auglabAug_v26_6_2_train090_val000 fold 0 (200ep)"
FOLD_SLOT_GPU="0,0,0" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_21_train_auglabAug_v26_6_2_train090_val000.sh" \
    "${RUN_AUG_090_000}"

echo "[GPU0 7/7] NEW auglabAug_v26_6_2_train090_val100 fold 0 (200ep)"
FOLD_SLOT_GPU="0,0,0" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_22_train_auglabAug_v26_6_2_train090_val100.sh" \
    "${RUN_AUG_090_100}"

echo "[GPU0 $(date '+%H:%M:%S')] all 7 folds done"
) >> /tmp/chaos_gpu0_queue.log 2>&1 &

# ── GPU 1 (slot 1) ─────────────────────────────────────────────────────────
(
set -euo pipefail
echo "[GPU1 $(date '+%H:%M:%S')] queue start"

echo "[GPU1 1/7] v26_6_2_train090_val100 fold 2 (200ep, resume)"
FOLD_SLOT_GPU="2,1,1" NNUNET_NUM_EPOCHS=200 LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_06_train_exp0_v26_6_2_90_100.sh" \
    "chaos_v26_6_2_train090_val100_20260615_213615"

echo "[GPU1 2/7] auglabAug_synthseg_EM fold 3 (200ep, resume)"
FOLD_SLOT_GPU="3,1,1" NNUNET_NUM_EPOCHS=200 LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_08_train_exp2_auglabAug_synthseg_EM.sh" \
    "chaos_auglabAug_synthseg_EM_train100_val000_20260615_213615"

echo "[GPU1 3/7] v26_6_2_train025_val000 fold 3 (200ep, resume)"
FOLD_SLOT_GPU="3,1,1" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_12_train_v26_6_2_train025_val000.sh" \
    "chaos_v26_6_2_train025_val000_20260616_010628"

echo "[GPU1 4/7] NEW v26_6_2_train025_val100 fold 1 (200ep)"
FOLD_SLOT_GPU="1,1,1" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_19_train_v26_6_2_train025_val100.sh" \
    "${RUN_025_100}"

echo "[GPU1 5/7] NEW auglabAug_v26_6_2_train025_val100 fold 1 (200ep)"
FOLD_SLOT_GPU="1,1,1" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_20_train_auglabAug_v26_6_2_train025_val100.sh" \
    "${RUN_AUG_025_100}"

echo "[GPU1 6/7] NEW auglabAug_v26_6_2_train090_val000 fold 1 (200ep)"
FOLD_SLOT_GPU="1,1,1" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_21_train_auglabAug_v26_6_2_train090_val000.sh" \
    "${RUN_AUG_090_000}"

echo "[GPU1 7/7] NEW auglabAug_v26_6_2_train090_val100 fold 1 (200ep)"
FOLD_SLOT_GPU="1,1,1" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_22_train_auglabAug_v26_6_2_train090_val100.sh" \
    "${RUN_AUG_090_100}"

echo "[GPU1 $(date '+%H:%M:%S')] all 7 folds done"
) >> /tmp/chaos_gpu1_queue.log 2>&1 &

# ── GPU 2 (slot 2) ─────────────────────────────────────────────────────────
(
set -euo pipefail
echo "[GPU2 $(date '+%H:%M:%S')] queue start"

echo "[GPU2 1/7] v26_6_2_train090_val100 fold 3 (200ep, resume)"
FOLD_SLOT_GPU="3,2,2" NNUNET_NUM_EPOCHS=200 LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_06_train_exp0_v26_6_2_90_100.sh" \
    "chaos_v26_6_2_train090_val100_20260615_213615"

echo "[GPU2 2/7] auglabAug_v26_6_2_train050_val000 fold 1 (200ep, resume)"
FOLD_SLOT_GPU="1,2,2" NNUNET_NUM_EPOCHS=200 LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_09_train_auglabAug_v26_6_2_train050_val000.sh" \
    "chaos_auglabAug_v26_6_2_train050_val000_20260615_213615"

echo "[GPU2 3/7] auglabAug_v26_6_2_train025_val000 fold 2 (200ep, resume)"
FOLD_SLOT_GPU="2,2,2" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_13_train_auglabAug_v26_6_2_train025_val000.sh" \
    "chaos_auglabAug_v26_6_2_train025_val000_20260616_010628"

echo "[GPU2 4/7] NEW v26_6_2_train025_val100 fold 2 (200ep)"
FOLD_SLOT_GPU="2,2,2" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_19_train_v26_6_2_train025_val100.sh" \
    "${RUN_025_100}"

echo "[GPU2 5/7] NEW auglabAug_v26_6_2_train025_val100 fold 2 (200ep)"
FOLD_SLOT_GPU="2,2,2" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_20_train_auglabAug_v26_6_2_train025_val100.sh" \
    "${RUN_AUG_025_100}"

echo "[GPU2 6/7] NEW auglabAug_v26_6_2_train090_val000 fold 2 (200ep)"
FOLD_SLOT_GPU="2,2,2" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_21_train_auglabAug_v26_6_2_train090_val000.sh" \
    "${RUN_AUG_090_000}"

echo "[GPU2 7/7] NEW auglabAug_v26_6_2_train090_val100 fold 2 (200ep)"
FOLD_SLOT_GPU="2,2,2" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_22_train_auglabAug_v26_6_2_train090_val100.sh" \
    "${RUN_AUG_090_100}"

echo "[GPU2 $(date '+%H:%M:%S')] all 7 folds done"
) >> /tmp/chaos_gpu2_queue.log 2>&1 &

# ── GPU 3 (slot 3) ─────────────────────────────────────────────────────────
(
set -euo pipefail
echo "[GPU3 $(date '+%H:%M:%S')] queue start"

echo "[GPU3 1/7] auglabAug_v26_6_2_train050_val000 fold 2 (200ep, resume)"
FOLD_SLOT_GPU="2,3,3" NNUNET_NUM_EPOCHS=200 LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_09_train_auglabAug_v26_6_2_train050_val000.sh" \
    "chaos_auglabAug_v26_6_2_train050_val000_20260615_213615"

echo "[GPU3 2/7] auglabAug_v26_6_2_train050_val000 fold 3 (200ep, resume)"
FOLD_SLOT_GPU="3,3,3" NNUNET_NUM_EPOCHS=200 LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_09_train_auglabAug_v26_6_2_train050_val000.sh" \
    "chaos_auglabAug_v26_6_2_train050_val000_20260615_213615"

echo "[GPU3 3/7] auglabAug_v26_6_2_train025_val000 fold 3 (200ep, resume)"
FOLD_SLOT_GPU="3,3,3" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_13_train_auglabAug_v26_6_2_train025_val000.sh" \
    "chaos_auglabAug_v26_6_2_train025_val000_20260616_010628"

echo "[GPU3 4/7] NEW v26_6_2_train025_val100 fold 3 (200ep)"
FOLD_SLOT_GPU="3,3,3" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_19_train_v26_6_2_train025_val100.sh" \
    "${RUN_025_100}"

echo "[GPU3 5/7] NEW auglabAug_v26_6_2_train025_val100 fold 3 (200ep)"
FOLD_SLOT_GPU="3,3,3" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_20_train_auglabAug_v26_6_2_train025_val100.sh" \
    "${RUN_AUG_025_100}"

echo "[GPU3 6/7] NEW auglabAug_v26_6_2_train090_val000 fold 3 (200ep)"
FOLD_SLOT_GPU="3,3,3" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_21_train_auglabAug_v26_6_2_train090_val000.sh" \
    "${RUN_AUG_090_000}"

echo "[GPU3 7/7] NEW auglabAug_v26_6_2_train090_val100 fold 3 (200ep)"
FOLD_SLOT_GPU="3,3,3" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_22_train_auglabAug_v26_6_2_train090_val100.sh" \
    "${RUN_AUG_090_100}"

echo "[GPU3 $(date '+%H:%M:%S')] all 7 folds done"
) >> /tmp/chaos_gpu3_queue.log 2>&1 &

echo "[$(date '+%H:%M:%S')] All 4 GPU queues launched (PIDs: $(jobs -p | tr '\n' ' '))"
echo "  Monitor: tail -f /tmp/chaos_gpu{0,1,2,3}_queue.log"
echo "  Fold logs: /tmp/nnunet_chaos_*/fold*.log"
