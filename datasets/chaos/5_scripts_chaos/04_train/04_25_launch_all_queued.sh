#!/usr/bin/env bash
# Master sequential queue — resume all incomplete runs then train all batch5 experiments.
# LAUNCH_WAIT=1 blocks each entry until all its folds finish before the next starts.
# Supersedes 04_23 and 04_24 (those are kept for reference).
#
# Phase 1 — resume incomplete runs (hardcoded existing run IDs):
#   [1/9] chaos_v26_6_2_train025_val000_20260616_010628        folds 2,3
#   [2/9] chaos_v26_6_2_train090_val100_20260615_213615        folds 1,2,3
#   [3/9] chaos_auglabAug_synthseg_EM_train100_val000_20260615_213615  folds 2,3
#   [4/9] chaos_auglabAug_v26_6_2_train025_val000_20260616_010628      folds 2,3
#   [5/9] chaos_auglabAug_v26_6_2_train050_val000_20260615_213615      folds 1,2,3
#
# Phase 2 — new batch5 runs (auto-generate RUN_ID at launch time, 4 folds each):
#   [6/9] chaos_v26_6_2_train025_val100_<TS>
#   [7/9] chaos_auglabAug_v26_6_2_train025_val100_<TS>
#   [8/9] chaos_auglabAug_v26_6_2_train090_val000_<TS>
#   [9/9] chaos_auglabAug_v26_6_2_train090_val100_<TS>
#
# Usage (run from project root, e.g. via run_in_background):
#   bash datasets/chaos/5_scripts_chaos/04_train/04_25_launch_all_queued.sh

set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_QUEUE="/tmp/nnunet_chaos_queue_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_QUEUE") 2>&1

echo "[$(date '+%H:%M:%S')] === Master queue started (9 runs, sequential) ==="
echo "  Log: $LOG_QUEUE"

# ── Phase 1: Resume incomplete runs ──────────────────────────────────────────

echo ""
echo "[$(date '+%H:%M:%S')] [1/9] chaos_v26_6_2_train025_val000_20260616_010628 — folds 2,3"
FOLD_SLOT_GPU="2,2,2 3,3,3" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_12_train_v26_6_2_train025_val000.sh" \
    "chaos_v26_6_2_train025_val000_20260616_010628"

echo ""
echo "[$(date '+%H:%M:%S')] [2/9] chaos_v26_6_2_train090_val100_20260615_213615 — folds 1,2,3"
FOLD_SLOT_GPU="1,1,1 2,2,2 3,3,3" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_06_train_exp0_v26_6_2_90_100.sh" \
    "chaos_v26_6_2_train090_val100_20260615_213615"

echo ""
echo "[$(date '+%H:%M:%S')] [3/9] chaos_auglabAug_synthseg_EM_train100_val000_20260615_213615 — folds 2,3"
FOLD_SLOT_GPU="2,2,2 3,3,3" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_08_train_exp2_auglabAug_synthseg_EM.sh" \
    "chaos_auglabAug_synthseg_EM_train100_val000_20260615_213615"

echo ""
echo "[$(date '+%H:%M:%S')] [4/9] chaos_auglabAug_v26_6_2_train025_val000_20260616_010628 — folds 2,3"
FOLD_SLOT_GPU="2,2,2 3,3,3" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_13_train_auglabAug_v26_6_2_train025_val000.sh" \
    "chaos_auglabAug_v26_6_2_train025_val000_20260616_010628"

echo ""
echo "[$(date '+%H:%M:%S')] [5/9] chaos_auglabAug_v26_6_2_train050_val000_20260615_213615 — folds 1,2,3"
FOLD_SLOT_GPU="1,1,1 2,2,2 3,3,3" LAUNCH_WAIT=1 \
    bash "${SCRIPTS_DIR}/04_09_train_auglabAug_v26_6_2_train050_val000.sh" \
    "chaos_auglabAug_v26_6_2_train050_val000_20260615_213615"

# ── Phase 2: New batch5 runs (all 4 folds, 1/GPU, 200 epochs) ────────────────

echo ""
echo "[$(date '+%H:%M:%S')] [6/9] NEW chaos_v26_6_2_train025_val100"
LAUNCH_WAIT=1 bash "${SCRIPTS_DIR}/04_19_train_v26_6_2_train025_val100.sh"

echo ""
echo "[$(date '+%H:%M:%S')] [7/9] NEW chaos_auglabAug_v26_6_2_train025_val100"
LAUNCH_WAIT=1 bash "${SCRIPTS_DIR}/04_20_train_auglabAug_v26_6_2_train025_val100.sh"

echo ""
echo "[$(date '+%H:%M:%S')] [8/9] NEW chaos_auglabAug_v26_6_2_train090_val000"
LAUNCH_WAIT=1 bash "${SCRIPTS_DIR}/04_21_train_auglabAug_v26_6_2_train090_val000.sh"

echo ""
echo "[$(date '+%H:%M:%S')] [9/9] NEW chaos_auglabAug_v26_6_2_train090_val100"
LAUNCH_WAIT=1 bash "${SCRIPTS_DIR}/04_22_train_auglabAug_v26_6_2_train090_val100.sh"

echo ""
echo "[$(date '+%H:%M:%S')] === Master queue complete. ==="
