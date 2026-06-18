#!/usr/bin/env bash
# Resume all CHAOS experiments that are missing one or more folds, queued sequentially.
# LAUNCH_WAIT=1 blocks until all folds of a run finish before starting the next.
# FOLD_SLOT_GPU overrides SINGLE_FOLD/SINGLE_SLOT/SINGLE_GPU in 04_00_common.sh —
# only the missing folds are launched (existing checkpoints auto-detected and skipped).
#
# Runs (and original epoch targets):
#   chaos_v26_6_2_train025_val000_20260616_010628        folds 2,3   (200 ep)
#   chaos_v26_6_2_train090_val100_20260615_213615        folds 1,2,3 (300 ep)
#   chaos_auglabAug_synthseg_EM_train100_val000_20260615_213615  folds 2,3   (300 ep)
#   chaos_auglabAug_v26_6_2_train025_val000_20260616_010628      folds 2,3   (200 ep)
#   chaos_auglabAug_v26_6_2_train050_val000_20260615_213615      folds 1,2,3 (300 ep)
#
# Usage:
#   bash 04_24_launch_resume_incomplete.sh

set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[$(date '+%H:%M:%S')] Resuming 5 incomplete runs (queued, LAUNCH_WAIT=1)"

# 1. v26_6_2_train025_val000 — folds 2,3 missing
echo "[$(date '+%H:%M:%S')] [1/5] chaos_v26_6_2_train025_val000_20260616_010628 — folds 2,3"
FOLD_SLOT_GPU="2,2,2 3,3,3" LAUNCH_WAIT=1 \
  bash "${SCRIPTS_DIR}/04_12_train_v26_6_2_train025_val000.sh" \
  "chaos_v26_6_2_train025_val000_20260616_010628"

# 2. v26_6_2_train090_val100 — folds 1,2,3 missing
echo "[$(date '+%H:%M:%S')] [2/5] chaos_v26_6_2_train090_val100_20260615_213615 — folds 1,2,3"
FOLD_SLOT_GPU="1,1,1 2,2,2 3,3,3" LAUNCH_WAIT=1 \
  bash "${SCRIPTS_DIR}/04_06_train_exp0_v26_6_2_90_100.sh" \
  "chaos_v26_6_2_train090_val100_20260615_213615"

# 3. auglabAug_synthseg_EM_train100_val000 — folds 2,3 missing
echo "[$(date '+%H:%M:%S')] [3/5] chaos_auglabAug_synthseg_EM_train100_val000_20260615_213615 — folds 2,3"
FOLD_SLOT_GPU="2,2,2 3,3,3" LAUNCH_WAIT=1 \
  bash "${SCRIPTS_DIR}/04_08_train_exp2_auglabAug_synthseg_EM.sh" \
  "chaos_auglabAug_synthseg_EM_train100_val000_20260615_213615"

# 4. auglabAug_v26_6_2_train025_val000 — folds 2,3 missing
echo "[$(date '+%H:%M:%S')] [4/5] chaos_auglabAug_v26_6_2_train025_val000_20260616_010628 — folds 2,3"
FOLD_SLOT_GPU="2,2,2 3,3,3" LAUNCH_WAIT=1 \
  bash "${SCRIPTS_DIR}/04_13_train_auglabAug_v26_6_2_train025_val000.sh" \
  "chaos_auglabAug_v26_6_2_train025_val000_20260616_010628"

# 5. auglabAug_v26_6_2_train050_val000 — folds 1,2,3 missing
echo "[$(date '+%H:%M:%S')] [5/5] chaos_auglabAug_v26_6_2_train050_val000_20260615_213615 — folds 1,2,3"
FOLD_SLOT_GPU="1,1,1 2,2,2 3,3,3" LAUNCH_WAIT=1 \
  bash "${SCRIPTS_DIR}/04_09_train_auglabAug_v26_6_2_train050_val000.sh" \
  "chaos_auglabAug_v26_6_2_train050_val000_20260615_213615"

echo "[$(date '+%H:%M:%S')] All incomplete runs resumed."
