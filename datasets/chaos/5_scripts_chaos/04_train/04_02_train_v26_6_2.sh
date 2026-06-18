#!/usr/bin/env bash
# Train V26_6_2 (whole-image K-means synthesis + per-label remap) on CHAOS MR T1-in.
# Domain-randomization synthesis baseline-comparison for cross-modality generalization.
#
# Fresh run (4 folds × 500 epochs, 2 folds/GPU on GPU0+GPU1):
#   FOLD_SLOT_GPU="0,0,0 1,1,0 2,2,1 3,3,1" bash 04_02_train_v26_6_2.sh
#
# RESUME / EXTEND (see 04_00_common.sh): pass the existing RUN_ID; raise
# NNUNET_NUM_EPOCHS to train past the original PolyLR horizon. Fire-and-exit — run
# foreground; do NOT wrap in `timeout`/`nohup &`.
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="v26_6_2_train090_val100"
TRAINER="nnUNetTrainerCHAOSV26_6_2"
DATASET_ID="060"
# Worker-side spatial aug pipeline → batchgenerators DA workers (4 folds share 64 CPUs).
DA_WORKERS="${DA_WORKERS:-16}"
LOG_DIR="/tmp/nnunet_chaos_v26_6_2"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-500}"
source "$(dirname "$0")/04_00_common.sh" "$@"
