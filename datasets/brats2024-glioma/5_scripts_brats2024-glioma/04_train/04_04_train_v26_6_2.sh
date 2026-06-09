#!/usr/bin/env bash
# Train V26_6_2 (whole-image synthesis + per-label remap) on BraTS 2024 Glioma.
#
# Fresh run:
#   bash 04_04_train_v26_6_2.sh                       # new RUN_ID, trains to NNUNET_NUM_EPOCHS (default 1000)
#
# RESUME / EXTEND an existing run (see 04_00_common.sh "RESUME NOTES" for the full story):
#   NNUNET_NUM_EPOCHS=1600 bash 04_04_train_v26_6_2.sh <RUN_ID>
#     - Pass the existing RUN_ID → common.sh resumes each fold from checkpoint_latest
#       (or checkpoint_final for a completed run), GPU-pinned, then EXITS (fire-and-exit;
#       the set_slot jobs keep running). Just run it foreground — it returns after the
#       ~45s launch stagger. Do NOT wrap in `timeout` (kills the folds) or `nohup … &`
#       (reaped before it fires). See RESUME NOTES #1.
#     - To train PAST the original length, raise NNUNET_NUM_EPOCHS (the PolyLR horizon).
#     - WandB auto-resumes the same run via each fold's wandb/ dir. If that dir was
#       deleted, pass RESUME_WANDB_IDS="id0 id1 id2 id3" (one cloud id per fold).
#
# Example (this run): resume to 1600 keeping the original wandb runs:
#   RESUME_WANDB_IDS="rx5xtndf mymxmv1h uem9bokf i1evwzl4" NNUNET_NUM_EPOCHS=1600 \
#     bash 04_04_train_v26_6_2.sh v26_6_2_20260608_003445

source "$(dirname "$0")/../00_utils/env.sh"

METHOD="v26_6_2"
TRAINER="nnUNetTrainerBraTS2024GliomaV26_6_2"
DATASET_ID="051"
DA_WORKERS=16
LOG_DIR="/tmp/nnunet_brats2024_v26_6_2"
export nnUNet_compile=1
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-1000}"   # override to extend on resume
source "$(dirname "$0")/04_00_common.sh" "$@"
