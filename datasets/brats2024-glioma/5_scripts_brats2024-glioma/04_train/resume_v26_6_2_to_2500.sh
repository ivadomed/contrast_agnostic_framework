#!/usr/bin/env bash
# Explicit one-off resume: v26_6_2 folds 0-3 from epoch 1600 → 2500 (+900 epochs),
# GPU-pinned (fold f → GPU f), continuing the ORIGINAL wandb runs by id.
#
# initial_lr=0.005 so the PolyLR restart LR at epoch 1600 is gentle:
#   0.005 * (1 - 1600/2500)^0.9  ≈  0.00199   (≤ 0.002, as requested)
#   0.002 * (1 - 2500/3000)^0.9 =  
# ⚠️ DO NOT run this with `bash thisfile.sh`, `timeout …`, or `tmux …`.
# set_slot/ml_job uses `systemd-run --pty --wait`, so each training job is tied to its
# launching client's PTY. A sub-script (or timeout, or a Bash-tool-spawned tmux) gets
# reaped when it exits/returns, the PTY closes, and the jobs die ~1 epoch in (silently).
# The ONLY launch that survives is INLINE in a persistent shell. So either:
#   • SOURCE it from an interactive/persistent shell:   source thisfile.sh
#   • or paste the for-loop below directly into the shell.
# (Verified: identical nohup+set_slot dies via `bash file.sh`, survives inline.)
set -uo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project

RES="$PWD/datasets/brats2024-glioma/8_results_brats2024-glioma/01_predictions/nnUNet/brats2024-glioma_v26_6_2_train090_val000_20260608_003445"
RAW="$PWD/datasets/brats2024-glioma/2_nnUNet_brats2024-glioma/raw"
PRE="$PWD/datasets/brats2024-glioma/2_nnUNet_brats2024-glioma/preprocessed"
PP="$PWD/datasets/brats2024-glioma/5_scripts_brats2024-glioma"
TDIR="$RES/Dataset051_BraTS2024GliomaT1n/nnUNetTrainerBraTS2024GliomaV26_6_2__nnUNetPlans__3d_fullres"
IDS=(rx5xtndf mymxmv1h uem9bokf i1evwzl4)
LOGD=/tmp/nnunet_v26_6_2_resume2500
mkdir -p "$LOGD"

for f in 0 1 2 3; do
  # resume exactly from the completed-1600 weights: checkpoint_final → checkpoint_latest
  cp -f "$TDIR/fold_$f/checkpoint_final.pth" "$TDIR/fold_$f/checkpoint_latest.pth"
  # nohup is REQUIRED: without it the set_slot client catches SIGHUP when this
  # orchestrator exits and the training job dies with it (~1 epoch in). With nohup
  # the per-fold set_slot job detaches and survives. (Proven: 1600 resume used nohup.)
  nohup set_slot "$f" env CUDA_VISIBLE_DEVICES="$f" \
    nnUNet_raw="$RAW" nnUNet_preprocessed="$PRE" nnUNet_results="$RES" \
    NNUNET_PROJECT_ROOT="$PWD" PYTHONPATH="$PP" \
    nnUNet_n_proc_DA=16 nnUNet_compile=1 \
    NNUNET_NUM_EPOCHS=3000 NNUNET_INITIAL_LR=0.002 \
    nnUNet_wandb_enabled=1 nnUNet_wandb_project=mri_synthesis_seg_brats2024-glioma \
    nnUNet_wandb_run_name="brats2024-glioma_v26_6_2_train090_val000_20260608_003445_fold${f}" nnUNet_wandb_run_id="${IDS[$f]}" \
    .venv/bin/nnUNetv2_train 051 3d_fullres "$f" --c \
      -tr nnUNetTrainerBraTS2024GliomaV26_6_2 -p nnUNetPlans -num_gpus 1 \
    > "$LOGD/fold${f}.log" 2>&1 &
  echo "fold$f → GPU $f  (wandb ${IDS[$f]})  log: $LOGD/fold${f}.log"
  sleep 5      # stagger: simultaneous set_slot/systemd-run setups race
done
# Fire-and-exit: each fold is an independent set_slot job (own systemd slice) that
# keeps running after this orchestrator exits. Brief settle so the last fold's
# systemd-run job fully detaches before we return. Watch: tail -f $LOGD/fold0.log
sleep 20
echo "all 4 folds launched and detached (1600→2500, lr_start≈0.00199); logs: $LOGD/fold{0..3}.log"
