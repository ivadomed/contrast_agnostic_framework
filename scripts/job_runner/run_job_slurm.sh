#!/usr/bin/env bash
# run_job backend: Slurm. Cluster-neutral — used on both Vulcan and Killarney
# (and any other Slurm cluster). Submits each call as its own sbatch job —
# true fire-and-forget (the scheduler keeps it alive, not this shell), since
# there's no persistent multi-GPU session to background into the way
# set_slot has. --wait maps to `sbatch --wait` to block until the job
# finishes (needed for call sites that today background+wait on a *group* of
# launches, e.g. "launch 4 folds, wait for all 4 before the next step").
#
# Only the GPU gres type (RUN_JOB_GPU_TYPE) is cluster-specific; everything
# else (cpus/mem-per-gpu, account, time) is env-parametrised with defaults
# below. Do NOT fork a per-cluster copy of this file — export the
# RUN_JOB_* vars instead (e.g. in a per-cluster env.sh sourced before
# run_job.sh) so this stays the single backend-neutral interface.
#
# Per-node specs confirmed via
#   sinfo -o "%G" --Node | sort -u   and   sinfo -o "%N %c %m %G"
# Vulcan (2026-06-19): gpu:l40s:4, 64 CPUs / ~503G mem / 4 GPUs per node.
# Killarney (2026-07-06): gpu:l40s:4 (kn001-168, 64 CPU/515000M/4 GPU) and
#   gpu:h100:8 (kn169-178, 48+ CPU/2060000M/8 GPU) — defaults below target
#   the l40s nodes (same 16 cpu/gpu, ~110G/gpu math as Vulcan). Re-confirm
#   with the same commands if this ever looks stale — never trust these
#   numbers blindly (see CLAUDE.md).
#
# WandB: Slurm compute nodes have no internet access, so jobs submitted here
# run with WANDB_MODE=offline. Sync later from the login node with
# scripts/job_runner/wandb_sync.sh.

: "${RUN_JOB_ACCOUNT:=aip-jcohen}"
: "${RUN_JOB_TIME_DEFAULT:=24:00:00}"
: "${RUN_JOB_CPUS_PER_GPU:=16}"     # 64 cpus / 4 GPUs per L40S node
: "${RUN_JOB_MEM_PER_GPU:=110G}"    # ~503G/node / 4 GPUs, with headroom
: "${RUN_JOB_CPUS_DEFAULT:=4}"      # used when --gpus 0 (CPU-only jobs)
: "${RUN_JOB_MEM_DEFAULT:=16G}"
: "${RUN_JOB_GPU_TYPE:=l40s}"       # gres GPU type; set empty for clusters using plain "gpu:N"
: "${RUN_JOB_EXCLUDE_NODES:=}"      # optional comma-separated nodelist to avoid (e.g. a
                                    # node just found to throw CUDA ECC errors) — maps to
                                    # sbatch --exclude; empty (default) omits the flag

run_job() {
    local name="job" gpus=0 cpus="" mem="" time="${RUN_JOB_TIME_DEFAULT}" log="" wait_flag=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --name) name="$2"; shift 2 ;;
            --gpus) gpus="$2"; shift 2 ;;
            --cpus) cpus="$2"; shift 2 ;;
            --mem)  mem="$2"; shift 2 ;;
            --time) time="$2"; shift 2 ;;
            --slot) shift 2 ;;   # not meaningful on Slurm — it assigns GPUs itself
            --log)  log="$2"; shift 2 ;;
            --wait) wait_flag=1; shift ;;
            --) shift; break ;;
            *) echo "run_job (slurm): unknown arg '$1'" >&2; return 1 ;;
        esac
    done

    if [ -z "${cpus}" ]; then
        if [ "${gpus}" -gt 0 ]; then cpus=$((RUN_JOB_CPUS_PER_GPU * gpus)); else cpus="${RUN_JOB_CPUS_DEFAULT}"; fi
    fi
    if [ -z "${mem}" ]; then
        if [ "${gpus}" -gt 0 ]; then mem="$(( ${RUN_JOB_MEM_PER_GPU%G} * gpus ))G"; else mem="${RUN_JOB_MEM_DEFAULT}"; fi
    fi
    : "${log:=/dev/null}"

    local script
    script="$(mktemp --suffix=.sh)"
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=${name}"
        echo "#SBATCH --account=${RUN_JOB_ACCOUNT}"
        echo "#SBATCH --time=${time}"
        echo "#SBATCH --cpus-per-task=${cpus}"
        echo "#SBATCH --mem=${mem}"
        if [ "${gpus}" -gt 0 ]; then
            echo "#SBATCH --gres=gpu${RUN_JOB_GPU_TYPE:+:$RUN_JOB_GPU_TYPE}:${gpus}"
        fi
        if [ -n "${RUN_JOB_EXCLUDE_NODES}" ]; then
            echo "#SBATCH --exclude=${RUN_JOB_EXCLUDE_NODES}"
        fi
        echo "#SBATCH --output=${log}"
        # nnUNet_wandb_mode, NOT WANDB_MODE: this project's WandbLogger patch
        # (src/nnunet/patches/nnunet_logger.py) reads its own nnUNet_wandb_mode
        # var (default "online"), ignoring the generic wandb-SDK WANDB_MODE
        # entirely. Setting only WANDB_MODE here silently no-ops: wandb.init()
        # still runs mode="online" -> tries to authenticate -> every fold dies
        # at trainer __init__ with "No API key configured" before Epoch 0
        # (found on Killarney 2026-07-06 launching the open-ms matrix). Set
        # both so it's correct however a given run reads it.
        echo "export WANDB_MODE=offline"
        echo "export nnUNet_wandb_mode=offline"
        printf '%q ' "$@"
        echo
    } > "${script}"

    if [ "${wait_flag}" = "1" ]; then
        sbatch --wait "${script}"
        local rc=$?
        rm -f "${script}"
        return "${rc}"
    else
        sbatch "${script}"
        rm -f "${script}"
    fi
}
