#!/usr/bin/env bash
# Neutral job-launch interface, sourced from each dataset's 00_utils/env.sh.
# Exposes run_job() with one calling convention; the backend (set_slot on the
# original persistent workstation, Slurm on Vulcan) is auto-detected, or
# pinned explicitly via RUN_JOB_BACKEND=set_slot|slurm.
#
#   run_job --name NAME [--gpus N] [--cpus N] [--mem SIZE] [--time HH:MM:SS]
#           [--slot SPEC] [--log FILE] [--wait] -- CMD...
#
# --gpus/--cpus/--mem/--time: only consulted by the slurm backend (a set_slot
#   slot already carries a fixed GPU+CPU bundle).
# --slot SPEC: only consulted by the set_slot backend (which physical
#   slot/GPU range, e.g. "0" or "0-1") — Slurm assigns GPUs itself.
# --log FILE: redirect stdout+stderr.
# --wait: block until the job finishes. Default is fire-and-exit (the job
#   keeps running after this call returns) — for Slurm that means each call
#   is its own independent sbatch job, since there's no persistent session to
#   background into the way set_slot has.

_RUN_JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${RUN_JOB_BACKEND:-}" ]; then
    if command -v set_slot >/dev/null 2>&1; then
        RUN_JOB_BACKEND=set_slot
    elif command -v sbatch >/dev/null 2>&1; then
        RUN_JOB_BACKEND=slurm
    else
        echo "run_job.sh: no backend found (neither 'set_slot' nor 'sbatch' is on PATH)" >&2
        return 1 2>/dev/null || exit 1
    fi
fi

case "${RUN_JOB_BACKEND}" in
    set_slot) source "${_RUN_JOB_DIR}/run_job_set_slot.sh" ;;
    slurm)    source "${_RUN_JOB_DIR}/run_job_slurm.sh" ;;
    *)
        echo "run_job.sh: unknown RUN_JOB_BACKEND='${RUN_JOB_BACKEND}' (expected set_slot|slurm)" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac
