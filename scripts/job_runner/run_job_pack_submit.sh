#!/usr/bin/env bash
# run_job_pack_submit.sh — companion to run_job's NODE-PACK mode (RUN_JOB_PACK_DIR).
#
# On clusters that allocate GPUs only by whole node (e.g. tamia: "h100 GPUs are
# only allocated by node", 4 GPUs/node) AND cap GPU walltime below what a full
# training needs (tamia max 24h; brats 2500ep @ ~63s/ep with 2 folds/GPU ≈ 44h),
# the workflow is:
#
#   1. Populate a pack dir by running the usual method wrappers with
#      RUN_JOB_PACK_DIR set — run_job RECORDS each fold's command into
#      <dir>/<foldname>.sh + a line in <dir>/index.tsv (cols:
#      cmdfile <TAB> log <TAB> name <TAB> donefile[checkpoint_final.pth]),
#      instead of submitting. The recorded cmd detects resume at RUNTIME
#      (train_common.sh pack mode) so it is safe to re-run across a chain.
#   2. Call THIS script: it submits a CHAIN of PACK_CHAIN whole-node jobs
#      (each --dependency=afterany:<prev>), all running the same recorded folds
#      pinned across the node's GPUs round-robin (6 over 4 = 2,2,1,1). Job N+1
#      resumes job N's checkpoints. Each job first checks the done-markers and
#      exits immediately if every fold already has checkpoint_final (so the tail
#      of the chain is a cheap no-op once training finishes).
#
# Each fold trains at its own default config (batch size unchanged) — this is
# purely a placement/utilisation + walltime-chaining layer.
#
# Usage:  bash run_job_pack_submit.sh <PACK_DIR>
#
# Tunables (env; tamia-oriented defaults):
#   PACK_NODE_GPUS=4  PACK_GPU_TYPE=h100  PACK_CPUS=48  PACK_MEM=0
#   PACK_TIME=23:59:00      per-job walltime (keep < cluster cap so it routes to the long partition)
#   PACK_CHAIN=3            number of chained jobs (>= ceil(total_hours / PACK_TIME))
#   PACK_ACCOUNT=aip-jcohen PACK_USE_MPS=0  PACK_JOB_NAME=nodepack
set -euo pipefail

PACK_DIR="${1:?usage: run_job_pack_submit.sh <PACK_DIR>}"
INDEX="${PACK_DIR}/index.tsv"
[ -f "${INDEX}" ] || { echo "ERROR: no index.tsv in ${PACK_DIR} (nothing recorded)" >&2; exit 1; }
N_FOLDS="$(grep -c . "${INDEX}")"
[ "${N_FOLDS}" -gt 0 ] || { echo "ERROR: index.tsv is empty" >&2; exit 1; }

PACK_NODE_GPUS="${PACK_NODE_GPUS:-4}"
PACK_GPU_TYPE="${PACK_GPU_TYPE:-h100}"
PACK_CPUS="${PACK_CPUS:-48}"
PACK_MEM="${PACK_MEM:-0}"
PACK_TIME="${PACK_TIME:-23:59:00}"
PACK_CHAIN="${PACK_CHAIN:-3}"
PACK_ACCOUNT="${PACK_ACCOUNT:-aip-jcohen}"
PACK_USE_MPS="${PACK_USE_MPS:-0}"
PACK_JOB_NAME="${PACK_JOB_NAME:-nodepack}"

echo "[pack] ${N_FOLDS} folds over ${PACK_NODE_GPUS} GPU(s) (round-robin i%G), chain of ${PACK_CHAIN} x ${PACK_TIME}:"
_i=0
while IFS=$'\t' read -r _cmd _log _name _done; do
    [ -n "${_cmd}" ] || continue
    echo "  GPU $((_i % PACK_NODE_GPUS))  <-  ${_name}"
    _i=$((_i + 1))
done < "${INDEX}"

# One persistent job-body script, reused by every job in the chain.
BODY="${PACK_DIR}/_packjob_body.sh"
{
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=${PACK_JOB_NAME}"
    echo "#SBATCH --account=${PACK_ACCOUNT}"
    echo "#SBATCH --time=${PACK_TIME}"
    echo "#SBATCH --nodes=1"
    echo "#SBATCH --gpus-per-node=${PACK_GPU_TYPE}:${PACK_NODE_GPUS}"
    echo "#SBATCH --cpus-per-task=${PACK_CPUS}"
    echo "#SBATCH --mem=${PACK_MEM}"
    echo "#SBATCH --output=${PACK_DIR}/pack_%j.out"
    cat <<SBODY
set -uo pipefail
G=${PACK_NODE_GPUS}
INDEX='${INDEX}'
USE_MPS=${PACK_USE_MPS}

# Skip if every fold already finished (chain tail becomes a cheap no-op).
all_done=1
while IFS=\$'\t' read -r cmdfile log name donefile; do
    [ -n "\$cmdfile" ] || continue
    if [ -z "\$donefile" ] || [ ! -f "\$donefile" ]; then all_done=0; fi
done < "\$INDEX"
if [ "\$all_done" = "1" ]; then
    echo "[pack-job] all folds already have checkpoint_final — nothing to do, exiting."
    exit 0
fi

echo "[pack-job] host=\$(hostname) job=\$SLURM_JOB_ID gpus:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

if [ "\${USE_MPS}" = "1" ]; then
    export CUDA_MPS_PIPE_DIRECTORY="\${SLURM_TMPDIR:-/tmp}/mps_pipe_\$SLURM_JOB_ID"
    export CUDA_MPS_LOG_DIRECTORY="\${SLURM_TMPDIR:-/tmp}/mps_log_\$SLURM_JOB_ID"
    mkdir -p "\$CUDA_MPS_PIPE_DIRECTORY" "\$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d && echo "[pack-job] CUDA MPS daemon started"
fi

declare -a PIDS NAMES GPUS
i=0
while IFS=\$'\t' read -r cmdfile log name donefile; do
    [ -n "\$cmdfile" ] || continue
    gpu=\$(( i % G ))
    plog="\${cmdfile%.sh}.log"   # persistent (pack dir on scratch), not node-local /tmp
    echo "[pack-job] launch '\$name' on GPU \$gpu -> \$plog"
    CUDA_VISIBLE_DEVICES=\$gpu bash "\$cmdfile" > "\$plog" 2>&1 &
    PIDS[\$i]=\$!; NAMES[\$i]="\$name"; GPUS[\$i]=\$gpu
    i=\$(( i + 1 ))
    sleep 5
done < "\$INDEX"

echo "[pack-job] \$i folds launched; waiting..."
rc_all=0
for j in "\${!PIDS[@]}"; do
    if wait "\${PIDS[\$j]}"; then echo "[pack-job] OK   '\${NAMES[\$j]}' (GPU \${GPUS[\$j]})"
    else rc=\$?; rc_all=1; echo "[pack-job] FAIL '\${NAMES[\$j]}' (GPU \${GPUS[\$j]}) exit=\$rc"; fi
done

if [ "\${USE_MPS}" = "1" ]; then echo quit | nvidia-cuda-mps-control || true; fi
echo "[pack-job] done rc_all=\$rc_all"
exit \$rc_all
SBODY
} > "${BODY}"

# PACK_DEPENDENCY (optional): Slurm dependency string (e.g. "afterany:383046") applied
# to ONLY the first job in this chain — lets this pack run natively wait on another
# job already in THIS cluster's queue (e.g. a training pack chain), via Slurm's own
# dependency mechanism, with zero external polling. Subsequent chain jobs still depend
# on their own predecessor as before. Empty (default): first job submits immediately.
PACK_DEPENDENCY="${PACK_DEPENDENCY:-}"

echo "[pack] submitting dependency chain of ${PACK_CHAIN} whole-node job(s) (${PACK_GPU_TYPE}:${PACK_NODE_GPUS}, ${PACK_TIME} each, MPS=${PACK_USE_MPS})..."
prev=""
for ((c=1; c<=PACK_CHAIN; c++)); do
    if [ -z "${prev}" ]; then
        if [ -n "${PACK_DEPENDENCY}" ]; then
            out="$(sbatch --dependency="${PACK_DEPENDENCY}" "${BODY}")"
        else
            out="$(sbatch "${BODY}")"
        fi
    else
        out="$(sbatch --dependency=afterany:"${prev}" "${BODY}")"
    fi
    echo "  [chain ${c}/${PACK_CHAIN}] ${out}${prev:+  (after ${prev})}"
    prev="$(echo "${out}" | grep -oE '[0-9]+' | tail -1)"
done
echo "[pack] chain submitted; last job id=${prev}"
