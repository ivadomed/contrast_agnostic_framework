#!/usr/bin/env bash
# TAMIA whole-node pack launcher for picai-prostate — ONE pack (= one 4xH100 node).
# Normally invoked by 04_21_tamia_pack_launch_all.sh, which defines all 6 packs; run it
# directly only to (re)launch or extend a single pack.
#
# WHY PACKS AT ALL: tamia allocates H100s only by WHOLE NODE and caps GPU walltime at 24h,
# while a 2000-epoch picai-prostate fold needs far longer. So each pack records its folds
# (RUN_JOB_PACK_DIR → run_job records instead of submitting), pins them across the node's 4
# GPUs, and submits a CHAIN of whole-node jobs that resume each other from checkpoint_latest
# (scripts/job_runner/run_job_pack_submit.sh). A chain job whose folds all have
# checkpoint_final exits immediately, so the tail of the chain is a cheap no-op.
#
# RUN_IDs are FIXED (persisted in the pack dir's RUN_IDS.env) so every job in the chain — and
# any later re-invocation to extend it — targets the SAME run directories and RESUMES rather
# than restarting. Recording happens ONCE per pack dir; re-invoking with the same PACK_DIR
# reuses the recording and just submits a fresh chain.
#
# Usage:
#   bash 04_20_tamia_pack.sh <pack_name> <gpu_map> <chain_len> <wrapper.sh> [wrapper2.sh ...]
#     pack_name   short id, becomes the pack dir name + Slurm job name
#     gpu_map     PACK_GPU_MAP: one GPU index per recorded fold, in record order, or "-"
#                 for the default round-robin (i%4). See run_job_pack_submit.sh.
#     chain_len   number of chained 23:59:00 whole-node jobs
#   PACK_DIR=<existing> bash 04_20_tamia_pack.sh ...   # extend/resume an existing pack
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"

PACK_NAME="${1:?usage: 04_20_tamia_pack.sh <pack_name> <gpu_map> <chain_len> <wrapper.sh>...}"
GPU_MAP_ARG="${2:?missing gpu_map ('-' for round-robin)}"
CHAIN_LEN="${3:?missing chain_len}"
shift 3
[ "$#" -gt 0 ] || { echo "ERROR: no method wrappers given" >&2; exit 1; }

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"
PACK_DIR="${PACK_DIR:-${SCRATCH}/picai-prostate/_packruns/${PACK_NAME}}"
mkdir -p "${PACK_DIR}"
echo "[tamia-pack] ${PACK_NAME}: PACK_DIR=${PACK_DIR}"

# Fixed RUN_IDs, one per wrapper, persisted on first call. The id is derived from the
# wrapper's own filename (04_NN_train_<contrast>_<method>.sh) so it matches exactly what
# train_common.sh would have auto-generated — and so a resume names the same directory.
RUNIDS="${PACK_DIR}/RUN_IDS.env"
if [ ! -f "${RUNIDS}" ]; then
    TS="$(date +%Y%m%d_%H%M%S)"
    : > "${RUNIDS}"
    for w in "$@"; do
        base="$(basename "$w" .sh)"
        rest="${base#04_*_train_}"          # <contrast>_<method>
        contrast="${rest%%_*}"
        method="${rest#*_}"
        printf '%s\n' "picai-prostate_${contrast}_${method}_${TS}" >> "${RUNIDS}"
    done
    echo "[tamia-pack] generated RUN_IDs → ${RUNIDS}"
else
    echo "[tamia-pack] reusing RUN_IDs from ${RUNIDS}"
fi
mapfile -t RUN_IDS < "${RUNIDS}"
if [ "${#RUN_IDS[@]}" -ne "$#" ]; then
    echo "ERROR: ${RUNIDS} has ${#RUN_IDS[@]} ids but $# wrappers were given — the pack dir" \
         "was built from a different wrapper list. Use a fresh PACK_DIR." >&2
    exit 1
fi
for id in "${RUN_IDS[@]}"; do echo "    ${id}"; done

# Record every wrapper's folds ONCE (each in its own subshell → no env bleed between them).
if [ -f "${PACK_DIR}/index.tsv" ]; then
    echo "[tamia-pack] index.tsv exists — reusing existing recording (resume/extend)."
else
    i=0
    for w in "$@"; do
        echo "[tamia-pack] recording $(basename "$w") as ${RUN_IDS[$i]}"
        RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/$(basename "$w")" "${RUN_IDS[$i]}"
        i=$((i + 1))
    done
    echo "[tamia-pack] recorded folds:"; cut -f3 "${PACK_DIR}/index.tsv" | sed 's/^/  /'
fi

PACK_GPU_MAP=""
[ "${GPU_MAP_ARG}" != "-" ] && PACK_GPU_MAP="${GPU_MAP_ARG}"

PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_CPUS=48 PACK_MEM=0 \
PACK_TIME="${PACK_TIME:-23:59:00}" PACK_CHAIN="${CHAIN_LEN}" \
PACK_GPU_MAP="${PACK_GPU_MAP}" PACK_USE_MPS="${PACK_USE_MPS:-0}" \
PACK_JOB_NAME="picai_${PACK_NAME}" \
  bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PACK_DIR}"
