#!/usr/bin/env bash
# TamIA SIZING PROBE for isles2022 (run ON TamIA; do this BEFORE the real pack launch, and re-do it for every new dataset -- a probe's numbers do NOT transfer).
# It is a short dry run of the REAL mix at the REAL contention level, not a hand-picked 2-method test on idle GPUs (that was 2.6-8.5x too
# optimistic on atlas-liver-hcc): fold 0 of all 6 headline methods x both contrasts = 12 folds packed on ONE whole H100 node (3 per GPU,
# round-robin), PROBE_EPOCHS epochs each, into a THROWAWAY results base ($SCRATCH/isles2022/_probe) so it cannot pollute real checkpoints.
# Read the numbers with 04_25_probe_report.sh <pack dir>: median contended epoch time per method (-> 2000-epoch wall time) + peak GPU memory.
#   bash 04_24_tamia_sizing_probe.sh            # PROBE_EPOCHS=20 PACK_TIME=00:50:00 by default
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/${USER}}"
# whole-node H100: 12 CPUs + 115 GiB per GPU (GiB, never raw MB); set BEFORE env.sh (common_env freezes run_job defaults)
export RUN_JOB_CPUS_PER_GPU=12 RUN_JOB_MEM_PER_GPU=115G
source "${HERE}/../00_utils/env.sh"
source "${ROOT}/scripts/cluster/tamia_env_isles2022.sh"
PROBE="${SCRATCH}/isles2022/_probe"
export PREDICTIONS_ROOT="${PROBE}/01_predictions"        # throwaway base; env.sh derives nnUNet_results from it
PACK_DIR="${PROBE}/pack_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${PACK_DIR}" "${PREDICTIONS_ROOT}"
export RUN_JOB_PACK_DIR="${PACK_DIR}" TRAIN_FOLDS="0" NNUNET_NUM_EPOCHS="${PROBE_EPOCHS:-20}"
echo "[probe] PACK_DIR=${PACK_DIR} epochs=${NNUNET_NUM_EPOCHS} results base=${PREDICTIONS_ROOT}"
n=0
for C in dwi flair; do
    if [ "${C}" = dwi ]; then WRAPPERS=( "${HERE}"/04_0[1-6]_train_dwi_*.sh ); else WRAPPERS=( "${HERE}"/04_{08,09,10,11,12,13}_train_flair_*.sh ); fi
    [ "${#WRAPPERS[@]}" = 6 ] || { echo "ERROR: expected 6 ${C} headline wrappers, got ${#WRAPPERS[@]}" >&2; exit 1; }
    for w in "${WRAPPERS[@]}"; do
        M="$(grep -m1 -E '^METHOD="' "$w" | sed -E 's/^METHOD="([^"]*)".*/\1/')"
        echo "[probe] record ${C}/${M}"
        # subshell: each wrapper sources its own env (env.sh for dwi, env_flair.sh for flair); exports above (tamia paths, PREDICTIONS_ROOT) persist
        ( bash "$w" "isles2022_${C}_${M}_PROBE" )
    done
done
N=$(grep -c . "${PACK_DIR}/index.tsv"); echo "[probe] recorded ${N} fold commands (expect 12)"; [ "${N}" = "12" ] || { echo "ERROR: expected 12" >&2; exit 1; }
# contrast sanity: each recorded cmd must name exactly one of the two contrasts' result dirs
for f in "${PACK_DIR}"/*.sh; do case "$f" in *_packjob_body.sh) continue;; esac
    grep -q "isles2022_model/dwi/" "$f" && grep -q "isles2022_model/flair/" "$f" && { echo "ERROR: $f mixes contrasts" >&2; exit 1; }
    grep -q "isles2022_model/\(dwi\|flair\)/" "$f" || { echo "ERROR: $f names no contrast results dir" >&2; exit 1; }
done
PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${PACK_TIME:-00:50:00}" PACK_CHAIN=1 PACK_JOB_NAME=isles2022_probe \
    bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PACK_DIR}"
echo "[probe] submitted. Report when done:  bash ${HERE}/04_25_probe_report.sh ${PACK_DIR}"
