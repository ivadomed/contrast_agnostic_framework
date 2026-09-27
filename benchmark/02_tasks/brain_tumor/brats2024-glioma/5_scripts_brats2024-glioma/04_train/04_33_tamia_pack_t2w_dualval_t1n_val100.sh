#!/usr/bin/env bash
# TAMIA whole-node pack launcher (H100 nodes are allocated only by whole node,
# and GPU walltime is capped at 24h < the ~44h a 2500-epoch brats run needs).
# Runs all 6 folds of our two auglabAug_v26_6_2 brats runs on ONE 4xH100 node:
#   - t2w dual-val   (04_32 -> nnUNetTrainerBraTS2024GliomaT2wAugLabDualVal, folds 0 1 2)
#   - t1n val100     (04_30 -> nnUNetTrainerBraTS2024GliomaAugLabValSynth,  folds 0 1 2)
# packed across the 4 GPUs round-robin (2,2,1,1), as a dependency CHAIN of
# whole-node jobs that resume each other via checkpoint_latest (run_job_pack_submit.sh).
# Each fold trains at its OWN default config (batch size unchanged) — pure placement.
#
# RUN_IDs are FIXED (persisted in the pack dir) so every job in the chain — and any
# manual re-invocation to extend the chain — targets the SAME run directories and
# resumes rather than restarting. Recording happens ONCE per pack dir; re-invoking
# with the same PACK_DIR reuses the recording and just submits a fresh chain.
#
# Usage (run ON tamia):
#   bash 04_33_tamia_pack_t2w_dualval_t1n_val100.sh                 # fresh launch (new pack dir)
#   PACK_DIR=<existing> bash 04_33_...                              # extend chain / resume
#   PACK_CHAIN=4 PACK_USE_MPS=1 bash 04_33_...                      # tune chain length / MPS
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"

PACK_DIR="${PACK_DIR:-/scratch/p/paulh/brats2024-glioma/_packruns/pack_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${PACK_DIR}"
echo "[tamia-pack] PACK_DIR=${PACK_DIR}"

# Fixed RUN_IDs (persist once, reuse forever) so the whole chain resumes the same run.
RUNIDS="${PACK_DIR}/RUN_IDS.env"
if [ -f "${RUNIDS}" ]; then
    # shellcheck disable=SC1090
    source "${RUNIDS}"
    echo "[tamia-pack] reusing RUN_IDs from ${RUNIDS}"
else
    TS="$(date +%Y%m%d_%H%M%S)"
    T2W_RUN_ID="brats2024-glioma_t2w_auglabAug_v26_6_2_train050_val000_${TS}"
    T1N_RUN_ID="brats2024-glioma_t1n_auglabAug_v26_6_2_train050_val100_${TS}"
    printf 'T2W_RUN_ID=%s\nT1N_RUN_ID=%s\n' "${T2W_RUN_ID}" "${T1N_RUN_ID}" > "${RUNIDS}"
    echo "[tamia-pack] generated RUN_IDs -> ${RUNIDS}"
fi
echo "  t2w: ${T2W_RUN_ID}"
echo "  t1n: ${T1N_RUN_ID}"

# Record the 6 folds ONCE (each wrapper in pack-record mode, own subshell → no env bleed).
if [ -f "${PACK_DIR}/index.tsv" ]; then
    echo "[tamia-pack] index.tsv exists — reusing existing recording (resume/extend)."
else
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_32_train_t2w_auglabAug_v26_6_2_dualval.sh"           "${T2W_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_30_train_t1n_auglabAug_v26_6_2_train050_val100.sh"    "${T1N_RUN_ID}"
    echo "[tamia-pack] recorded folds:"; cut -f3 "${PACK_DIR}/index.tsv" | sed 's/^/  /'
fi

# Submit the whole-node dependency chain.
PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${PACK_TIME:-23:59:00}" PACK_CHAIN="${PACK_CHAIN:-3}" \
PACK_USE_MPS="${PACK_USE_MPS:-0}" PACK_JOB_NAME="${PACK_JOB_NAME:-brats_pack_t2w_t1n}" \
  bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PACK_DIR}"
