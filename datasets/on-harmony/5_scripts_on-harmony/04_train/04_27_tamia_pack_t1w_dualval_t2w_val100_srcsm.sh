#!/usr/bin/env bash
# TAMIA whole-node pack launcher (H100 nodes are allocated only by whole node, and GPU
# walltime is capped at 24h). Moves ALL pending on-harmony training off Killarney onto
# ONE 4xH100 node, packed across the 4 GPUs round-robin (7 folds -> 2,2,2,1), as a
# dependency CHAIN of whole-node jobs that resume each other via checkpoint_latest
# (run_job_pack_submit.sh). Each fold trains at its OWN default config (batch size
# unchanged) — pure placement/utilisation, per project convention.
#
#   - T1w dual-val   (04_26 -> nnUNetTrainerOnHarmonyAugLabDualVal, folds 0 1 2)
#                    neither train050_val000 nor train050_val100 existed yet for T1w —
#                    DualVal trains once and gets BOTH.
#   - T2w val100     (04_24 -> nnUNetTrainerOnHarmonyAugLabValSynth, folds 0 1 2)
#                    T2w's val000 sibling already exists (trained 2026-07-10) — this is
#                    a REGULAR standalone val100 training, not dual-val.
#   - T1w srcsm      (04_20 -> nnUNetTrainerOnHarmonyAugLabDefault, fold 2 only)
#                    folds 0/1 already have checkpoint_final (done); fold 2 resumes from
#                    ~epoch 1773/2000 (checkpoint_latest, synced from Killarney). Fixed
#                    existing RUN_ID: on-harmony_T1w_srcsm_20260709_122115.
#
# MPS OFF (PACK_USE_MPS=0, the run_job_pack_submit.sh default) — tested MPS=1 here first
# (utilisation lesson from the brats tamia run: 2 folds/GPU without MPS averaged ~50-65%
# compute, <20% VRAM) but it HUNG all 7 folds outright (zero epoch progress, 40min
# timeout) in a smoke test; identical pack with MPS=0 trained normally within minutes.
# Root cause not yet debugged — see project_tamia_node_pack memory correction
# (2026-07-27). Plain time-slicing (no MPS) is what this pack actually runs on. Batch
# size / training config UNCHANGED throughout — only placement, per explicit
# instruction not to touch the science.
#
# RUN_IDs are FIXED (persisted in the pack dir) so every job in the chain — and any
# manual re-invocation to extend the chain — targets the SAME run directories and
# resumes rather than restarting. Recording happens ONCE per pack dir; re-invoking with
# the same PACK_DIR reuses the recording and just submits a fresh chain.
#
# Usage (run ON tamia):
#   bash 04_27_tamia_pack_t1w_dualval_t2w_val100_srcsm.sh                 # fresh launch
#   PACK_DIR=<existing> bash 04_27_...                                    # extend chain / resume
#   PACK_CHAIN=4 bash 04_27_...                                           # tune chain length
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"

PACK_DIR="${PACK_DIR:-/scratch/p/paulh/on-harmony/_packruns/pack_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${PACK_DIR}"
echo "[tamia-pack] PACK_DIR=${PACK_DIR}"

# Fixed RUN_IDs (persist once, reuse forever) so the whole chain resumes the same run.
# SRCSM_RUN_ID is NOT generated — it's the EXISTING run being resumed (see header).
RUNIDS="${PACK_DIR}/RUN_IDS.env"
if [ -f "${RUNIDS}" ]; then
    # shellcheck disable=SC1090
    source "${RUNIDS}"
    echo "[tamia-pack] reusing RUN_IDs from ${RUNIDS}"
else
    TS="$(date +%Y%m%d_%H%M%S)"
    T1W_RUN_ID="on-harmony_T1w_auglabAug_v26_6_2_train050_val000_${TS}"
    T2W_RUN_ID="on-harmony_T2w_auglabAug_v26_6_2_train050_val100_${TS}"
    SRCSM_RUN_ID="on-harmony_T1w_srcsm_20260709_122115"
    printf 'T1W_RUN_ID=%s\nT2W_RUN_ID=%s\nSRCSM_RUN_ID=%s\n' "${T1W_RUN_ID}" "${T2W_RUN_ID}" "${SRCSM_RUN_ID}" > "${RUNIDS}"
    echo "[tamia-pack] generated RUN_IDs -> ${RUNIDS}"
fi
echo "  T1w dualval: ${T1W_RUN_ID}"
echo "  T2w val100:  ${T2W_RUN_ID}"
echo "  T1w srcsm:   ${SRCSM_RUN_ID} (resume, fold 2 only)"

# Record the 7 folds ONCE (each wrapper in pack-record mode, own subshell → no env bleed).
if [ -f "${PACK_DIR}/index.tsv" ]; then
    echo "[tamia-pack] index.tsv exists — reusing existing recording (resume/extend)."
else
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_26_train_t1w_auglabAug_v26_6_2_dualval.sh"       "${T1W_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_24_train_t2w_auglabAug_v26_6_2_train050_val100.sh" "${T2W_RUN_ID}"
    # srcsm: only fold 2 needs work (folds 0/1 already have checkpoint_final). Use
    # SINGLE_FOLD to record just that one fold against the existing RUN_ID.
    SINGLE_FOLD=2 RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_20_train_t1w_srcsm.sh" "${SRCSM_RUN_ID}"
    echo "[tamia-pack] recorded folds:"; cut -f3 "${PACK_DIR}/index.tsv" | sed 's/^/  /'
fi

# Submit the whole-node dependency chain. MPS on by default (see header).
PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${PACK_TIME:-23:59:00}" PACK_CHAIN="${PACK_CHAIN:-3}" \
PACK_USE_MPS="${PACK_USE_MPS:-0}" PACK_JOB_NAME="${PACK_JOB_NAME:-onharmony_pack}" \
  bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PACK_DIR}"
