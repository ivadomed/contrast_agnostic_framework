#!/usr/bin/env bash
# TAMIA SIZING PROBE for autopet CT — REQUIRED before the real pack launch, on BOTH
# modalities (this probes CT only; re-run against env_pet.sh for PET before packing that
# run-all too — SUV intensity/noise characteristics differ from CT's HU scale, so a CT
# measurement must NOT be assumed to transfer to PET, any more than another dataset's
# srcsm ratio transfers here).
#
# CLAUDE.md mandates measuring rather than assuming, and a project memory
# ([[project_tamia_pack_contention_sizing]]) records the specific way this goes wrong: a
# probe that runs N folds on N IDLE GPUs is 2.6-8.5x too optimistic versus the real
# launch, which puts several folds on the SAME GPU. So this probe reproduces the intended
# contention exactly — folds packed onto the node at the same folds-per-GPU ratio the
# real launch will use.
#
# Whole-body PET/CT volumes (skull-base to mid-thigh) are far larger than any other
# dataset trained in this project so far — do not assume ANY prior dataset's per-epoch
# time, VRAM footprint, or srcsm ratio transfers here. This probe is not optional.
#
# Results go to a THROWAWAY results base so they can never pollute real checkpoints.
#
# Usage (run ON tamia, after sourcing env.sh + tamia_env_autopet.sh):
#   bash 04_15_tamia_sizing_probe.sh
#   PROBE_EPOCHS=5 PROBE_TIME=01:00:00 bash 04_15_tamia_sizing_probe.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"

PROBE_EPOCHS="${PROBE_EPOCHS:-6}"
PROBE_TIME="${PROBE_TIME:-01:30:00}"
PROBE_BASE="${PROBE_BASE:-${SCRATCH:-/scratch/p/paulh}/autopet/_probe_$(date +%Y%m%d_%H%M%S)}"
PACK_DIR="${PROBE_BASE}/pack"
mkdir -p "${PACK_DIR}"

# THROWAWAY results root — nothing here is ever read by predict/evaluate.
export PREDICTIONS_ROOT="${PROBE_BASE}/01_predictions"
export nnUNet_results="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/nnUNet"
export RESULTS_DIR="${PROBE_BASE}"
mkdir -p "${nnUNet_results}"
echo "[probe] throwaway results base: ${PROBE_BASE}"

# 4 methods x 2 folds = 8 fold-jobs over 4 GPUs = 2 folds/GPU, the layout the real suite
# pack will use. TRAIN_FOLDS deliberately restricted (the shared driver defaults to
# "0 1 2"); this measures cost, it does not produce results.
export TRAIN_FOLDS="0 1"
TS="$(date +%Y%m%d_%H%M%S)"
# Spans the CHEAPEST (baseline), the SUSPECTED-EXPENSIVE (srcsm), the MAIN CONTENDER
# (synthseg_EM) and OURS (DualVal — a second validation pass every epoch, so it is the
# other candidate straggler and must be measured, not assumed cheap).
NNUNET_NUM_EPOCHS="${PROBE_EPOCHS}" RUN_JOB_PACK_DIR="${PACK_DIR}" \
    bash "${HERE}/04_01_train_ct_baseline.sh"                       "probe_baseline_${TS}"
NNUNET_NUM_EPOCHS="${PROBE_EPOCHS}" RUN_JOB_PACK_DIR="${PACK_DIR}" \
    bash "${HERE}/04_04_train_ct_synthseg_EM.sh"                    "probe_synthseg_EM_${TS}"
NNUNET_NUM_EPOCHS="${PROBE_EPOCHS}" RUN_JOB_PACK_DIR="${PACK_DIR}" \
    bash "${HERE}/04_05_train_ct_srcsm.sh"                          "probe_srcsm_${TS}"
NNUNET_NUM_EPOCHS="${PROBE_EPOCHS}" RUN_JOB_PACK_DIR="${PACK_DIR}" \
    bash "${HERE}/04_06_train_ct_auglabAug_v26_6_2_dualval.sh"      "probe_ours_val000_${TS}"

echo "[probe] recorded folds:"; cut -f3 "${PACK_DIR}/index.tsv" | sed 's/^/  /'
echo "${PROBE_BASE}" > "${HERE}/.last_probe_base"
