#!/bin/bash
# RECOVERY job (CPU): the on-harmony T1w rung-5 predictions were scored while still in RAS space (the 05_02 resample-to-native
# post-step is a silent no-op in pack-record mode). Resample them in place, then re-run the standard on-harmony eval + audit.
# The caller must already have moved the invalid metrics dir aside (06_01 silently reuses existing CSVs).
#SBATCH --account=aip-jcohen
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
set -uo pipefail
: "${ROOT:?}"; cd /project/aip-jcohen/paulh/mri_synthesis_project
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/${USER}}"; source "${ROOT}/RUN_IDS.env"
D=benchmark/02_tasks/brain_healthy/on-harmony/5_scripts_on-harmony
( source "$D/00_utils/env.sh"; source scripts/cluster/tamia_env_onharmony.sh; export TRAINING_CONTRAST=T1w
  base="${PREDICTIONS_ROOT}/${MODEL_TYPE}/T1w/nnUNet/${ONH_RUN}"
  for k in 0 1 2; do for item in T1w T2w bold dwi_ap epi_ap gre_echo1_mag; do
    .venv/bin/python "$D/05_predict/05_02_resample_predictions_to_native.py" "$base/fold$k/final/$item" "${PREDICTIONS_ROOT}/${MODEL_TYPE}/_test_set/$item/images_native" || exit 1
  done; done ) || { echo "[onh-fix] resample FAILED"; exit 1; }
SKIP="chaos brats" bash scripts/cluster/tamia_rung5_eval_job.sh
