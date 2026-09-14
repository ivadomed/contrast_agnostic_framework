#!/bin/bash
#SBATCH --job-name=ispy2_pack_predict_own
#SBATCH --account=aip-jcohen
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --output=/scratch/p/paulh/ispy2/_packruns/predict_own_slurm_%j.out
# I-SPY2 OWN-MODEL predict, single TamIA whole-node H100 job -- the headline
# ispy2-internal cross-contrast eval (2026-09-06, post-pivot: I-SPY2 trains,
# predicts on its OWN held-out test cases -- 84 patients, ~102/168 (t1wce/t2w)
# bilateral+unilateral FOV-variant cases, see 4_splits_ispy2/test_cases.json,
# disjoint from every fold's train+val set -- verified 2026-09-06).
# Covers ONLY the 8 headline RUN_IDs from the 3 TamIA packs confirmed complete
# via real checkpoint_final.pth on disk (basicA=t1wce x4, basicC=t2w x4):
# NOT basicB/ladderT1wce (still training as of this writing) and NOT the
# ladderT2w ablation rungs (out of scope for the two headline eval targets).
# Mirrors 05_20_tamia_pack_predict_evaluate.sh's proven pack pattern (TamIA
# rejects a plain run_job/sbatch single-GPU request -- must record via
# RUN_JOB_PACK_DIR and run a bounded worker pool across the node's 4 GPUs).
# Evaluate is done SEPARATELY, CPU-only, via 06_evaluate/06_01_evaluate_own_run.sh
# (never on a GPU node -- see CLAUDE.md's aggregation/eval resource-usage rule).
set -uo pipefail
HERE="/project/aip-jcohen/paulh/mri_synthesis_project/datasets/ispy2/5_scripts_ispy2"
PROJECT_ROOT="/project/aip-jcohen/paulh/mri_synthesis_project"
PACK_DIR="/scratch/p/paulh/ispy2/_packruns/predict_own_pack_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${PACK_DIR}"

echo "[pack] host=$(hostname) job=${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

source "${HERE}/00_utils/env.sh"
source "${PROJECT_ROOT}/scripts/cluster/tamia_env_ispy2.sh"
echo "[pack] nnUNet_raw=${nnUNet_raw}"
echo "[pack] PREDICTIONS_ROOT=${PREDICTIONS_ROOT}"

N_T1WCE=$(find "${nnUNet_raw}/imagesTs_t1wce" -name '*.nii.gz' 2>/dev/null | wc -l)
N_T2W=$(find "${nnUNet_raw}/imagesTs_t2w" -name '*.nii.gz' 2>/dev/null | wc -l)
echo "[pack] N_T1WCE=${N_T1WCE} N_T2W=${N_T2W} (own held-out test cases)"
[ "${N_T1WCE}" = "102" ] || { echo "[pack] ERROR: expected 102 t1wce test cases, found ${N_T1WCE}" >&2; exit 1; }
[ "${N_T2W}" = "168" ] || { echo "[pack] ERROR: expected 168 t2w test cases, found ${N_T2W}" >&2; exit 1; }

TS=20260905_163655
T0=$(date +%s)
echo "[pack] ==== PHASE 1: record 8 headline runs x 3 folds (=24 fold-jobs), items=t1wce+t2w ===="
PACK_DIR_T1WCE="${PACK_DIR}/t1wce"
PACK_DIR_T2W="${PACK_DIR}/t2w"
mkdir -p "${PACK_DIR_T1WCE}" "${PACK_DIR_T2W}"
cd "${HERE}/05_predict"
for job in \
    "05_21_predict_own_t1wce_baseline.sh ispy2_t1wce_baseline_${TS}" \
    "05_22_predict_own_t1wce_auglab_default.sh ispy2_t1wce_auglab_default_${TS}" \
    "05_23_predict_own_t1wce_synthseg_noEM.sh ispy2_t1wce_synthseg_noEM_${TS}" \
    "05_24_predict_own_t1wce_srcsm.sh ispy2_t1wce_srcsm_${TS}" ; do
    wrapper="${job%% *}"; run_id="${job#* }"
    RUN_JOB_PACK_DIR="${PACK_DIR_T1WCE}" bash "${wrapper}" "${run_id}" all
done
for job in \
    "05_25_predict_own_t2w_synthseg_noEM.sh ispy2_t2w_synthseg_noEM_${TS}" \
    "05_26_predict_own_t2w_synthseg_EM.sh ispy2_t2w_synthseg_EM_${TS}" \
    "05_27_predict_own_t2w_srcsm.sh ispy2_t2w_srcsm_${TS}" \
    "05_28_predict_own_t2w_auglabAug_v26_6_2_train050_val000.sh ispy2_t2w_auglabAug_v26_6_2_train050_val000_${TS}" ; do
    wrapper="${job%% *}"; run_id="${job#* }"
    RUN_JOB_PACK_DIR="${PACK_DIR_T2W}" bash "${wrapper}" "${run_id}" all
done

cat "${PACK_DIR_T1WCE}/index.tsv" "${PACK_DIR_T2W}/index.tsv" > "${PACK_DIR}/index.tsv" 2>/dev/null || true
N_RECORDED=$(grep -c . "${PACK_DIR}/index.tsv" 2>/dev/null || echo 0)
echo "[pack] recorded ${N_RECORDED} fold-predict commands -> ${PACK_DIR}/index.tsv (expect 24)"
[ "${N_RECORDED}" -gt 0 ] || { echo "[pack] ERROR: nothing recorded" >&2; exit 1; }

echo "[pack] ==== PHASE 2: bounded worker pool, 2 concurrent tasks PER GPU (8-way total) ===="
mapfile -t LINES < "${PACK_DIR}/index.tsv"
N=${#LINES[@]}
SLOTS=8
G=4
worker() {
    local slot="$1" idx="$1"
    local gpu=$(( slot % G ))
    while [ "${idx}" -lt "${N}" ]; do
        IFS=$'\t' read -r cmdfile log name donefile <<< "${LINES[$idx]}"
        local plog="${cmdfile%.sh}.log"
        local t0=$(date +%s)
        echo "[pack] slot${slot}(GPU${gpu}): launch '${name}' -> ${plog}"
        CUDA_VISIBLE_DEVICES=${gpu} bash "${cmdfile}" > "${plog}" 2>&1
        echo "[pack] slot${slot}(GPU${gpu}): '${name}' exit=$? ($(( $(date +%s) - t0 ))s)"
        idx=$(( idx + SLOTS ))
    done
}
pids=()
for slot in 0 1 2 3 4 5 6 7; do worker "${slot}" & pids+=($!); done
for p in "${pids[@]}"; do wait "${p}"; done
echo "[pack] predict phase done ($(( $(date +%s) - T0 ))s elapsed since job start)"

echo "[pack] ==== PHASE 3: verify EXACT file counts (t1wce=${N_T1WCE}, t2w=${N_T2W} per fold per item) ===="
verify_fail=0
for d in "${PREDICTIONS_ROOT}"/ispy2_model/*/*/*/fold*/t1wce; do
    [ -d "${d}" ] || continue
    n=$(find "${d}" -name '*.nii.gz' 2>/dev/null | wc -l)
    if [ "${n}" != "${N_T1WCE}" ]; then echo "[pack] BAD COUNT (${n}/${N_T1WCE}): ${d}" >&2; verify_fail=1; fi
done
for d in "${PREDICTIONS_ROOT}"/ispy2_model/*/*/*/fold*/t2w; do
    [ -d "${d}" ] || continue
    n=$(find "${d}" -name '*.nii.gz' 2>/dev/null | wc -l)
    if [ "${n}" != "${N_T2W}" ]; then echo "[pack] BAD COUNT (${n}/${N_T2W}): ${d}" >&2; verify_fail=1; fi
done
echo "[pack] predict verification done (verify_fail=${verify_fail})"
echo "[pack] ==== DONE ===="
