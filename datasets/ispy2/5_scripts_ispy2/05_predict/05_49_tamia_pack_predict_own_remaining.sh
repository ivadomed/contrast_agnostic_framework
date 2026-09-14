#!/bin/bash
#SBATCH --job-name=ispy2_pack_predict_own_remaining
#SBATCH --account=aip-jcohen
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --output=/scratch/p/paulh/ispy2/_packruns/predict_own_remaining_slurm_%j.out
# I-SPY2 OWN-MODEL predict for every run that was still missing as of
# 2026-09-06's verification pass: t1wce synthseg_EM, t1wce OURS
# (auglabAug_v26_6_2_train050_val000), t1wce ladder rung 5
# (v26_6_2_train050_val100); t2w baseline, t2w auglab_default, t2w ladder
# rungs 2-5 (baseline_kmeans, +label_remap, +voronoi, v26_6_2_train050_val100).
# Completes the full 6-method x 2-modality suite + full ladder on the own-eval
# side. 9 runs x 3 folds = 27 fold-jobs, packed 4-way on one whole-node H100 job
# (mirrors 05_29/05_38's proven pattern).
set -uo pipefail
HERE="/project/aip-jcohen/paulh/mri_synthesis_project/datasets/ispy2/5_scripts_ispy2"
PROJECT_ROOT="/project/aip-jcohen/paulh/mri_synthesis_project"
PACK_DIR="/scratch/p/paulh/ispy2/_packruns/predict_own_remaining_pack_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${PACK_DIR}"

echo "[pack] host=$(hostname) job=${SLURM_JOB_ID:-?}"
source "${HERE}/00_utils/env.sh"
source "${PROJECT_ROOT}/scripts/cluster/tamia_env_ispy2.sh"

N_T1WCE=$(find "${nnUNet_raw}/imagesTs_t1wce" -name '*.nii.gz' 2>/dev/null | wc -l)
N_T2W=$(find "${nnUNet_raw}/imagesTs_t2w" -name '*.nii.gz' 2>/dev/null | wc -l)
[ "${N_T1WCE}" = "102" ] || { echo "[pack] ERROR: expected 102 t1wce test cases, found ${N_T1WCE}" >&2; exit 1; }
[ "${N_T2W}" = "168" ] || { echo "[pack] ERROR: expected 168 t2w test cases, found ${N_T2W}" >&2; exit 1; }

TS=20260905_163655
cd "${HERE}/05_predict"
echo "[pack] ==== PHASE 1: record 9 runs x 3 folds (=27 fold-jobs) ===="
for job in \
    "05_40_predict_own_t1wce_synthseg_EM.sh ispy2_t1wce_synthseg_EM_${TS}" \
    "05_41_predict_own_t1wce_auglabAug_v26_6_2_train050_val000.sh ispy2_t1wce_auglabAug_v26_6_2_train050_val000_${TS}" \
    "05_42_predict_own_t1wce_v26_6_2_train050_val100_ladder.sh ispy2_t1wce_v26_6_2_train050_val100_${TS}" \
    "05_43_predict_own_t2w_baseline.sh ispy2_t2w_baseline_${TS}" \
    "05_44_predict_own_t2w_auglab_default.sh ispy2_t2w_auglab_default_${TS}" \
    "05_45_predict_own_t2w_baseline_kmeans.sh ispy2_t2w_baseline_kmeans_${TS}" \
    "05_46_predict_own_t2w_baseline_kmeans_label_remap.sh ispy2_t2w_baseline_kmeans_label_remap_${TS}" \
    "05_47_predict_own_t2w_baseline_kmeans_label_remap_voronoi.sh ispy2_t2w_baseline_kmeans_label_remap_voronoi_${TS}" \
    "05_48_predict_own_t2w_v26_6_2_train050_val100_ladder.sh ispy2_t2w_v26_6_2_train050_val100_${TS}" ; do
    wrapper="${job%% *}"; run_id="${job#* }"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${wrapper}" "${run_id}" all
done
N_RECORDED=$(grep -c . "${PACK_DIR}/index.tsv" 2>/dev/null || echo 0)
echo "[pack] recorded ${N_RECORDED} fold-predict commands (expect 27)"
[ "${N_RECORDED}" -gt 0 ] || { echo "[pack] ERROR: nothing recorded" >&2; exit 1; }

echo "[pack] ==== PHASE 2: worker pool, round-robin over 4 GPUs (27 tasks) ===="
mapfile -t LINES < "${PACK_DIR}/index.tsv"
N=${#LINES[@]}
G=4
worker() {
    local slot="$1" idx="$1" gpu=$(( slot % G ))
    while [ "${idx}" -lt "${N}" ]; do
        IFS=$'\t' read -r cmdfile log name donefile <<< "${LINES[$idx]}"
        local plog="${cmdfile%.sh}.log"
        echo "[pack] slot${slot}(GPU${gpu}): launch '${name}'"
        CUDA_VISIBLE_DEVICES=${gpu} bash "${cmdfile}" > "${plog}" 2>&1
        echo "[pack] slot${slot}(GPU${gpu}): '${name}' exit=$?"
        idx=$(( idx + G ))
    done
}
pids=()
for slot in 0 1 2 3; do worker "${slot}" & pids+=($!); done
for p in "${pids[@]}"; do wait "${p}"; done
echo "[pack] predict phase done"

echo "[pack] ==== PHASE 3: verify EXACT file counts ===="
verify_fail=0
declare -A M2C=(
    [t1wce_synthseg_EM]=t1wce [t1wce_auglabAug_v26_6_2_train050_val000]=t1wce [t1wce_v26_6_2_train050_val100]=t1wce
    [t2w_baseline]=t2w [t2w_auglab_default]=t2w [t2w_baseline_kmeans]=t2w
    [t2w_baseline_kmeans_label_remap]=t2w [t2w_baseline_kmeans_label_remap_voronoi]=t2w [t2w_v26_6_2_train050_val100]=t2w
)
for key in "${!M2C[@]}"; do
    train_contrast="${M2C[$key]}"
    method="${key#${train_contrast}_}"
    for F in 0 1 2; do
        for item_n_pair in "t1wce ${N_T1WCE}" "t2w ${N_T2W}"; do
            item="${item_n_pair%% *}"; expect="${item_n_pair#* }"
            d="${PREDICTIONS_ROOT}/ispy2_model/${train_contrast}/auglab/ispy2_${train_contrast}_${method}_${TS}/fold${F}/${item}"
            [ -d "${d}" ] || d="${PREDICTIONS_ROOT}/ispy2_model/${train_contrast}/nnUNet/ispy2_${train_contrast}_${method}_${TS}/fold${F}/${item}"
            n=$(find "${d}" -name '*.nii.gz' 2>/dev/null | wc -l)
            [ "${n}" = "${expect}" ] || { echo "[pack] BAD COUNT (${n}/${expect}): ${d}" >&2; verify_fail=1; }
        done
    done
done
echo "[pack] verification done (verify_fail=${verify_fail})"
echo "[pack] ==== DONE ===="
