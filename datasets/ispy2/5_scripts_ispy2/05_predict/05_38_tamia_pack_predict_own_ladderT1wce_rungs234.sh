#!/bin/bash
#SBATCH --job-name=ispy2_pack_predict_own_ladder
#SBATCH --account=aip-jcohen
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --output=/scratch/p/paulh/ispy2/_packruns/predict_own_ladder_slurm_%j.out
# I-SPY2 OWN-MODEL predict for the ladderT1wce causal-ablation pack's rungs 2-4
# ONLY (baseline_kmeans, +label_remap, +voronoi) -- confirmed complete via real
# checkpoint_final.pth on disk 2026-09-06. Rung 5 (v26_6_2_train050_val100) is
# excluded here per explicit instruction not to touch it yet even though this
# run's own verification also found checkpoint_final.pth for it (flagged
# separately, not acted on). Mirrors 05_29_tamia_pack_predict_own.sh's pattern
# exactly, smaller run list.
set -uo pipefail
HERE="/project/aip-jcohen/paulh/mri_synthesis_project/datasets/ispy2/5_scripts_ispy2"
PROJECT_ROOT="/project/aip-jcohen/paulh/mri_synthesis_project"
PACK_DIR="/scratch/p/paulh/ispy2/_packruns/predict_own_ladder_pack_$(date +%Y%m%d_%H%M%S)"
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
echo "[pack] ==== PHASE 1: record 3 ladder-rung runs x 3 folds (=9 fold-jobs) ===="
for job in \
    "05_30_predict_own_t1wce_baseline_kmeans.sh ispy2_t1wce_baseline_kmeans_${TS}" \
    "05_31_predict_own_t1wce_baseline_kmeans_label_remap.sh ispy2_t1wce_baseline_kmeans_label_remap_${TS}" \
    "05_32_predict_own_t1wce_baseline_kmeans_label_remap_voronoi.sh ispy2_t1wce_baseline_kmeans_label_remap_voronoi_${TS}" ; do
    wrapper="${job%% *}"; run_id="${job#* }"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${wrapper}" "${run_id}" all
done
N_RECORDED=$(grep -c . "${PACK_DIR}/index.tsv" 2>/dev/null || echo 0)
echo "[pack] recorded ${N_RECORDED} fold-predict commands (expect 9)"
[ "${N_RECORDED}" -gt 0 ] || { echo "[pack] ERROR: nothing recorded" >&2; exit 1; }

echo "[pack] ==== PHASE 2: worker pool, 1 task per GPU (4-way, only 9 tasks) ===="
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
for m in baseline_kmeans baseline_kmeans_label_remap baseline_kmeans_label_remap_voronoi; do
    for F in 0 1 2; do
        for item_n_pair in "t1wce ${N_T1WCE}" "t2w ${N_T2W}"; do
            item="${item_n_pair%% *}"; expect="${item_n_pair#* }"
            d="${PREDICTIONS_ROOT}/ispy2_model/t1wce/auglab/ispy2_t1wce_${m}_${TS}/fold${F}/${item}"
            n=$(find "${d}" -name '*.nii.gz' 2>/dev/null | wc -l)
            [ "${n}" = "${expect}" ] || { echo "[pack] BAD COUNT (${n}/${expect}): ${d}" >&2; verify_fail=1; }
        done
    done
done
echo "[pack] verification done (verify_fail=${verify_fail})"
echo "[pack] ==== DONE ===="
