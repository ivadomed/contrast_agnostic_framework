#!/bin/bash
#SBATCH --job-name=msdspleen_pack_predict_eval
#SBATCH --account=aip-jcohen
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --output=/scratch/p/paulh/msd-spleen/_packruns/slurm_%j.out
# MSD-SPLEEN pack PREDICT + EVALUATE, single TamIA whole-node H100 job (run ON tamia:
#   sbatch 05_20_tamia_pack_predict_evaluate.sh
# ). Predicts all 8 chaos methods x 2 contrasts x folds 0-2 (=48 fold-tasks), then
# evaluates all 48 -- all within one whole-node allocation so no GPU sits idle and no
# CPU-only eval work wastes a second whole-node reservation.
#
# TWO BUGS HIT (and fixed here) while first running this — kept as comments so a
# future edit doesn't reintroduce them:
#  1. Env-sourcing order: the TamIA scratch-path override
#     (scripts/cluster/tamia_env_msd-spleen_chaoscross.sh) must be sourced BEFORE
#     Phase 1, not just before the evaluate phase -- each 05_0X_predict_chaos_*.sh
#     wrapper runs as a FRESH subprocess that re-sources env.sh itself; only an
#     already-exported override survives that re-source (env.sh's ${VAR:-default}
#     guards). Sourcing the override late left Phase 1 silently falling back to the
#     git-repo-relative default path, where no chaos checkpoints exist on tamia.
#  2. Unbounded concurrency: launching all fold-predict commands with only a 3s
#     stagger (no real throttling) let up to ~42 concurrent nnUNetv2_predict
#     processes each spawn -npp 12 -nps 6 (18) preprocessing workers -- hundreds of
#     processes fighting over the node's RAM. Some workers got OOM-killed mid-run,
#     but nnUNetv2_predict didn't propagate that as a nonzero exit code, so the
#     launcher logged "OK" for folds that had actually written only a few of the
#     41 test cases. ALWAYS verify output file COUNTS after predicting, never just
#     the wrapper's exit code. Fixed with a true bounded 4-way worker pool (one
#     task at a time per GPU, waits for completion before starting the next).
#  Separately, the CPU-only evaluate phase (Dice/HD95, some memory-heavy due to
#  HD95 distance-transform computation) OOM'd the node at 12 concurrent x 4 internal
#  workers = 48 processes. Fixed by dropping to 4 concurrent x 2 workers = 8.
set -uo pipefail
HERE_MSD="/project/aip-jcohen/paulh/mri_synthesis_project/datasets/msd-spleen/5_scripts_msd-spleen"
HERE_CHAOS="/project/aip-jcohen/paulh/mri_synthesis_project/datasets/chaos/5_scripts_chaos"
PROJECT_ROOT="/project/aip-jcohen/paulh/mri_synthesis_project"
PACK_DIR="/scratch/p/paulh/msd-spleen/_packruns/predict_pack_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${PACK_DIR}"

echo "[pack] host=$(hostname) job=${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# Bug #1 fix: source BEFORE Phase 1, not just before evaluate.
source "${HERE_MSD}/00_utils/env.sh"
source /project/aip-jcohen/paulh/mri_synthesis_project/scripts/cluster/tamia_env_msd-spleen_chaoscross.sh
echo "[pack] CHAOS_DATASET_ROOT=${CHAOS_DATASET_ROOT}"
echo "[pack] nnUNet_raw=${nnUNet_raw}"
echo "[pack] PREDICTIONS_ROOT=${PREDICTIONS_ROOT}"

echo "[pack] ==== PHASE 1: record all 16 runs x 3 folds (=48) via RUN_JOB_PACK_DIR ===="
cd "${HERE_MSD}/05_predict"
for wrapper in \
    "05_02_predict_chaos_baseline.sh" \
    "05_03_predict_chaos_v26_6_2.sh" \
    "05_04_predict_chaos_synthseg_EM.sh" \
    "05_05_predict_chaos_auglab_default.sh" \
    "05_06_predict_chaos_synthseg_noEM.sh" \
    "05_07_predict_chaos_srcsm.sh" \
    "05_08_predict_chaos_auglabAug_v26_6_2_train050_val000.sh" \
    "05_09_predict_chaos_auglabAug_v26_6_2_train050_val100.sh" \
    "05_11_predict_chaos_t2spir_baseline.sh" \
    "05_12_predict_chaos_t2spir_v26_6_2.sh" \
    "05_13_predict_chaos_t2spir_synthseg_EM.sh" \
    "05_14_predict_chaos_t2spir_auglab_default.sh" \
    "05_15_predict_chaos_t2spir_synthseg_noEM.sh" \
    "05_16_predict_chaos_t2spir_srcsm.sh" \
    "05_17_predict_chaos_t2spir_auglabAug_v26_6_2_train050_val000.sh" \
    "05_18_predict_chaos_t2spir_auglabAug_v26_6_2_train050_val100.sh" ; do
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${wrapper}" "" all ct
done

N_RECORDED=$(grep -c . "${PACK_DIR}/index.tsv" 2>/dev/null || echo 0)
echo "[pack] recorded ${N_RECORDED} fold-predict commands -> ${PACK_DIR}/index.tsv"
[ "${N_RECORDED}" -gt 0 ] || { echo "[pack] ERROR: nothing recorded (missing chaos checkpoints on tamia?)" >&2; exit 1; }

echo "[pack] ==== PHASE 2: TRUE bounded 4-way worker pool (Bug #2 fix) ===="
mapfile -t LINES < "${PACK_DIR}/index.tsv"
N=${#LINES[@]}
G=4
worker() {
    local gpu="$1" idx="$1"
    while [ "${idx}" -lt "${N}" ]; do
        IFS=$'\t' read -r cmdfile log name donefile <<< "${LINES[$idx]}"
        local plog="${cmdfile%.sh}.log"
        echo "[pack] GPU${gpu}: launch '${name}' -> ${plog}"
        CUDA_VISIBLE_DEVICES=${gpu} bash "${cmdfile}" > "${plog}" 2>&1
        echo "[pack] GPU${gpu}: '${name}' exit=$?"
        idx=$(( idx + G ))
    done
}
pids=()
for gpu in 0 1 2 3; do worker "${gpu}" & pids+=($!); done
for p in "${pids[@]}"; do wait "${p}"; done
echo "[pack] predict phase done"

echo "[pack] ==== PHASE 3: verify file counts (never trust exit code alone -- Bug #2) ===="
verify_fail=0
while IFS=$'\t' read -r cmdfile log name donefile; do
    [ -n "${cmdfile}" ] || continue
    # name encodes e.g. msd_spleen_predict_baseline_fold2 -- just report counts under
    # each run's fold dirs broadly; a targeted per-(run,fold) check is done by the
    # caller if resuming a partial run (see the fix-up job pattern in git history).
    :
done < "${PACK_DIR}/index.tsv"
find "${PREDICTIONS_ROOT}" -mindepth 5 -maxdepth 5 -type d -name ct 2>/dev/null | while read -r d; do
    n=$(find "${d}" -name '*.nii.gz' 2>/dev/null | wc -l)
    [ "${n}" -eq 41 ] || echo "[pack] WARNING: ${d} has ${n}/41 case files -- re-run its predict wrapper for this fold" >&2
done

echo "[pack] ==== PHASE 4: evaluate (CPU-only, direct python -- no nested sbatch, throttled to avoid OOM) ===="
EVALUATE_PY="${HERE_CHAOS}/06_evaluate/06_00_evaluate.py"
FOV_JSON="${HERE_CHAOS}/06_evaluate/chaos_fov_margins.json"
LABEL_MAP='{"spleen": [4, 1]}'
GT_DIR="${nnUNet_raw}/labelsTs_ct"

run_eval() {   # $1=contrast $2=category $3=run_id $4=fold
    local contrast="$1" category="$2" run_id="$3" fold="$4"
    local pred_dir="${PREDICTIONS_ROOT}/chaos_model/${contrast}/${category}/${run_id}/fold${fold}/ct"
    local eval_dir="${METRICS_ROOT}/chaos_model/${contrast}/${category}_${run_id}/fold${fold}"
    if [ -f "${eval_dir}/ct_metrics.csv" ]; then
        echo "[pack] skip (already done): ${run_id} fold${fold}"; return 0
    fi
    if [ ! -d "${pred_dir}" ] || [ "$(find "${pred_dir}" -name '*.nii.gz' 2>/dev/null | wc -l)" -ne 41 ]; then
        echo "  ! ${run_id} fold${fold}: incomplete/missing predictions at ${pred_dir} -- skipping" >&2
        return 1
    fi
    mkdir -p "${eval_dir}"
    local mm
    mm=$("${PROJECT_ROOT}/.venv/bin/python" -c "import json;d=json.load(open('${FOV_JSON}'))['${contrast}']['spleen'];print(d['sup_mm'],d['inf_mm'])")
    "${PROJECT_ROOT}/.venv/bin/python" "${EVALUATE_PY}" \
        --pred_dir "${pred_dir}" --gt_dir "${GT_DIR}" --label_map "${LABEL_MAP}" \
        --name ct --out_csv "${eval_dir}/ct_metrics.csv" --workers 2 \
        --fov_anchor_gt_ids 1 --fov_sup_mm ${mm% *} --fov_inf_mm ${mm#* } \
        > "${eval_dir}/eval.log" 2>&1
    "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/summarize_fold.py" \
        "${eval_dir}" "${run_id}" "${fold}" --group-col modality --groups-word Modalities \
        --title-suffix " | MSD-SPLEEN CT | spleen only" --groups ct >> "${eval_dir}/eval.log" 2>&1
    echo "[pack] eval done: ${run_id} fold${fold}"
}
export -f run_eval
export EVALUATE_PY FOV_JSON LABEL_MAP GT_DIR PREDICTIONS_ROOT METRICS_ROOT PROJECT_ROOT

cat <<'TASKS' | while read -r contrast category run_id; do
t1in nnUNet chaos_t1in_baseline_20260614_153230
t1in nnUNet chaos_t1in_v26_6_2_train050_val100_20260615_213615
t1in auglab chaos_t1in_synthseg_EM_train100_val000_20260611_120000
t1in auglab chaos_t1in_auglab_default_20260611_120000
t1in auglab chaos_t1in_synthseg_noEM_train100_val000_20260611_120000
t1in auglab chaos_t1in_srcsm_20260710_011817
t1in auglab chaos_t1in_auglabAug_v26_6_2_train050_val000_20260615_213615
t1in auglab chaos_t1in_auglabAug_v26_6_2_train050_val100_20260616_112420
t2spir nnUNet chaos_t2spir_baseline_20260620_111146
t2spir nnUNet chaos_t2spir_v26_6_2_train050_val100_20260620_112122
t2spir auglab chaos_t2spir_synthseg_EM_20260620_112357
t2spir auglab chaos_t2spir_auglab_default_20260620_112240
t2spir auglab chaos_t2spir_synthseg_noEM_20260620_112515
t2spir auglab chaos_t2spir_srcsm_20260709_121945
t2spir auglab chaos_t2spir_auglabAug_v26_6_2_train050_val000_20260710_054202
t2spir auglab chaos_t2spir_auglabAug_v26_6_2_train050_val100_20260723_194053
TASKS
    for fold in 0 1 2; do echo "${contrast} ${category} ${run_id} ${fold}"; done
done > "${PACK_DIR}/eval_joblist.txt"

cat "${PACK_DIR}/eval_joblist.txt" | xargs -P 4 -L 1 bash -c 'run_eval "$@"' _
rc_eval=$?
echo "[pack] evaluate phase done rc=${rc_eval}"
echo "[pack] ==== DONE ===="
