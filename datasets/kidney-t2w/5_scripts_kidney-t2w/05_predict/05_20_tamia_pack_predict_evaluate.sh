#!/bin/bash
#SBATCH --job-name=kidney_t2w_pack_predict_eval
#SBATCH --account=aip-jcohen
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --output=/scratch/p/paulh/kidney-t2w/_packruns/slurm_%j.out
# KIDNEY-T2W pack PREDICT + EVALUATE, single TamIA whole-node H100 job.
# 100 tiny T2w volumes, single "t2" item, 7 methods x 2 contrasts x 3 folds = 42
# fold-tasks. Far smaller than cirrmri-liver's 628-case/16-run job (which took
# ~1.5-2h at 8-way concurrency) -- 1h budget is generous, not tight; adjust upward
# only if the actual per-task timing logged below says otherwise.
#
# Every lesson from msd-spleen/cirrmri-liver's onboarding applied from the START
# (see project_msd_spleen_onboarding.md / project_cirrmri_liver_onboarding.md
# memories for the full incident history):
#  1. Source the TamIA scratch-path env override BEFORE Phase 1 (predict), not
#     just before evaluate -- each predict wrapper is a fresh subprocess that
#     re-sources env.sh itself; only an already-exported override survives that.
#  2. TRUE bounded worker pool (one task at a time per slot, waiting for
#     completion before starting the next) -- NOT a stagger-launch-everything
#     loop, which oversubscribes CPU/RAM via nnU-Net's internal preprocessing
#     workers and can silently truncate predictions without a nonzero exit code.
#  3. Evaluate calls the custom kidney evaluator shim DIRECTLY (bypassing
#     06_01_evaluate_run.sh and run_job entirely) -- run_job's internal sbatch
#     call would itself fail on tamia (whole-node-only scheduler rejects a nested
#     --gpus request) even for a CPU-only job, and nesting more sbatch
#     submissions from inside a running job just adds needless queue overhead.
#  4. Evaluate concurrency throttled to avoid the OOM cirrmri-liver's first eval
#     attempt hit at unbounded concurrency.
#  5. Verify EXACT file/row counts for every output, every time -- never trust
#     existence or a self-reported exit code alone.
set -uo pipefail
HERE_KT="/project/aip-jcohen/paulh/mri_synthesis_project/datasets/kidney-t2w/5_scripts_kidney-t2w"
HERE_CHAOS="/project/aip-jcohen/paulh/mri_synthesis_project/datasets/chaos/5_scripts_chaos"
PROJECT_ROOT="/project/aip-jcohen/paulh/mri_synthesis_project"
PACK_DIR="/scratch/p/paulh/kidney-t2w/_packruns/predict_pack_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${PACK_DIR}"

echo "[pack] host=$(hostname) job=${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

source "${HERE_KT}/00_utils/env.sh"
source "${PROJECT_ROOT}/scripts/cluster/tamia_env_kidney-t2w_chaoscross.sh"
echo "[pack] CHAOS_DATASET_ROOT=${CHAOS_DATASET_ROOT}"
echo "[pack] nnUNet_raw=${nnUNet_raw}"
echo "[pack] PREDICTIONS_ROOT=${PREDICTIONS_ROOT}"

T0=$(date +%s)
echo "[pack] ==== PHASE 1: record all 14 runs x 3 folds (=42), item=t2 ===="
cd "${HERE_KT}/05_predict"
for wrapper in \
    "05_02_predict_chaos_baseline.sh" \
    "05_03_predict_chaos_synthseg_EM.sh" \
    "05_04_predict_chaos_auglab_default.sh" \
    "05_05_predict_chaos_synthseg_noEM.sh" \
    "05_06_predict_chaos_srcsm.sh" \
    "05_07_predict_chaos_auglabAug_v26_6_2_train050_val000.sh" \
    "05_08_predict_chaos_auglabAug_v26_6_2_train050_val100.sh" \
    "05_11_predict_chaos_t2spir_baseline.sh" \
    "05_12_predict_chaos_t2spir_synthseg_EM.sh" \
    "05_13_predict_chaos_t2spir_auglab_default.sh" \
    "05_14_predict_chaos_t2spir_synthseg_noEM.sh" \
    "05_15_predict_chaos_t2spir_srcsm.sh" \
    "05_16_predict_chaos_t2spir_auglabAug_v26_6_2_train050_val000.sh" \
    "05_17_predict_chaos_t2spir_auglabAug_v26_6_2_train050_val100.sh" ; do
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${wrapper}" "" all t2
done

N_RECORDED=$(grep -c . "${PACK_DIR}/index.tsv" 2>/dev/null || echo 0)
echo "[pack] recorded ${N_RECORDED} fold-predict commands -> ${PACK_DIR}/index.tsv"
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

echo "[pack] ==== PHASE 3: verify EXACT file counts (100 per fold, item=t2) ===="
verify_fail=0
for d in "${PREDICTIONS_ROOT}"/chaos_model/*/*/*/fold*/t2; do
    [ -d "${d}" ] || continue
    n=$(find "${d}" -name '*.nii.gz' 2>/dev/null | wc -l)
    if [ "${n}" -ne 100 ]; then
        echo "[pack] BAD COUNT (${n}/100): ${d}" >&2
        verify_fail=1
    fi
done
echo "[pack] predict verification done (verify_fail=${verify_fail})"

echo "[pack] ==== PHASE 4: evaluate (CPU-only, direct python calls, throttled) ===="
EVALUATE_PY="${HERE_KT}/06_evaluate/06_00_evaluate_kidney_t2w.py"
FOV_JSON="${HERE_CHAOS}/06_evaluate/chaos_fov_margins.json"
GT_DIR="${nnUNet_raw}/labelsTs_t2"

run_eval() {   # $1=contrast $2=category $3=run_id $4=fold
    local contrast="$1" category="$2" run_id="$3" fold="$4"
    local pred_dir="${PREDICTIONS_ROOT}/chaos_model/${contrast}/${category}/${run_id}/fold${fold}/t2"
    local eval_dir="${METRICS_ROOT}/chaos_model/${contrast}/${category}_${run_id}/fold${fold}"
    local n
    n=$(find "${pred_dir}" -name '*.nii.gz' 2>/dev/null | wc -l)
    if [ "${n}" -ne 100 ]; then
        echo "  ! ${run_id} fold${fold}: ${n}/100 predictions -- refusing to evaluate" >&2
        return 1
    fi
    mkdir -p "${eval_dir}"
    local mm
    mm=$("${PROJECT_ROOT}/.venv/bin/python" -c "import json;d=json.load(open('${FOV_JSON}'))['${contrast}']['kidney'];print(d['sup_mm'],d['inf_mm'])")
    "${PROJECT_ROOT}/.venv/bin/python" "${EVALUATE_PY}" \
        --pred_dir "${pred_dir}" --gt_dir "${GT_DIR}" \
        --name t2 --out_csv "${eval_dir}/t2_metrics.csv" --workers 2 \
        --fov_anchor_gt_ids 1 --fov_sup_mm ${mm% *} --fov_inf_mm ${mm#* } \
        > "${eval_dir}/t2_eval.log" 2>&1
    local rows
    rows=$(( $(wc -l < "${eval_dir}/t2_metrics.csv") - 1 ))
    echo "[pack] eval done: ${run_id} fold${fold} (${rows}/100 rows)"
    [ "${rows}" -eq 100 ] || { echo "[pack] WARNING: ${run_id} fold${fold} wrote only ${rows}/100 rows" >&2; return 1; }
}
export -f run_eval
export EVALUATE_PY FOV_JSON GT_DIR PREDICTIONS_ROOT METRICS_ROOT PROJECT_ROOT

cat > "${PACK_DIR}/eval_tasks.txt" <<'TASKS'
t1in nnUNet chaos_t1in_baseline_20260614_153230
t1in auglab chaos_t1in_synthseg_EM_train100_val000_20260611_120000
t1in auglab chaos_t1in_auglab_default_20260611_120000
t1in auglab chaos_t1in_synthseg_noEM_train100_val000_20260611_120000
t1in auglab chaos_t1in_srcsm_20260710_011817
t1in auglab chaos_t1in_auglabAug_v26_6_2_train050_val000_20260615_213615
t1in auglab chaos_t1in_auglabAug_v26_6_2_train050_val100_20260616_112420
t2spir nnUNet chaos_t2spir_baseline_20260620_111146
t2spir auglab chaos_t2spir_auglab_default_20260620_112240
t2spir auglab chaos_t2spir_synthseg_noEM_20260620_112515
t2spir auglab chaos_t2spir_synthseg_EM_20260620_112357
t2spir auglab chaos_t2spir_srcsm_20260709_121945
t2spir auglab chaos_t2spir_auglabAug_v26_6_2_train050_val000_20260710_054202
t2spir auglab chaos_t2spir_auglabAug_v26_6_2_train050_val100_20260723_194053
TASKS

: > "${PACK_DIR}/eval_joblist.txt"
while read -r contrast category run_id; do
    [ -n "${contrast}" ] || continue
    for fold in 0 1 2; do
        echo "${contrast} ${category} ${run_id} ${fold}" >> "${PACK_DIR}/eval_joblist.txt"
    done
done < "${PACK_DIR}/eval_tasks.txt"

cat "${PACK_DIR}/eval_joblist.txt" | xargs -P 4 -L 1 bash -c 'run_eval "$@"' _
rc_eval=$?
echo "[pack] evaluate phase done rc=${rc_eval}"

echo "[pack] ==== PHASE 5: merge each fold's t2 into eval_all.csv (summarize_fold.py) ===="
find "${METRICS_ROOT}" -mindepth 3 -maxdepth 3 -type d | while read -r run_dir; do
    run_name="$(basename "${run_dir}")"
    run_id="${run_name#nnUNet_}"; run_id="${run_id#auglab_}"
    for fold_dir in "${run_dir}"/fold*/; do
        [ -d "${fold_dir}" ] || continue
        fold="$(basename "${fold_dir}" | sed 's/fold//')"
        if [ -f "${fold_dir}t2_metrics.csv" ]; then
            "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/summarize_fold.py" \
                "${fold_dir}" "${run_id}" "${fold}" --group-col modality --groups-word Modalities \
                --title-suffix " | KIDNEY-T2W MRI | kidney only" --groups t2 > /dev/null 2>&1
        fi
    done
done
echo "[pack] eval_all.csv merge done"

echo "[pack] ==== FINAL COMPREHENSIVE CHECK: all metrics files, exact row counts ===="
find "${METRICS_ROOT}" -name "t2_metrics.csv" 2>/dev/null | sort | while read -r f; do
    n=$(( $(wc -l < "$f") - 1 ))
    [ "${n}" -eq 100 ] || echo "[pack] BAD ROW COUNT (${n}/100): ${f}"
done
total=$(find "${METRICS_ROOT}" -name "t2_metrics.csv" 2>/dev/null | wc -l)
echo "[pack] total metrics files: ${total} (expect 42 = 14 runs x 3 folds)"
echo "[pack] ==== DONE (verify_fail=${verify_fail}, eval rc=${rc_eval}) ===="
