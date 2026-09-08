#!/bin/bash
#SBATCH --job-name=tf2_predict_eval
#SBATCH --account=aip-jcohen
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --output=/scratch/p/paulh/toothfairy2/_packruns/predict_slurm_%j.out
# toothfairy2 PREDICT + EVALUATE, single TamIA whole-node H100 job.
#
# Covers BOTH evaluation axes in one pass:
#   - own held-out CBCT test set (71 sealed cases)      -> IN-DOMAIN reference
#   - hanseg FOV-matched head-and-neck CT (42 cases)    -> CROSS-MODALITY (the headline)
#
# Follows ispy2/ambl's proven 05_20 shape (record via RUN_JOB_PACK_DIR, then a bounded
# worker pool inside ONE whole-node job, then exact-count verification, then direct
# python eval calls). That shape exists because **TamIA rejects a plain per-fold
# single-GPU run_job/sbatch request**, so the normal `bash 05_XX_predict_*.sh` path
# cannot be used here directly.
#
# SELECTING WHICH RUNS: set TF2_ONLY to a space-separated list of method tokens to
# restrict the run set (default: every run present in the pack RUN_IDS.env files whose
# training has finished). This is what lets the job be submitted twice — once now for
# the finished methods, once --dependency-gated for the stragglers.
#
#   sbatch --export=ALL,SUITE_A=...,SUITE_B=...,LADDER=... 05_31_tamia_predict_evaluate.sh
#   sbatch --dependency=afterany:<trainjob> --export=ALL,...,TF2_ONLY="ours rung5" 05_31_...
#
# ⚠️ A run whose training has NOT finished is SKIPPED, loudly, not predicted from a
# partial checkpoint — CLAUDE.md's stale-RUN_ID / partial-metrics lesson (a TamIA pack
# once silently truncated 14/48 fold-metrics and it was only caught by a human noticing
# "OURS looks worse"). Every fold is required to have checkpoint_final.pth AND every
# predict dir is verified to hold EXACTLY the expected case count before evaluation.
set -uo pipefail
PROJECT_ROOT="/project/aip-jcohen/paulh/mri_synthesis_project"
TF2="${PROJECT_ROOT}/datasets/toothfairy2/5_scripts_toothfairy2"
HS="${PROJECT_ROOT}/datasets/hanseg/5_scripts_hanseg"
PACK_DIR="/scratch/p/paulh/toothfairy2/_packruns/predict_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${PACK_DIR}"
echo "[pp] host=$(hostname) job=${SLURM_JOB_ID:-?} pack=${PACK_DIR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

: "${SUITE_A:?need SUITE_A pack dir}"; : "${SUITE_B:?need SUITE_B pack dir}"; : "${LADDER:?need LADDER pack dir}"
TF2_ONLY="${TF2_ONLY:-}"
get() { grep -E "^$2=" "$1/RUN_IDS.env" 2>/dev/null | head -1 | cut -d= -f2-; }

BASELINE=$(get "$SUITE_A" BASELINE_RUN_ID);  AUGDEF=$(get "$SUITE_A" AUGLAB_DEFAULT_RUN_ID)
SSNOEM=$(get "$SUITE_A" SYNTHSEG_NOEM_RUN_ID); SSEM=$(get "$SUITE_B" SYNTHSEG_EM_RUN_ID)
SRCSM=$(get "$SUITE_B" SRCSM_RUN_ID);        OURS0=$(get "$SUITE_B" OURS_RUN_ID)
OURS1="${OURS0/_val000_/_val100_}"
R2=$(get "$LADDER" R2_RUN_ID); R3=$(get "$LADDER" R3_RUN_ID)
R4=$(get "$LADDER" R4_RUN_ID); R5=$(get "$LADDER" R5_RUN_ID)

# token : own-wrapper : hanseg-wrapper : run_id : category : metrics-subdir
ROWS=(
 "baseline:05_02_predict_baseline.sh:05_02_predict_baseline.sh:${BASELINE}:nnUNet:"
 "auglab_default:05_03_predict_auglab_default.sh:05_03_predict_auglab_default.sh:${AUGDEF}:auglab:"
 "synthseg_noEM:05_04_predict_synthseg_noEM.sh:05_04_predict_synthseg_noEM.sh:${SSNOEM}:auglab:"
 "synthseg_EM:05_05_predict_synthseg_EM.sh:05_05_predict_synthseg_EM.sh:${SSEM}:auglab:"
 "srcsm:05_06_predict_srcsm.sh:05_06_predict_srcsm.sh:${SRCSM}:auglab:"
 "ours:05_07_predict_ours_val000.sh:05_07_predict_ours_val000.sh:${OURS0}:auglab:"
 "ours100:05_08_predict_ours_val100.sh:05_08_predict_ours_val100.sh:${OURS1}:auglab:"
 "rung2:05_09_predict_baseline_kmeans.sh:05_09_predict_baseline_kmeans.sh:${R2}:auglab:ablations"
 "rung3:05_10_predict_baseline_kmeans_label_remap.sh:05_10_predict_baseline_kmeans_label_remap.sh:${R3}:auglab:ablations"
 "rung4:05_11_predict_baseline_kmeans_label_remap_voronoi.sh:05_11_predict_baseline_kmeans_label_remap_voronoi.sh:${R4}:auglab:ablations"
 "rung5:05_12_predict_v26_6_2_train050_val100.sh:05_12_predict_v26_6_2_train050_val100.sh:${R5}:nnUNet:ablations"
)

want() { [ -z "${TF2_ONLY}" ] && return 0; case " ${TF2_ONLY} " in *" $1 "*) return 0;; *) return 1;; esac; }

# ── Phase 0: inputs + which runs are actually trained ────────────────────────
( set -e; cd "${PROJECT_ROOT}"
  source "${TF2}/00_utils/env.sh"; source "${PROJECT_ROOT}/scripts/cluster/tamia_env_toothfairy2.sh"
  bash "${TF2}/05_predict/05_00_build_test_inputs.sh" )
TF2_RAW=/scratch/p/paulh/toothfairy2/2_nnUNet/raw/Dataset110_ToothFairy2CBCT
HS_RAW=/scratch/p/paulh/hanseg/2_nnUNet/raw
N_OWN=$(ls "${TF2_RAW}/imagesTs_cbct" 2>/dev/null | wc -l)
N_HS=$(ls "${HS_RAW}/imagesTs_ct" 2>/dev/null | wc -l)
echo "[pp] own test cases=${N_OWN} (expect 71)   hanseg cases=${N_HS} (expect 42)"
[ "${N_OWN}" = "71" ] || { echo "[pp] ERROR: own test inputs not 71" >&2; exit 1; }
[ "${N_HS}"  = "42" ] || { echo "[pp] ERROR: hanseg inputs not 42" >&2; exit 1; }

TF2_PRED_ROOT=/scratch/p/paulh/toothfairy2/8_results/01_predictions/toothfairy2_model/cbct
SELECTED=()
for row in "${ROWS[@]}"; do
  IFS=: read -r TOK OWNW HSW RID CAT SUB <<< "${row}"
  want "${TOK}" || continue
  [ -n "${RID}" ] || { echo "[pp] skip ${TOK}: no RUN_ID"; continue; }
  # ours100 is the DualVal hard-linked mirror: it exists only once ours has finished.
  SRC_RID="${RID}"; [ "${TOK}" = "ours100" ] && SRC_RID="${OURS0}"
  NF=$(find "${TF2_PRED_ROOT}"/*/"${SRC_RID}" -name checkpoint_final.pth 2>/dev/null | wc -l)
  if [ "${NF}" != "3" ]; then
    echo "[pp] SKIP ${TOK} (${RID}): only ${NF}/3 folds have checkpoint_final.pth — training unfinished"
    continue
  fi
  SELECTED+=("${row}")
  echo "[pp] selected ${TOK} -> ${RID} (${CAT}${SUB:+,${SUB}})"
done
[ ${#SELECTED[@]} -gt 0 ] || { echo "[pp] nothing to do"; exit 0; }

# ── Phase 1: record predict commands (own + hanseg in SEPARATE subshells) ────
# Separate subshells + separate pack subdirs: the two datasets both export
# nnUNet_raw/PREDICTIONS_ROOT, and sourcing one env after the other in the same shell
# is the documented env-clobbering bug that has hit every cross-eval dataset here.
PD_OWN="${PACK_DIR}/own"; PD_HS="${PACK_DIR}/hanseg"; mkdir -p "${PD_OWN}" "${PD_HS}"
( cd "${PROJECT_ROOT}"
  source "${TF2}/00_utils/env.sh"; source "${PROJECT_ROOT}/scripts/cluster/tamia_env_toothfairy2.sh"
  for row in "${SELECTED[@]}"; do
    IFS=: read -r TOK OWNW HSW RID CAT SUB <<< "${row}"
    RUN_JOB_PACK_DIR="${PD_OWN}" bash "${TF2}/05_predict/${OWNW}" "${RID}" all
  done ) > "${PACK_DIR}/record_own.log" 2>&1
( cd "${PROJECT_ROOT}"
  source "${HS}/00_utils/env.sh"; source "${PROJECT_ROOT}/scripts/cluster/tamia_env_hanseg.sh"
  for row in "${SELECTED[@]}"; do
    IFS=: read -r TOK OWNW HSW RID CAT SUB <<< "${row}"
    RUN_JOB_PACK_DIR="${PD_HS}" bash "${HS}/05_predict/${HSW}" "${RID}" all
  done ) > "${PACK_DIR}/record_hanseg.log" 2>&1
cat "${PD_OWN}/index.tsv" "${PD_HS}/index.tsv" > "${PACK_DIR}/index.tsv" 2>/dev/null
N=$(grep -c . "${PACK_DIR}/index.tsv" 2>/dev/null || echo 0)
echo "[pp] recorded ${N} fold-predict tasks (expect ${#SELECTED[@]} runs x 3 folds x 2 datasets = $(( ${#SELECTED[@]} * 6 )))"
[ "${N}" -gt 0 ] || { echo "[pp] ERROR: nothing recorded; see ${PACK_DIR}/record_*.log" >&2; tail -20 "${PACK_DIR}"/record_*.log; exit 1; }

# ── Phase 2: bounded worker pool, 2 concurrent tasks per GPU ─────────────────
mapfile -t LINES < "${PACK_DIR}/index.tsv"
NT=${#LINES[@]}; SLOTS=8; G=4
worker() {
  local slot="$1" idx="$1"; local gpu=$(( slot % G ))
  while [ "${idx}" -lt "${NT}" ]; do
    IFS=$'\t' read -r cmdfile log name donefile <<< "${LINES[$idx]}"
    local plog="${cmdfile%.sh}.log"; local t0=$(date +%s)
    echo "[pp] slot${slot}(GPU${gpu}) launch ${name}"
    CUDA_VISIBLE_DEVICES=${gpu} bash "${cmdfile}" > "${plog}" 2>&1
    echo "[pp] slot${slot}(GPU${gpu}) ${name} exit=$? ($(( $(date +%s) - t0 ))s)"
    idx=$(( idx + SLOTS ))
  done
}
pids=(); for s in 0 1 2 3 4 5 6 7; do worker "$s" & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done
echo "[pp] predict phase complete"

# ── Phase 3+4: verify exact counts, then evaluate ───────────────────────────
# Evaluation calls the evaluator PYTHON DIRECTLY rather than going through
# 06_01_evaluate_run.sh. Those wrappers dispatch via run_job -> sbatch, i.e. a NESTED
# job submission from inside this compute node, which CLAUDE.md records as unreliable
# on TamIA (CPU-only submission there has failed and succeeded on identical scripts).
# Same reason ispy2/ambl's 05_20 evaluates inline. The metrics layout written here is
# byte-identical to what the wrappers produce, so aggregation is unaffected.
PY="${PROJECT_ROOT}/.venv/bin/python"
SUMMARIZE="${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/summarize_fold.py"
DJ="${TF2_RAW}/dataset.json"
fail=0

echo "[pp] ==== PHASE 3+4a: own held-out CBCT (in-domain), 3 labels ===="
TF2_METRICS=/scratch/p/paulh/toothfairy2/8_results/02_metrics/toothfairy2_model/cbct
for row in "${SELECTED[@]}"; do
  IFS=: read -r TOK OWNW HSW RID CAT SUB <<< "${row}"
  for F in 0 1 2; do
    D="${TF2_PRED_ROOT}/${CAT}/${RID}/fold${F}/cbct"
    n=$(find "$D" -name '*.nii.gz' 2>/dev/null | wc -l)
    if [ "$n" != "71" ]; then echo "[pp] BAD OWN COUNT ${n}/71: $D" >&2; fail=1; continue; fi
    OUT="${TF2_METRICS}${SUB:+/${SUB}}/${CAT}_${RID}/fold${F}"; mkdir -p "$OUT"
    "$PY" "${TF2}/06_evaluate/06_00_evaluate.py" --pred_dir "$D" --gt_dir "${TF2_RAW}/labelsTs_cbct" \
      --dataset_json "$DJ" --labels mandible lower_teeth pharynx --name cbct \
      --out_csv "${OUT}/cbct_metrics.csv" --workers 6 > "${OUT}/cbct_eval.log" 2>&1 \
      || { echo "[pp] own eval FAILED ${RID} fold${F}" >&2; fail=1; continue; }
    rows=$(( $(wc -l < "${OUT}/cbct_metrics.csv") - 1 ))
    echo "[pp] own eval ${TOK} fold${F}: ${rows} rows"
    "$PY" "$SUMMARIZE" "$OUT" "$RID" "$F" --groups cbct --group-col contrast --groups-word Contrasts \
      >> "${OUT}/cbct_eval.log" 2>&1
  done
done

echo "[pp] ==== PHASE 3+4b: hanseg CT (CROSS-MODALITY), mandible union ===="
HS_PRED=/scratch/p/paulh/hanseg/8_results/01_predictions/toothfairy2_model/cbct
HS_METRICS=/scratch/p/paulh/hanseg/8_results/02_metrics/toothfairy2_model/cbct
for row in "${SELECTED[@]}"; do
  IFS=: read -r TOK OWNW HSW RID CAT SUB <<< "${row}"
  for F in 0 1 2; do
    D="${HS_PRED}/${CAT}/${RID}/fold${F}/ct"
    n=$(find "$D" -name '*.nii.gz' 2>/dev/null | wc -l)
    if [ "$n" != "42" ]; then echo "[pp] BAD HANSEG COUNT ${n}/42: $D" >&2; fail=1; continue; fi
    M="${D}_mandible_union"
    "$PY" "${HS}/05_predict/05_20_merge_mandible_union.py" --pred_dir "$D" --out_dir "$M" >/dev/null 2>&1 \
      || { echo "[pp] union merge FAILED ${RID} fold${F}" >&2; fail=1; continue; }
    OUT="${HS_METRICS}${SUB:+/${SUB}}/${CAT}_${RID}/fold${F}"; mkdir -p "$OUT"
    "$PY" "${HS}/06_evaluate/06_00_evaluate.py" --pred_dir "$M" --gt_dir "${HS_RAW}/labelsTs_ct" \
      --label_map '{"mandible": [1, 1]}' --name ct \
      --out_csv "${OUT}/ct_metrics.csv" --workers 6 > "${OUT}/ct_eval.log" 2>&1 \
      || { echo "[pp] hanseg eval FAILED ${RID} fold${F}" >&2; fail=1; continue; }
    rows=$(( $(wc -l < "${OUT}/ct_metrics.csv") - 1 ))
    echo "[pp] hanseg eval ${TOK} fold${F}: ${rows} rows"
    "$PY" "$SUMMARIZE" "$OUT" "$RID" "$F" --groups ct --group-col contrast --groups-word Contrasts \
      >> "${OUT}/ct_eval.log" 2>&1
  done
done
echo "[pp] eval fail flag = ${fail}"

echo "[pp] DONE — $(date)"
