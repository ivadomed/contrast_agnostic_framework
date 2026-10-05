#!/usr/bin/env bash
# TamIA launcher: FINAL ladder rung = the full OURS recipe (PALETTE + the usual AugLab augmentations; DualVal
# trainer where the dataset has one, val000 otherwise) + the boundary partial-volume option, for all 16 ladder
# training sets x 3 folds = 48 fold trainings, packed onto 5 whole-node 4xH100 pack chains.
# Each fold is recorded by its dataset's *_auglabAug_v26_6_2_pv_* wrapper = a copy of that dataset's current OURS
# wrapper with ONLY the train config swapped (OURS json + pv keys), METHOD naming (auglabAug_v26_6_2_pv_...), and
# for chaos T1in the epoch default 300->200 (matches its existing OURS run and the chaos policy). Trainer, epochs,
# validation config all come from the wrapper. Results land in the same `auglab/` category as the OURS runs.
#   bash scripts/cluster/tamia_pack_rung7_auglab_pv.sh record [PACK...]   # record + verify, no submit
#   bash scripts/cluster/tamia_pack_rung7_auglab_pv.sh submit [PACK...]   # submit chains (resume-safe)
# A pack whose index.tsv already exists is never re-recorded. Predict/eval: not queued here.
# PROBE=1 (any mode): rehearsal of the exact same recorded commands with 2 epochs, results under
#   ${SCRATCH}/_rung7_probe (throwaway), separate state file, 45 min x 1 job per pack -- to check startup, config,
#   placement and GPU memory BEFORE the real launch.
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
export SCRATCH="${SCRATCH:-/scratch/p/paulh}"
T=benchmark/02_tasks
STATE="${SCRATCH}/_packruns_rung7_auglab_pv_root${PROBE:+_probe}.txt"
MODE="${1:?usage: record|submit [PACK...]}"; shift
ALL_PACKS="A_brats B_onharmony C_openms_tf2 D_ispy2_chaos E_pelvic"
SEL="${*:-${ALL_PACKS}}"
NEWCFG=transform_params_gpu_default01-23_auglabAug_pv_ImageContrastV26_6_2GPUTransform_train050.json
want() { [[ " ${SEL} " == *" $1 "* ]] && { [ "${MODE}" != record ] || [ ! -f "${PACKS_ROOT}/$1/index.tsv" ]; }; }
if [ -f "${STATE}" ]; then PACKS_ROOT="$(cat "${STATE}")"
else
    [ "${MODE}" = record ] || { echo "nothing recorded yet"; exit 1; }
    PACKS_ROOT="${SCRATCH}/_packruns/rung7_auglab_pv${PROBE:+_probe}_$(date +%Y%m%d_%H%M%S)"; echo "${PACKS_ROOT}" > "${STATE}"
fi
echo "[rung7] PACKS_ROOT=${PACKS_ROOT}"

# pack | dataset scripts dir | new pv wrapper | original OURS wrapper | tamia env | expected epochs
ROWS=(
"A_brats|brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma|04_85_train_t1n_auglabAug_v26_6_2_pv_train050_val000.sh|04_28_train_t1n_auglabAug_v26_6_2_train050_val000.sh|tamia_env.sh|2500"
"A_brats|brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma|04_86_train_t2w_auglabAug_v26_6_2_pv_dualval.sh|04_32_train_t2w_auglabAug_v26_6_2_dualval.sh|tamia_env.sh|2500"
"A_brats|brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma|04_87_train_t2f_auglabAug_v26_6_2_pv_dualval.sh|04_55_train_t2f_auglabAug_v26_6_2_dualval.sh|tamia_env.sh|2500"
"A_brats|brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma|04_88_train_t1c_auglabAug_v26_6_2_pv_dualval.sh|04_75_train_t1c_auglabAug_v26_6_2_dualval.sh|tamia_env.sh|2500"
"B_onharmony|brain_healthy/on-harmony/5_scripts_on-harmony|04_50_train_t1w_auglabAug_v26_6_2_pv_dualval.sh|04_26_train_t1w_auglabAug_v26_6_2_dualval.sh|tamia_env_onharmony.sh|2000"
"B_onharmony|brain_healthy/on-harmony/5_scripts_on-harmony|04_51_train_t2w_auglabAug_v26_6_2_pv_train050_val000.sh|04_23_train_t2w_auglabAug_v26_6_2_train050_val000.sh|tamia_env_onharmony.sh|2000"
"B_onharmony|brain_healthy/on-harmony/5_scripts_on-harmony|04_52_train_dwi_ap_auglabAug_v26_6_2_pv_dualval.sh|04_41_train_dwi_ap_auglabAug_v26_6_2_dualval.sh|tamia_env_onharmony.sh|2000"
"C_openms_tf2|brain_ms/open-ms/5_scripts_open-ms|04_41_train_flair_auglabAug_v26_6_2_pv_train050_val000.sh|04_22_train_auglabAug_v26_6_2_train050_val000.sh|tamia_env_openms.sh|2000"
"C_openms_tf2|brain_ms/open-ms/5_scripts_open-ms|04_42_train_t1w_auglabAug_v26_6_2_pv_train050_val000.sh|04_23_train_t1w_auglabAug_v26_6_2_train050_val000.sh|tamia_env_openms.sh|2000"
"C_openms_tf2|mandible_healthy/toothfairy2/5_scripts_toothfairy2|04_16_train_cbct_auglabAug_v26_6_2_pv_dualval.sh|04_06_train_auglabAug_v26_6_2_dualval.sh|tamia_env_toothfairy2.sh|1000"
"D_ispy2_chaos|breast_cancer/ispy2/5_scripts_ispy2|04_28_train_t1wce_auglabAug_v26_6_2_pv_dualval.sh|04_06_train_t1wce_auglabAug_v26_6_2_dualval.sh|tamia_env_ispy2.sh|1000"
"D_ispy2_chaos|breast_cancer/ispy2/5_scripts_ispy2|04_29_train_t2w_auglabAug_v26_6_2_pv_dualval.sh|04_13_train_t2w_auglabAug_v26_6_2_dualval.sh|tamia_env_ispy2.sh|1000"
"D_ispy2_chaos|abdomen_healthy/chaos/5_scripts_chaos|04_61_train_t1in_auglabAug_v26_6_2_pv_train050_val000.sh|04_09_train_auglabAug_v26_6_2_train050_val000.sh|tamia_env_chaos.sh|200"
"D_ispy2_chaos|abdomen_healthy/chaos/5_scripts_chaos|04_62_train_t2spir_auglabAug_v26_6_2_pv_train050_val000.sh|04_44_train_t2spir_auglabAug_v26_6_2_train050_val000.sh|tamia_env_chaos.sh|200"
"E_pelvic|pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic|04_25_train_ct_auglabAug_v26_6_2_pv_dualval.sh|04_06_train_ct_auglabAug_v26_6_2_dualval.sh|tamia_env_totalseg-pelvic.sh|200"
"E_pelvic|pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic|04_26_train_mri_auglabAug_v26_6_2_pv_dualval.sh|04_13_train_mri_auglabAug_v26_6_2_dualval.sh|tamia_env_totalseg-pelvic.sh|200"
)
declare -A EXPECT=([A_brats]=12 [B_onharmony]=9 [C_openms_tf2]=9 [D_ispy2_chaos]=12 [E_pelvic]=6)
declare -A CHAIN=([A_brats]=8 [B_onharmony]=8 [C_openms_tf2]=6 [D_ispy2_chaos]=6 [E_pelvic]=3)

if [ "${MODE}" = record ]; then
  # decide which packs to record BEFORE the loop (the first wrapper of a pack creates its index.tsv)
  TODO=""; for p in ${SEL}; do want "$p" && TODO="${TODO} $p"; done
  for r in "${ROWS[@]}"; do IFS='|' read -r pack dir w orig tenv ep <<<"$r"
    [[ " ${TODO} " == *" ${pack} "* ]] || continue
    wrapper="$T/$dir/04_train/$w"
    envf="$(grep -o 'source "$(dirname "$0")/../00_utils/env[a-z0-9_]*\.sh"' "${wrapper}" | sed 's|.*/00_utils/||; s|"$||')"
    echo "[record] ${pack} <- ${w}  (env ${envf}, ${tenv})"
    (
        source "$T/$dir/00_utils/${envf}"; source "scripts/cluster/${tenv}"
        # probe: throwaway results tree (the wrappers' env files keep an already-exported PREDICTIONS_ROOT)
        [ -z "${PROBE:-}" ] || { export PREDICTIONS_ROOT="${SCRATCH}/_rung7_probe/pred"; mkdir -p "${PREDICTIONS_ROOT}"; }
        # category dir like the OURS runs (<PREDICTIONS_ROOT>/<model>/<contrast>/auglab), on scratch; a wrapper that
        # sets its own NNUNET_RESULTS_BASE overrides this with the same value
        export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"
        RUN_JOB_PACK_DIR="${PACKS_ROOT}/${pack}" bash "${wrapper}"
    )
  done
  if [ -n "${PROBE:-}" ]; then   # 2 epochs in every recorded command (some wrappers hard-set the epoch count, so patch the cmd files)
    python3 - "${PACKS_ROOT}" <<'PY'
import re, sys, glob
n = 0
for f in glob.glob(sys.argv[1] + "/*/fold*.sh"):
    s = open(f).read(); t, k = re.subn(r"NNUNET_NUM_EPOCHS=(\\?'?)\d+", r"NNUNET_NUM_EPOCHS=\g<1>2", s)
    open(f, "w").write(t); n += k
print(f"[probe] epochs patched to 2 in {n} places")
PY
    echo "[probe] epoch values now: $(grep -ho "NNUNET_NUM_EPOCHS=[^ ]*" "${PACKS_ROOT}"/*/fold*.sh | sort | uniq -c | tr '\n' ' ')"
  fi
  echo; echo "=== verification (every recorded fold) ==="; bad=0
  for p in ${SEL}; do
    n=$(wc -l < "${PACKS_ROOT}/$p/index.tsv"); echo "-- $p: ${n} folds (expect ${EXPECT[$p]})"
    [ "$n" = "${EXPECT[$p]}" ] || { echo "   FOLD COUNT MISMATCH"; bad=1; }
    while IFS=$'\t' read -r cmd log name done_; do
      clean="$(sed -e 's/\\n/\n/g' -e "s/\\\\'//g" "$cmd")"
      get() { printf '%s\n' "$clean" | sed -n "s/^ *export $1=//p" | head -1; }
      pre=$(get nnUNet_preprocessed); res=$(get nnUNet_results); cfg=$(get AUGLAB_PARAMS_GPU_JSON | sed 's|.*/||'); ep=$(get NNUNET_NUM_EPOCHS)
      tr=$(printf '%s\n' "$clean" | grep -o '\-tr [A-Za-z0-9_]*' | head -1 | cut -d' ' -f2)
      ok=OK
      case "$pre" in /scratch/*) [ -d "$pre" ] || ok="NO_PREPROC_DIR";; *) ok="PREPROC_NOT_SCRATCH";; esac
      case "$res" in /scratch/*) ;; *) ok="${ok},RESULTS_NOT_SCRATCH";; esac
      [[ "$res" == */auglab/* ]] || ok="${ok},NOT_AUGLAB_CATEGORY"
      [[ "$res" == *auglabAug_v26_6_2_pv_train050_val000_* ]] || ok="${ok},RUNID_NOT_PV"
      [ "$cfg" = "${NEWCFG}" ] || ok="${ok},WRONG_CFG($cfg)"
      grep -q "nnUNetv2_train" "$cmd" || ok="${ok},NO_TRAIN_CMD"
      [ -e "$res" ] && ok="${ok},RESULTS_DIR_ALREADY_EXISTS"
      printf '   %-6s %s\n      trainer=%s ep=%s cfg=%s\n      preproc=%s\n      results=%s\n' "$ok" "$name" "$tr" "$ep" "$cfg" "$pre" "$res"
      [ "$ok" = OK ] || bad=1
    done < "${PACKS_ROOT}/$p/index.tsv"
  done
  # trainer/epochs vs the original OURS wrappers: compare declared values textually (record-independent, per wrapper pair)
  echo; echo "=== wrapper pairs: new vs original OURS (trainer, epoch default, config) ==="
  for r in "${ROWS[@]}"; do IFS='|' read -r pack dir w orig tenv eep <<<"$r"
    a="$T/$dir/04_train/$w"; b="$T/$dir/04_train/$orig"
    ta=$(grep -o '^TRAINER="[^"]*"' "$a"); tb=$(grep -o '^TRAINER="[^"]*"' "$b")
    ea=$(grep -o 'NUM_EPOCHS[:=-]*[^}"]*' "$a" | head -1); eb=$(grep -o 'NUM_EPOCHS[:=-]*[^}"]*' "$b" | head -1)
    s=OK; [ "$ta" = "$tb" ] || s="TRAINER_DIFFERS"
    [ "$ea" = "$eb" ] || { [[ "$w" == *t1in* ]] && s="${s} (epochs ${eb}->${ea}: intended)" || s="${s} EPOCHS_DIFFER"; }
    case "$s" in OK*) ;; *) [ "$s" = "OK (epochs ${eb}->${ea}: intended)" ] || bad=1;; esac
    printf '   %-40s %s  %s\n' "$s" "$w" "$ta"
  done
  [ "$bad" = 0 ] && echo "VERIFY PASSED" || echo "VERIFY FAILED"; exit 0
fi

[ "${MODE}" = submit ] || { echo "unknown mode ${MODE}"; exit 2; }
for p in ${SEL}; do
  if [ -n "${PROBE:-}" ]; then _time="00:45:00"; _chain=1; _name="rung7probepv_${p}"
  else _time="${PACK_TIME:-23:59:00}"; _chain="${CHAIN[$p]}"; _name="rung7pv_${p}"; fi
  PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${_time}" PACK_CHAIN="${_chain}" \
  PACK_USE_MPS=0 PACK_JOB_NAME="${_name}" bash scripts/job_runner/run_job_pack_submit.sh "${PACKS_ROOT}/${p}"
  sleep 5
done
