#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#
# HARD-CROP FOV variant of the chaos cross-dataset evaluation (TamIA).
#
# The standing pipeline applies the CHAOS FOV slab as a MASK at EVALUATION time
# (zeroing pred+GT outside the slab) -- the network still *sees* the whole
# full-extent volume at inference. This variant instead CROPS the volume in
# place BEFORE prediction, so the model sees only the chaos-equivalent slab,
# matching chaos's own restricted FOV as an input condition rather than as a
# scoring rule.
#
# Results land in a DEDICATED `fov_crop/` metrics subdir (per CLAUDE.md: a
# non-headline result set gets its own subdir) so the existing headline
# masked-eval numbers are never touched.
#
# Driven entirely by env vars -- one script, all datasets:
#   DATASET       e.g. cirrmri-liver
#   ITEMS         space-separated test items, e.g. "t1 t2"  /  "ct mri"
#   ANCHOR        chaos_fov_margins.json anchor name: liver|kidney|spleen
#   ANCHOR_IDS    comma-sep anchor ids in THIS dataset's OWN GT numbering
#   CONTRASTS     which chaos training contrasts to run, e.g. "t1in t2spir"
#   EVAL_MODE     generic_labels | label_map | amos
#   EVAL_LABELS   generic_labels: chaos label name(s), e.g. "liver"
#   EVAL_LABEL_MAP  label_map: JSON, e.g. '{"spleen": [4, 1]}'
#   EVAL_PARALLEL optional, concurrent eval tasks (default 20). Evaluation is
#                 CPU-only and the node has 48 cores, so the old default of 6
#                 (x2 workers = 12 procs) left the node 75% idle: cirrmri's eval
#                 projected ~75 min at P=6 and finished in 4.5 min at P=20.
#                 For a big dataset prefer ALSO splitting by contrast into
#                 separate jobs (and run eval on a CPU-only allocation) rather
#                 than one whole-node GPU job doing everything serially.
set -uo pipefail
PROJECT_ROOT="/project/aip-jcohen/paulh/mri_synthesis_project"
SCRATCH="${SCRATCH:-/scratch/p/paulh}"
: "${DATASET:?}" "${ITEMS:?}" "${ANCHOR:?}" "${ANCHOR_IDS:?}" "${CONTRASTS:?}" "${EVAL_MODE:?}"

export NNUNET_PROJECT_ROOT="${PROJECT_ROOT}"
RAW="${SCRATCH}/${DATASET}/2_nnUNet_${DATASET}/raw"
RES="${SCRATCH}/${DATASET}/8_results_${DATASET}"
WORK="${SCRATCH}/${DATASET}/_fovcrop/${SLURM_JOB_ID:-manual}"
FOV_JSON="${PROJECT_ROOT}/datasets/chaos/5_scripts_chaos/06_evaluate/chaos_fov_margins.json"
CHAOS_SCRATCH="${SCRATCH}/chaos"
CHAOS_DATASET_JSON="${CHAOS_SCRATCH}/2_nnUNet_chaos/raw/Dataset060_CHAOS_MR_T1in/dataset.json"
mkdir -p "${WORK}"

echo "[fov] $(hostname) job=${SLURM_JOB_ID:-?} dataset=${DATASET} items='${ITEMS}' anchor=${ANCHOR}(${ANCHOR_IDS}) contrasts='${CONTRASTS}'"

# ── run roster: the canonical 8 per contrast (the ones feeding the tables) ───
runs_for() {   # $1 = contrast; emits "category|run_id|trainer" per line
  if [ "$1" = t1in ]; then cat <<'EOF'
nnUNet|chaos_t1in_baseline_20260614_153230|nnUNetTrainerCHAOSBaseline
auglab|chaos_t1in_auglab_default_20260611_120000|nnUNetTrainerCHAOSAugLabDefault
auglab|chaos_t1in_synthseg_noEM_train100_val000_20260611_120000|nnUNetTrainerCHAOSAugLabDefault
auglab|chaos_t1in_synthseg_EM_train100_val000_20260611_120000|nnUNetTrainerCHAOSAugLabDefault
nnUNet|chaos_t1in_v26_6_2_train050_val100_20260615_213615|nnUNetTrainerCHAOSV26_6_2_p50
auglab|chaos_t1in_srcsm_20260710_011817|nnUNetTrainerCHAOSAugLabDefault
auglab|chaos_t1in_auglabAug_v26_6_2_train050_val000_20260615_213615|nnUNetTrainerCHAOSAugLabV26_6_2
auglab|chaos_t1in_auglabAug_v26_6_2_train050_val100_20260616_112420|nnUNetTrainerCHAOSAugLabValSynth
EOF
  else cat <<'EOF'
nnUNet|chaos_t2spir_baseline_20260620_111146|nnUNetTrainerCHAOSBaseline
auglab|chaos_t2spir_auglab_default_20260620_112240|nnUNetTrainerCHAOSAugLabDefault
auglab|chaos_t2spir_synthseg_noEM_20260620_112515|nnUNetTrainerCHAOSAugLabDefault
auglab|chaos_t2spir_synthseg_EM_20260620_112357|nnUNetTrainerCHAOSAugLabDefault
nnUNet|chaos_t2spir_v26_6_2_train050_val100_20260620_112122|nnUNetTrainerCHAOSV26_6_2_p50
auglab|chaos_t2spir_srcsm_20260709_121945|nnUNetTrainerCHAOSAugLabDefault
auglab|chaos_t2spir_auglabAug_v26_6_2_train050_val000_20260710_054202|nnUNetTrainerCHAOSAugLabV26_6_2
auglab|chaos_t2spir_auglabAug_v26_6_2_train050_val100_20260723_194053|nnUNetTrainerCHAOSAugLabValSynth
EOF
  fi
}
ds_id_for() { [ "$1" = t1in ] && echo 60 || echo 61; }

# ── PHASE 1: build hard-cropped inputs, per (contrast, item) ─────────────────
echo "[fov] ==== PHASE 1: hard-crop volumes to the chaos ${ANCHOR} slab ===="
"${PROJECT_ROOT}/.venv/bin/python" - "$WORK" "$RAW" "$FOV_JSON" "$ANCHOR" "$ANCHOR_IDS" "$ITEMS" "$CONTRASTS" <<'PYEOF'
import sys, os, glob, json
import numpy as np, SimpleITK as sitk
sys.path.insert(0, "/project/aip-jcohen/paulh/mri_synthesis_project/datasets/00_commun_scripts/00_00_utils")
from fov import si_axis_sign, bbox_si, slab_keep_range   # shared geometry, not reimplemented

WORK, RAW, FOV_JSON, ANCHOR, ANCHOR_IDS, ITEMS, CONTRASTS = sys.argv[1:8]
anchor_ids = [int(x) for x in ANCHOR_IDS.split(",")]
margins = json.load(open(FOV_JSON))

for contrast in CONTRASTS.split():
    m = margins[contrast][ANCHOR]
    sup_mm, inf_mm = float(m["sup_mm"]), float(m["inf_mm"])
    for item in ITEMS.split():
        oi = f"{WORK}/{contrast}/imagesTs_{item}"; og = f"{WORK}/{contrast}/labelsTs_{item}"
        os.makedirs(oi, exist_ok=True); os.makedirs(og, exist_ok=True)
        cases = sorted(os.path.basename(p)[:-len("_0000.nii.gz")]
                       for p in glob.glob(f"{RAW}/imagesTs_{item}/*_0000.nii.gz"))
        kept, skipped, ratios = 0, 0, []
        for c in cases:
            img = sitk.ReadImage(f"{RAW}/imagesTs_{item}/{c}_0000.nii.gz")
            gt  = sitk.ReadImage(f"{RAW}/labelsTs_{item}/{c}.nii.gz")
            np_axis, sup_dir, sp = si_axis_sign(img)
            gt_arr = sitk.GetArrayFromImage(gt)
            bb = bbox_si(np.isin(gt_arr, anchor_ids), np_axis)
            if bb is None:            # no anchor organ -> cannot place the slab
                skipped += 1; continue
            lo, hi = slab_keep_range(gt_arr.shape, np_axis, sup_dir, sp,
                                     bb[0], bb[1], sup_mm, inf_mm)
            itk_axis = 2 - np_axis    # GetArrayFromImage reverses axis order
            sl = [slice(None)] * 3
            sl[itk_axis] = slice(lo, hi + 1)     # sitk slicing fixes origin for us
            sitk.WriteImage(img[tuple(sl)], f"{oi}/{c}_0000.nii.gz")
            sitk.WriteImage(gt[tuple(sl)],  f"{og}/{c}.nii.gz")
            ratios.append((hi - lo + 1) / gt_arr.shape[np_axis]); kept += 1
        print(f"[fov]   {contrast}/{item}: cropped {kept}, skipped(no-anchor) {skipped}, "
              f"mean kept-fraction along S-I = {np.mean(ratios) if ratios else float('nan'):.3f}",
              flush=True)
PYEOF

# ── PHASE 2: predict on the cropped inputs, bounded 8-way pool (2/GPU) ───────
echo "[fov] ==== PHASE 2: predict (8 runs x 3 folds x items x contrasts) ===="
JOBS=()
for contrast in ${CONTRASTS}; do
  while IFS='|' read -r cat rid tr; do
    [ -n "$cat" ] || continue
    for f in 0 1 2; do for it in ${ITEMS}; do
      JOBS+=("${contrast}|${cat}|${rid}|${tr}|${f}|${it}")
    done; done
  done < <(runs_for "$contrast")
done
N=${#JOBS[@]}; echo "[fov] ${N} predict tasks"

worker() {
  local slot="$1" idx="$1" gpu=$(( $1 % 4 ))
  while [ "${idx}" -lt "${N}" ]; do
    IFS='|' read -r contrast cat rid tr f it <<< "${JOBS[$idx]}"
    local out="${RES}/01_predictions/chaos_model/${contrast}/${cat}/${rid}/fold${f}/${it}_fovcrop"
    local inp="${WORK}/${contrast}/imagesTs_${it}"
    mkdir -p "${out}"
    local t0=$(date +%s)
    nnUNet_results="${CHAOS_SCRATCH}/8_results_chaos/01_predictions/chaos_model/${contrast}/${cat}/${rid}" \
    CUDA_VISIBLE_DEVICES=${gpu} \
      "${PROJECT_ROOT}/.venv/bin/nnUNetv2_predict" -i "${inp}" -o "${out}" \
      -d "$(ds_id_for "$contrast")" -c 3d_fullres -tr "${tr}" -f "${f}" --disable_tta \
      -chk checkpoint_best.pth -npp 3 -nps 3 \
      > "${WORK}/predict_${contrast}_${rid}_f${f}_${it}.log" 2>&1
    echo "[fov] slot${slot}(GPU${gpu}) ${contrast}/${rid##*_}/f${f}/${it} rc=$? ($(( $(date +%s) - t0 ))s)"
    idx=$(( idx + 8 ))
  done
}
pids=(); for s in 0 1 2 3 4 5 6 7; do worker "$s" & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done

# ── PHASE 3: evaluate (NO fov flags -- the volumes are already cropped) ──────
echo "[fov] ==== PHASE 3: evaluate cropped predictions (no FOV masking) ===="
case "${EVAL_MODE}" in
  amos) EVAL_PY="${PROJECT_ROOT}/datasets/amos/5_scripts_amos/06_evaluate/06_00_evaluate_amos.py";;
  *)    EVAL_PY="${PROJECT_ROOT}/datasets/chaos/5_scripts_chaos/06_evaluate/06_00_evaluate.py";;
esac

eval_one() {   # contrast cat rid fold item
  local contrast="$1" cat="$2" rid="$3" f="$4" it="$5"
  local pred="${RES}/01_predictions/chaos_model/${contrast}/${cat}/${rid}/fold${f}/${it}_fovcrop"
  local gt="${WORK}/${contrast}/labelsTs_${it}"
  local ed="${RES}/02_metrics/chaos_model/${contrast}/fov_crop/${cat}_${rid}/fold${f}"
  local n; n=$(find "${pred}" -name '*.nii.gz' 2>/dev/null | wc -l)
  [ "${n}" -gt 0 ] || { echo "[fov] SKIP empty ${contrast}/${rid}/f${f}/${it}" >&2; return 1; }
  mkdir -p "${ed}"
  local extra=()
  case "${EVAL_MODE}" in
    generic_labels) extra=(--dataset_json "${CHAOS_DATASET_JSON}" --labels ${EVAL_LABELS});;
    label_map)      extra=(--label_map "${EVAL_LABEL_MAP}");;
  esac
  "${PROJECT_ROOT}/.venv/bin/python" "${EVAL_PY}" \
      --pred_dir "${pred}" --gt_dir "${gt}" --name "${it}" \
      --out_csv "${ed}/${it}_metrics.csv" --workers 2 "${extra[@]}" \
      > "${ed}/${it}_eval.log" 2>&1
  echo "[fov] eval ${contrast}/${rid}/f${f}/${it}: $(( $(wc -l < "${ed}/${it}_metrics.csv") - 1 ))/${n} rows"
}
export -f eval_one; export RES WORK PROJECT_ROOT EVAL_PY EVAL_MODE CHAOS_DATASET_JSON
export EVAL_LABELS="${EVAL_LABELS:-}" EVAL_LABEL_MAP="${EVAL_LABEL_MAP:-}"
printf '%s\n' "${JOBS[@]}" | awk -F'|' '{print $1,$2,$3,$5,$6}' \
  | xargs -P "${EVAL_PARALLEL:-20}" -L 1 bash -c 'eval_one "$@"' _
echo "[fov] evaluate phase done"

# ── PHASE 4: merge -> eval_all.csv (depth 4: .../<contrast>/fov_crop/<run>) ──
echo "[fov] ==== PHASE 4: summarize_fold -> eval_all.csv ===="
find "${RES}/02_metrics" -mindepth 4 -maxdepth 4 -type d -path '*/fov_crop/*' | while read -r rd; do
  rid="$(basename "${rd}")"; rid="${rid#nnUNet_}"; rid="${rid#auglab_}"
  for fd in "${rd}"/fold*/; do
    [ -d "${fd}" ] || continue
    have=(); for it in ${ITEMS}; do [ -f "${fd}${it}_metrics.csv" ] && have+=("${it}"); done
    [ ${#have[@]} -gt 0 ] || continue
    "${PROJECT_ROOT}/.venv/bin/python" \
      "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/summarize_fold.py" \
      "${fd}" "${rid}" "$(basename "${fd%/}" | sed 's/fold//')" \
      --group-col modality --groups-word Modalities \
      --title-suffix " | ${DATASET} | FOV HARD-CROP" --groups "${have[@]}" >/dev/null 2>&1
  done
done

# ── PHASE 5: report cropped vs existing masked numbers ──────────────────────
echo "[fov] ==== PHASE 5: RESULTS (hard-crop vs masked-eval) ===="
"${PROJECT_ROOT}/.venv/bin/python" - "${RES}/02_metrics" "${DATASET}" "${CONTRASTS}" "${ITEMS}" <<'PYEOF'
import sys, os, csv, glob, statistics as st
from collections import defaultdict
METRICS, DATASET, CONTRASTS, ITEMS = sys.argv[1:5]
VULCAN = f"/project/aip-jcohen/paulh/mri_synthesis_project/datasets/{DATASET}/8_results_{DATASET}/02_metrics"

def load(base):
    out=defaultdict(lambda: defaultdict(list))     # run -> item -> dices
    for csvp in glob.glob(f"{base}/fold*/eval_all.csv"):
        for r in csv.DictReader(open(csvp)):
            out[os.path.basename(base)][r["group"]].append(float(r["dice"]))
    return out

for contrast in CONTRASTS.split():
    print(f"\n=== {DATASET} / chaos_{contrast} : mean Dice x100 ===")
    items = ITEMS.split()
    hdr = f"{'run':<62}" + "".join(f"{i+' crop':>12}{i+' mask':>12}" for i in items)
    print(hdr); print("-"*len(hdr))
    crop_root = f"{METRICS}/chaos_model/{contrast}/fov_crop"
    for rd in sorted(glob.glob(f"{crop_root}/*")):
        if not os.path.isdir(rd): continue
        run = os.path.basename(rd)
        c = load(rd).get(run, {})
        m = load(f"{VULCAN}/chaos_model/{contrast}/{run}").get(run, {})
        line = f"{run[:60]:<62}"
        for it in items:
            cv = f"{st.mean(c[it])*100:.1f}" if c.get(it) else "--"
            mv = f"{st.mean(m[it])*100:.1f}" if m.get(it) else "--"
            line += f"{cv:>12}{mv:>12}"
        print(line)
PYEOF
echo "[fov] DONE"
