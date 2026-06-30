#!/usr/bin/env bash
# Evaluate a trained on-harmony model on the cross-contrast test set, writing into the
# STANDARD results layout (identical to chaos/brats):
#   predictions → PREDICTIONS_ROOT/<model>/<train_contrast>/<category>/<RUN_ID>/fold{k}/<test_contrast>/
#   metrics     → METRICS_ROOT/<model>/<train_contrast>/<category>_<RUN_ID>/fold{k}/eval_all.csv
# Aggregate across runs with the SHARED scripts/evaluate/aggregate_from_config.py via
# 06_06_aggregate_from_config.sh — exactly the same as chaos.
#
# TWO MODES:
#   LAUNCHER:  bash 06_01_evaluate_testset.sh <RUN_ID>
#       Assembles the shared cross-contrast test set once (cheap), then fans out ONE GPU
#       job per fold (the job IS the compute — no idle CPU coordinator).
#   WORKER:    bash 06_01_evaluate_testset.sh <RUN_ID> <FOLD>   (runs inside a 1-GPU job)
#       Predicts every contrast for its fold → resamples to native → shared evaluate.py
#       (Dice+HD95) → summarize_fold → fold{k}/eval_all.csv (group=test contrast).
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

RUN_ID="${1:?Usage: $0 <RUN_ID> [FOLD]}"
FOLD="${2:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=".venv/bin/python"

BIDS="${BIDS_ROOT}"
TEST_CASES="${PROJECT_ROOT}/datasets/on-harmony/4_splits_on-harmony/test_cases.json"

# Standard layout (mirror chaos): the trained model is CO-LOCATED with its predictions under
#   01_predictions/<model>/<train_contrast>/<nnUNet|auglab>/<RUN_ID>/DatasetXXX.../
# Training contrast comes from the RUN_ID; discover which category dir actually holds this run
# (nnUNet vs auglab) by looking for the trainer dir under each — same RUN_ID is unique.
TRAIN_CONTRAST="$(echo "$RUN_ID" | grep -oE 'T[12]w' | head -1)"; TRAIN_CONTRAST="${TRAIN_CONTRAST:-T1w}"
RUN_BASE=""; CATEGORY=""
for CAT in nnUNet auglab; do
    cand="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAIN_CONTRAST}/${CAT}/${RUN_ID}"
    if ls -d "${cand}"/*/*__nnUNetPlans__3d_fullres >/dev/null 2>&1; then
        RUN_BASE="$cand"; CATEGORY="$CAT"; break
    fi
done
[ -z "$RUN_BASE" ] && { echo "ERROR: no trained model for ${RUN_ID} under ${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAIN_CONTRAST}/{nnUNet,auglab}/"; exit 1; }

RESULTS_DIR="${RUN_BASE}"                       # trained model lives here (co-located w/ predictions, chaos-style)
export nnUNet_raw nnUNet_preprocessed
export nnUNet_results="${RUN_BASE}"
export NNUNET_PROJECT_ROOT="${PROJECT_ROOT}"

[ -f "$TEST_CASES" ] || { echo "ERROR: $TEST_CASES not found"; exit 1; }

# Trainer + dataset auto-discovered from the run dir.
_TDIR="$(ls -d "${RESULTS_DIR}"/*/*__nnUNetPlans__3d_fullres 2>/dev/null | head -1)"
[ -z "$_TDIR" ] && { echo "ERROR: no trainer dir under ${RESULTS_DIR}"; exit 1; }
TRAINER="$(basename "${_TDIR}" | sed 's/__nnUNetPlans__3d_fullres$//')"
DS_NAME="$(basename "$(dirname "${_TDIR}")")"
DATASET_ID="$(echo "${DS_NAME}" | sed -E 's/^Dataset([0-9]+)_.*/\1/')"
DJ="${nnUNet_raw}/${DS_NAME}/dataset.json"

TESTSET="${PREDICTIONS_ROOT}/${MODEL_TYPE}/_test_set"                                   # shared, run-independent
PRED_BASE="${RUN_BASE}"                                                                  # predictions co-located w/ model (chaos-style)
METRICS_DIR="${METRICS_ROOT}/${MODEL_TYPE}/${TRAIN_CONTRAST}/${CATEGORY}_${RUN_ID}"     # per-run metrics
CONTRAST_LIST="T1w T2w bold dwi_ap epi_ap gre_echo1_mag"

# ════════════════════════════════════════════════════════════════════════════
# LAUNCHER — assemble the shared test set once, then one GPU job per fold
# ════════════════════════════════════════════════════════════════════════════
if [ -z "$FOLD" ]; then
    echo "[$(date '+%H:%M:%S')] LAUNCH eval ${RUN_ID}  trainer=${TRAINER}  -> ${TRAIN_CONTRAST}/${CATEGORY}"
    export BIDS TEST_CASES TESTSET
    # Shared cross-contrast test set (images 4D→3D + 31-class GT). Idempotent: built once,
    # reused by every run/model (the test set is identical across them).
    $PY - <<'PYEOF'
import json, shutil, os
import numpy as np, nibabel as nib
from pathlib import Path
bids       = Path(os.environ["BIDS"])
testset    = Path(os.environ["TESTSET"])
test_cases = json.loads(Path(os.environ["TEST_CASES"]).read_text())
_FS_IDS_31 = [2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,26,28,41,42,43,44,46,47,49,50,51,52,53,54,58,60]
_MAXFS=max(_FS_IDS_31); _LUT=np.zeros(_MAXFS+2, np.uint8)
for i,fs in enumerate(_FS_IDS_31): _LUT[fs]=i+1
def remap(nib_img):
    a=np.asarray(nib_img.dataobj).astype(np.int32)
    return nib.Nifti1Image(_LUT[np.clip(a,0,_MAXFS+1)].astype(np.uint8), nib_img.affine, nib_img.header)
def find_gre(sd):
    swi=sd/"swi"
    if not swi.exists(): return None
    c=sorted(swi.glob("*echo-1*part-mag*GRE.nii.gz")); return c[0] if c else None
CONTRASTS={"T1w":lambda s,e,b:b/s/e/"anat"/f"{s}_{e}_T1w.nii.gz",
           "T2w":lambda s,e,b:b/s/e/"anat"/f"{s}_{e}_T2w.nii.gz",
           "bold":lambda s,e,b:b/s/e/"func"/f"{s}_{e}_task-rest_bold.nii.gz",
           "dwi_ap":lambda s,e,b:b/s/e/"dwi"/f"{s}_{e}_dir-AP_dwi.nii.gz",
           "epi_ap":lambda s,e,b:b/s/e/"fmap"/f"{s}_{e}_dir-AP_epi.nii.gz",
           "gre_echo1_mag":lambda s,e,b:find_gre(b/s/e)}
found=0
for contrast,fn in CONTRASTS.items():
    idir=testset/contrast/"images_native"; idir.mkdir(parents=True,exist_ok=True)
    rdir=testset/contrast/"images_ras";    rdir.mkdir(parents=True,exist_ok=True)
    gdir=testset/contrast/"gt_native";     gdir.mkdir(parents=True,exist_ok=True)
    for tc in test_cases:
        s,e=tc["subject"],tc["session"]; src=fn(s,e,bids)
        if src is None or not src.exists(): continue
        cid=f"{s}_{e}_{contrast}"; out=idir/f"{cid}_0000.nii.gz"
        if not out.exists():
            n=nib.load(str(src))
            if n.ndim>3:
                arr=np.asarray(n.dataobj); v=arr.mean(axis=-1) if contrast=="bold" else arr[...,0]
                nib.save(nib.Nifti1Image(v.astype(np.float32),n.affine,n.header),str(out))
            else:
                shutil.copy2(src,out)
        # RAS-canonical copy = the PREDICTION input. nnU-Net does NOT reorient, and training data
        # is RAS; feeding native LAS (bold/dwi/epi) mirrors the brain for the network, so only
        # mirror-augmented models cope (artifact that inflates v26_6_2 over the rest). Reorienting
        # to RAS makes every model see the orientation it was trained on. No-op for already-RAS
        # contrasts (T1w/T2w/gre). Predictions are resampled back to native space for GT comparison.
        rout=rdir/f"{cid}_0000.nii.gz"
        if not rout.exists():
            nib.save(nib.as_closest_canonical(nib.load(str(out))), str(rout))
        found+=1
        rel=src.relative_to(bids)
        m=bids/"derivatives"/"synthseg_masks"/rel.parent/(rel.name[:-len(".nii.gz")]+"_synthseg.nii.gz")
        gout=gdir/f"{cid}.nii.gz"
        if m.exists() and not gout.exists():
            nib.save(remap(nib.load(str(m))), str(gout))
print(f"shared test set ready ({found} image refs).")
PYEOF
    echo "[$(date '+%H:%M:%S')] fanning out 4 per-fold GPU jobs"
    for F in 0 1 2 3; do
        run_job --name "onheval_${RUN_ID:0:26}_f${F}" --gpus 1 --slot "${F}" --time "${ONHEVAL_TIME:-01:00:00}" \
            --log "${SCRATCH:-/tmp}/onheval_${RUN_ID}_fold${F}.log" -- \
            bash "${HERE}/06_01_evaluate_testset.sh" "${RUN_ID}" "${F}"
    done
    echo "[$(date '+%H:%M:%S')] submitted. metrics → ${METRICS_DIR}/fold*/eval_all.csv"
    exit 0
fi

# ════════════════════════════════════════════════════════════════════════════
# WORKER (FOLD set) — 1-GPU job: predict+resample+eval THIS fold into standard dirs
# ════════════════════════════════════════════════════════════════════════════
export PYTHONPATH="${PROJECT_ROOT}/src/nnunet:${PROJECT_ROOT}/SynthSeg:${PYTHONPATH:-}"
echo "[$(date '+%H:%M:%S')] WORKER ${RUN_ID} fold${FOLD} -> ${TRAIN_CONTRAST}/${CATEGORY} (-d ${DATASET_ID})"
FOLD_METRICS="${METRICS_DIR}/fold${FOLD}"; mkdir -p "$FOLD_METRICS"

for CONTRAST in $CONTRAST_LIST; do
    IMG_DIR="${TESTSET}/${CONTRAST}/images_native"   # native geometry: resample ref + matches GT
    IMG_RAS="${TESTSET}/${CONTRAST}/images_ras"       # RAS-canonical: prediction input (matches RAS training)
    GT_DIR="${TESTSET}/${CONTRAST}/gt_native"
    [ -d "$IMG_RAS" ] && N=$(ls "$IMG_RAS" 2>/dev/null | wc -l) || N=0
    [ "$N" -eq 0 ] && { echo "  [fold${FOLD}] skip $CONTRAST (no images)"; continue; }
    [ -d "$GT_DIR" ] && NGT=$(ls "$GT_DIR" 2>/dev/null | wc -l) || NGT=0

    # Predict on the RAS-canonical input (so the network sees the orientation it was trained on).
    # Raw nnU-Net output (RAS geometry + sidecar JSONs) is a throwaway intermediate → goes to
    # transient node-local scratch. We persist only ONE dir per contrast: the prediction resampled
    # back to NATIVE geometry (matches gt_native), exactly like chaos's single dir per test contrast.
    PRED_RAW="${SLURM_TMPDIR:-${SCRATCH:-/tmp}}/onhpred_${RUN_ID}_f${FOLD}_${CONTRAST}"
    rm -rf "$PRED_RAW"; mkdir -p "$PRED_RAW"
    if ! .venv/bin/nnUNetv2_predict -i "$IMG_RAS" -o "$PRED_RAW" \
            -d "${DATASET_ID}" -c 3d_fullres -f "${FOLD}" -tr "${TRAINER}" -p nnUNetPlans; then
        echo "  ! [fold${FOLD}] predict failed for $CONTRAST (continuing)"; rm -rf "$PRED_RAW"; continue
    fi

    PRED_DIR="${PRED_BASE}/fold${FOLD}/${CONTRAST}"; mkdir -p "$PRED_DIR"
    IN_DIR="$PRED_RAW" OUT_DIR="$PRED_DIR" REF_DIR="$IMG_DIR" $PY - <<'PYEOF2'
import os, SimpleITK as sitk
from pathlib import Path
pd=Path(os.environ["IN_DIR"]); nd=Path(os.environ["OUT_DIR"]); idr=Path(os.environ["REF_DIR"])
for p in sorted(pd.glob("*.nii.gz")):
    cid=p.stem.replace(".nii",""); ref=idr/f"{cid}_0000.nii.gz"
    if not ref.exists(): continue
    pr=sitk.ReadImage(str(p),sitk.sitkUInt8); rf=sitk.ReadImage(str(ref))
    sitk.WriteImage(sitk.Resample(pr,rf,sitk.Transform(),sitk.sitkNearestNeighbor,0,pr.GetPixelID()), str(nd/p.name))
PYEOF2
    rm -rf "$PRED_RAW"

    if [ "$NGT" -gt 0 ]; then
        $PY "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/evaluate.py" \
            --pred_dir "$PRED_DIR" --gt_dir "$GT_DIR" \
            --out_csv "$FOLD_METRICS/${CONTRAST}_metrics.csv" --name "$CONTRAST" --dataset_json "$DJ" \
            || echo "  ! [fold${FOLD}] eval failed for $CONTRAST (continuing)"
    fi
    echo "[$(date '+%H:%M:%S')]   [fold${FOLD}] done $CONTRAST"
done

present=()
for c in $CONTRAST_LIST; do [ -f "$FOLD_METRICS/${c}_metrics.csv" ] && present+=("$c"); done
if [ ${#present[@]} -gt 0 ]; then
    $PY "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/summarize_fold.py" \
        "$FOLD_METRICS" "$RUN_ID" "$FOLD" --group-col contrast --groups-word Contrasts --groups "${present[@]}"
    echo "[$(date '+%H:%M:%S')] [fold${FOLD}] complete → $FOLD_METRICS/eval_all.csv"
else
    echo "[$(date '+%H:%M:%S')] [fold${FOLD}] no metrics produced"
fi
