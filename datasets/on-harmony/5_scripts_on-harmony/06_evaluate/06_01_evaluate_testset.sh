#!/usr/bin/env bash

source "$(dirname "$0")/../00_utils/env.sh"
# Evaluate a trained method on the test set.
#
# Usage
#   bash 04_evaluate_testset.sh <RUN_ID>
#   e.g.: bash 04_evaluate_testset.sh baseline_20260529_233632
#         bash 04_evaluate_testset.sh v26_6_gpuaug_20260531_110150
#
# Steps
#   1. Assemble test images + remap/copy GT into per-contrast directories
#   2. Per fold: nnUNetv2_predict → predictions_1mm/fold_k/
#   3. Ensemble 4 folds: nnUNetv2_ensemble → predictions_1mm/ensemble/
#   4. Resample predictions to native space (SimpleITK NN, no ANTs CLI needed)
#   5. nnUNetv2_evaluate_folder on native-space predictions vs native-space GT
set -euo pipefail
cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <RUN_ID>"
    exit 1
fi

RUN_ID="$1"
# Extract method prefix: strip _YYYYMMDD_HHMMSS suffix and any extra qualifiers
# e.g. "v26_6_gpuaug_20260531_110150" → "v26_6_gpuaug" → strip to "v26_6"
_RAW="${RUN_ID%%_2026*}"   # remove timestamp suffix

PY=".venv/bin/python"
BIDS="data/ON-Harmony"
MASKS_REG="$BIDS/derivatives/synthseg_registered"
TEST_CASES="data/splits/test_cases.json"
RESULTS_DIR="${nnUNet_results}/${RUN_ID}"
EVAL_DIR="eval/onharmony/${RUN_ID}"

export nnUNet_raw="${nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results}/${RUN_ID}"
export NNUNET_PROJECT_ROOT="$(pwd)"
export PYTHONPATH="$(pwd)/src/nnunet:$(pwd)/SynthSeg:${PYTHONPATH:-}"

[ -f "$TEST_CASES" ] || { echo "ERROR: $TEST_CASES not found"; exit 1; }
[ -d "$RESULTS_DIR" ] || { echo "ERROR: $RESULTS_DIR not found"; exit 1; }

# ── Determine trainer from RUN_ID prefix (glob-style matching) ────────────────
case "$_RAW" in
    baseline*)   TRAINER="nnUNetTrainerOnHarmonyBaseline" ;;
    v26_6*)      TRAINER="nnUNetTrainerOnHarmonyV26_6" ;;
    synthseg_a*) TRAINER="nnUNetTrainerOnHarmonySynthSegA" ;;
    synthseg_b*) TRAINER="nnUNetTrainerOnHarmonySynthSegB" ;;
    *)           echo "ERROR: unknown method prefix '$_RAW' in RUN_ID='$RUN_ID'"; exit 1 ;;
esac

echo "[$(date '+%H:%M:%S')] Evaluating RUN_ID=${RUN_ID}  trainer=${TRAINER}"
mkdir -p "$EVAL_DIR"

export BIDS EVAL_DIR
# ── Step 1: Assemble test images and GT ───────────────────────────────────────
$PY - <<'PYEOF'
import json, shutil, sys
import numpy as np
import nibabel as nib
from pathlib import Path
import os

bids       = Path(os.environ.get("BIDS", "data/ON-Harmony"))
masks_reg  = bids / "derivatives" / "synthseg_registered"
eval_dir   = Path(os.environ.get("EVAL_DIR", "eval/onharmony/run"))
test_cases = json.loads(Path("data/splits/test_cases.json").read_text())

# FreeSurfer → 7-class remap (matches 01_convert_dataset.py)
_MAX = 60
_LUT = np.zeros(_MAX + 2, dtype=np.uint8)
for fs_id, cls in {0:0,2:2,3:1,4:3,5:3,7:6,8:6,10:4,11:4,12:4,13:4,14:3,15:3,
                   16:5,17:4,18:4,26:4,28:4,41:2,42:1,43:3,44:3,46:6,47:6,
                   49:4,50:4,51:4,52:4,53:4,54:4,58:4,60:4}.items():
    _LUT[min(fs_id, _MAX + 1)] = cls

def remap_fs_labels(nib_img):
    arr = np.asarray(nib_img.dataobj).astype(np.int32)
    remapped = _LUT[np.clip(arr, 0, _MAX + 1)].astype(np.uint8)
    return nib.Nifti1Image(remapped, nib_img.affine, nib_img.header)

# GRE: OXF1PRI uses coil-specific naming; glob to find echo-1 mag
def find_gre(session_dir):
    swi = session_dir / "swi"
    if not swi.exists():
        return None
    # Prefer simple echo-1 mag, then coil-prefixed
    candidates = sorted(swi.glob("*echo-1*part-mag*GRE.nii.gz"))
    return candidates[0] if candidates else None

CONTRASTS = {
    "T1w":          lambda sub, ses, bdir: bdir / sub / ses / "anat" / f"{sub}_{ses}_T1w.nii.gz",
    "T2w":          lambda sub, ses, bdir: bdir / sub / ses / "anat" / f"{sub}_{ses}_T2w.nii.gz",
    "bold":         lambda sub, ses, bdir: bdir / sub / ses / "func" / f"{sub}_{ses}_task-rest_bold.nii.gz",
    "dwi_ap":       lambda sub, ses, bdir: bdir / sub / ses / "dwi"  / f"{sub}_{ses}_dir-AP_dwi.nii.gz",
    "epi_ap":       lambda sub, ses, bdir: bdir / sub / ses / "fmap" / f"{sub}_{ses}_dir-AP_epi.nii.gz",
    "gre_echo1_mag":lambda sub, ses, bdir: find_gre(bdir / sub / ses),
}

found, missing = 0, 0
for contrast, img_fn in CONTRASTS.items():
    img_dir = eval_dir / contrast / "images_native"
    gt_dir  = eval_dir / contrast / "gt_native"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    for tc in test_cases:
        sub, ses = tc["subject"], tc["session"]
        src = img_fn(sub, ses, bids)
        if src is None or not src.exists():
            missing += 1
            continue

        case_id = f"{sub}_{ses}_{contrast}"
        shutil.copy2(src, img_dir / f"{case_id}_0000.nii.gz")
        found += 1

        # GT in native space
        if contrast == "T1w":
            mask_src = bids / "derivatives" / "synthseg_masks" / sub / ses / "anat" / f"{sub}_{ses}_T1w_synthseg.nii.gz"
            if mask_src.exists():
                # REMAP FreeSurfer labels → 7-class (critical: synthseg_masks has FS IDs)
                img7 = remap_fs_labels(nib.load(str(mask_src)))
                nib.save(img7, str(gt_dir / f"{case_id}.nii.gz"))
        else:
            reg_seg = masks_reg / contrast / f"{sub}_{ses}_{contrast}_seg7.nii.gz"
            if reg_seg.exists():
                shutil.copy2(reg_seg, gt_dir / f"{case_id}.nii.gz")

print(f"Step 1 complete: {found} images found, {missing} missing.")
PYEOF

export BIDS EVAL_DIR

# ── Steps 2-5: Per contrast ───────────────────────────────────────────────────
for CONTRAST in T1w T2w bold dwi_ap epi_ap gre_echo1_mag; do
    IMG_DIR="$EVAL_DIR/$CONTRAST/images_native"
    GT_DIR="$EVAL_DIR/$CONTRAST/gt_native"

    [ -d "$IMG_DIR" ] && N=$(ls "$IMG_DIR" | wc -l) || N=0
    [ "$N" -eq 0 ] && { echo "  Skipping $CONTRAST (no images)"; continue; }

    # Check if GT is available (skip evaluation for contrasts without GT yet)
    [ -d "$GT_DIR" ] && NGT=$(ls "$GT_DIR" 2>/dev/null | wc -l) || NGT=0
    if [ "$NGT" -eq 0 ]; then
        echo "  Warning: $CONTRAST — no GT available (run 01b_register_test_gt.sh). Skipping metrics."
        GT_AVAILABLE=0
    else
        GT_AVAILABLE=1
    fi

    echo "[$(date '+%H:%M:%S')] $CONTRAST: $N images, $NGT GT masks"

    # ── Step 2: Predict per fold (4 folds in parallel on 4 GPUs) ───────────
    # set_slot uses sudo which strips env vars → wrap in bash -c with exports
    declare -A PIDS_FOLD
    _NNUNET_RAW="${nnUNet_raw}"
    _NNUNET_PRE="${nnUNet_preprocessed}"
    _NNUNET_RES="${nnUNet_results}/${RUN_ID}"
    _PYTHONPATH="$(pwd)/src/nnunet:$(pwd)/SynthSeg"
    for FOLD in 0 1 2 3; do
        PRED_DIR="$EVAL_DIR/$CONTRAST/predictions_1mm/fold_${FOLD}"
        mkdir -p "$PRED_DIR"
        set_slot ${FOLD} bash -c "
            export nnUNet_raw='${_NNUNET_RAW}'
            export nnUNet_preprocessed='${_NNUNET_PRE}'
            export nnUNet_results='${_NNUNET_RES}'
            export NNUNET_PROJECT_ROOT='$(pwd)'
            export PYTHONPATH='${_PYTHONPATH}:\${PYTHONPATH:-}'
            cd '$(pwd)'
            .venv/bin/nnUNetv2_predict \
                -i '${IMG_DIR}' \
                -o '${PRED_DIR}' \
                -d 030 -c 3d_fullres \
                -f ${FOLD} \
                -tr '${TRAINER}' \
                -p nnUNetPlans \
                --save_probabilities
        " > /tmp/predict_${RUN_ID}_${CONTRAST}_fold${FOLD}.log 2>&1 &
        PIDS_FOLD[$FOLD]=$!
    done
    for FOLD in 0 1 2 3; do wait "${PIDS_FOLD[$FOLD]}"; done
    echo "[$(date '+%H:%M:%S')]   Predictions done: $CONTRAST"

    # ── Step 3: Ensemble folds ──────────────────────────────────────────────
    ENS_DIR="$EVAL_DIR/$CONTRAST/predictions_1mm/ensemble"
    mkdir -p "$ENS_DIR"
    .venv/bin/nnUNetv2_ensemble \
        -i "$EVAL_DIR/$CONTRAST/predictions_1mm/fold_0" \
           "$EVAL_DIR/$CONTRAST/predictions_1mm/fold_1" \
           "$EVAL_DIR/$CONTRAST/predictions_1mm/fold_2" \
           "$EVAL_DIR/$CONTRAST/predictions_1mm/fold_3" \
        -o "$ENS_DIR" \
        -np 8 \
        > /tmp/ensemble_${RUN_ID}_${CONTRAST}.log 2>&1
    echo "[$(date '+%H:%M:%S')]   Ensemble done: $CONTRAST"

    # ── Step 4: Resample predictions to native space (SimpleITK NN) ────────
    NATIVE_PRED_DIR="$EVAL_DIR/$CONTRAST/predictions_native"
    mkdir -p "$NATIVE_PRED_DIR"
    $PY - <<PYEOF2
import SimpleITK as sitk
from pathlib import Path

ens_dir    = Path("$ENS_DIR")
native_dir = Path("$NATIVE_PRED_DIR")
img_dir    = Path("$IMG_DIR")

for pred_1mm in sorted(ens_dir.glob("*.nii.gz")):
    case_id = pred_1mm.stem.replace(".nii", "")
    ref_path = img_dir / f"{case_id}_0000.nii.gz"
    if not ref_path.exists():
        print(f"  No native ref for {case_id}, skipping")
        continue
    pred = sitk.ReadImage(str(pred_1mm), sitk.sitkUInt8)
    ref  = sitk.ReadImage(str(ref_path))
    resampled = sitk.Resample(
        pred, ref, sitk.Transform(),
        sitk.sitkNearestNeighbor, 0, pred.GetPixelID()
    )
    sitk.WriteImage(resampled, str(native_dir / pred_1mm.name))
print("Native-space resampling done: $CONTRAST")
PYEOF2

    # ── Step 5: Evaluate metrics in native space (only if GT available) ─────
    if [ "$GT_AVAILABLE" -eq 1 ]; then
        PLANS_FILE="$(find ${nnUNet_preprocessed}/Dataset030_OnHarmonyT1w -name 'nnUNetPlans.json' | head -1)"
        .venv/bin/nnUNetv2_evaluate_folder \
            "$GT_DIR" \
            "$NATIVE_PRED_DIR" \
            -djfile "${nnUNet_raw}/Dataset030_OnHarmonyT1w/dataset.json" \
            -pfile "$PLANS_FILE" \
            > "$EVAL_DIR/$CONTRAST/metrics.json" 2>/tmp/eval_${RUN_ID}_${CONTRAST}.log
        echo "[$(date '+%H:%M:%S')]   Metrics computed: $CONTRAST"
    else
        echo "[$(date '+%H:%M:%S')]   Metrics skipped (no GT): $CONTRAST"
    fi
done

echo ""
echo "[$(date '+%H:%M:%S')] Evaluation complete → $EVAL_DIR"
echo "Run 05_aggregate_results.py to summarise."
