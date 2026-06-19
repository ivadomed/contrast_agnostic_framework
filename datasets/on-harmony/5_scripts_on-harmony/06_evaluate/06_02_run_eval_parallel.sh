#!/usr/bin/env bash
# Evaluate baseline + V26_6 on all contrasts.
# - 4 folds per method run on 4 GPUs simultaneously (run_job --wait, background)
# - Majority vote ensemble (no softmax needed)
# - Dice computed directly (no nnUNetv2_evaluate_folder dependency)
# Usage: bash 06_02_run_eval_parallel.sh
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

PY=".venv/bin/python"
NNRAW="${nnUNet_raw}"
NNPRE="${nnUNet_preprocessed}"
ROOT="${PROJECT_ROOT}"

METHODS=(
    "baseline_20260529_233632:nnUNetTrainerOnHarmonyBaseline"
    "v26_6_gpuaug_20260531_110150:nnUNetTrainerOnHarmonyV26_6"
    "synthseg_a_20260601_154222:nnUNetTrainerOnHarmonySynthSegA"
    "synthseg_b_20260601_234120:nnUNetTrainerOnHarmonySynthSegB"
)
CONTRASTS="T1w T2w bold dwi_ap epi_ap gre_echo1_mag"

for METHOD_TRAINER in "${METHODS[@]}"; do
    METHOD_DIR="${METHOD_TRAINER%%:*}"
    TRAINER="${METHOD_TRAINER##*:}"
    NNRES="${nnUNet_results}/${METHOD_DIR}"
    EVAL="${ROOT}/eval/onharmony/${METHOD_DIR}"

    echo "[$(date '+%H:%M:%S')] === ${METHOD_DIR} ==="

    for CONTRAST in $CONTRASTS; do
        IMG_DIR="${EVAL}/${CONTRAST}/images_native"
        [ -d "$IMG_DIR" ] && N=$(ls "$IMG_DIR" 2>/dev/null | wc -l) || N=0
        [ "$N" -eq 0 ] && { echo "  Skipping $CONTRAST (no images)"; continue; }
        echo "[$(date '+%H:%M:%S')]   $CONTRAST: $N images — running 4-fold prediction..."

        # Skip if predictions already exist for all folds
        ALL_DONE=1
        for F in 0 1 2 3; do
            NC=$(ls "${EVAL}/${CONTRAST}/predictions_1mm/fold_${F}/"*.nii.gz 2>/dev/null | wc -l)
            [ "$NC" -lt "$N" ] && ALL_DONE=0
        done

        if [ "$ALL_DONE" -eq 0 ]; then
            declare -A PIDS
            for FOLD in 0 1 2 3; do
                PRED_DIR="${EVAL}/${CONTRAST}/predictions_1mm/fold_${FOLD}"
                mkdir -p "$PRED_DIR"
                run_job --name "onharmony_evpar_${METHOD_DIR}_${CONTRAST}_fold${FOLD}" \
                    --gpus 1 --slot "${FOLD}" --wait \
                    --log "/tmp/pred_${METHOD_DIR}_${CONTRAST}_f${FOLD}.log" -- \
                    bash -c "
                    export nnUNet_raw='${NNRAW}'
                    export nnUNet_preprocessed='${NNPRE}'
                    export nnUNet_results='${NNRES}'
                    export NNUNET_PROJECT_ROOT='${ROOT}'
                    export PYTHONPATH='${ROOT}/src/nnunet:${ROOT}/SynthSeg'
                    cd '${ROOT}'
                    .venv/bin/nnUNetv2_predict \
                        -i '${IMG_DIR}' \
                        -o '${PRED_DIR}' \
                        -d 030 -c 3d_fullres \
                        -f ${FOLD} -tr '${TRAINER}' -p nnUNetPlans
                " &
                PIDS[$FOLD]=$!
            done
            for F in 0 1 2 3; do wait "${PIDS[$F]}"; done
            echo "[$(date '+%H:%M:%S')]   Predictions done: $CONTRAST"
        else
            echo "[$(date '+%H:%M:%S')]   Predictions already exist: $CONTRAST"
        fi

        # Majority vote + resample + Dice — all in one Python call
        ENS_DIR="${EVAL}/${CONTRAST}/predictions_1mm/ensemble"
        NATIVE_DIR="${EVAL}/${CONTRAST}/predictions_native"
        GT_DIR="${EVAL}/${CONTRAST}/gt_native"
        mkdir -p "$ENS_DIR" "$NATIVE_DIR"

        $PY - << PYEOF
import nibabel as nib, numpy as np, SimpleITK as sitk, json
from pathlib import Path
N_CLS = 7
ens_dir    = Path("${ENS_DIR}")
native_dir = Path("${NATIVE_DIR}")
gt_dir     = Path("${GT_DIR}")
img_dir    = Path("${IMG_DIR}")
fold_dirs  = [Path("${EVAL}/${CONTRAST}/predictions_1mm/fold_\${f}") for f in range(4)]

# Majority vote
cases = sorted(fold_dirs[0].glob("*.nii.gz"))
for cp in cases:
    out = ens_dir / cp.name
    if out.exists(): continue
    arrays = [np.asarray(nib.load(str(fd/cp.name)).dataobj, dtype=np.uint8)
              for fd in fold_dirs if (fd/cp.name).exists()]
    stacked = np.stack(arrays,0)
    counts  = np.stack([(stacked==l).sum(0) for l in range(N_CLS)],0)
    voted   = counts.argmax(0).astype(np.uint8)
    ref     = nib.load(str(fold_dirs[0]/cp.name))
    nib.save(nib.Nifti1Image(voted, ref.affine, ref.header), str(out))

# Resample to native space
for pf in sorted(ens_dir.glob("*.nii.gz")):
    out = native_dir / pf.name
    if out.exists(): continue
    cid = pf.stem.replace(".nii","")
    rp  = img_dir / f"{cid}_0000.nii.gz"
    if not rp.exists(): continue
    p  = sitk.ReadImage(str(pf), sitk.sitkUInt8)
    r  = sitk.ReadImage(str(rp))
    rs = sitk.Resample(p, r, sitk.Transform(), sitk.sitkNearestNeighbor, 0, p.GetPixelID())
    sitk.WriteImage(rs, str(out))

# Dice
CLASS = {1:"CorticalGM",2:"WM",3:"CSF",4:"SubcortGM",5:"Brainstem",6:"Cerebellum"}
gt_files = {f.name: f for f in gt_dir.glob("*.nii.gz")}
dice_lists = {c:[] for c in CLASS}
for pf in sorted(native_dir.glob("*.nii.gz")):
    if pf.name not in gt_files: continue
    pred = np.asarray(nib.load(str(pf)).dataobj,         dtype=np.uint8)
    gt   = np.asarray(nib.load(str(gt_files[pf.name])).dataobj, dtype=np.uint8)
    for c in CLASS:
        pm, gm = pred==c, gt==c
        den = pm.sum()+gm.sum()
        dice_lists[c].append(2*(pm&gm).sum()/den if den>0 else float("nan"))

means = {c: float(np.nanmean(dice_lists[c])) for c in CLASS}
n = len([f for f in native_dir.glob("*.nii.gz") if f.name in gt_files])
print(f"  ${METHOD_DIR} ${CONTRAST}: {n} cases evaluated")
for c, nm in CLASS.items():
    print(f"    {nm:12s}: {means[c]:.3f}")
print(f"    {'Mean':12s}: {np.nanmean(list(means.values())):.3f}")

# Save per-contrast result
out_json = Path("${EVAL}/${CONTRAST}/dice.json")
out_json.write_text(json.dumps({"method":"${METHOD_DIR}","contrast":"${CONTRAST}","dice":means}, indent=2))
PYEOF
    done
done

echo ""
echo "[$(date '+%H:%M:%S')] All done. Aggregating..."

# Print summary table
$PY - << 'PYEOF'
import json, numpy as np
from pathlib import Path
METHODS = [
    "baseline_20260529_233632",
    "v26_6_gpuaug_20260531_110150",
    "synthseg_a_20260601_154222",
    "synthseg_b_20260601_234120",
]
LABELS = {
    "baseline_20260529_233632":     "Baseline",
    "v26_6_gpuaug_20260531_110150": "V26_6",
    "synthseg_a_20260601_154222":   "SynthSeg-A",
    "synthseg_b_20260601_234120":   "SynthSeg-B",
}
CONTRASTS = ["T1w","T2w","bold","dwi_ap","epi_ap","gre_echo1_mag"]

print("\n" + "="*90)
print(f"{'':12s} | {'Baseline':>9s} | {'V26_6':>9s} | {'SynthSeg-A':>10s} | {'SynthSeg-B':>10s}")
print("="*90)
for contrast in CONTRASTS:
    vals = {}
    for m in METHODS:
        p = Path(f"eval/onharmony/{m}/{contrast}/dice.json")
        if not p.exists(): continue
        d = json.loads(p.read_text())["dice"]
        vals[m] = np.nanmean(list(d.values()))
    row = [vals.get(m, float("nan")) for m in METHODS]
    print(f"{contrast:12s} | {row[0]:9.3f} | {row[1]:9.3f} | {row[2]:10.3f} | {row[3]:10.3f}")

# Overall mean
print("-"*90)
overall = []
for m in METHODS:
    all_dice = []
    for contrast in CONTRASTS:
        p = Path(f"eval/onharmony/{m}/{contrast}/dice.json")
        if not p.exists(): continue
        d = json.loads(p.read_text())["dice"]
        all_dice.append(np.nanmean(list(d.values())))
    overall.append(np.nanmean(all_dice) if all_dice else float("nan"))
print(f"{'MEAN':12s} | {overall[0]:9.3f} | {overall[1]:9.3f} | {overall[2]:10.3f} | {overall[3]:10.3f}")
PYEOF
