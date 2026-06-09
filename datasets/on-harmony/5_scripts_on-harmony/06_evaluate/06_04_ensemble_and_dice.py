#!/usr/bin/env python3
"""
Ensemble 4 folds, resample to native space, compute Dice.
Usage:
  python ensemble_and_dice.py <method_dir> [contrast1 contrast2 ...]
  python ensemble_and_dice.py --summary
"""
import json, sys
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from pathlib import Path

ROOT = Path("/home/ge.polymtl.ca/pahoa/mri_synthesis_project")
N_CLS = 7
CLASS = {1:"CorticalGM", 2:"WM", 3:"CSF", 4:"SubcortGM", 5:"Brainstem", 6:"Cerebellum"}
ALL_CONTRASTS = ["T1w","T2w","bold","dwi_ap","epi_ap","gre_echo1_mag"]
ALL_METHODS = [
    ("baseline_20260529_233632",     "Baseline"),
    ("v26_6_gpuaug_20260531_110150", "V26_6"),
    ("synthseg_a_20260601_154222",   "SynthSeg-A"),
    ("synthseg_b_20260601_234120",   "SynthSeg-B"),
]

def process(method_dir, contrasts):
    eval_base = ROOT / f"eval/onharmony/{method_dir}"
    for contrast in contrasts:
        img_dir    = eval_base / contrast / "images_native"
        gt_dir     = eval_base / contrast / "gt_native"
        ens_dir    = eval_base / contrast / "predictions_1mm" / "ensemble"
        native_dir = eval_base / contrast / "predictions_native"
        ens_dir.mkdir(parents=True, exist_ok=True)
        native_dir.mkdir(parents=True, exist_ok=True)

        fold_dirs = [eval_base / contrast / "predictions_1mm" / f"fold_{f}" for f in range(4)]
        cases = sorted(fold_dirs[0].glob("*.nii.gz")) if fold_dirs[0].exists() else []
        if not cases:
            print(f"  {method_dir} {contrast}: no predictions", flush=True); continue

        for cp in cases:
            out = ens_dir / cp.name
            if out.exists(): continue
            arrays = [np.asarray(nib.load(str(fd/cp.name)).dataobj, dtype=np.uint8)
                      for fd in fold_dirs if (fd/cp.name).exists()]
            stacked = np.stack(arrays, 0)
            counts  = np.stack([(stacked==l).sum(0) for l in range(N_CLS)], 0)
            voted   = counts.argmax(0).astype(np.uint8)
            ref     = nib.load(str(fold_dirs[0]/cp.name))
            nib.save(nib.Nifti1Image(voted, ref.affine, ref.header), str(out))

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

        gt_files = {f.name: f for f in gt_dir.glob("*.nii.gz")}
        if not gt_files:
            print(f"  {method_dir} {contrast}: no GT", flush=True); continue

        dice_lists = {c:[] for c in CLASS}
        for pf in sorted(native_dir.glob("*.nii.gz")):
            if pf.name not in gt_files: continue
            pred = np.asarray(nib.load(str(pf)).dataobj, dtype=np.uint8)
            gt   = np.asarray(nib.load(str(gt_files[pf.name])).dataobj, dtype=np.uint8)
            for c in CLASS:
                pm, gm = pred==c, gt==c
                den = pm.sum()+gm.sum()
                dice_lists[c].append(2*(pm&gm).sum()/den if den>0 else float("nan"))

        means = {c: float(np.nanmean(dice_lists[c])) for c in CLASS}
        m_all = float(np.nanmean(list(means.values())))
        print(f"  {method_dir} {contrast}: {m_all:.3f}", flush=True)
        out_json = eval_base / contrast / "dice.json"
        out_json.write_text(json.dumps({"method":method_dir,"contrast":contrast,"dice":means}, indent=2))


def summary():
    print("\n" + "="*90)
    labels = [lbl for _, lbl in ALL_METHODS]
    print(f"{'':12s} | " + " | ".join(f"{lbl:>10s}" for lbl in labels))
    print("="*90)
    overall = {m: [] for m, _ in ALL_METHODS}
    for contrast in ALL_CONTRASTS:
        row = []
        for m, _ in ALL_METHODS:
            p = ROOT / f"eval/onharmony/{m}/{contrast}/dice.json"
            if p.exists():
                d = json.loads(p.read_text())["dice"]
                v = float(np.nanmean(list(d.values())))
                row.append(v); overall[m].append(v)
            else:
                row.append(float("nan"))
        print(f"{contrast:12s} | " + " | ".join(f"{v:10.3f}" for v in row))
    print("-"*90)
    r = [float(np.nanmean(overall[m])) if overall[m] else float("nan") for m, _ in ALL_METHODS]
    print(f"{'MEAN':12s} | " + " | ".join(f"{v:10.3f}" for v in r))


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summary()
    else:
        method_dir = sys.argv[1]
        contrasts  = sys.argv[2:] if len(sys.argv) > 2 else ALL_CONTRASTS
        process(method_dir, contrasts)
