#!/usr/bin/env python3
"""
open-ms lesion-wise + size-stratified comparison: OUR method (auglabAug_v26) vs the
SynthSeg-EM contender. This is the metric that should expose the texture/spatial-structure
advantage — SynthSeg is texture-blind, so it should lose most on (a) cross-contrast
(FLAIR→T2W→T1W, lesions progressively more iso-intense) and (b) SMALL lesions, and in the
detection rate (catastrophic misses), even where bulk voxel-Dice looks close.

For each (method, test-contrast) it reports, pooled over the 4 folds × 8 held-out test
patients:
  * voxel Dice (per patient, mean)
  * lesion-wise detection: a GT connected-component is "detected" if ≥ MIN_OVERLAP of its
    voxels are predicted positive. Reports sensitivity overall and stratified by lesion
    size (small / medium / large by voxel count), plus false-positive lesion count.

Read-only over predictions in 8_results_open-ms/01_predictions/... and GT in
2_nnUNet_open-ms/raw/Dataset070_OpenMS_FLAIR/labelsTs_<contrast>/.

Usage:
  python 06_02_lesionwise_analysis.py <OURS_RUN_ID> <SYNTHSEG_RUN_ID> [--out CSV]
  (run IDs are the auglab/<RUN_ID> dir names; both live under .../flair/auglab/)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

DATASET_ROOT = Path(__file__).resolve().parents[2]            # datasets/open-ms
PRED_BASE = DATASET_ROOT / "8_results_open-ms/01_predictions/open_ms_model/flair/auglab"
GT_BASE = DATASET_ROOT / "2_nnUNet_open-ms/raw/Dataset070_OpenMS_FLAIR"
CONTRASTS = ["flair", "t2w", "t1w"]
MIN_OVERLAP = 0.10          # GT lesion detected if >=10% of its voxels predicted positive
SIZE_BINS = [(0, 50, "small(<50)"), (50, 500, "med(50-500)"), (500, 10**9, "large(>=500)")]


def dice(p, g):
    a, b = p.sum(), g.sum()
    if a == 0 and b == 0:
        return np.nan
    if a == 0 or b == 0:
        return 0.0
    return 2.0 * np.logical_and(p, g).sum() / (a + b)


def lesionwise(pred, gt):
    """Per-GT-lesion detection + size; plus #false-positive predicted components."""
    gt_cc, n_gt = ndimage.label(gt)
    out = []                                   # (size, detected_bool)
    for i in range(1, n_gt + 1):
        comp = gt_cc == i
        sz = int(comp.sum())
        det = (np.logical_and(comp, pred).sum() / sz) >= MIN_OVERLAP if sz else False
        out.append((sz, bool(det)))
    pred_cc, n_pred = ndimage.label(pred)
    fp = sum(1 for j in range(1, n_pred + 1)
             if np.logical_and(pred_cc == j, gt).sum() == 0)
    return out, fp


def analyze_run(run_id):
    res = {}                                   # contrast -> dict of metrics
    for contrast in CONTRASTS:
        gt_dir = GT_BASE / f"labelsTs_{contrast}"
        dices, lesions, fps = [], [], 0
        for fold in range(4):
            pdir = PRED_BASE / run_id / f"fold{fold}" / contrast
            if not pdir.exists():
                continue
            for pf in sorted(pdir.glob("*.nii.gz")):
                gf = gt_dir / pf.name
                if not gf.exists():
                    continue
                p = (sitk.GetArrayFromImage(sitk.ReadImage(str(pf))) > 0)
                g = (sitk.GetArrayFromImage(sitk.ReadImage(str(gf))) > 0)
                if p.shape != g.shape:
                    continue
                d = dice(p, g)
                if d == d:
                    dices.append(d)
                les, fp = lesionwise(p, g)
                lesions.extend(les); fps += fp
        res[contrast] = {"dice": dices, "lesions": lesions, "fp": fps,
                         "n_pred_vols": sum(1 for fold in range(4)
                                            for _ in (PRED_BASE / run_id / f"fold{fold}" / contrast).glob("*.nii.gz")
                                            if (PRED_BASE / run_id / f"fold{fold}" / contrast).exists())}
    return res


def sens_by_size(lesions):
    out = {}
    for lo, hi, name in SIZE_BINS:
        sub = [det for (sz, det) in lesions if lo <= sz < hi]
        out[name] = (np.mean(sub) if sub else np.nan, len(sub))
    overall = [det for (_, det) in lesions]
    out["all"] = (np.mean(overall) if overall else np.nan, len(overall))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ours_run_id")
    ap.add_argument("synthseg_run_id")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = {"ours(v26+auglab)": args.ours_run_id, "synthseg_EM": args.synthseg_run_id}
    data = {name: analyze_run(rid) for name, rid in runs.items()}

    rows = []
    print(f"\n{'='*78}\nopen-ms lesion-wise comparison (pooled over 4 folds × held-out test patients)")
    print(f"detection = >={int(MIN_OVERLAP*100)}% of a GT lesion's voxels predicted positive\n{'='*78}")
    for contrast in CONTRASTS:
        print(f"\n### test contrast: {contrast.upper()}")
        print(f"{'method':18s} {'vox Dice':>9s} {'det:all':>8s} {'small':>7s} {'med':>7s} {'large':>7s} {'FPles':>6s}")
        for name in runs:
            r = data[name][contrast]
            d = np.nanmean(r["dice"]) * 100 if r["dice"] else float("nan")
            ss = sens_by_size(r["lesions"])
            print(f"{name:18s} {d:9.1f} "
                  f"{ss['all'][0]*100:7.1f}% {ss['small(<50)'][0]*100:6.1f}% "
                  f"{ss['med(50-500)'][0]*100:6.1f}% {ss['large(>=500)'][0]*100:6.1f}% {r['fp']:6d}")
            rows.append({"contrast": contrast, "method": name, "vox_dice": round(d, 2),
                         "det_all": round(ss['all'][0]*100, 1), "n_lesions": ss['all'][1],
                         "det_small": round(ss['small(<50)'][0]*100, 1),
                         "det_med": round(ss['med(50-500)'][0]*100, 1),
                         "det_large": round(ss['large(>=500)'][0]*100, 1), "fp_lesions": r['fp']})

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
