#!/usr/bin/env python3
"""
TRUSTED CHIMERA evaluator (context-injection experiment).

Scores kidney segmentation STRICTLY on the US region pasted into the CT (see
05_predict/05_20_build_chimera_inputs.py). For each chimera case:

  * prediction  — chaos labels; kidney = union {2,3} (right+left), as elsewhere.
  * GT          — the placed US kidney mask (labelsTs_chimera) — only the inserted
                  side(s) are present, so one-sided patients are scored on their one
                  kidney automatically.
  * ROI         — roiTs_chimera, the region actually overwritten by US (1=R, 2=L).
                  We intersect the prediction with the ROI so the metric measures ONLY
                  the US part: prediction firing on the surrounding *CT* (e.g. the
                  other, non-inserted kidney) is excluded, not counted as FP.

Reported per case:
  kidney            Dice/HD95 of (pred∩ROI) vs placed-US-GT
  kidney_centerprior Dice of a matched-size ball at the ROI centroid vs GT — the
                  CONTROL. Chimera "success" only means something if it beats this
                  positional prior (the US kidney is ~centered in its fan/ROI).

Usage:
  python 06_00_evaluate_trusted_chimera.py --pred_dir D --gt_dir D --roi_dir D
         --out_csv C [--name chimera] [--workers N]
"""
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "00_commun_scripts" / "00_00_utils"))
import eval_metrics as em  # noqa: E402

KIDNEY_PRED_IDS = (2, 3)   # chaos right_kidney + left_kidney → merged kidney


def _ball(shape, center, n_vox):
    r = (3 * max(1, n_vox) / (4 * np.pi)) ** (1 / 3)
    zz, yy, xx = np.indices(shape)
    return ((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2) <= r ** 2


def _process(args):
    pf, gf, rf, name = args
    gt_img = sitk.ReadImage(gf)
    spacing = gt_img.GetSpacing()[::-1]
    gt = sitk.GetArrayFromImage(gt_img)
    pred = sitk.GetArrayFromImage(sitk.ReadImage(pf))
    roi = sitk.GetArrayFromImage(sitk.ReadImage(rf)) > 0
    case = Path(pf).name[:-7]

    # Crop everything to the ROI bounding box (+2 vox margin). All metrics are
    # restricted to the ROI anyway, so this is EXACT — and it drops the working arrays
    # from the full CT grid (~244 M voxels) to the fan footprint, keeping memory/time
    # low (makes a single-fold eval runnable even on the login node).
    if roi.any():
        idx = np.argwhere(roi)
        lo = np.maximum(idx.min(0) - 2, 0)
        hi = np.minimum(idx.max(0) + 3, roi.shape)
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        pred, gt, roi = pred[sl], gt[sl], roi[sl]

    gk = gt > 0
    pk = np.isin(pred, KIDNEY_PRED_IDS) & roi          # strictly the US region
    rows = [{"group": name, "case": case, "label": "kidney",
             "dice": em.dice_score(pk, gk), "hd95": em.hd95(pk, gk, spacing)}]
    # center-prior control: matched-size ball at the ROI centroid, clipped to ROI
    if roi.any() and gk.any():
        cen = np.argwhere(roi).mean(0)
        blob = _ball(roi.shape, cen, int(gk.sum())) & roi
        rows.append({"group": name, "case": case, "label": "kidney_centerprior",
                     "dice": em.dice_score(blob, gk), "hd95": float("nan")})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--roi_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--name", default="chimera")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    pred_dir, gt_dir, roi_dir = Path(args.pred_dir), Path(args.gt_dir), Path(args.roi_dir)
    preds = sorted(pred_dir.glob("*.nii.gz"))
    if not preds:
        raise SystemExit(f"No predictions in {pred_dir}")
    work = [(str(p), str(gt_dir / p.name), str(roi_dir / p.name), args.name)
            for p in preds if (gt_dir / p.name).exists() and (roi_dir / p.name).exists()]

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(_process, w) for w in work]):
            rows.extend(fut.result())

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "case", "label", "dice", "hd95"])
        w.writeheader(); w.writerows(rows)

    for lab in ("kidney", "kidney_centerprior"):
        d = np.array([r["dice"] for r in rows if r["label"] == lab], float)
        if len(d):
            print(f"  {lab:18} (n={np.isfinite(d).sum():3d}): Dice {np.nanmean(d)*100:5.1f} ± {np.nanstd(d)*100:4.1f}")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
