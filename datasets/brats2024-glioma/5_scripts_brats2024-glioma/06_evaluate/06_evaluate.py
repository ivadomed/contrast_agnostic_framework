#!/usr/bin/env python3
"""
Method-agnostic segmentation evaluation: compare a directory of predictions against
ground-truth labels and report per-case, per-label Dice and HD95.

Knows nothing about which model/contrast produced the predictions — it just matches
`<case>.nii.gz` in --pred_dir to the same name in --gt_dir.

Usage:
  python 06_evaluate.py --pred_dir <dir> --gt_dir <dir> --dataset_json <dataset.json> \
                        --out_csv <metrics.csv> [--name <label>]

Metrics:
  Dice  — 2|P∩G| / (|P|+|G|).  1.0 when both empty; 0.0 when only one is empty.
  HD95  — 95th-percentile symmetric Hausdorff distance (mm, via GT voxel spacing).
          NaN when either mask is empty for that label (distance undefined).

Foreground labels and names are read from dataset.json ("labels").
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from monai.metrics import compute_hausdorff_distance


def dice_score(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    p, g = pred_bin.sum(), gt_bin.sum()
    if p == 0 and g == 0:
        return float("nan")          # label absent in both → not informative
    if p == 0 or g == 0:
        return 0.0                   # one empty, the other not → worst overlap
    inter = np.logical_and(pred_bin, gt_bin).sum()
    return float(2.0 * inter / (p + g))


def hd95(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing: tuple) -> float:
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float("nan")          # distance undefined if either is empty
    p = torch.from_numpy(pred_bin[None, None].astype(np.uint8))
    g = torch.from_numpy(gt_bin[None, None].astype(np.uint8))
    d = compute_hausdorff_distance(p, g, include_background=True,
                                   percentile=95, spacing=list(spacing))
    v = float(d.item())
    return v if np.isfinite(v) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--dataset_json", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--name", default="", help="optional tag stored in a 'group' column")
    args = ap.parse_args()

    labels = json.loads(Path(args.dataset_json).read_text())["labels"]
    # foreground only: {name: int}, drop background (0) and skip region-lists
    fg = {name: int(v) for name, v in labels.items()
          if isinstance(v, int) and int(v) != 0}

    pred_dir, gt_dir = Path(args.pred_dir), Path(args.gt_dir)
    preds = sorted(pred_dir.glob("*.nii.gz"))
    if not preds:
        raise SystemExit(f"No predictions (*.nii.gz) in {pred_dir}")

    rows, missing_gt = [], []
    for pf in preds:
        case = pf.name[:-7]                       # strip .nii.gz
        gf = gt_dir / pf.name
        if not gf.exists():
            missing_gt.append(case)
            continue
        gt_img = sitk.ReadImage(str(gf))
        spacing = gt_img.GetSpacing()[::-1]        # sitk (x,y,z) → numpy (z,y,x)
        gt = sitk.GetArrayFromImage(gt_img).astype(np.int16)
        pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pf))).astype(np.int16)
        if pred.shape != gt.shape:
            raise SystemExit(f"Shape mismatch {case}: pred {pred.shape} vs gt {gt.shape}")
        for name, lab in fg.items():
            pb, gb = (pred == lab), (gt == lab)
            rows.append({"group": args.name, "case": case, "label": name,
                         "dice": dice_score(pb, gb), "hd95": hd95(pb, gb, spacing)})

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "case", "label", "dice", "hd95"])
        w.writeheader(); w.writerows(rows)

    # console summary (nan-aware mean over cases where the label is present)
    print(f"[{args.name or pred_dir.name}] {len(preds)} cases, labels: {list(fg)}")
    if missing_gt:
        print(f"  WARNING: {len(missing_gt)} preds had no GT: {missing_gt[:5]}"
              f"{'…' if len(missing_gt) > 5 else ''}")
    for name in fg:
        dvals = np.array([r["dice"] for r in rows if r["label"] == name], float)
        hvals = np.array([r["hd95"] for r in rows if r["label"] == name], float)
        n = int(np.isfinite(dvals).sum())
        print(f"  {name:6s} (n={n:3d}): "
              f"Dice {np.nanmean(dvals):.4f}±{np.nanstd(dvals):.4f}  "
              f"HD95 {np.nanmean(hvals):6.2f}±{np.nanstd(hvals):5.2f} mm")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
