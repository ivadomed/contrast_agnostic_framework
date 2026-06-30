#!/usr/bin/env python3
"""
Shared segmentation-metric core for all datasets' evaluation scripts.

This is the single source of truth for Dice / HD95 and the per-case scoring loop
that was previously copy-pasted into every dataset's `06_00_evaluate*.py`. It is
deliberately *model-agnostic and dataset-agnostic*: it never reads dataset.json,
never knows organ names, and never cares which model produced the predictions. It
only takes a list of label triples and two directories of matching `<case>.nii.gz`.

Label triples
-------------
Each scoreable label is a triple ``(name, pred_id, gt_id)``:
  * `name`    — string stored in the CSV `label` column.
  * `pred_id` — integer label value in the *prediction* volume, OR a list/tuple of
                integers to MERGE (their union is the foreground). e.g. chaos splits
                the kidney into right_kidney(2)+left_kidney(3); scoring against a
                single-kidney GT uses pred_id=[2, 3] (see TRUSTED 06_00_evaluate).
  * `gt_id`   — integer label value in the *ground-truth* volume (or a list to merge).

For same-label-space evaluation (BraTS, CHAOS, SLIVER07) `pred_id == gt_id`. For
cross-dataset evaluation (e.g. AMOS GT scored against chaos-trained predictions)
the two differ — that is the whole reason this is a triple and not a single id.
The dataset-specific scripts build the triples; this module just applies them.

Metrics
-------
  Dice  — 2|P∩G| / (|P|+|G|).  NaN when both empty; 0.0 when only one is empty.
  HD95  — 95th-percentile symmetric Hausdorff distance (mm, via GT voxel spacing).
          NaN when either mask is empty for that label (distance undefined).
"""
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from monai.metrics import compute_hausdorff_distance

CSV_FIELDS = ["group", "case", "label", "dice", "hd95"]


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


def _process_case(args: tuple) -> list:
    """Top-level so it is picklable for ProcessPoolExecutor.

    args = (pred_path, gt_path, triples, name_tag) where triples is a list of
    (label_name, pred_id, gt_id).
    """
    pf_str, gf_str, triples, name_tag = args
    pf, gf = Path(pf_str), Path(gf_str)
    gt_img = sitk.ReadImage(str(gf))
    spacing = gt_img.GetSpacing()[::-1]        # sitk (x,y,z) → numpy (z,y,x)
    gt = sitk.GetArrayFromImage(gt_img).astype(np.int16)
    pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pf))).astype(np.int16)
    if pred.shape != gt.shape:
        raise RuntimeError(f"Shape mismatch {pf.name}: pred {pred.shape} vs gt {gt.shape}")
    case = pf.name[:-7]                         # strip .nii.gz
    rows = []
    for name, pred_id, gt_id in triples:
        # pred_id / gt_id may be a single int or a list/tuple to merge (union).
        pb = np.isin(pred, pred_id) if isinstance(pred_id, (list, tuple)) else (pred == pred_id)
        gb = np.isin(gt, gt_id) if isinstance(gt_id, (list, tuple)) else (gt == gt_id)
        rows.append({"group": name_tag, "case": case, "label": name,
                     "dice": dice_score(pb, gb), "hd95": hd95(pb, gb, spacing)})
    return rows


def run_evaluation(pred_dir, gt_dir, out_csv, name, triples, workers=8,
                   summary_note: str = "") -> list:
    """Score every `<case>.nii.gz` in pred_dir against the same name in gt_dir.

    triples: list of (label_name, pred_id, gt_id). Writes a CSV with columns
    group,case,label,dice,hd95 to out_csv and prints a per-label summary.
    Returns the list of row dicts.
    """
    pred_dir, gt_dir, out = Path(pred_dir), Path(gt_dir), Path(out_csv)
    preds = sorted(pred_dir.glob("*.nii.gz"))
    if not preds:
        raise SystemExit(f"No predictions (*.nii.gz) in {pred_dir}")

    missing_gt = [pf.name[:-7] for pf in preds if not (gt_dir / pf.name).exists()]
    if missing_gt:
        print(f"  WARNING: {len(missing_gt)} preds had no GT: {missing_gt[:5]}"
              f"{'…' if len(missing_gt) > 5 else ''}")

    work = [(str(pf), str(gt_dir / pf.name), triples, name)
            for pf in preds if (gt_dir / pf.name).exists()]

    rows: list = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_process_case, w): w[0] for w in work}
        for fut in as_completed(futs):
            rows.extend(fut.result())

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader(); w.writerows(rows)

    label_names = [t[0] for t in triples]
    print(f"[{name or pred_dir.name}] {len(preds)} cases, labels: {label_names}")
    if summary_note:
        print(f"  {summary_note}")
    for lab in label_names:
        dvals = np.array([r["dice"] for r in rows if r["label"] == lab], float)
        hvals = np.array([r["hd95"] for r in rows if r["label"] == lab], float)
        n = int(np.isfinite(dvals).sum())
        print(f"  {lab:15s} (n={n:3d}): "
              f"Dice {np.nanmean(dvals):.4f}±{np.nanstd(dvals):.4f}  "
              f"HD95 {np.nanmean(hvals):6.2f}±{np.nanstd(hvals):5.2f} mm")
    print(f"  → {out}")
    return rows
