#!/usr/bin/env python3
"""
AMOS-specific evaluator for chaos-trained model predictions.

IMPORTANT — LABEL REMAP:
  Predictions come from chaos-trained models (chaos label space):
    0=background, 1=liver, 2=right_kidney, 3=left_kidney, 4=spleen
  AMOS GT uses a different label space (15 organs):
    0=bg, 1=spleen, 2=right_kidney, 3=left_kidney, 4=gall_bladder, 5=esophagus,
    6=liver, 7=stomach, 8=aorta, 9=postcava, 10=pancreas, 11=right_adrenal_gland,
    12=left_adrenal_gland, 13=duodenum, 14=bladder, 15=prostate_uterus

  Direct comparison of label IDs would be WRONG (e.g. chaos pred=1 is liver but
  amos gt=1 is spleen). This script maps each scoreable organ name to its correct
  label ID in EACH space, never conflating the two.

Scoreable organs (chaos can predict these 4; AMOS GT has all 4):
  organ          chaos_id  amos_id
  ─────────────  ────────  ───────
  liver             1         6
  right_kidney      2         2   ← same index, but coincidence — still mapped by name
  left_kidney       3         3   ← same
  spleen            4         1

Usage:
  python 06_00_evaluate_amos.py \\
      --pred_dir <dir>   \\   chaos-label predictions (*.nii.gz)
      --gt_dir   <dir>   \\   AMOS GT labels (*.nii.gz, same filenames)
      --out_csv  <path>  \\   output CSV (group, case, label, dice, hd95)
      [--name <tag>]     \\   stored in 'group' column (e.g. "ct" or "mri")
      [--workers N]
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from monai.metrics import compute_hausdorff_distance

# ── label map: organ name → (chaos_pred_id, amos_gt_id) ─────────────────────
# Generated from chaos dataset.json + AMOS dataset.json.
# These are the ONLY organs chaos can predict that AMOS also annotates.
ORGAN_MAP = {
    "liver":        (1, 6),
    "right_kidney": (2, 2),
    "left_kidney":  (3, 3),
    "spleen":       (4, 1),
}


def dice_score(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    p, g = pred_bin.sum(), gt_bin.sum()
    if p == 0 and g == 0:
        return float("nan")
    if p == 0 or g == 0:
        return 0.0
    inter = np.logical_and(pred_bin, gt_bin).sum()
    return float(2.0 * inter / (p + g))


def hd95(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing: tuple) -> float:
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float("nan")
    p = torch.from_numpy(pred_bin[None, None].astype(np.uint8))
    g = torch.from_numpy(gt_bin[None, None].astype(np.uint8))
    d = compute_hausdorff_distance(p, g, include_background=True,
                                   percentile=95, spacing=list(spacing))
    v = float(d.item())
    return v if np.isfinite(v) else float("nan")


def _process_case(args: tuple) -> list[dict]:
    pf_str, gf_str, name_tag = args
    pf, gf = Path(pf_str), Path(gf_str)

    gt_img = sitk.ReadImage(str(gf))
    spacing = gt_img.GetSpacing()[::-1]          # sitk (x,y,z) → numpy (z,y,x)
    gt = sitk.GetArrayFromImage(gt_img).astype(np.int16)
    pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pf))).astype(np.int16)

    if pred.shape != gt.shape:
        raise RuntimeError(f"Shape mismatch {pf.name}: pred {pred.shape} vs gt {gt.shape}")

    case = pf.name[:-7]  # strip .nii.gz
    rows = []
    for organ, (chaos_id, amos_id) in ORGAN_MAP.items():
        pb = (pred == chaos_id)
        gb = (gt   == amos_id)
        rows.append({
            "group": name_tag, "case": case, "label": organ,
            "dice": dice_score(pb, gb),
            "hd95": hd95(pb, gb, spacing),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir",  required=True,
                    help="Directory of chaos-model predictions (chaos label IDs)")
    ap.add_argument("--gt_dir",    required=True,
                    help="Directory of AMOS GT labels (AMOS label IDs)")
    ap.add_argument("--out_csv",   required=True)
    ap.add_argument("--name",      default="",
                    help="Tag stored in the 'group' column (e.g. 'ct' or 'mri')")
    ap.add_argument("--workers",   type=int, default=8)
    args = ap.parse_args()

    pred_dir, gt_dir = Path(args.pred_dir), Path(args.gt_dir)
    preds = sorted(pred_dir.glob("*.nii.gz"))
    if not preds:
        raise SystemExit(f"No predictions (*.nii.gz) in {pred_dir}")

    missing_gt = [pf.name for pf in preds if not (gt_dir / pf.name).exists()]
    if missing_gt:
        print(f"  WARNING: {len(missing_gt)} preds have no GT match: {missing_gt[:5]}")

    work = [(str(pf), str(gt_dir / pf.name), args.name)
            for pf in preds if (gt_dir / pf.name).exists()]

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_process_case, w): w[0] for w in work}
        for fut in as_completed(futs):
            rows.extend(fut.result())

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "case", "label", "dice", "hd95"])
        w.writeheader()
        w.writerows(rows)

    # Print summary
    organs = list(ORGAN_MAP)
    print(f"[{args.name or pred_dir.name}] {len(preds)} cases | organs: {organs}")
    print(f"  NOTE: chaos pred-ids remapped to AMOS GT-ids per organ (see ORGAN_MAP)")
    for organ in organs:
        dvals = np.array([r["dice"] for r in rows if r["label"] == organ], float)
        hvals = np.array([r["hd95"] for r in rows if r["label"] == organ], float)
        n = int(np.isfinite(dvals).sum())
        chaos_id, amos_id = ORGAN_MAP[organ]
        print(f"  {organ:15s} (chaos={chaos_id}→amos={amos_id}) "
              f"Dice {np.nanmean(dvals):.4f}±{np.nanstd(dvals):.4f}  "
              f"HD95 {np.nanmean(hvals):6.2f}±{np.nanstd(hvals):.2f} mm  n={n}")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
