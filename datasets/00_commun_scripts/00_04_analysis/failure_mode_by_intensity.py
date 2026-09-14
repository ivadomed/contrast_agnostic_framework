#!/usr/bin/env python3
"""
Where does a model's false-positive mass land, as a function of local image
intensity? Reads predictions and images already on disk -- no inference.

Motivation (2026-09-01, atlas-liver-hcc vs lld-mmri-hcc): a model trained on
fat-SUPPRESSED CE-T1w (fat dark) is hypothesised to fail on Dixon IN-phase
(fat bright) because bright fat adjacent to the liver is a configuration it
never saw, while Dixon OUT-of-phase partially nulls fat and so stays closer to
the training appearance. That hypothesis makes a specific, falsifiable
prediction about WHERE the errors go: on in-phase the false positives should
pile up in bright (fatty) tissue, on out-of-phase they should not.

Reports, per contrast: predicted/GT volume ratio, and the median normalised
image intensity under false-positive voxels vs under true-positive voxels and
vs the body as a whole. FP intensity >> body median means the model is firing
on bright tissue.

Usage:
  python failure_mode_by_intensity.py --pred-root DIR --image-root DIR \
      --contrast NAME=PRED_SUBDIR:IMG_GLOB_TEMPLATE ... --out DIR [--max-cases 40]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi


def body_mask_3d(vol: np.ndarray) -> np.ndarray:
    pos = vol[vol > 0]
    if pos.size < 100:
        return np.zeros_like(vol, dtype=bool)
    m = vol > max(np.percentile(pos, 30), vol.max() * 0.06)
    return ndi.binary_fill_holes(m)


def analyse_case(img_p: Path, pred_p: Path, gt_p: Path, tumour_label: int):
    try:
        img = nib.load(str(img_p)).get_fdata().astype(np.float32)
        pred = nib.load(str(pred_p)).get_fdata()
        gt = nib.load(str(gt_p)).get_fdata()
    except Exception:
        return None
    if img.shape != pred.shape or img.shape != gt.shape:
        return None

    p = pred == tumour_label
    g = gt > 0
    body = body_mask_3d(img)
    if body.sum() < 1000:
        return None

    vals = img[body]
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        return None
    norm = np.clip((img - lo) / (hi - lo), 0, 1)

    fp = p & ~g
    tp = p & g
    return {
        "gt_vox": int(g.sum()),
        "pred_vox": int(p.sum()),
        "vol_ratio": float(p.sum() / max(g.sum(), 1)),
        "fp_intensity": float(np.median(norm[fp])) if fp.sum() > 20 else np.nan,
        "tp_intensity": float(np.median(norm[tp])) if tp.sum() > 20 else np.nan,
        "body_intensity": float(np.median(norm[body])),
        "fp_frac_bright": float((norm[fp] > 0.6).mean()) if fp.sum() > 20 else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", action="append", required=True,
                    help="NAME=PRED_DIR::IMG_GLOB::GT_DIR")
    ap.add_argument("--tumour-label", type=int, default=2)
    ap.add_argument("--max-cases", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in args.contrast:
        name, _, rest = spec.partition("=")
        pred_dir, img_glob, gt_dir = rest.split("::")
        preds = sorted(glob.glob(f"{pred_dir}/*.nii.gz"))[: args.max_cases]
        recs = []
        for pp in preds:
            stem = Path(pp).name.replace(".nii.gz", "")
            gt = Path(gt_dir) / f"{stem}.nii.gz"
            imgs = glob.glob(img_glob.replace("{case}", stem))
            if not gt.exists() or not imgs:
                continue
            r = analyse_case(Path(imgs[0]), Path(pp), gt, args.tumour_label)
            if r:
                recs.append(r)
        if not recs:
            print(f"[{name}] no usable cases", flush=True)
            continue
        agg = {"contrast": name, "n": len(recs)}
        for k in ["vol_ratio", "fp_intensity", "tp_intensity", "body_intensity", "fp_frac_bright"]:
            vals = np.array([r[k] for r in recs], dtype=float)
            vals = vals[~np.isnan(vals)]
            agg[k] = float(np.median(vals)) if vals.size else float("nan")
        rows.append(agg)
        print(f"[{name}] n={agg['n']} vol_ratio={agg['vol_ratio']:.2f} "
              f"fp_int={agg['fp_intensity']:.3f} body_int={agg['body_intensity']:.3f}", flush=True)

    lines = ["# Failure mode by local intensity", "",
             "`vol_ratio` = predicted / ground-truth lesion volume (>>1 = over-segmentation). "
             "`fp_int` / `tp_int` = median normalised image intensity under false-positive / "
             "true-positive voxels; `body_int` = median over the whole body. "
             "`fp_bright` = fraction of false-positive voxels in bright tissue (normalised >0.6). "
             "FP intensity well above body median means the model is firing on bright tissue.", "",
             "| contrast | n | vol_ratio | fp_int | tp_int | body_int | fp_bright |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['contrast']} | {r['n']} | {r['vol_ratio']:.2f} | {r['fp_intensity']:.3f} | "
                     f"{r['tp_intensity']:.3f} | {r['body_intensity']:.3f} | {r['fp_frac_bright']:.3f} |")

    (out_dir / "failure_mode.md").write_text("\n".join(lines) + "\n")
    (out_dir / "failure_mode.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
