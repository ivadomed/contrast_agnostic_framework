#!/usr/bin/env python3
"""
QA visual for the translation experiment: render ORIGINAL vs TRANSLATED image+GT
overlays side by side so the 50%-left shift (and identical GT shift) can be eyeballed
before committing the predict sweep. Renders an axial slice (the L-R shift is in-plane).

Saves a PNG (no on-screen display) to:
  8_results_chaos/02_metrics/chaos_model/exp_translation_<NNN>_qa/<ds>_<mod>_translation.png

Usage:
  python 03_03_visualize_translation.py [--ds Dataset060_CHAOS_MR_T1in] [--mod t1in]
                                        [--frac 0.5] [--n 3]
"""
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET_ROOT = Path(__file__).resolve().parents[2]
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_chaos" / "raw"
OUT_ROOT     = DATASET_ROOT / "8_results_chaos" / "02_metrics" / "chaos_model"


def load(p):
    return np.asanyarray(nib.load(str(p)).dataobj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="Dataset060_CHAOS_MR_T1in")
    ap.add_argument("--mod", default="t1in")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=3, help="number of cases to show")
    args = ap.parse_args()
    tag = f"translation_{round(args.frac*100):03d}"

    base = NNUNET_RAW / args.ds
    img_o, lab_o = base / f"imagesTs_{args.mod}", base / f"labelsTs_{args.mod}"
    img_t, lab_t = base / f"imagesTs_{args.mod}_{tag}", base / f"labelsTs_{args.mod}_{tag}"
    if not img_t.is_dir():
        raise SystemExit(f"translated dir missing: {img_t} — run 03_02 first.")

    cases = sorted(p.name for p in lab_o.glob("*.nii.gz"))[:args.n]
    fig, ax = plt.subplots(len(cases), 2, figsize=(8, 4*len(cases)))
    if len(cases) == 1:
        ax = ax[None, :]
    for r, case in enumerate(cases):
        stem = case[:-7]
        io, go = load(img_o/f"{stem}_0000.nii.gz"), load(lab_o/case)
        it, gt = load(img_t/f"{stem}_0000.nii.gz"), load(lab_t/case)
        z = int(np.argmax((go > 0).sum(axis=(0, 1))))   # slice with most GT (axis2 = S)
        for c, (I, G, ttl) in enumerate([(io, go, "original"), (it, gt, f"translated ({tag})")]):
            a = ax[r, c]; a.axis("off"); a.set_title(f"{stem} — {ttl}", fontsize=9)
            a.imshow(I[:, :, z].T, cmap="gray", origin="lower")
            if (G[:, :, z] > 0).any():
                a.contour(G[:, :, z].T, colors="lime", linewidths=1.0)
    fig.suptitle(f"{args.ds} / {args.mod}: original vs 50%-left translated (image+GT shifted together)")
    out_dir = OUT_ROOT / f"exp_{tag}_qa"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.ds}_{args.mod}_{tag}.png"
    plt.tight_layout(); plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
