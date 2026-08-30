#!/usr/bin/env python3
"""
Supplementary figure for the BraTS T2w->T1n per-label breakdown: zoomed
before/after crops of the fill-swap step (noise-fill -> real-fill) on the two
labels that behave oppositely -- SNFH (oedema, where real-fill significantly
WORSENS OOD Dice, -9.8, p=4.5e-05) and ET (enhancing tumour, where the same
step is flat/non-significant, +0.9, p=0.38). Companion to the per-label
tables in sec/X_suppl.tex; case and rung choice documented in
paper/compute_ladder_significance.py's sibling analysis and
datasets/brats2024-glioma/.../ablations/t1n_t2w_transfer_per_label/summary.md.

Usage:
  .venv/bin/python make_snfh_et_zoom_figure.py
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
CASE = "BraTSGLI02108101"

RAW051 = REPO / "datasets/brats2024-glioma/2_nnUNet_brats2024-glioma/raw/Dataset051_BraTS2024GliomaT1n"
GT_DIR = REPO / "datasets/brats2024-glioma/2_nnUNet_brats2024-glioma/raw/Dataset052_BraTS2024GliomaT2w/labelsTr"
P_REAL = REPO / "datasets/brats2024-glioma/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t2w/nnUNet/brats2024-glioma_t2w_v26_6_2_train050_val100_20260620_125217/fold0/t1n"
P_NOISE = REPO / "datasets/brats2024-glioma/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t2w/auglab/brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_20260805_020659/fold0/t1n"

OUT = Path(__file__).resolve().parent / "cvpr_format_latex" / "figures" / "snfh_et_zoom.pdf"

LABEL_COLORS = {1: (0.56, 0.27, 0.68), 2: (0.16, 0.50, 0.73), 3: (0.75, 0.22, 0.17), 4: (0.84, 0.65, 0.06)}
SNFH, ET = 2, 3


def load(p):
    return np.asarray(nib.load(str(p)).dataobj)


t1n_img = load(RAW051 / f"imagesTs_t1n/{CASE}_0000.nii.gz")
gt = load(GT_DIR / f"{CASE}.nii.gz")
pred_real = load(P_REAL / f"{CASE}.nii.gz")
pred_noise = load(P_NOISE / f"{CASE}.nii.gz")

areas = (gt > 0).sum(axis=(0, 1))
z = int(np.argmax(areas))

mask3d = gt > 0
lbl3d, n = ndimage.label(mask3d)
sizes = ndimage.sum(mask3d, lbl3d, range(1, n + 1))
order = np.argsort(sizes)[::-1]
big_id = order[0] + 1  # SNFH-dominant lesion
small_candidates = [i + 1 for i in order[1:] if sizes[i] > 500]
small_id = small_candidates[0] if small_candidates else order[1] + 1  # ET-dominant lesion


def bbox_2d(mask2d, margin=10):
    ys, xs = np.where(mask2d)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin, mask2d.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin, mask2d.shape[1])
    return y0, y1, x0, x1


big_bbox = bbox_2d(lbl3d[:, :, z] == big_id)
small_bbox = bbox_2d(lbl3d[:, :, z] == small_id)


def norm(sl):
    lo, hi = np.percentile(sl, [1, 99])
    return np.clip((sl - lo) / max(hi - lo, 1e-6), 0, 1)


def crop(a, bbox):
    y0, y1, x0, x1 = bbox
    return a[y0:y1, x0:x1]


def dice_label(pred, gt_slice, label_val):
    p = pred == label_val
    g = gt_slice == label_val
    denom = p.sum() + g.sum()
    return 2 * np.logical_and(p, g).sum() / denom if denom > 0 else float("nan")


def panel(ax, pred_slice, gt_slice, img_slice, label_val, bbox, title):
    im = crop(img_slice, bbox); gtl = crop(gt_slice, bbox); pr = crop(pred_slice, bbox)
    ax.imshow(np.rot90(norm(im)), cmap="gray")
    col = LABEL_COLORS[label_val]
    layer = np.zeros((*gtl.shape, 4))
    tp = (pr == label_val) & (gtl == label_val)
    fp = (pr == label_val) & (gtl != label_val)
    layer[tp] = (*col, 0.55)
    layer[fp] = (0.55, 0.55, 0.55, 0.5)
    ax.imshow(np.rot90(layer))
    m = np.rot90(gtl == label_val)
    if m.any():
        ax.contour(m, colors=[col], linewidths=1.5, levels=[0.5])
    d = dice_label(pr, gtl, label_val)
    ax.set_title(f"{title}\nlabel Dice={d:.2f}", fontsize=10)
    ax.axis("off")


fig, axes = plt.subplots(2, 2, figsize=(6.6, 7.0))
panel(axes[0, 0], pred_noise[:, :, z], gt[:, :, z], t1n_img[:, :, z], SNFH, big_bbox, "SNFH -- before (noise-fill)")
panel(axes[0, 1], pred_real[:, :, z], gt[:, :, z], t1n_img[:, :, z], SNFH, big_bbox, "SNFH -- after (real-fill)")
panel(axes[1, 0], pred_noise[:, :, z], gt[:, :, z], t1n_img[:, :, z], ET, small_bbox, "ET -- before (noise-fill)")
panel(axes[1, 1], pred_real[:, :, z], gt[:, :, z], t1n_img[:, :, z], ET, small_bbox, "ET -- after (real-fill)")

fig.suptitle(f"{CASE}, fold 0 -- T2w-trained model evaluated on T1n", fontsize=10.5, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=300)
print("wrote", OUT)

# print the population-level numbers this figure illustrates, for the caption
print(f"SNFH: n=70, Delta Dice (voronoi->real) = -9.8, p=4.5e-05 (significant worsening)")
print(f"ET:   n=70, Delta Dice (voronoi->real) = +0.9, p=0.38 (not significant)")
