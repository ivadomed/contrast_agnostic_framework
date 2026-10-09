#!/usr/bin/env python3
"""Orientation/overlay gallery: axial slice through the pancreas centroid (+ a sagittal view) with the mask contour, for SOURCE training cases and this companion's test cases
(already LPS). Look at it: aff2axcodes is blind to 180-degree flips. Usage: 02_04_render_check.py <src_raw_Dataset> <test_raw> <out_dir>"""
import sys, random
from pathlib import Path
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
src, tst, out = map(Path, sys.argv[1:4]); out.mkdir(parents=True, exist_ok=True); random.seed(0)
def panel(ax_a, ax_s, img, lab, title):
    I = np.asanyarray(nib.load(img).dataobj).astype(np.float32); M = np.asanyarray(nib.load(lab).dataobj) > 0
    c = [int(round(i.mean())) for i in np.where(M)]
    lo, hi = np.percentile(I, [1, 99.5])
    for ax, im, m in ((ax_a, I[:, :, c[2]], M[:, :, c[2]]), (ax_s, I[c[0], :, :], M[c[0], :, :])):
        ax.imshow(np.rot90(im), cmap="gray", vmin=lo, vmax=hi); ax.contour(np.rot90(m), [0.5], colors="r", linewidths=0.8); ax.axis("off")
    ax_a.set_title(title, fontsize=6)
sets = {"source": [(src / "imagesTr" / p.name.replace(".nii.gz", "_0000.nii.gz"), p) for p in sorted((src / "labelsTr").glob("*.nii.gz"))]}
for d in sorted(tst.glob("labelsTs_*")):
    sets[d.name.replace("labelsTs_", "test_")] = [(tst / d.name.replace("labelsTs", "imagesTs") / p.name.replace(".nii.gz", "_0000.nii.gz"), p) for p in sorted(d.glob("*.nii.gz"))]
for name, items in sets.items():
    sel = random.sample(items, min(len(items), 12 if name == "source" else 24))
    n = len(sel); cols = 6; rows = int(np.ceil(n / 3))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axs = np.atleast_2d(axs)
    for k, (i, l) in enumerate(sel):
        r, c0 = divmod(k, 3); panel(axs[r, c0 * 2], axs[r, c0 * 2 + 1], i, l, l.name.replace(".nii.gz", "")[-14:])
    for ax in axs.ravel(): ax.axis("off")
    fig.tight_layout(); fig.savefig(out / f"gallery_{name}.png", dpi=110); plt.close(fig)
print("done")
