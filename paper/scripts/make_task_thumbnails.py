#!/usr/bin/env python3
"""
One example image per ablation task, for the insets of fig:ladder
(make_per_contrast_curves.py places them in each panel's empty space).

Per task: one training-modality volume + its reference labels, reoriented to RAS,
the axial slice with the largest labeled area, cropped to the body, labels drawn
as a translucent overlay (one color for single-target tasks, a qualitative map
for multi-label ones). Writes figures/task_thumbs/<Task>.png.

Usage:  .venv/bin/python paper/scripts/make_task_thumbnails.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
T = REPO / "benchmark" / "02_tasks"
OUT = REPO / "paper" / "cvpr_format_latex" / "figures" / "task_thumbs"

R = "2_nnUNet_{ds}/raw/{did}"
# task -> (image, label, keep-label-ids or None for all > 0)
TASKS = {
    "CHAOS": (T / "abdomen_healthy/chaos" / R.format(ds="chaos", did="Dataset060_CHAOS_MR_T1in") / "imagesTr/MR01_0000.nii.gz",
              T / "abdomen_healthy/chaos" / R.format(ds="chaos", did="Dataset060_CHAOS_MR_T1in") / "labelsTr/MR01.nii.gz", None),
    "ON-Harmony": (None, None, None),   # resolved below (first case of Dataset031)
    "Mandible": (T / "mandible_healthy/toothfairy2/1_BIDS_toothfairy2/maxillofacial-toothfairy2/sub-ToothFairy2F001/anat/sub-ToothFairy2F001_acq-cbct_ct.nii.gz",
                 T / "mandible_healthy/toothfairy2/1_BIDS_toothfairy2/maxillofacial-toothfairy2/derivatives/labels/sub-ToothFairy2F001/anat/sub-ToothFairy2F001_acq-cbct_ct_label-maxillofacial_seg.nii.gz",
                 [1]),
    "Pelvis": (None, None, None),
    "BraTS-GLI": (None, None, None),
    "Open-MS": (None, None, None),
    "Breast": (None, None, None),
}
FIRST = {   # task -> (raw dataset dir); the first imagesTr case is used
    "ON-Harmony": T / "brain_healthy/on-harmony" / R.format(ds="on-harmony", did="Dataset031_OnHarmonyT1w31"),
    "Pelvis": T / "pelvis_healthy/totalseg-pelvic" / R.format(ds="totalseg-pelvic", did="Dataset130_TotalsegPelvic_CT"),
    "BraTS-GLI": T / "brain_tumor/brats2024-glioma" / R.format(ds="brats2024-glioma", did="Dataset053_BraTS2024GliomaT2f"),
    "Open-MS": T / "brain_ms/open-ms" / R.format(ds="open-ms", did="Dataset070_OpenMS_FLAIR"),
    "Breast": T / "breast_cancer/ispy2" / R.format(ds="ispy2", did="Dataset100_ISPY2T1wce"),
}
SEARCH = {"Breast", "BraTS-GLI", "Open-MS"}   # single-target tasks: pick the case whose target is most visible
CT_WINDOW = {"Pelvis": (-160, 240)}           # HU window (soft tissue + bone) instead of percentiles
LABEL_CROP = {"Mandible"}                     # crop around the label rather than the whole field of view
SINGLE = "#f08a24"   # single-target overlay color


def resolve(task):
    img, lab, keep = TASKS[task]
    if img is not None:
        return img, lab, keep
    d = FIRST[task]
    imgs = sorted((d / "imagesTr").glob("*_0000.nii.gz"))
    lab_of = lambda im: d / "labelsTr" / im.name.replace("_0000.nii.gz", ".nii.gz")
    if task not in SEARCH:
        return imgs[0], lab_of(imgs[0]), None
    best, score = imgs[0], -1.0
    for im in imgs[:12]:   # largest target share of its best axial slice
        lab = load_ras(lab_of(im)) > 0
        a = lab.sum(axis=(0, 1))
        z = int(np.argmax(a))
        body = (load_ras(im)[:, :, z] > 0).sum()
        sc = a[z] / max(body, 1)
        if sc > score:
            best, score = im, sc
    return best, lab_of(best), None


def load_ras(p):
    im = nib.as_closest_canonical(nib.load(str(p)))
    return np.asarray(im.dataobj, dtype=np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        img_p, lab_p, keep = resolve(task)
        img, lab = load_ras(img_p), load_ras(lab_p).astype(np.int32)
        if keep is not None:
            lab = np.where(np.isin(lab, keep), lab, 0)
        z = int(np.argmax((lab > 0).sum(axis=(0, 1))))
        sl, ls = img[:, :, z], lab[:, :, z]
        # crop to the body (intensity above the 5th percentile of non-zero voxels)
        nz = sl[sl > 0]
        body = sl > (np.percentile(nz, 20) if nz.size else 0)
        ys, xs = np.where((ls > 0) if task in LABEL_CROP else (body | (ls > 0)))
        pad = 18 if task in LABEL_CROP else 4
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, sl.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, sl.shape[1])
        sl, ls = sl[y0:y1, x0:x1], ls[y0:y1, x0:x1]
        if task in CT_WINDOW:
            lo, hi = CT_WINDOW[task]
        else:
            lo, hi = np.percentile(sl[sl > 0] if (sl > 0).any() else sl, [1, 99.5])
        sl = np.clip((sl - lo) / max(hi - lo, 1e-6), 0, 1)
        # RAS array -> display: rotate so anterior is up, patient's right on the left
        sl, ls = np.rot90(sl), np.rot90(ls)
        labels = [v for v in np.unique(ls) if v > 0]
        fig, ax = plt.subplots(figsize=(1.6, 1.6 * sl.shape[0] / sl.shape[1]), dpi=220)
        ax.imshow(sl, cmap="gray", interpolation="bilinear")
        if len(labels) <= 1:
            ov = np.ma.masked_where(ls == 0, ls)
            ax.imshow(ov, cmap=ListedColormap([SINGLE]), alpha=0.55, interpolation="nearest")
            ax.contour(ls > 0, levels=[0.5], colors=[SINGLE], linewidths=0.6)
        else:
            cmap = plt.get_cmap("tab20", max(len(labels), 2))
            idx = np.zeros_like(ls)
            for i, v in enumerate(labels, start=1):
                idx[ls == v] = i
            ax.imshow(np.ma.masked_where(idx == 0, idx), cmap=cmap, alpha=0.5,
                      interpolation="nearest", vmin=1, vmax=max(len(labels), 2))
        ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)
        out = OUT / f"{task}.png"
        fig.savefig(out, dpi=220)
        plt.close(fig)
        print(f"{task:11s} slice z={z} labels={len(labels)} <- {img_p.name} -> {out.name}")


if __name__ == "__main__":
    main()
