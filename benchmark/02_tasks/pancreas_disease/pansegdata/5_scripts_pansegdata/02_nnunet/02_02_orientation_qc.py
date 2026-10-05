#!/usr/bin/env python3
"""Signed orientation QC of the CONVERTED nnU-Net data (Dataset150 T1WCE / Dataset151 T2W): axial + coronal + sagittal slices through the
pancreas mask centroid for several training cases per center and contrast, drawn exactly as stored in LPS (array axis 0 = L, 1 = P, 2 = S), with
anatomical side markers computed from the array orientation. aff2axcodes is blind to 180-degree flips, so this is judged by eye:
  axial:   anterior at the TOP of the picture, patient RIGHT on the image LEFT (radiological), spine at the bottom, liver on the image left;
  coronal: superior at the top;  sagittal: anterior to the left (axis 1 = posterior runs left->right).
Also checks numerically that the mask centroid sits in the expected half of the body (liver/pancreas side) using the brightest large organ is NOT
used; the visual check is the gate. Output: 9_tests_pansegdata/orientation_qc_<contrast>.png.
Usage: python 02_02_orientation_qc.py"""
import json
from pathlib import Path
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = Path(__file__).resolve().parents[2]
RAW = DS / "2_nnUNet_pansegdata" / "raw"
OUT = DS / "9_tests_pansegdata"; OUT.mkdir(exist_ok=True)
part = json.loads((DS / "4_splits_pansegdata" / "partition.json").read_text())
site = part["site"]
rng = np.random.default_rng(0)
for contrast, ds in (("t1wce", "Dataset150_PanSegData_T1WCE"), ("t2w", "Dataset151_PanSegData_T2W")):
    picks = []
    for s in sorted(set(site[c] for c in part["train_pool"])):   # only centers present in the (MCF-excluded) split
        cs = [c for c in part["train_pool"] if site[c] == s]
        picks += list(rng.choice(cs, 3 if s == "NYU" else 2, replace=False))
    fig, axs = plt.subplots(len(picks), 3, figsize=(7.5, 2.6 * len(picks)))
    for i, cid in enumerate(picks):
        im = nib.load(str(RAW / ds / "imagesTr" / f"{cid}_0000.nii.gz")); x = np.asanyarray(im.dataobj)
        m = np.asanyarray(nib.load(str(RAW / ds / "labelsTr" / f"{cid}.nii.gz")).dataobj) > 0
        ax_ = "".join(nib.aff2axcodes(im.affine)); z = im.header.get_zooms()[:3]
        c = np.round(np.argwhere(m).mean(0)).astype(int)
        lo, hi = np.percentile(x, [1, 99.5])
        views = [("axial", x[:, :, c[2]], m[:, :, c[2]], (z[0], z[1])), ("coronal", x[:, c[1], :], m[:, c[1], :], (z[0], z[2])), ("sagittal", x[c[0], :, :], m[c[0], :, :], (z[1], z[2]))]
        for j, (nm, sl, ml, (a, b)) in enumerate(views):
            ax = axs[i][j]
            org = "upper" if nm == "axial" else "lower"  # axial: P increases downward -> anterior UP (radiological); cor/sag: S increases upward
            ax.imshow(sl.T, cmap="gray", vmin=lo, vmax=hi, origin=org, aspect=b / a)
            ax.contour(ml.T.astype(float), levels=[0.5], colors="r", linewidths=0.6, origin=org)
            ax.set_title(f"{cid[11:]} {nm} [{ax_}]", fontsize=6); ax.axis("off")
    fig.suptitle(f"{contrast}: converted arrays (LPS). Expected: axial anterior UP + patient-left on image RIGHT (liver image-left); coronal superior up; sagittal anterior LEFT", fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / f"orientation_qc_{contrast}.png", dpi=100); plt.close(fig)
    print("wrote", OUT / f"orientation_qc_{contrast}.png", "cases", [p[11:] for p in picks])
