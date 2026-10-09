#!/usr/bin/env python3
"""Per-case FOV geometry of the pancreas relative to the image box, for the SOURCE training set (imagesTr/labelsTr) and this companion's test items (imagesTs_*/labelsTs_*).
Everything in mm along L-R / A-P / S-I (LPS-reoriented data: array axes 0,1,2 = L,P,S): FOV extent, pancreas bbox/centroid distance to the low edge (inferior / anterior-right ends are
array index 0 for axis 2 / ...), and the pancreas bbox size. Output TSV (LF). Usage: 02_03_fov_per_case.py <raw_dir_src_Dataset> <raw_dir_test> <out.tsv>"""
import sys, csv
from pathlib import Path
import numpy as np, nibabel as nib

src, tst, out = map(Path, sys.argv[1:4])
rows = []
def one(group, img, lab):
    I = nib.load(img); L = nib.load(lab); ax = nib.aff2axcodes(I.affine)
    a = np.asanyarray(L.dataobj) > 0; z = np.array(I.header.get_zooms()[:3]); sh = np.array(I.shape[:3])
    ext = sh * z; ix = np.where(a); lo = np.array([i.min() for i in ix]) * z; hi = (np.array([i.max() for i in ix]) + 1) * z; cen = np.array([i.mean() for i in ix]) * z
    r = dict(group=group, case=lab.name.replace(".nii.gz", ""), axcodes="".join(ax))
    for k, n in enumerate(("x", "y", "z")):
        r.update({f"ext_{n}": f"{ext[k]:.0f}", f"sp_{n}": f"{z[k]:.2f}", f"cen_{n}": f"{cen[k]:.0f}", f"lo_{n}": f"{lo[k]:.0f}", f"hi_gap_{n}": f"{ext[k]-hi[k]:.0f}", f"pan_{n}": f"{hi[k]-lo[k]:.0f}", f"cenfrac_{n}": f"{cen[k]/ext[k]:.2f}"})
    rows.append(r)
for lab in sorted((src / "labelsTr").glob("*.nii.gz")):
    one("source_train", src / "imagesTr" / (lab.name.replace(".nii.gz", "_0000.nii.gz")), lab)
for d in sorted(tst.glob("labelsTs_*")):
    for lab in sorted(d.glob("*.nii.gz")):
        one(d.name.replace("labelsTs_", "test_"), tst / d.name.replace("labelsTs", "imagesTs") / (lab.name.replace(".nii.gz", "_0000.nii.gz")), lab)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0]), delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(rows)
print("rows", len(rows))
