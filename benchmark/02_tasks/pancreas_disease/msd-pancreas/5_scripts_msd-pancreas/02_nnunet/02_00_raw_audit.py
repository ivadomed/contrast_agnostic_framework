#!/usr/bin/env python3
"""Raw audit of MSD Task07 Pancreas (CT, MSKCC; labels 1=pancreas 2=cancer): per case grid, orientation, spacing, label values, pancreas/cancer volumes, union component count, border contact,
HU physics (median HU in pancreas / cancer mask, p1/p99 of the image). Output TSV (LF). Usage: 02_00_raw_audit.py <Task07_Pancreas_dir> <out.tsv>"""
import sys, csv
from pathlib import Path
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
root, out = Path(sys.argv[1]), Path(sys.argv[2]); rows = []
cols = ["case", "shape", "zooms", "axcodes", "fov_mm", "same_grid", "label_vals", "pan_ml", "cancer_ml", "union_ml", "union_ncomp", "union_touches_border", "hu_pan_med", "hu_cancer_med", "hu_p1", "hu_p99", "err"]
for f in sorted((root / "labelsTr").glob("*.nii.gz")):
    r = {"case": f.name.replace(".nii.gz", "")}
    try:
        L = nib.load(f); I = nib.load(root / "imagesTr" / f.name); a = np.asanyarray(L.dataobj); v = np.asanyarray(I.dataobj).astype(np.float32); z = np.array(I.header.get_zooms()[:3])
        vox = float(np.prod(L.header.get_zooms()[:3])) / 1000
        u = a > 0
        r.update(shape="x".join(map(str, I.shape)), zooms="x".join(f"{k:.2f}" for k in z), axcodes="".join(nib.aff2axcodes(I.affine)), fov_mm="x".join(f"{s*k:.0f}" for s, k in zip(I.shape, z)),
                 same_grid=int(L.shape == I.shape and np.allclose(L.affine, I.affine, atol=1e-3)), label_vals=",".join(map(str, np.unique(a))),
                 pan_ml=f"{(a==1).sum()*vox:.1f}", cancer_ml=f"{(a==2).sum()*vox:.1f}", union_ml=f"{u.sum()*vox:.1f}")
        if u.sum():
            r["union_ncomp"] = ndi.label(u)[1]; ix = np.where(u); r["union_touches_border"] = int(any(i.min() == 0 or i.max() == u.shape[k] - 1 for k, i in enumerate(ix)))
            r["hu_pan_med"] = f"{np.median(v[a==1]):.0f}" if (a == 1).any() else ""; r["hu_cancer_med"] = f"{np.median(v[a==2]):.0f}" if (a == 2).any() else ""
        r["hu_p1"], r["hu_p99"] = f"{np.percentile(v,1):.0f}", f"{np.percentile(v,99):.0f}"
    except Exception as e:
        r["err"] = repr(e)
    rows.append(r)
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, cols, delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(rows)
print("cases", len(rows), "errors", sum(1 for r in rows if r.get("err")), "nonempty", sum(1 for r in rows if float(r.get("union_ml") or 0) > 0))
