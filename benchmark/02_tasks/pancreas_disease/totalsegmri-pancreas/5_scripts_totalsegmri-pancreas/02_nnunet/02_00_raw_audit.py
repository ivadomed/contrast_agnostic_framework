#!/usr/bin/env python3
"""Raw audit of TotalSegmentator MRI v2.0.0 (Zenodo 10.5281/zenodo.14710732): per case image grid, orientation, pancreas mask size/extent + meta.csv
fields, to decide usable N for the pancreas eval companion. Output: audit TSV (LF line endings). Usage: 02_00_raw_audit.py <extracted_dir> <out.tsv>"""
import sys, csv
from pathlib import Path
import numpy as np, nibabel as nib
from scipy import ndimage as ndi

root, out = Path(sys.argv[1]), Path(sys.argv[2])
meta = {r["image_id"]: r for r in csv.DictReader(open(root / "meta.csv", encoding="utf-8-sig"), delimiter=";")}
cols = ["case", "split", "institute", "manufacturer", "field", "seq", "tr", "te", "slice_thk_meta", "source", "shape", "zooms", "axcodes",
        "fov_mm", "mask_vox", "mask_ml", "mask_ncomp", "mask_z_mm", "mask_inside_fov_border", "mask_vals", "mask_same_grid", "img_p1", "img_p99", "err"]
rows = []
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    m = meta.get(d.name, {}); r = {"case": d.name}
    try:
        im = nib.load(d / "mri.nii.gz"); sg = nib.load(d / "segmentations" / "pancreas.nii.gz")
        a = np.asanyarray(sg.dataobj); z = im.header.get_zooms()[:3]
        r.update(split=m.get("split"), institute=m.get("institute"), manufacturer=m.get("manufacturer"), field=m.get("magnetic_field_strength"),
                 seq=m.get("scanning_sequence"), tr=m.get("repetition_time"), te=m.get("echo_time"), slice_thk_meta=m.get("slice_thickness"), source=m.get("source"),
                 shape="x".join(map(str, im.shape)), zooms="x".join(f"{v:.2f}" for v in z), axcodes="".join(nib.aff2axcodes(im.affine)),
                 fov_mm="x".join(f"{s*v:.0f}" for s, v in zip(im.shape[:3], z)),
                 mask_same_grid=int(sg.shape == im.shape and np.allclose(sg.affine, im.affine, atol=1e-3)),
                 mask_vals=",".join(map(str, np.unique(a)[:5])))
        b = a > 0; n = int(b.sum()); r["mask_vox"] = n
        r["mask_ml"] = f"{n*np.prod(sg.header.get_zooms()[:3])/1000:.1f}"
        if n:
            lab, nc = ndi.label(b); r["mask_ncomp"] = nc
            zs = np.where(b.any((0, 1)))[0]; r["mask_z_mm"] = f"{(zs.max()-zs.min()+1)*sg.header.get_zooms()[2]:.0f}"
            ix = np.where(b)
            r["mask_inside_fov_border"] = int(any(ix[k].min() == 0 or ix[k].max() == b.shape[k]-1 for k in range(3)))
        v = np.asanyarray(im.dataobj).astype(np.float32); r["img_p1"], r["img_p99"] = f"{np.percentile(v,1):.0f}", f"{np.percentile(v,99):.0f}"
    except Exception as e:
        r["err"] = repr(e)
    rows.append(r)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, cols, delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(rows)
print("cases", len(rows), "nonempty", sum(1 for r in rows if int(r.get("mask_vox") or 0) > 0), "errors", sum(1 for r in rows if r.get("err")))
