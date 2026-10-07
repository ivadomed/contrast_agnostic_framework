#!/usr/bin/env python3
"""Image-based contrast class for TotalSegmentator-MRI cases with a non-empty pancreas mask (meta.csv TR/TE are unreliable, e.g. TR 0.01).
Uses organ intensity ratios (extended: vessels/kidney/muscle for contrast phase) (median inside each organ mask): spleen/liver is >1 on T2-weighted and <~1 on T1-weighted (pre-contrast), pancreas/liver
and CSF-free robust stats complement it. Extracts liver+spleen masks from the zip first (the extract step only took the pancreas).
Usage: 02_01_contrast_classify.py <zip> <extracted_dir> <audit.tsv> <out.tsv>"""
import sys, csv, zipfile
from pathlib import Path
import numpy as np, nibabel as nib

zpath, root, audit, out = map(Path, sys.argv[1:5])
ORG = ["pancreas", "liver", "spleen", "aorta", "portal_vein_and_splenic_vein", "inferior_vena_cava", "kidney_left", "kidney_right", "autochthon_left", "autochthon_right"]
rows = [r for r in csv.DictReader(open(audit), delimiter="\t") if int(r["mask_vox"] or 0) > 0]
need = [f"{r['case']}/segmentations/{o}.nii.gz" for r in rows for o in ORG[1:]]
with zipfile.ZipFile(zpath) as z:
    names = set(z.namelist())
    for n in need:
        if n in names and not (root / n).exists():
            z.extract(n, root)
def med(img, seg):
    m = np.asanyarray(seg.dataobj) > 0
    return (float(np.median(img[m])), int(m.sum())) if m.sum() >= 200 else (np.nan, int(m.sum()))
res = []
for r in rows:
    d = root / r["case"]; img = np.asanyarray(nib.load(d / "mri.nii.gz").dataobj).astype(np.float32); o = {"case": r["case"]}
    for k in ORG:
        p = d / "segmentations" / f"{k}.nii.gz"
        o[f"{k}_med"], o[f"{k}_n"] = med(img, nib.load(p)) if p.exists() else (np.nan, 0)
    bg = float(np.percentile(img, 99.5)); o["p995"] = bg
    o["spleen_over_liver"] = o["spleen_med"] / o["liver_med"] if o["liver_med"] and o["liver_med"] > 0 else np.nan
    o["panc_over_liver"] = o["pancreas_med"] / o["liver_med"] if o["liver_med"] and o["liver_med"] > 0 else np.nan
    mus = np.nanmean([o["autochthon_left_med"], o["autochthon_right_med"]]) if (o["autochthon_left_n"] or o["autochthon_right_n"]) else np.nan
    o["muscle_med"] = mus
    for k in ("aorta", "portal_vein_and_splenic_vein", "inferior_vena_cava", "kidney_left", "kidney_right", "liver", "pancreas", "spleen"):
        o[f"{k}_over_muscle"] = o[f"{k}_med"] / mus if mus and mus > 0 else np.nan
    res.append(o)
cols = list(res[0])
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, cols, delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(res)
print("rows", len(res), "with liver+spleen", sum(1 for o in res if o["spleen_over_liver"] == o["spleen_over_liver"]))
