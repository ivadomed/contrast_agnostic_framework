"""Audit the raw ISLES-2022 tree: usable-N (non-empty masks), axcodes, grids, spacing,
lesion volume, DWI/ADC/FLAIR/mask grid equality, and a physics check (lesion mean
DWI high / ADC low / FLAIR high vs brain). Writes a TSV next to the raw tree."""
import csv, sys
from pathlib import Path
import numpy as np, nibabel as nib

ROOT = Path(__file__).resolve().parents[2] / "0_raw_isles2022" / "ISLES-2022"
OUT = ROOT.parent / "isles2022_raw_audit.tsv"
rows = []
for sd in sorted(ROOT.glob("sub-*")):
    s = sd.name
    p = {"dwi": sd/"ses-0001/dwi"/f"{s}_ses-0001_dwi.nii.gz",
         "adc": sd/"ses-0001/dwi"/f"{s}_ses-0001_adc.nii.gz",
         "flair": sd/"ses-0001/anat"/f"{s}_ses-0001_FLAIR.nii.gz",
         "msk": ROOT/"derivatives"/s/"ses-0001"/f"{s}_ses-0001_msk.nii.gz"}
    im = {k: nib.load(v) for k, v in p.items()}
    d = {k: np.asanyarray(v.dataobj) for k, v in im.items()}
    m = d["msk"] > 0
    brain = d["dwi"] > 0
    r = {"case": s, "shape": "x".join(map(str, im["dwi"].shape)),
         "spacing": "x".join(f"{z:.2f}" for z in im["dwi"].header.get_zooms()[:3]),
         "axcodes": "".join(nib.aff2axcodes(im["dwi"].affine)),
         **{f"{k}_on_msk_grid": bool(im[k].shape == im["msk"].shape and np.allclose(im[k].affine, im["msk"].affine, atol=1e-3)) for k in ("dwi", "adc", "flair")},
         "flair_shape": "x".join(map(str, im["flair"].shape)), "flair_axcodes": "".join(nib.aff2axcodes(im["flair"].affine)),
         "mask_vals": ",".join(map(str, np.unique(d["msk"]).astype(int)[:5])),
         "lesion_ml": round(float(m.sum()*np.prod(im["msk"].header.get_zooms()[:3])/1000), 3),
         "n_vox": int(m.sum())}
    if m.any():
        for k in ("dwi", "adc", "flair"):
            if not r[f"{k}_on_msk_grid"]:
                continue
            b = d[k][(d[k] > 0) & ~m]
            r[f"{k}_les_over_brain"] = round(float(d[k][m].mean()/max(b.mean(), 1e-6)), 3)
    rows.append(r)
keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
with open(OUT, "w") as f:
    w = csv.DictWriter(f, fieldnames=keys, delimiter="\t"); w.writeheader(); w.writerows(rows)
v = np.array([r["lesion_ml"] for r in rows])
print("cases", len(rows), "empty masks", int((v == 0).sum()), "tiny(<0.05ml)", int((v < 0.05).sum()))
print("lesion ml median/mean/max", np.median(v), v.mean(), v.max())
from collections import Counter
print("axcodes", Counter(r["axcodes"] for r in rows)); 
print("spacing top", Counter(r["spacing"] for r in rows).most_common(6))
print("mask vals", Counter(r["mask_vals"] for r in rows))
for k in ("dwi", "adc", "flair"):
    x = np.array([r[f"{k}_les_over_brain"] for r in rows if f"{k}_les_over_brain" in r])
    print(k, "lesion/brain ratio median", np.median(x), "frac>1", (x > 1).mean())
