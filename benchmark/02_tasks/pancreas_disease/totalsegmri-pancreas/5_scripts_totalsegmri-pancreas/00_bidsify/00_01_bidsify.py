#!/usr/bin/env python3
"""TotalSegmentator MRI v2.0.0 (Zenodo 10.5281/zenodo.14710732) -> BIDS leaf 1_BIDS_totalsegmri-pancreas/abdomen-totalsegmri/ (pancreas part only).
Keeps the 182 of 616 exams whose pancreas mask is non-empty (the other 434 do not show the pancreas). Images and masks are copied byte-for-byte; only names change.
Per-case class (item) and acquisition plane are derived here from header-level metadata + organ-intensity ratios and stored in participants.tsv:
  plane   : thickest voxel axis (ratio >= 1.8) -> axial (S/I), coronal (A/P), sagittal (L/R); otherwise "3D" (near-isotropic)
  class   : t1gre  = scanning_sequence exactly "GR"
            t2like = scanning_sequence without any GR and spleen/liver median intensity >= 1.3 (T2-weighted organ contrast; meta TR/TE are unreliable)
            other  = everything else
Usage: 00_01_bidsify.py <extracted_dir> <audit.tsv> <classify.tsv> <bids_leaf_dir> <license_note_file>"""
import csv, shutil, sys, json, collections
from pathlib import Path

ext, audit, clf, leaf, lic = map(Path, sys.argv[1:6])
A = {r["case"]: r for r in csv.DictReader(open(audit), delimiter="\t")}
C = {r["case"]: r for r in csv.DictReader(open(clf), delimiter="\t")}
def plane(zooms, ax):
    z = [float(v) for v in zooms.split("x")]
    if max(z) / min(z) < 1.8: return "3D"
    return {"S": "axial", "I": "axial", "A": "coronal", "P": "coronal", "L": "sagittal", "R": "sagittal"}[ax[z.index(max(z))]]
def cls(a, c):
    seq = a["seq"]; s = c.get("spleen_over_liver", "nan"); s = float(s) if s not in ("", "nan") else None
    if seq == "GR": return "t1gre"
    if "GR" not in seq and s is not None and s >= 1.3: return "t2like"
    return "other"
SUFFIX = {"t1gre": ("gre", "T1w"), "t2like": ("t2like", "T2w"), "other": ("unclassified", "MRI")}
rows = []
for case in sorted(A):
    a = A[case]
    if int(a["mask_vox"] or 0) == 0: continue
    c = C[case]; k = cls(a, c); acq, suf = SUFFIX[k]; sub = "sub-TS" + case[1:]
    d = leaf / sub / "anat"; dl = leaf / "derivatives" / "labels" / sub / "anat"; d.mkdir(parents=True, exist_ok=True); dl.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ext / case / "mri.nii.gz", d / f"{sub}_acq-{acq}_{suf}.nii.gz")
    shutil.copyfile(ext / case / "segmentations" / "pancreas.nii.gz", dl / f"{sub}_acq-{acq}_label-pancreas_seg.nii.gz")
    sl = c["spleen_over_liver"]
    rows.append(dict(participant_id=sub, source_case=case, source_split=a["split"], institute=a["institute"], manufacturer=a["manufacturer"], field_T=a["field"],
                     scanning_sequence=a["seq"], source=a["source"], shape=a["shape"], zooms_mm=a["zooms"], axcodes=a["axcodes"], plane=plane(a["zooms"], a["axcodes"]),
                     pancreas_ml=a["mask_ml"], mask_touches_border=a["mask_inside_fov_border"], spleen_over_liver=f"{float(sl):.2f}" if sl not in ("", "nan") else "n/a", item_class=k))
with open(leaf / "participants.tsv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0]), delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(rows)
shutil.copyfile(lic, leaf / "LICENSE")
(leaf / "dataset_description.json").write_text(json.dumps({"Name": "TotalSegmentator MRI v2.0.0, pancreas subset", "BIDSVersion": "1.9.0", "DatasetType": "raw",
    "License": "CC-BY-NC-SA-2.0", "Authors": ["Jakob Wasserthal", "Tugba Akinci D'Antonoli"], "SourceDatasets": [{"DOI": "10.5281/zenodo.14710732"}],
    "ReferencesAndLinks": ["https://arxiv.org/abs/2405.19492"]}, indent=1))
print("subjects", len(rows), dict(collections.Counter(r["item_class"] for r in rows)), dict(collections.Counter(r["plane"] for r in rows)))
