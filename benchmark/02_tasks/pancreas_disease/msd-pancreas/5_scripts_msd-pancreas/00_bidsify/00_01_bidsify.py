#!/usr/bin/env python3
"""MSD Task07 Pancreas (portal-venous CT, MSKCC; labels 1 = pancreas, 2 = cancer) -> BIDS leaf 1_BIDS_msd-pancreas/onc-pancreas-msd/. The 281 labeled training cases only (the 139 'test' cases have no
public labels). Images and labels are byte-for-byte copies; names change, ids are kept (sub-MSD<NNNN> = pancreas_<NNN>). participants.tsv carries the per-case audit numbers.
Usage: 00_01_bidsify.py <Task07_Pancreas_dir> <audit.tsv> <bids_leaf_dir> <license_note_file>"""
import csv, json, shutil, sys
from pathlib import Path
src, audit, leaf, lic = map(Path, sys.argv[1:5])
rows = []
for r in csv.DictReader(open(audit), delimiter="\t"):
    n = int(r["case"].split("_")[1]); sub = f"sub-MSD{n:04d}"
    d = leaf / sub / "anat"; dl = leaf / "derivatives" / "labels" / sub / "anat"; d.mkdir(parents=True, exist_ok=True); dl.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / "imagesTr" / f"{r['case']}.nii.gz", d / f"{sub}_acq-portalvenous_CT.nii.gz")
    shutil.copyfile(src / "labelsTr" / f"{r['case']}.nii.gz", dl / f"{sub}_acq-portalvenous_label-pancreascancer_dseg.nii.gz")
    rows.append(dict(participant_id=sub, source_case=r["case"], shape=r["shape"], zooms_mm=r["zooms"], axcodes=r["axcodes"], fov_mm=r["fov_mm"], pancreas_ml=r["pan_ml"], cancer_ml=r["cancer_ml"],
                     union_ml=r["union_ml"], union_ncomp=r["union_ncomp"], union_touches_border=r["union_touches_border"], hu_pancreas_median=r["hu_pan_med"], hu_cancer_median=r["hu_cancer_med"]))
with open(leaf / "participants.tsv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0]), delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(rows)
shutil.copyfile(lic, leaf / "LICENSE")
(leaf / "dataset_description.json").write_text(json.dumps({"Name": "Medical Segmentation Decathlon Task07 Pancreas (labeled CT cases)", "BIDSVersion": "1.9.0", "DatasetType": "raw",
    "License": "CC-BY-SA-4.0", "Authors": ["Amber L. Simpson", "Michela Antonelli", "Spyridon Bakas", "et al. (Medical Segmentation Decathlon)"],
    "ReferencesAndLinks": ["https://arxiv.org/abs/1902.09063", "https://doi.org/10.1038/s41467-022-30695-9"]}, indent=1))
print("subjects", len(rows))
