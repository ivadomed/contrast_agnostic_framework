#!/usr/bin/env python3
"""
BIDSify CirrMRI600+ (T1_3D + T2_3D, already extracted to 0_raw by 00_00_download.py).

CIRRMRI-LIVER is EVALUATION-ONLY here (see datasets/cirrmri-liver/README.md): the
337 patients (310 T1w-labeled, 318 T2w-labeled, 291 with both) become OUR test set
for MRI->MRI (different scanner/protocol/pathology cohort) generalization of
chaos-trained models. The dataset's own train/valid/test split is IGNORED and all
three are combined per contrast -- we don't train here, so that split has no meaning
for us; every case becomes part of our single test set.

Raw layout (per contrast, already extracted):
  0_raw_cirrmri-liver/Cirrhosis_{T1,T2}_3D/Cirrhosis_{T1,T2}_3D/
    {train,valid,test}_images/<patient_id>.nii.gz
    {train,valid,test}_masks/<patient_id>.nii.gz   (binary liver mask, already {0,1})
Patient ids are shared across T1/T2 (291 of 337 patients have both contrasts) --
each becomes ONE BIDS subject with both anat images where available.

BIDS output (1_BIDS_cirrmri-liver/cirrmri-liver/):
  sub-CR{id}/anat/sub-CR{id}_T1w.nii.gz  (+ .json)         <- Cirrhosis_T1_3D/*_images
  sub-CR{id}/anat/sub-CR{id}_T2w.nii.gz  (+ .json)         <- Cirrhosis_T2_3D/*_images
  derivatives/manual_masks/sub-CR{id}/anat/sub-CR{id}_{T1w,T2w}_dseg.nii.gz
  dataset_description.json   participants.tsv

Label map matches chaos's OWN liver numbering directly (both are id=1) -- unlike
msd-spleen (chaos spleen=4 vs this dataset's own spleen=1, needed --label_map),
evaluation here can use the SAME simpler --dataset_json --labels liver mode
sliver07 uses, since pred_id == gt_id == 1 requires no cross-space remap:
  0 = background, 1 = liver

Images/masks were confirmed already stored in LPS (project canonical) -- see
03_preprocess/03_00_check_orientation.py for the idempotent verification (no
reorientation needed, unlike msd-spleen which shipped RAS).

Usage:
  python 00_01_bidsify.py
"""
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

DATASET_ROOT = Path(__file__).resolve().parents[2]   # .../datasets/cirrmri-liver/
RAW_ROOT = DATASET_ROOT / "0_raw_cirrmri-liver"
BIDS_ROOT = DATASET_ROOT / "1_BIDS_cirrmri-liver" / "cirrmri-liver"
DERIVATIVES_DIR = BIDS_ROOT / "derivatives" / "manual_masks"

CONTRASTS = {"T1": "Cirrhosis_T1_3D", "T2": "Cirrhosis_T2_3D"}
SPLITS = ("train", "valid", "test")
LABELS = {"background": 0, "liver": 1}


def write_nii(img: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _sidecar(descr: str) -> dict:
    return {"Modality": "MR", "SeriesDescription": descr, "CirrMRI600PlusSource": True}


def collect_ids(contrast_dir_name: str) -> dict:
    """Return {patient_id: (image_path, mask_path)} across all 3 original splits."""
    base = RAW_ROOT / contrast_dir_name / contrast_dir_name
    found = {}
    for split in SPLITS:
        img_dir = base / f"{split}_images"
        msk_dir = base / f"{split}_masks"
        for img_path in sorted(img_dir.glob("*.nii.gz")):
            pid = img_path.name.replace(".nii.gz", "")
            msk_path = msk_dir / img_path.name
            if not msk_path.exists():
                print(f"    WARNING: {contrast_dir_name} {split} pid={pid}: no matching mask, skipping")
                continue
            if pid in found:
                print(f"    WARNING: {contrast_dir_name} pid={pid} appears in multiple splits, keeping first")
                continue
            found[pid] = (img_path, msk_path)
    return found


def bidsify_case(pid: str, contrast: str, img_path: Path, msk_path: Path) -> None:
    label = f"CR{int(pid):03d}"
    sub = f"sub-{label}"
    anat = BIDS_ROOT / sub / "anat"
    deriv = DERIVATIVES_DIR / sub / "anat"
    suffix = f"{contrast}w"   # -> "T1w"/"T2w"

    img = sitk.ReadImage(str(img_path))
    seg = sitk.ReadImage(str(msk_path))

    arr = (sitk.GetArrayFromImage(seg) > 0).astype(np.uint8) * LABELS["liver"]
    img_shape = sitk.GetArrayFromImage(img).shape
    if arr.shape != img_shape:
        raise ValueError(f"{label} {contrast}: image/mask shape mismatch {img_shape} vs {arr.shape}")
    lab = sitk.GetImageFromArray(arr)
    lab.CopyInformation(img)

    write_nii(img, anat / f"{sub}_{suffix}.nii.gz")
    write_json(anat / f"{sub}_{suffix}.json", _sidecar(f"CirrMRI600+ cirrhotic-liver {contrast}-weighted MRI"))
    write_nii(lab, deriv / f"{sub}_{suffix}_dseg.nii.gz")


def bidsify() -> None:
    BIDS_ROOT.mkdir(parents=True, exist_ok=True)
    _write_dataset_description()
    _write_derivatives_description()

    per_contrast = {}
    for contrast, dirname in CONTRASTS.items():
        found = collect_ids(dirname)
        per_contrast[contrast] = found
        print(f"  {contrast}: {len(found)} cases (across train/valid/test)")

    all_pids = sorted(set(per_contrast["T1"]) | set(per_contrast["T2"]), key=lambda x: int(x))
    print(f"  {len(all_pids)} unique patients total")

    participants = []
    for pid in all_pids:
        contrasts_present = []
        for contrast in ("T1", "T2"):
            if pid in per_contrast[contrast]:
                img_path, msk_path = per_contrast[contrast][pid]
                bidsify_case(pid, contrast, img_path, msk_path)
                contrasts_present.append(contrast)
        participants.append({"label": f"CR{int(pid):03d}", "cirrmri_id": pid,
                              "contrasts": "+".join(contrasts_present)})
        print(f"    CR{int(pid):03d} done ({'+'.join(contrasts_present)})")

    _write_participants_tsv(participants)
    print(f"  BIDSify done -> {BIDS_ROOT}")


def _write_dataset_description() -> None:
    write_json(BIDS_ROOT / "dataset_description.json", {
        "Name": "CirrMRI600+ (T1w+T2w cirrhotic-liver MRI subset)",
        "BIDSVersion": "1.9.0",
        "License": "CC BY-NC 4.0 (see 0_raw_cirrmri-liver/LICENSE.txt)",
        "Authors": ["Jha, D.", "Susladkar, O. K.", "Gorade, V.", "Keles, E.",
                    "Antalek, M.", "Bagci, U.", "et al."],
        "ReferencesAndLinks": [
            "https://osf.io/cuk24/",
            "Jha et al., Large Scale MRI Collection and Segmentation of Cirrhotic "
            "Liver, Scientific Data 12, 896 (2025). https://doi.org/10.1038/s41597-025-05201-7",
        ],
        "DatasetType": "raw",
        "Usage": "EVALUATION ONLY -- test set for chaos-trained models (MRI->MRI generalization, liver label).",
    })


def _write_derivatives_description() -> None:
    write_json(DERIVATIVES_DIR / "dataset_description.json", {
        "Name": "CirrMRI600+ Manual Liver Masks",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "CirrMRI600+ (Jha et al. 2025)"}],
        "LabelMap": LABELS,
    })


def _write_participants_tsv(participants: list) -> None:
    lines = ["participant_id\tmodality\tcirrmri_id\tcontrasts\tsplit"]
    for p in participants:
        lines.append(f"sub-{p['label']}\tMR\t{p['cirrmri_id']}\t{p['contrasts']}\ttest")
    (BIDS_ROOT / "participants.tsv").write_text("\n".join(lines) + "\n")


def main() -> None:
    print("=" * 64)
    print("CirrMRI600+ -- BIDSify (T1w + T2w, combined splits)")
    print("=" * 64)
    bidsify()


if __name__ == "__main__":
    main()
