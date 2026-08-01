#!/usr/bin/env python3
"""
BIDSify the T2-weighted Kidney MRI Segmentation dataset (Healthy_Control + CKD,
already extracted to 0_raw by 00_00_download.py).

KIDNEY-T2W is EVALUATION-ONLY here (see datasets/kidney-t2w/README.md). Raw layout
is a flat dir per group with filenames like:
  HC_01_T2W.nii.gz            HC_01_T2W_mask.nii.gz          (single scan)
  HC_26_T2W_3.nii.gz          HC_26_T2W_3_mask.nii.gz        (repeat scan #3 of 5;
                                                               10 subjects were
                                                               scanned 5x each for
                                                               precision assessment)
Each (subject, repeat) pair is its own independent acquisition with its own mask --
treated as its own test case here (same convention as e.g. TRUSTED's per-kidney
US cases: independent files, not deduplicated by subject).

BIDS output (1_BIDS_kidney-t2w/kidney-t2w/):
  sub-{HC|CKD}{NN}[rN]/anat/sub-..._T2w.nii.gz  (+ .json)
  derivatives/manual_masks/sub-.../anat/..._T2w_dseg.nii.gz
  dataset_description.json   participants.tsv

Label map: 0 = background, 1 = kidney (already binary in the source masks --
both kidneys merged into one label, like TRUSTED's CT/US kidney masks). chaos
predicts right_kidney(2)/left_kidney(3) SEPARATELY, so evaluation uses the same
cross-label-space merge TRUSTED uses (chaos {2,3} union -> this GT's id 1), via a
dedicated 06_00_evaluate_kidney_t2w.py shim (not the generic --label_map CLI,
which only accepts a single int, not a list, for pred_id).

Usage:
  python 00_01_bidsify.py
"""
import json
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk

DATASET_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = DATASET_ROOT / "0_raw_kidney-t2w"
BIDS_ROOT = DATASET_ROOT / "1_BIDS_kidney-t2w" / "kidney-t2w"
DERIVATIVES_DIR = BIDS_ROOT / "derivatives" / "manual_masks"

GROUPS = {"HC": "Healthy_Control", "CKD": "CKD"}
LABELS = {"background": 0, "kidney": 1}

CASE_RE = re.compile(r"^(HC|CKD)_(\d+)_T2W(?:_(\d+))?\.nii\.gz$")


def write_nii(img: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _sidecar(descr: str) -> dict:
    return {"Modality": "MR", "SeriesDescription": descr, "KidneyT2WSource": True}


def collect_cases(group_dir: Path):
    """Yield (label, img_path, mask_path) for every image file, matched by name."""
    for img_path in sorted(group_dir.glob("*.nii.gz")):
        if "_mask" in img_path.name:
            continue
        m = CASE_RE.match(img_path.name)
        if not m:
            print(f"    WARNING: unrecognized filename pattern, skipping: {img_path.name}")
            continue
        group, subj, repeat = m.group(1), m.group(2), m.group(3)
        label = f"{group}{int(subj):02d}" + (f"r{repeat}" if repeat else "")
        mask_path = img_path.with_name(img_path.name.replace(".nii.gz", "_mask.nii.gz"))
        if not mask_path.exists():
            print(f"    WARNING: no matching mask for {img_path.name}, skipping")
            continue
        yield label, img_path, mask_path


def bidsify_case(label: str, img_path: Path, mask_path: Path, participants: list) -> None:
    sub = f"sub-{label}"
    anat = BIDS_ROOT / sub / "anat"
    deriv = DERIVATIVES_DIR / sub / "anat"

    img = sitk.ReadImage(str(img_path))
    seg = sitk.ReadImage(str(mask_path))

    arr = (sitk.GetArrayFromImage(seg) > 0).astype(np.uint8) * LABELS["kidney"]
    img_shape = sitk.GetArrayFromImage(img).shape
    if arr.shape != img_shape:
        raise ValueError(f"{label}: image/mask shape mismatch {img_shape} vs {arr.shape}")
    lab = sitk.GetImageFromArray(arr)
    lab.CopyInformation(img)

    write_nii(img, anat / f"{sub}_T2w.nii.gz")
    write_json(anat / f"{sub}_T2w.json", _sidecar("T2-weighted kidney MRI (T2-weighted Kidney MRI Segmentation dataset)"))
    write_nii(lab, deriv / f"{sub}_T2w_dseg.nii.gz")

    participants.append({"label": label})


def bidsify() -> None:
    BIDS_ROOT.mkdir(parents=True, exist_ok=True)
    _write_dataset_description()
    _write_derivatives_description()

    participants: list = []
    for group, dirname in GROUPS.items():
        group_dir = RAW_ROOT / dirname
        n = 0
        for label, img_path, mask_path in collect_cases(group_dir):
            bidsify_case(label, img_path, mask_path, participants)
            n += 1
        print(f"  {group}: {n} cases")

    _write_participants_tsv(participants)
    print(f"  BIDSify done -> {BIDS_ROOT} ({len(participants)} total cases)")


def _write_dataset_description() -> None:
    write_json(BIDS_ROOT / "dataset_description.json", {
        "Name": "T2-weighted Kidney MRI Segmentation (Healthy Control + CKD)",
        "BIDSVersion": "1.9.0",
        "License": "CC-BY-4.0",
        "ReferencesAndLinks": [
            "https://zenodo.org/records/5153568",
            "Automated Kidney Segmentation by Mask R-CNN in T2-weighted MRI (arXiv:2108.12506)",
        ],
        "DatasetType": "raw",
        "Usage": "EVALUATION ONLY — test set for chaos-trained models (MRI->MRI generalization, kidney label).",
    })


def _write_derivatives_description() -> None:
    write_json(DERIVATIVES_DIR / "dataset_description.json", {
        "Name": "T2-weighted Kidney MRI Segmentation — Manual Kidney Masks",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "T2-weighted Kidney MRI Segmentation dataset (Zenodo 5153568)"}],
        "LabelMap": LABELS,
    })


def _write_participants_tsv(participants: list) -> None:
    lines = ["participant_id\tmodality\tsplit"]
    for p in participants:
        lines.append(f"sub-{p['label']}\tMR\ttest")
    (BIDS_ROOT / "participants.tsv").write_text("\n".join(lines) + "\n")


def main() -> None:
    print("=" * 64)
    print("T2-weighted Kidney MRI Segmentation — BIDSify")
    print("=" * 64)
    bidsify()


if __name__ == "__main__":
    main()
