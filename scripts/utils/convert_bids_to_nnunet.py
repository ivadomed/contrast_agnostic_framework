#!/usr/bin/env python3
"""Convert MS-MULTI-SPINE BIDS dataset to nnUNetv2 raw format.

This export creates one training case per (subject, modality):
  imagesTr/sub-XXX_<modality>_0000.nii.gz
  labelsTr/sub-XXX_<modality>.nii.gz

Only modalities that physically exist for a subject are exported.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

nnUNet_raw = Path(os.environ.get("nnUNet_raw", str(PROJECT_ROOT / "data" / "nnUNet_raw")))
nnUNet_preprocessed = Path(
    os.environ.get("nnUNet_preprocessed", str(PROJECT_ROOT / "data" / "nnUNet_preprocessed"))
)
nnUNet_results = Path(os.environ.get("nnUNet_results", str(PROJECT_ROOT / "results" / "nnUNet")))

MODALITY_MAP = {
    "t2w": "T2w",
    "mp2rage": "MP2RAGE",
    "stir": "STIR",
    "psir": "PSIR",
}
DESC_PRIORITY = ["desc-preprocReg", "desc-preproc", None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MS-MULTI-SPINE BIDS to nnUNet raw.")
    parser.add_argument("--bids-root", type=str, default="data/ms_multi_spine", help="BIDS root directory.")
    parser.add_argument("--dataset_id", "--dataset-id", type=int, default=104)
    parser.add_argument("--dataset_name", "--dataset-name", type=str, default="MSMultiSpine")
    parser.add_argument(
        "--audit-json",
        type=str,
        default="results/eval/ms_multi_spine_audit.json",
        help="Audit JSON path; if present, thick_axis_first policy is read from it.",
    )
    parser.add_argument(
        "--thick-axis-first",
        action="store_true",
        help="Force through-plane axis to axis 0 for all outputs.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing output dataset folder before conversion.",
    )
    return parser.parse_args()


def _pick_image(subject_dir: Path, bids_modality: str) -> tuple[Path | None, str]:
    anat = subject_dir / "anat"
    if not anat.exists():
        return None, ""
    for desc in DESC_PRIORITY:
        if desc is None:
            name = f"{subject_dir.name}_{bids_modality}.nii.gz"
            key = "raw"
        else:
            name = f"{subject_dir.name}_{desc}_{bids_modality}.nii.gz"
            key = desc
        p = anat / name
        if p.exists():
            return p, key
    return None, ""


def _pick_label(label_subject_dir: Path, desc_key: str) -> Path | None:
    anat = label_subject_dir / "anat"
    if not anat.exists():
        return None
    candidates: list[Path] = []
    if desc_key == "desc-preprocReg":
        candidates.append(anat / f"{label_subject_dir.name}_desc-preprocReg_T2w_label-lesion_seg.nii.gz")
        candidates.append(anat / f"{label_subject_dir.name}_desc-preproc_T2w_label-lesion_seg.nii.gz")
        candidates.append(anat / f"{label_subject_dir.name}_T2w_label-lesion_seg.nii.gz")
    elif desc_key == "desc-preproc":
        candidates.append(anat / f"{label_subject_dir.name}_desc-preproc_T2w_label-lesion_seg.nii.gz")
        candidates.append(anat / f"{label_subject_dir.name}_desc-preprocReg_T2w_label-lesion_seg.nii.gz")
        candidates.append(anat / f"{label_subject_dir.name}_T2w_label-lesion_seg.nii.gz")
    else:
        candidates.append(anat / f"{label_subject_dir.name}_T2w_label-lesion_seg.nii.gz")
        candidates.append(anat / f"{label_subject_dir.name}_desc-preprocReg_T2w_label-lesion_seg.nii.gz")
        candidates.append(anat / f"{label_subject_dir.name}_desc-preproc_T2w_label-lesion_seg.nii.gz")

    for p in candidates:
        if p.exists():
            return p
    return None


def _choose_thick_axis_first(audit_json: Path, force_flag: bool) -> bool:
    if force_flag:
        return True
    if audit_json.exists():
        payload = json.loads(audit_json.read_text(encoding="utf-8"))
        return bool(payload.get("requires_thick_axis_first", False))
    return False


def _prepare_affine_and_data(path: Path, thick_axis_first: bool) -> tuple[np.ndarray, np.ndarray, int]:
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI at {path}, got shape {data.shape}")

    spacing = list(float(z) for z in img.header.get_zooms()[:3])
    thick_axis = int(np.argmax(spacing))
    if thick_axis_first and thick_axis != 0:
        perm = [thick_axis] + [i for i in range(3) if i != thick_axis]
        data = np.transpose(data, tuple(perm))
        spacing = [spacing[i] for i in perm]

    # Preserve real voxel spacing explicitly; never write identity for this dataset.
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0]).astype(np.float32)
    return data, affine, thick_axis


def _prepare_label(path: Path, thick_axis_first: bool, thick_axis_from_image: int) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    label = np.asarray(img.get_fdata())
    if label.ndim != 3:
        raise ValueError(f"Expected 3D label NIfTI at {path}, got shape {label.shape}")

    # Enforce binary class-level labels by collapsing all non-zero instance IDs.
    rounded = np.rint(label).astype(np.int64)
    unique = np.unique(rounded)
    if np.any(rounded < 0):
        raise ValueError(f"Negative labels found in {path}: unique={unique.tolist()}")

    spacing = list(float(z) for z in img.header.get_zooms()[:3])
    if thick_axis_first and thick_axis_from_image != 0:
        perm = [thick_axis_from_image] + [i for i in range(3) if i != thick_axis_from_image]
        rounded = np.transpose(rounded, tuple(perm))
        spacing = [spacing[i] for i in perm]

    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0]).astype(np.float32)
    label = (rounded > 0).astype(np.uint8)
    return label, affine


def _write_dataset_json(dataset_dir: Path, n_training: int) -> None:
    payload = {
        "channel_names": {"0": "MRI"},
        "labels": {
            "background": 0,
            "lesion": 1,
        },
        "numTraining": n_training,
        "file_ending": ".nii.gz",
        "name": dataset_dir.name,
        "description": "MS-MULTI-SPINE BIDS export with modality-specific training cases.",
        "reference": "MS-Multi-Spine-Challenge-2024",
        "licence": "Unknown",
        "release": "2026-04-18",
    }
    out = dataset_dir / "dataset.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    bids_root = PROJECT_ROOT / args.bids_root
    if not bids_root.exists():
        raise SystemExit(f"BIDS root not found: {bids_root}")

    dataset_dir = nnUNet_raw / f"Dataset{int(args.dataset_id):03d}_{args.dataset_name}"
    if args.clean and dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    for d in [nnUNet_raw, nnUNet_preprocessed, nnUNet_results, images_tr, labels_tr, images_ts]:
        d.mkdir(parents=True, exist_ok=True)

    thick_axis_first = _choose_thick_axis_first(PROJECT_ROOT / args.audit_json, args.thick_axis_first)

    label_root = bids_root / "derivatives" / "labels"
    subject_dirs = sorted([p for p in bids_root.glob("sub-*") if p.is_dir()])

    n_ok = 0
    n_missing_label = 0
    exported_by_modality = {k: 0 for k in MODALITY_MAP}

    for subject_dir in subject_dirs:
        label_subject_dir = label_root / subject_dir.name
        for mod_key, bids_modality in MODALITY_MAP.items():
            image_path, desc_key = _pick_image(subject_dir, bids_modality)
            if image_path is None:
                continue

            label_path = _pick_label(label_subject_dir, desc_key)
            if label_path is None:
                n_missing_label += 1
                print(f"[WARN] Missing label for {subject_dir.name}:{mod_key}; skipping")
                continue

            img_data, img_affine, thick_axis = _prepare_affine_and_data(image_path, thick_axis_first)
            lbl_data, lbl_affine = _prepare_label(label_path, thick_axis_first, thick_axis)

            if img_data.shape != lbl_data.shape:
                print(
                    f"[WARN] Geometry mismatch {subject_dir.name}:{mod_key} "
                    f"img={img_data.shape} lbl={lbl_data.shape}; skipping"
                )
                continue

            case_id = f"{subject_dir.name}_{mod_key}"
            out_img = images_tr / f"{case_id}_0000.nii.gz"
            out_lbl = labels_tr / f"{case_id}.nii.gz"
            nib.save(nib.Nifti1Image(img_data, img_affine), str(out_img))
            nib.save(nib.Nifti1Image(lbl_data, lbl_affine), str(out_lbl))

            n_ok += 1
            exported_by_modality[mod_key] += 1

    _write_dataset_json(dataset_dir, n_ok)

    print("=" * 72)
    print("MS-MULTI-SPINE -> nnUNet conversion")
    print(f"Output dataset : {dataset_dir}")
    print(f"thick_axis_first={thick_axis_first}")
    print(f"Exported cases : {n_ok}")
    print(f"By modality    : {exported_by_modality}")
    print(f"Skipped (missing label): {n_missing_label}")
    print("=" * 72)


if __name__ == "__main__":
    main()
