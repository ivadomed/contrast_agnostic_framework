#!/usr/bin/env python3
"""Prepare the RSNA lumbar spine competition archive for the project pipeline.

The script unpacks the Kaggle archive, converts the three target series per study
into a 4D NIfTI volume, and builds a sparse label volume from the coordinate CSV.
It is designed to feed the existing nnU-Net conversion flow under data/LumbarSpine.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_SERIES = (
    "Sagittal T1",
    "Sagittal T2/STIR",
    "Axial T2",
)
SEVERITY_TO_CLASS = {
    "Normal/Mild": 1,
    "Moderate": 2,
    "Severe": 3,
}
CONDITION_TO_COLUMN = {
    "Spinal Canal Stenosis": "spinal_canal_stenosis",
    "Left Neural Foraminal Narrowing": "left_neural_foraminal_narrowing",
    "Right Neural Foraminal Narrowing": "right_neural_foraminal_narrowing",
    "Left Subarticular Stenosis": "left_subarticular_stenosis",
    "Right Subarticular Stenosis": "right_subarticular_stenosis",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        default="/tmp/rsna-lumbar-spine-degenerative-classification/rsna-2024-lumbar-spine-degenerative-classification.zip",
        help="Path to the Kaggle competition zip archive.",
    )
    parser.add_argument(
        "--extract-dir",
        default="/tmp/rsna-lumbar-spine-degenerative-classification/extracted",
        help="Directory where the archive will be unpacked.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "LumbarSpine"),
        help="Project-relative output directory containing imagesTr/labelsTr.",
    )
    parser.add_argument(
        "--study-limit",
        type=int,
        default=0,
        help="Optional limit for quick smoke tests; 0 processes all studies.",
    )
    return parser.parse_args()


def ensure_unpacked(archive: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".rsna_extracted"
    if marker.exists():
        return extract_dir

    print(f"Extracting {archive} -> {extract_dir}")
    with zipfile.ZipFile(str(archive), "r") as handle:
        handle.extractall(str(extract_dir))
    marker.write_text("ok\n", encoding="utf-8")
    return extract_dir


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))



def make_series_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    by_study: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        by_study[row["study_id"]][row["series_description"]] = row["series_id"]
    return by_study



def make_label_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        study_id = row["study_id"]
        for condition_slug in CONDITION_TO_COLUMN.values():
            for level in ("L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"):
                column = f"{condition_slug}_{level.lower().replace('/', '_')}"
                severity = row.get(column)
                if severity:
                    key = f"{condition_slug}_{level.lower().replace('/', '_')}"
                    lookup[study_id][key] = severity
    return lookup



def locate_series_dir(extract_dir: Path, study_id: str, series_id: str) -> Path:
    return extract_dir / "train_images" / study_id / series_id



def convert_series_to_nifti(series_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "dcm2niix",
            "-z",
            "y",
            "-b",
            "n",
            "-o",
            str(output_dir),
            str(series_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    nii_files = sorted(output_dir.glob("*.nii.gz"))
    if not nii_files:
        raise RuntimeError(f"dcm2niix produced no NIfTI file for {series_dir}\n{result.stdout}\n{result.stderr}")
    return nii_files[0]



def _safe_world_to_voxel(affine: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    voxel = np.linalg.inv(affine) @ np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
    return voxel[:3]



def build_sample(
    study_id: str,
    series_map: dict[str, str],
    coord_rows: list[dict[str, str]],
    label_map: dict[str, dict[str, str]],
    extract_dir: Path,
    output_dir: Path,
    tmp_dir: Path,
) -> bool:
    ordered_series: list[tuple[str, str]] = []
    for description in TARGET_SERIES:
        series_id = series_map.get(description)
        if not series_id:
            print(f"  [WARN] {study_id}: missing series '{description}'")
            return False
        ordered_series.append((description, series_id))

    converted_paths: list[tuple[str, nib.Nifti1Image]] = []
    for description, series_id in ordered_series:
        series_dir = locate_series_dir(extract_dir, study_id, series_id)
        if not series_dir.exists():
            print(f"  [WARN] {study_id}: missing DICOM directory for {description} at {series_dir}")
            return False
        nifti_path = convert_series_to_nifti(series_dir, tmp_dir / study_id / series_id)
        converted_paths.append((series_id, nib.load(str(nifti_path))))

    reference = converted_paths[0][1]
    reference_target = (reference.shape, reference.affine)
    resampled_channels: list[np.ndarray] = []
    for _, img in converted_paths:
        if img.shape != reference.shape or not np.allclose(img.affine, reference.affine):
            img = resample_from_to(img, reference_target, order=1)
        data = img.get_fdata(dtype=np.float32)
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        resampled_channels.append(data)

    stacked = np.stack(resampled_channels, axis=-1).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_tr = output_dir / "imagesTr"
    labels_tr = output_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    img_header = reference.header.copy()
    img_header.set_data_shape(stacked.shape)
    nib.save(nib.Nifti1Image(stacked, reference.affine, img_header), str(images_tr / f"{study_id}.nii.gz"))

    label_volume = np.zeros(reference.shape, dtype=np.uint8)
    label_rows = [row for row in coord_rows if row["study_id"] == study_id]
    series_images = {series_id: img for series_id, img in converted_paths}
    study_labels = label_map.get(study_id, {})
    for row in label_rows:
        series_id = row["series_id"]
        series_img = series_images.get(series_id)
        if series_img is None:
            continue
        severity_name = study_labels.get(
            f"{CONDITION_TO_COLUMN[row['condition']]}_{row['level'].lower().replace('/', '_')}"
        )
        class_id = SEVERITY_TO_CLASS.get(severity_name or "", 0)
        if class_id == 0:
            continue

        source_voxel = np.array(
            [float(row["x"]), float(row["y"]), float(row["instance_number"]) - 1.0],
            dtype=np.float64,
        )
        world = series_img.affine @ np.array([source_voxel[0], source_voxel[1], source_voxel[2], 1.0])
        ref_voxel = _safe_world_to_voxel(reference.affine, world[:3])
        idx = tuple(np.rint(ref_voxel).astype(int))
        if all(0 <= idx[axis] < label_volume.shape[axis] for axis in range(3)):
            label_volume[idx] = max(label_volume[idx], class_id)

    label_header = reference.header.copy()
    label_header.set_data_shape(label_volume.shape)
    nib.save(nib.Nifti1Image(label_volume, reference.affine, label_header), str(labels_tr / f"{study_id}.nii.gz"))
    return True



def main() -> None:
    args = parse_args()
    archive = Path(args.archive)
    extract_dir = Path(args.extract_dir)
    output_dir = Path(args.output_dir)

    if not archive.exists():
        raise SystemExit(f"Archive not found: {archive}")

    unpacked = ensure_unpacked(archive, extract_dir)
    train_series_rows = read_csv_dicts(unpacked / "train_series_descriptions.csv")
    train_label_rows = read_csv_dicts(unpacked / "train_label_coordinates.csv")
    train_rows = read_csv_dicts(unpacked / "train.csv")

    series_lookup = make_series_lookup(train_series_rows)
    label_lookup = make_label_lookup(train_rows)
    coord_by_study: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in train_label_rows:
        coord_by_study[row["study_id"]].append(row)

    study_ids = sorted(series_lookup)
    if args.study_limit and args.study_limit > 0:
        study_ids = study_ids[: int(args.study_limit)]

    with tempfile.TemporaryDirectory(prefix="rsna-dcm2niix-") as tmp_root:
        tmp_dir = Path(tmp_root)
        processed = 0
        for study_id in study_ids:
            ok = build_sample(
                study_id=study_id,
                series_map=series_lookup[study_id],
                coord_rows=coord_by_study.get(study_id, []),
                label_map=label_lookup,
                extract_dir=unpacked,
                output_dir=output_dir,
                tmp_dir=tmp_dir,
            )
            if ok:
                processed += 1
            if processed and processed % 25 == 0:
                print(f"Processed {processed}/{len(study_ids)} studies")

    print(f"Prepared {processed}/{len(study_ids)} studies under {output_dir}")
    print("Use tools/generate_rsna_split.py to create the deterministic split.")
    print(f"Source train.csv rows: {len(train_rows)}")


if __name__ == "__main__":
    sys.exit(main())
