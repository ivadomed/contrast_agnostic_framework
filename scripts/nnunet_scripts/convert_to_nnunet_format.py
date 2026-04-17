#!/usr/bin/env python3
"""Convert a configured dataset/contrast into nnUNetv2 raw format."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_registry import get_dataset_nnunet_id, get_dataset_spec

# nnUNet env vars (fall back to project-local paths if unset)
nnUNet_raw = Path(os.environ.get("nnUNet_raw", str(PROJECT_ROOT / "data" / "nnUNet_raw")))
nnUNet_preprocessed = Path(
    os.environ.get("nnUNet_preprocessed", str(PROJECT_ROOT / "data" / "nnUNet_preprocessed"))
)
nnUNet_results = Path(os.environ.get("nnUNet_results", str(PROJECT_ROOT / "results" / "nnUNet")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert dataset data to nnUNetv2 format.")
    parser.add_argument("--dataset", type=str, default="brats", help="Dataset key from src.data_registry.")
    parser.add_argument("--contrast", type=str, default="t1w", help="Contrast channel to export.")
    parser.add_argument(
        "--dataset-id",
        "--dataset_id",
        type=str,
        default=None,
        help="nnUNet dataset id (e.g., 022). Overrides registry lookup.",
    )
    parser.add_argument(
        "--dataset-name",
        "--dataset_name",
        type=str,
        default=None,
        help="nnUNet dataset folder name (e.g., Dataset022_BraTST1w_gen_raw).",
    )
    parser.add_argument(
        "--channel-name",
        "--channel_name",
        type=str,
        default=None,
        help="Optional channel display name written to dataset.json.",
    )
    parser.add_argument(
        "--task-name",
        "--task_name",
        type=str,
        default=None,
        help="Override task folder under data/nnUNet source (default from registry).",
    )
    parser.add_argument(
        "--split-file",
        "--split_file",
        type=str,
        default=None,
        help="Override split json path (default: splits/brats_subject_split.json).",
    )
    return parser.parse_args()


def _pretty_channel_name(contrast: str) -> str:
    lowered = str(contrast).strip().lower()
    mapping = {"t1w": "T1w", "t2w": "T2w", "flair": "Flair", "t1gd": "T1gd"}
    return mapping.get(lowered, lowered.upper())


def _default_dataset_name(dataset: str, dataset_id: str, channel_name: str) -> str:
    if dataset.lower() == "brats":
        return f"Dataset{dataset_id}_BraTS{channel_name}_gen_raw"
    return f"Dataset{dataset_id}_{dataset.capitalize()}{channel_name}_gen_raw"


def load_split(split_file: Path) -> tuple[list[str], list[str], list[str]]:
    with split_file.open("r", encoding="utf-8") as fh:
        split = json.load(fh)
    if isinstance(split, list):
        if not split:
            return [], [], []
        first = split[0] or {}
        return (
            list(first.get("train", [])),
            list(first.get("val", [])),
            list(first.get("test", [])),
        )
    return (
        split.get("train_subjects", []),
        split.get("val_subjects", []),
        split.get("test_subjects", []),
    )


def extract_channel(img_nib: nib.Nifti1Image, channel_idx: int) -> np.ndarray:
    data = img_nib.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        if data.shape[-1] > channel_idx:
            return data[..., channel_idx]
        if data.shape[0] > channel_idx:
            return data[channel_idx]
        raise ValueError(f"Channel index {channel_idx} out of bounds for shape {data.shape}")
    if data.ndim == 3:
        return data
    raise ValueError(f"Unexpected ndim={data.ndim} in {img_nib.get_filename()}")


def remap_labels(lbl: np.ndarray, label_mapping: dict[int, int]) -> np.ndarray:
    remapped = lbl.copy()
    for src, dst in label_mapping.items():
        remapped[remapped == int(src)] = int(dst)
    return remapped


def convert_subject(
    subject_id: str,
    channel_idx: int,
    images_out: Path,
    labels_out: Path | None,
    data_dir: Path,
    label_mapping: dict[int, int],
) -> bool:
    src_img = data_dir / "imagesTr" / f"{subject_id}.nii.gz"
    src_lbl = data_dir / "labelsTr" / f"{subject_id}.nii.gz"

    if not src_img.exists():
        print(f"  [SKIP] {subject_id}: image not found at {src_img}")
        return False

    img_nib = nib.load(str(src_img))
    channel = extract_channel(img_nib, channel_idx)
    affine = img_nib.affine
    header = img_nib.header.copy()
    header.set_data_shape(channel.shape)
    out_img = images_out / f"{subject_id}_0000.nii.gz"
    nib.save(nib.Nifti1Image(channel, affine, header), str(out_img))

    if labels_out is not None:
        if not src_lbl.exists():
            print(f"  [WARN] {subject_id}: label not found, skipping")
        else:
            lbl_nib = nib.load(str(src_lbl))
            lbl = np.round(lbl_nib.get_fdata()).astype(np.uint8)
            lbl = remap_labels(lbl, label_mapping)
            nib.save(
                nib.Nifti1Image(lbl, lbl_nib.affine, lbl_nib.header),
                str(labels_out / f"{subject_id}.nii.gz"),
            )

    return True


def _labels_payload(num_classes: int) -> dict[str, int]:
    payload = {"background": 0}
    for idx in range(1, int(num_classes)):
        payload[f"class_{idx}"] = idx
    return payload


def write_dataset_json(dataset_dir: Path, dataset_name: str, channel_name: str, n_training: int, num_classes: int) -> None:
    payload = {
        "channel_names": {"0": channel_name},
        "labels": _labels_payload(num_classes),
        "numTraining": n_training,
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": f"Converted single-channel export ({channel_name}) for nnUNetv2.",
        "reference": "Project internal conversion",
        "licence": "Unknown",
        "release": "2026",
    }
    out = dataset_dir / "dataset.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  Wrote {out}  (numTraining={n_training})")


def main() -> None:
    args = parse_args()
    dataset_key = str(args.dataset).strip().lower()
    if dataset_key == "brats":
        dataset_name_hint = str(args.dataset_name or "").strip().lower()
        task_name_hint = str(args.task_name or "").strip().lower()
        dataset_id_hint = str(args.dataset_id or "").strip()
        if (
            dataset_name_hint == "spiderspine"
            or task_name_hint == "spiderspine"
            or dataset_name_hint == "spider_spine"
            or task_name_hint == "spider_spine"
            or dataset_id_hint == "102"
        ):
            dataset_key = "spider_spine"
    dataset_spec = get_dataset_spec(dataset_key)

    task_name = str(args.task_name or dataset_spec.get("task_name", "Task01_BrainTumour"))
    data_dir = PROJECT_ROOT / "data" / task_name
    split_file = Path(args.split_file) if args.split_file else (PROJECT_ROOT / "splits" / "brats_subject_split.json")

    available_contrasts = [str(c).strip().lower() for c in dataset_spec.get("contrasts", [])]
    contrast = str(args.contrast).strip().lower()
    if contrast not in available_contrasts:
        if dataset_key == "spider_spine" and available_contrasts:
            contrast = available_contrasts[0]
        else:
            valid = ", ".join(available_contrasts)
            raise ValueError(f"Unsupported contrast '{args.contrast}' for dataset '{dataset_key}'. Expected one of: {valid}")

    channel_idx = available_contrasts.index(contrast)
    dataset_id = str(args.dataset_id or get_dataset_nnunet_id(dataset_key, contrast))
    channel_name = str(args.channel_name or _pretty_channel_name(contrast))
    dataset_name = str(args.dataset_name or _default_dataset_name(dataset_key, dataset_id, channel_name))
    label_mapping = {int(k): int(v) for k, v in dict(dataset_spec.get("label_mapping") or {}).items()}
    num_classes = int(dataset_spec.get("num_classes", 2))

    dataset_dir = nnUNet_raw / dataset_name
    train_subjects, val_subjects, test_subjects = load_split(split_file)
    labeled = train_subjects + val_subjects

    print("=" * 60)
    print(f"Dataset key : {dataset_key}")
    print(f"Task name   : {task_name}")
    print(f"Contrast    : {contrast} (channel {channel_idx})")
    print(f"Dataset ID  : {dataset_id}")
    print(f"Dataset name: {dataset_name}")
    print(f"Output      : {dataset_dir}")
    print(f"imagesTr/labelsTr: {len(labeled)} subjects")
    print(f"imagesTs         : {len(test_subjects)} subjects")
    print("=" * 60)

    for d in [nnUNet_raw, nnUNet_preprocessed, nnUNet_results]:
        d.mkdir(parents=True, exist_ok=True)

    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    for d in [images_tr, labels_tr, images_ts]:
        d.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    print("\nConverting train+val subjects -> imagesTr / labelsTr ...")
    for subj in labeled:
        ok = convert_subject(
            subj,
            channel_idx,
            images_out=images_tr,
            labels_out=labels_tr,
            data_dir=data_dir,
            label_mapping=label_mapping,
        )
        if ok:
            n_ok += 1
        if n_ok % 50 == 0 and n_ok > 0:
            print(f"  {n_ok}/{len(labeled)} done")
    print(f"  Finished: {n_ok}/{len(labeled)} OK")

    n_ts_ok = 0
    print("\nConverting test subjects -> imagesTs (images only) ...")
    for subj in test_subjects:
        ok = convert_subject(
            subj,
            channel_idx,
            images_out=images_ts,
            labels_out=None,
            data_dir=data_dir,
            label_mapping=label_mapping,
        )
        if ok:
            n_ts_ok += 1
    print(f"  Finished: {n_ts_ok}/{len(test_subjects)} OK")

    print("\nWriting dataset.json ...")
    write_dataset_json(dataset_dir, dataset_name, channel_name, n_ok, num_classes)

    print("\nConversion complete.")
    print(f"\nNext step:\n  nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity")


if __name__ == "__main__":
    main()
