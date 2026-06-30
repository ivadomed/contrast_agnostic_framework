#!/usr/bin/env python3
"""
Convert BraTS 2024 glioma BIDS data to nnUNet raw format.

BIDS files are uncompressed .nii; this script gzips them on the fly.
All 700 subjects have segmentation labels → all go to imagesTr/labelsTr.

Usage:
  python 02_00_convert.py [--dataset-id 050] [--jobs N]

Reads from: ../../1_BIDS_brats2024-glioma/glioma-brain-brats2024/
Writes to:  ../../2_nnUNet_brats2024-glioma/raw/Dataset050_BraTS2024Glioma/
"""

import argparse
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]  # …/datasets/brats2024-glioma/
sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_00_utils"))
from nnunet_convert_lib import gzip_copy, run_threaded_conversion, write_dataset_json  # noqa: E402
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_brats2024-glioma" / "glioma-brain-brats2024"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_brats2024-glioma" / "raw"
DERIV_DIR    = BIDS_ROOT / "derivatives" / "manual_masks"

# BIDS suffix → nnUNet channel index
CHANNEL_MAP = {
    "_T1w.nii":               "0000",   # T1 native
    "_ce-gadolinium_T1w.nii": "0001",   # T1 with contrast
    "_T2w.nii":               "0002",   # T2w
    "_FLAIR.nii":             "0003",   # T2 FLAIR
}


def convert_subject(sub_name: str, images_tr: Path, labels_tr: Path) -> str:
    """Convert one subject; return case_id."""
    case_id = sub_name.removeprefix("sub-")
    anat_dir = BIDS_ROOT / sub_name / "anat"

    for suffix, channel in CHANNEL_MAP.items():
        src = anat_dir / f"{sub_name}{suffix}"
        if not src.exists():
            raise FileNotFoundError(f"Missing: {src}")
        gzip_copy(src, images_tr / f"{case_id}_{channel}.nii.gz")

    seg_src = DERIV_DIR / sub_name / "anat" / f"{sub_name}_dseg.nii"
    if not seg_src.exists():
        raise FileNotFoundError(f"Missing seg: {seg_src}")
    gzip_copy(seg_src, labels_tr / f"{case_id}.nii.gz")

    return case_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=16,
                        help="parallel gzip workers (default 16)")
    args = parser.parse_args()

    ds_name   = f"Dataset{args.dataset_id:03d}_BraTS2024Glioma"
    out_dir   = NNUNET_RAW / ds_name
    images_tr = out_dir / "imagesTr"
    labels_tr = out_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    subjects = sorted(p.name for p in BIDS_ROOT.glob("sub-*") if p.is_dir())
    print(f"Converting {len(subjects)} subjects with {args.jobs} workers …")

    case_ids = run_threaded_conversion(
        subjects, lambda s: convert_subject(s, images_tr, labels_tr),
        args.jobs, progress_every=100)

    write_dataset_json(
        out_dir,
        channel_names={"0": "T1n", "1": "T1c", "2": "T2w", "3": "T2f"},
        labels={"background": 0, "NCR": 1, "SNFH": 2, "ET": 3, "RC": 4},
        num_training=len(case_ids),
        name="BraTS2024Glioma",
        description="BraTS 2024 Glioma Challenge Dataset",
        reference="https://www.synapse.org/#!Synapse:syn51156910/wiki/",
        licence="CC BY 4.0",
        release="1.0",
    )
    print(f"\nDone. {len(case_ids)} cases → {out_dir}")


if __name__ == "__main__":
    main()
