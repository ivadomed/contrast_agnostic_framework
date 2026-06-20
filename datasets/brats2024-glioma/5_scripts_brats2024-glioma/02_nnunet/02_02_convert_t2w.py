#!/usr/bin/env python3
"""
Convert BraTS 2024 glioma BIDS data to nnUNet raw format — T2w only.

Produces Dataset052_BraTS2024GliomaT2w with a single channel (T2 weighted / T2w).
Parallel to 02_01_convert_t1n.py (Dataset051, T1n) — same case set, same label source,
single channel replaced.

Usage:
  python 02_02_convert_t2w.py [--dataset-id 052] [--jobs N]

Reads from: ../../1_BIDS_brats2024-glioma/glioma-brain-brats2024/
Writes to:  ../../2_nnUNet_brats2024-glioma/raw/Dataset052_BraTS2024GliomaT2w/
"""

import argparse
import gzip
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]  # …/datasets/brats2024-glioma/
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_brats2024-glioma" / "glioma-brain-brats2024"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_brats2024-glioma" / "raw"
DERIV_DIR    = BIDS_ROOT / "derivatives" / "manual_masks"


def gzip_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    with open(src, "rb") as f_in, gzip.open(dst, "wb", compresslevel=1) as f_out:
        shutil.copyfileobj(f_in, f_out)


def convert_subject(sub_name: str, images_tr: Path, labels_tr: Path) -> str:
    case_id = sub_name.removeprefix("sub-")
    anat_dir = BIDS_ROOT / sub_name / "anat"

    src = anat_dir / f"{sub_name}_T2w.nii"
    if not src.exists():
        raise FileNotFoundError(f"Missing T2w: {src}")
    gzip_copy(src, images_tr / f"{case_id}_0000.nii.gz")

    seg_src = DERIV_DIR / sub_name / "anat" / f"{sub_name}_dseg.nii"
    if not seg_src.exists():
        raise FileNotFoundError(f"Missing seg: {seg_src}")
    gzip_copy(seg_src, labels_tr / f"{case_id}.nii.gz")

    return case_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=int, default=52)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()

    ds_name   = f"Dataset{args.dataset_id:03d}_BraTS2024GliomaT2w"
    out_dir   = NNUNET_RAW / ds_name
    images_tr = out_dir / "imagesTr"
    labels_tr = out_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    subjects = sorted(p.name for p in BIDS_ROOT.glob("sub-*") if p.is_dir())
    print(f"Converting {len(subjects)} subjects (T2w only) with {args.jobs} workers …")

    case_ids = []
    failed   = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(convert_subject, sub, images_tr, labels_tr): sub
                   for sub in subjects}
        for i, fut in enumerate(as_completed(futures), 1):
            sub = futures[fut]
            try:
                case_ids.append(fut.result())
            except Exception as e:
                failed.append((sub, str(e)))
            if i % 100 == 0 or i == len(subjects):
                print(f"  {i}/{len(subjects)}")

    if failed:
        print(f"\nFAILED ({len(failed)}):")
        for sub, err in failed:
            print(f"  {sub}: {err}")
        raise SystemExit(1)

    dataset_json = {
        "name": "BraTS2024GliomaT2w",
        "description": "BraTS 2024 Glioma — T2w only (synthesis comparison from T2w input)",
        "reference": "https://www.synapse.org/#!Synapse:syn51156910/wiki/",
        "licence": "CC BY 4.0",
        "release": "1.0",
        "channel_names": {"0": "T2w"},
        "labels": {"background": 0, "NCR": 1, "SNFH": 2, "ET": 3, "RC": 4},
        "numTraining": len(case_ids),
        "file_ending": ".nii.gz",
    }
    (out_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print(f"\nDone. {len(case_ids)} cases → {out_dir}")


if __name__ == "__main__":
    main()
