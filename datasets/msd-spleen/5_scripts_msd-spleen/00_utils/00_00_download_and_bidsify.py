#!/usr/bin/env python3
"""
Download the Medical Segmentation Decathlon Task09_Spleen (CT spleen segmentation)
and convert its NIfTI volumes -> BIDS.

MSD-SPLEEN is EVALUATION-ONLY here (see datasets/msd-spleen/README.md): the 41
labeled "training" CT volumes become OUR test set for MR->CT generalization of
chaos-trained models. The 20 unlabeled "test" volumes (no public GT) are NOT used
and NOT extracted.

Raw layout (msd-for-monai S3 mirror, Task09_Spleen.tar, CC-BY-SA 4.0):
  0_raw_msd-spleen/
    Task09_Spleen/imagesTr/spleen_NN.nii.gz   (CT, HU)
    Task09_Spleen/labelsTr/spleen_NN.nii.gz   (binary spleen mask)
    Task09_Spleen/dataset.json
The label volume shares the image's grid (MSD ships them pre-registered).

BIDS output (1_BIDS_msd-spleen/msd-spleen/):
  sub-SP{NN}/anat/sub-SP{NN}_CT.nii  (+ .json)              <- imagesTr/spleen_NN
  derivatives/manual_masks/sub-SP{NN}/anat/sub-SP{NN}_CT_dseg.nii  <- labelsTr/spleen_NN
  dataset_description.json   participants.tsv

Label map matches chaos so chaos-trained predictions score directly:
  0 = background, 1 = spleen   (MSD binary mask -> spleen = label 1; chaos itself
  emits spleen as label 4 — 06_evaluate maps chaos id 4 -> this GT's id 1).

Usage:
  python 00_00_download_and_bidsify.py                 # download (if needed) + BIDSify
  python 00_00_download_and_bidsify.py --skip-download  # 0_raw already populated
  python 00_00_download_and_bidsify.py --skip-bids
"""
import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import SimpleITK as sitk

DATASET_ROOT    = Path(__file__).resolve().parents[2]   # …/datasets/msd-spleen/
RAW_ROOT        = DATASET_ROOT / "0_raw_msd-spleen"
TASK_DIR        = RAW_ROOT / "Task09_Spleen"
IMAGES_DIR      = TASK_DIR / "imagesTr"
LABELS_DIR      = TASK_DIR / "labelsTr"
BIDS_ROOT       = DATASET_ROOT / "1_BIDS_msd-spleen" / "msd-spleen"
DERIVATIVES_DIR = BIDS_ROOT / "derivatives" / "manual_masks"

# Public, unauthenticated mirror used by MONAI's DecathlonDataset (confirmed reachable,
# 1,610,352,640 bytes, Content-Type application/x-tar, no WAF/auth gate — unlike the
# springernature-hosted TRUSTED download, this needed no workaround).
TAR_URL  = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
TAR_NAME = "Task09_Spleen.tar"
# Only the labeled half is extracted. imagesTs/ (20 unlabeled test volumes, no public
# GT) is intentionally skipped — mirrors the sliver07/chaos sealed-test policy.
EXTRACT_MEMBERS_PREFIXES = ("Task09_Spleen/imagesTr/", "Task09_Spleen/labelsTr/",
                            "Task09_Spleen/dataset.json")

# Label map (also written into the BIDS derivatives description). Matches chaos: spleen = 1
# in THIS dataset's own GT numbering (chaos's own spleen id, 4, is remapped at eval time).
LABELS = {"background": 0, "spleen": 1}


# ── download ──────────────────────────────────────────────────────────────────

def download(raw_root: Path) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    dst = raw_root / TAR_NAME
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  {TAR_NAME}: already present -> {dst}")
    else:
        print(f"  downloading {TAR_NAME} (~1.5 GB) …")
        urllib.request.urlretrieve(TAR_URL, dst)
        print(f"    -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")

    print(f"  extracting labeled half (imagesTr/labelsTr/dataset.json) …")
    with tarfile.open(dst) as tf:
        members = [m for m in tf.getmembers()
                   if m.name.startswith(EXTRACT_MEMBERS_PREFIXES)]
        tf.extractall(raw_root, members=members)
    print(f"  Raw download done (20 unlabeled test volumes skipped) -> {raw_root}")


# ── helpers ─────────────────────────────────────────────────────────────────

def write_nii(img: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _sidecar(descr: str) -> dict:
    return {"Modality": "CT", "SeriesDescription": descr, "MSDSpleenSource": True}


# ── BIDSify ──────────────────────────────────────────────────────────────────

def case_ids(images_dir: Path) -> list[int]:
    # MSD case IDs are non-contiguous (e.g. spleen_2, spleen_3, spleen_6, …) — some
    # numbers are reserved for the unlabeled test split. Skip macOS resource-fork
    # junk (._spleen_NN.nii.gz) that some tar producers include.
    ids = sorted(int(p.stem.replace(".nii", "").replace("spleen_", ""))
                 for p in images_dir.glob("spleen_*.nii.gz")
                 if not p.name.startswith("."))
    if not ids:
        print(f"ERROR: no spleen_*.nii.gz under {images_dir}", file=sys.stderr)
        sys.exit(1)
    return ids


def bidsify_case(n: int, participants: list[dict]) -> None:
    label = f"SP{n:02d}"
    sub = f"sub-{label}"
    anat = BIDS_ROOT / sub / "anat"
    deriv = DERIVATIVES_DIR / sub / "anat"

    img_nii = IMAGES_DIR / f"spleen_{n}.nii.gz"
    seg_nii = LABELS_DIR / f"spleen_{n}.nii.gz"
    if not img_nii.exists() or not seg_nii.exists():
        raise FileNotFoundError(f"{label}: missing image/label ({img_nii.name})")

    img = sitk.ReadImage(str(img_nii))
    seg = sitk.ReadImage(str(seg_nii))

    # Binary spleen mask -> label 1 (matches LABELS["spleen"]). Copy the image's
    # geometry onto the relabeled mask so GT and prediction share an identical grid.
    arr = (sitk.GetArrayFromImage(seg) > 0).astype(np.uint8) * LABELS["spleen"]
    img_arr_shape = sitk.GetArrayFromImage(img).shape
    if arr.shape != img_arr_shape:
        raise ValueError(f"{label}: image/label shape mismatch {img_arr_shape} vs {arr.shape}")
    lab = sitk.GetImageFromArray(arr)
    lab.CopyInformation(img)

    write_nii(img, anat / f"{sub}_CT.nii")
    write_json(anat / f"{sub}_CT.json", _sidecar("MSD Task09_Spleen contrast-enhanced abdominal CT"))
    write_nii(lab, deriv / f"{sub}_CT_dseg.nii")

    participants.append({"label": label, "msd_id": str(n)})


def bidsify() -> None:
    BIDS_ROOT.mkdir(parents=True, exist_ok=True)
    _write_dataset_description()
    _write_derivatives_description()

    ids = case_ids(IMAGES_DIR)
    print(f"  BIDSifying {len(ids)} CT patients …")
    participants: list[dict] = []
    for n in ids:
        bidsify_case(n, participants)
        print(f"    SP{n:02d} done")

    _write_participants_tsv(participants)
    print(f"  BIDSify done -> {BIDS_ROOT}")


def _write_dataset_description() -> None:
    write_json(BIDS_ROOT / "dataset_description.json", {
        "Name": "Medical Segmentation Decathlon Task09_Spleen (labeled CT half)",
        "BIDSVersion": "1.9.0",
        "License": "CC-BY-SA 4.0",
        "Authors": ["Antonelli, M.", "Reinke, A.", "Bakas, S.", "et al. (MSD consortium)"],
        "ReferencesAndLinks": [
            "http://medicaldecathlon.com/",
            "https://arxiv.org/abs/1902.09063",
            "Antonelli et al., Nature Communications 13, 4128 (2022)",
        ],
        "DatasetType": "raw",
        "Usage": "EVALUATION ONLY — test set for chaos-trained models (MR->CT generalization, spleen label).",
    })


def _write_derivatives_description() -> None:
    write_json(DERIVATIVES_DIR / "dataset_description.json", {
        "Name": "MSD Task09_Spleen Manual Spleen Masks",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "Medical Segmentation Decathlon"}],
        "LabelMap": LABELS,
    })


def _write_participants_tsv(participants: list[dict]) -> None:
    lines = ["participant_id\tmodality\tmsd_id\tsplit"]
    for p in sorted(participants, key=lambda x: int(x["msd_id"])):
        # split=test: MSD-SPLEEN is only ever used as a test set here.
        lines.append(f"sub-{p['label']}\tCT\t{p['msd_id']}\ttest")
    (BIDS_ROOT / "participants.tsv").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true",
                    help="skip S3 download/extract (0_raw already populated)")
    ap.add_argument("--skip-bids", action="store_true", help="skip BIDS conversion")
    args = ap.parse_args()

    print("=" * 64)
    print("MSD Task09_Spleen — Download (labeled half) + NIfTI->BIDSify")
    print("=" * 64)

    if not args.skip_download:
        print(f"\n[1/2] Downloading from S3 -> {RAW_ROOT}")
        download(RAW_ROOT)
    else:
        print("\n[1/2] Skipping download (--skip-download)")

    if not args.skip_bids:
        print(f"\n[2/2] NIfTI->BIDSifying -> {BIDS_ROOT}")
        bidsify()
    else:
        print("\n[2/2] Skipping BIDS conversion (--skip-bids)")


if __name__ == "__main__":
    main()
