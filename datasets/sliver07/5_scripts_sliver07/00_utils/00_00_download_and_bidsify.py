#!/usr/bin/env python3
"""
Download SLIVER07 (MICCAI 2007 liver CT segmentation challenge) from Zenodo and
convert the MetaImage (.mhd/.raw) volumes → NIfTI BIDS.

SLIVER07 is EVALUATION-ONLY here (see datasets/sliver07/README.md): the 20 labeled
"training" CT volumes become OUR test set for MR→CT generalization of chaos-trained
models. The 10 unlabeled "test-scans" (no public GT) are NOT used and NOT downloaded.

Raw layout (Zenodo record 2597908, training-scans.zip + training-labels.zip):
  0_raw_sliver07/
    scan/  liver-orig0NN.mhd  liver-orig0NN.raw    (CT, MET_SHORT)
    label/ liver-seg0NN.mhd   liver-seg0NN.raw     (binary liver, MET_CHAR)
The label volume is co-registered with its scan (identical DimSize / spacing).

BIDS output (1_BIDS_sliver07/sliver07-liver/):
  sub-LV{NN}/anat/sub-LV{NN}_CT.nii  (+ .json)              <- liver-orig0NN
  derivatives/manual_masks/sub-LV{NN}/anat/sub-LV{NN}_CT_dseg.nii  <- liver-seg0NN
  dataset_description.json   participants.tsv

Label map matches chaos so chaos-trained predictions score directly:
  0 = background, 1 = liver   (SLIVER07 binary mask → liver = label 1).

Usage:
  python 00_00_download_and_bidsify.py                 # download (if needed) + BIDSify
  python 00_00_download_and_bidsify.py --skip-download  # 0_raw already populated
  python 00_00_download_and_bidsify.py --skip-bids
"""
import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

DATASET_ROOT    = Path(__file__).resolve().parents[2]   # …/datasets/sliver07/
RAW_ROOT        = DATASET_ROOT / "0_raw_sliver07"
SCAN_DIR        = RAW_ROOT / "scan"
LABEL_DIR       = RAW_ROOT / "label"
BIDS_ROOT       = DATASET_ROOT / "1_BIDS_sliver07" / "sliver07-liver"
DERIVATIVES_DIR = BIDS_ROOT / "derivatives" / "manual_masks"

ZENODO_RECORD = "2597908"
ZENODO_BASE   = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"
# Only the labeled half + license. The 10 unlabeled test-scans are intentionally skipped.
ZENODO_FILES  = ("training-labels.zip", "training-scans.zip", "license.txt")

# Label map (also written into the BIDS derivatives description). Matches chaos: liver = 1.
LABELS = {"background": 0, "liver": 1}


# ── download ──────────────────────────────────────────────────────────────────

def download(raw_root: Path) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    for fname in ZENODO_FILES:
        dst = raw_root / fname
        if dst.exists() and dst.stat().st_size > 0:
            print(f"  {fname}: already present → {dst}")
            continue
        url = f"{ZENODO_BASE}/{fname}/content"
        print(f"  downloading {fname} …")
        urllib.request.urlretrieve(url, dst)
        print(f"    → {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    # Extract scans/labels (zips lay out into scan/ and label/).
    for fname in ("training-labels.zip", "training-scans.zip"):
        zp = raw_root / fname
        if not zp.exists():
            continue
        print(f"  extracting {fname} …")
        with zipfile.ZipFile(zp) as z:
            z.extractall(raw_root)
    print(f"  Raw download done (10 unlabeled test-scans skipped) → {raw_root}")


# ── helpers ─────────────────────────────────────────────────────────────────

def write_nii(img: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _sidecar(descr: str) -> dict:
    return {"Modality": "CT", "SeriesDescription": descr, "SLIVER07Source": True}


# ── BIDSify ──────────────────────────────────────────────────────────────────

def case_ids(scan_dir: Path) -> list[int]:
    ids = sorted(int(p.stem.replace("liver-orig", "")) for p in scan_dir.glob("liver-orig*.mhd"))
    if not ids:
        print(f"ERROR: no liver-orig*.mhd under {scan_dir}", file=sys.stderr)
        sys.exit(1)
    return ids


def bidsify_case(n: int, participants: list[dict]) -> None:
    label = f"LV{n:02d}"
    sub = f"sub-{label}"
    anat = BIDS_ROOT / sub / "anat"
    deriv = DERIVATIVES_DIR / sub / "anat"

    scan_mhd = SCAN_DIR / f"liver-orig{n:03d}.mhd"
    seg_mhd  = LABEL_DIR / f"liver-seg{n:03d}.mhd"
    if not scan_mhd.exists() or not seg_mhd.exists():
        raise FileNotFoundError(f"{label}: missing scan/label ({scan_mhd.name} / {seg_mhd.name})")

    img = sitk.ReadImage(str(scan_mhd))
    seg = sitk.ReadImage(str(seg_mhd))

    # Binary liver mask → label 1 (matches chaos LABELS["liver"]). Copy scan geometry so
    # GT and prediction share an identical grid (label is already co-registered, but we
    # re-stamp the scan's header to guard against tiny float drift between the two .mhd).
    arr = (sitk.GetArrayFromImage(seg) > 0).astype(np.uint8) * LABELS["liver"]
    if arr.shape != sitk.GetArrayFromImage(img).shape:
        raise ValueError(f"{label}: scan/label shape mismatch "
                         f"{sitk.GetArrayFromImage(img).shape} vs {arr.shape}")
    lab = sitk.GetImageFromArray(arr)
    lab.CopyInformation(img)

    write_nii(img, anat / f"{sub}_CT.nii")
    write_json(anat / f"{sub}_CT.json", _sidecar("SLIVER07 contrast-enhanced abdominal CT"))
    write_nii(lab, deriv / f"{sub}_CT_dseg.nii")

    participants.append({"label": label, "sliver07_id": f"{n:03d}"})


def bidsify() -> None:
    BIDS_ROOT.mkdir(parents=True, exist_ok=True)
    _write_dataset_description()
    _write_derivatives_description()

    ids = case_ids(SCAN_DIR)
    print(f"  BIDSifying {len(ids)} CT patients …")
    participants: list[dict] = []
    for n in ids:
        bidsify_case(n, participants)
        print(f"    LV{n:02d} done")

    _write_participants_tsv(participants)
    print(f"  BIDSify done → {BIDS_ROOT}")


def _write_dataset_description() -> None:
    write_json(BIDS_ROOT / "dataset_description.json", {
        "Name": "SLIVER07 — Segmentation of the Liver Competition 2007 (labeled CT half)",
        "BIDSVersion": "1.9.0",
        "License": "SLIVER07 Challenge Rules (see 0_raw_sliver07/license.txt)",
        "Authors": ["T. Heimann", "B. van Ginneken", "M. A. Styner", "et al."],
        "ReferencesAndLinks": [
            "https://sliver07.grand-challenge.org/",
            "https://zenodo.org/records/2597908",
            "Heimann et al., IEEE TMI 28(8):1251-1265, 2009",
        ],
        "DatasetType": "raw",
        "Usage": "EVALUATION ONLY — test set for chaos-trained models (MR→CT generalization).",
    })


def _write_derivatives_description() -> None:
    write_json(DERIVATIVES_DIR / "dataset_description.json", {
        "Name": "SLIVER07 Manual Liver Masks",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "SLIVER07 Challenge"}],
        "LabelMap": LABELS,
    })


def _write_participants_tsv(participants: list[dict]) -> None:
    lines = ["participant_id\tmodality\tsliver07_id\tsplit"]
    for p in sorted(participants, key=lambda x: x["label"]):
        # split=test: SLIVER07 is only ever used as a test set here.
        lines.append(f"sub-{p['label']}\tCT\t{p['sliver07_id']}\ttest")
    (BIDS_ROOT / "participants.tsv").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true",
                    help="skip Zenodo download/extract (0_raw already populated)")
    ap.add_argument("--skip-bids", action="store_true", help="skip BIDS conversion")
    args = ap.parse_args()

    print("=" * 64)
    print("SLIVER07 — Download (labeled half) + MetaImage→NIfTI BIDSify")
    print("=" * 64)

    if not args.skip_download:
        print(f"\n[1/2] Downloading from Zenodo → {RAW_ROOT}")
        download(RAW_ROOT)
    else:
        print("\n[1/2] Skipping download (--skip-download)")

    if not args.skip_bids:
        print(f"\n[2/2] MetaImage→NIfTI BIDSifying → {BIDS_ROOT}")
        bidsify()
    else:
        print("\n[2/2] Skipping BIDS conversion (--skip-bids)")


if __name__ == "__main__":
    main()
