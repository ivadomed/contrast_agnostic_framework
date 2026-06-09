#!/usr/bin/env python3
"""
Download BraTS 2024 glioma dataset from Kaggle and convert to BIDS format.

Steps:
  1. Download via kagglehub (cached in ~/.cache/kagglehub/)
  2. Copy raw data → 0_raw_brats2024-glioma/
  3. BIDSify        → 1_BIDS_brats2024-glioma/glioma-brain-brats2024/

Raw data layout (BraTS 2024 convention):
  <kagglehub_cache>/
    BraTS-GLI-XXXXX-XXX/
      BraTS-GLI-XXXXX-XXX-t1c.nii    # T1w with contrast (gadolinium)
      BraTS-GLI-XXXXX-XXX-t1n.nii    # T1w native (no contrast)
      BraTS-GLI-XXXXX-XXX-t2f.nii    # T2 FLAIR
      BraTS-GLI-XXXXX-XXX-t2w.nii    # T2w
      BraTS-GLI-XXXXX-XXX-seg.nii    # segmentation label (training only)

BIDS output layout:
  1_BIDS_brats2024-glioma/glioma-brain-brats2024/
    dataset_description.json
    participants.tsv
    sub-<label>/anat/
      sub-<label>_T1w.nii                  (t1n)
      sub-<label>_T1w.json
      sub-<label>_ce-gadolinium_T1w.nii    (t1c)
      sub-<label>_ce-gadolinium_T1w.json
      sub-<label>_T2w.nii                  (t2w)
      sub-<label>_T2w.json
      sub-<label>_FLAIR.nii                (t2f)
      sub-<label>_FLAIR.json
    derivatives/manual_masks/
      sub-<label>/anat/
        sub-<label>_T1w_seg.nii            (seg)

Usage:
  .venv/bin/python 01_00_download_and_bidsify.py [--skip-copy] [--skip-bids]
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import kagglehub

DATASET_ROOT = Path(__file__).resolve().parents[2]  # …/datasets/brats2024-glioma/
RAW_ROOT = DATASET_ROOT / "0_raw_brats2024-glioma"
BIDS_ROOT = DATASET_ROOT / "1_BIDS_brats2024-glioma" / "glioma-brain-brats2024"
DERIVATIVES_DIR = BIDS_ROOT / "derivatives" / "manual_masks"

KAGGLE_DATASET = "i212385nomanarif/2024-brats-glioma"

MODALITY_MAP = {
    "t1n": "_T1w",
    "t1c": "_ce-gadolinium_T1w",
    "t2w": "_T2w",
    "t2f": "_FLAIR",
}

SUBJECT_RE = re.compile(r"^BraTS-GLI-\d{5}-\d{3}$")


def brats_id_to_bids_label(brats_id: str) -> str:
    """'BraTS-GLI-00001-000' → 'BraTSGLI00001000'"""
    return re.sub(r"[^A-Za-z0-9]", "", brats_id)


def make_sidecar(bids_suffix: str) -> dict:
    base: dict = {"BraTS2024Source": True}
    if "gadolinium" in bids_suffix:
        base["ContrastBolusIngredient"] = "GADOLINIUM"
    return base


def find_subject_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_dir() and SUBJECT_RE.match(p.name))


def copy_raw(kaggle_path: Path, raw_root: Path):
    """Copy raw BraTS subjects from kagglehub cache → 0_raw_brats2024-glioma."""
    subjects = find_subject_dirs(kaggle_path)
    if not subjects:
        print(f"ERROR: no BraTS-GLI-* dirs found under {kaggle_path}")
        sys.exit(1)
    print(f"  Copying {len(subjects)} subjects to {raw_root} …")
    raw_root.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(subjects):
        dst = raw_root / src.name
        if dst.exists():
            continue
        shutil.copytree(src, dst)
        if (i + 1) % 100 == 0 or (i + 1) == len(subjects):
            print(f"    {i + 1}/{len(subjects)}")
    print(f"  Raw copy done → {raw_root}")


def bidsify(raw_root: Path, bids_root: Path, deriv_dir: Path):
    subjects = find_subject_dirs(raw_root)
    if not subjects:
        print(f"ERROR: no BraTS-GLI-* dirs found under {raw_root}")
        sys.exit(1)

    bids_root.mkdir(parents=True, exist_ok=True)
    deriv_dir.mkdir(parents=True, exist_ok=True)

    _write_dataset_description(bids_root)
    _write_derivatives_description(deriv_dir)

    participants = []
    print(f"  BIDSifying {len(subjects)} subjects …")
    for i, subj_dir in enumerate(subjects):
        brats_id = subj_dir.name
        bids_label = brats_id_to_bids_label(brats_id)
        sub = f"sub-{bids_label}"
        anat_dir = bids_root / sub / "anat"
        deriv_anat = deriv_dir / sub / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)

        for src in sorted(subj_dir.glob("*.nii")):
            modality = src.stem.split("-")[-1]  # e.g. "t1c", "seg"
            if modality == "seg":
                deriv_anat.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, deriv_anat / f"{sub}_dseg.nii")
            elif modality in MODALITY_MAP:
                bids_suffix = MODALITY_MAP[modality]
                shutil.copy2(src, anat_dir / f"{sub}{bids_suffix}.nii")
                (anat_dir / f"{sub}{bids_suffix}.json").write_text(
                    json.dumps(make_sidecar(bids_suffix), indent=2)
                )
            else:
                print(f"  [WARN] unknown modality '{modality}' in {src.name}")

        participants.append({"bids_label": bids_label, "brats_id": brats_id})
        if (i + 1) % 100 == 0 or (i + 1) == len(subjects):
            print(f"    {i + 1}/{len(subjects)}")

    _write_participants_tsv(bids_root, participants)
    print(f"  BIDSify done → {bids_root}")


def _write_dataset_description(bids_root: Path):
    (bids_root / "dataset_description.json").write_text(json.dumps({
        "Name": "BraTS 2024 Glioma",
        "BIDSVersion": "1.9.0",
        "License": "CC BY 4.0",
        "Authors": ["BraTS 2024 Challenge Organizers"],
        "ReferencesAndLinks": [
            "https://www.synapse.org/#!Synapse:syn51156910/wiki/",
            "https://www.kaggle.com/datasets/i212385nomanarif/2024-brats-glioma",
        ],
        "DatasetType": "raw",
    }, indent=2))


def _write_derivatives_description(deriv_dir: Path):
    (deriv_dir / "dataset_description.json").write_text(json.dumps({
        "Name": "BraTS 2024 Glioma Manual Masks",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "BraTS 2024 Challenge"}],
    }, indent=2))


def _write_participants_tsv(bids_root: Path, participants: list[dict]):
    lines = ["participant_id\tbrats_id"]
    for p in participants:
        lines.append(f"sub-{p['bids_label']}\t{p['brats_id']}")
    (bids_root / "participants.tsv").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-copy", action="store_true",
                        help="skip copying raw data (0_raw already populated)")
    parser.add_argument("--skip-bids", action="store_true",
                        help="skip BIDS conversion")
    args = parser.parse_args()

    print("=" * 60)
    print("BraTS 2024 Glioma — Download + Copy Raw + BIDSify")
    print("=" * 60)

    # 1. Download (returns cached path if already done)
    print("\n[1/3] Downloading from Kaggle Hub …")
    kaggle_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"  Kaggle cache: {kaggle_path}")

    # 2. Copy to 0_raw
    if not args.skip_copy:
        print(f"\n[2/3] Copying raw data → {RAW_ROOT}")
        copy_raw(kaggle_path, RAW_ROOT)
    else:
        print(f"\n[2/3] Skipping raw copy (--skip-copy)")

    # 3. BIDSify
    if not args.skip_bids:
        print(f"\n[3/3] BIDSifying → {BIDS_ROOT}")
        bidsify(RAW_ROOT, BIDS_ROOT, DERIVATIVES_DIR)
    else:
        print(f"\n[3/3] Skipping BIDS conversion (--skip-bids)")

    print("\nAll done.")
    print(f"  Raw : {RAW_ROOT}")
    print(f"  BIDS: {BIDS_ROOT}")


if __name__ == "__main__":
    main()
