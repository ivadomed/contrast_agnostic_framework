#!/usr/bin/env python3
"""
Build per-modality nnUNet test-input dirs for AMOS evaluation.

Reads from the BIDS layout (1_BIDS_amos/amos-abdominal/) and selects only
the validation-split cases (split=val in participants.tsv: 100 CT + 20 MRI).
The training split is reserved for future native AMOS training and is excluded.

ID convention:  CT  = amos_0001 … amos_0499  (id < 500)
                MRI = amos_0500 … amos_0599  (id >= 500)

Each case is written as a single-channel input (_0000) to match the chaos
model's single-channel expectation (trained on MR T1-DUAL in-phase).

Reads:
  1_BIDS_amos/amos-abdominal/sub-AM{id:04d}/anat/sub-AM{id:04d}_{CT|T2w}.nii.gz
  1_BIDS_amos/amos-abdominal/derivatives/manual_masks/
    sub-AM{id:04d}/anat/sub-AM{id:04d}_{CT|T2w}_dseg.nii.gz

Writes:
  2_nnUNet_amos/raw/imagesTs_ct/{case}_0000.nii.gz
                    labelsTs_ct/{case}.nii.gz
                    imagesTs_mri/{case}_0000.nii.gz
                    labelsTs_mri/{case}.nii.gz

Usage:
    python 05_00_build_test_inputs.py                    # all modalities
    python 05_00_build_test_inputs.py --modalities ct
    python 05_00_build_test_inputs.py --clean            # remove stale training-set
                                                         # files from a prior run
"""
import argparse
import csv
from pathlib import Path

DATASET_ROOT    = Path(__file__).resolve().parents[2]
BIDS_ROOT       = DATASET_ROOT / "1_BIDS_amos" / "amos-abdominal"
DERIV_DIR       = BIDS_ROOT / "derivatives" / "manual_masks"
AMOS22          = DATASET_ROOT / "0_raw_amos" / "amos22"
NNUNET_RAW      = DATASET_ROOT / "2_nnUNet_amos" / "raw"

ALL_MODALITIES  = ("ct", "mri")


def _is_ct(case_id: int) -> bool:
    return case_id < 500


def _bids_suffix(case_id: int) -> str:
    return "CT" if _is_ct(case_id) else "T2w"


def _val_cases_from_bids():
    """Yield (case_id, modality_str, img_path, lab_path) for val-split BIDS cases."""
    tsv = BIDS_ROOT / "participants.tsv"
    if not tsv.exists():
        raise SystemExit(
            f"ERROR: {tsv} not found.\n"
            f"Run 00_01_bidsify.py first to populate 1_BIDS_amos/."
        )
    with tsv.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["split"] != "val":
                continue
            sub     = row["participant_id"]
            case_id = int(row["amos22_id"].replace("amos_", ""))
            suffix  = _bids_suffix(case_id)
            mod     = "ct" if _is_ct(case_id) else "mri"
            img     = BIDS_ROOT / sub / "anat" / f"{sub}_{suffix}.nii.gz"
            lab     = DERIV_DIR / sub / "anat"  / f"{sub}_{suffix}_dseg.nii.gz"
            if img.exists() and lab.exists():
                yield case_id, mod, img, lab
            else:
                print(f"  WARN: BIDS files missing for {sub} — skipping")


def _link(src: Path, dst: Path) -> None:
    """Hard-link src → dst; fall back to copy if cross-device. Skip if dst exists."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.hardlink_to(src)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def _training_case_ids() -> set[int]:
    """Return case IDs present in imagesTr (must not be in test dirs)."""
    ids: set[int] = set()
    img_dir = AMOS22 / "imagesTr"
    if img_dir.exists():
        for f in img_dir.glob("amos_*.nii.gz"):
            ids.add(int(f.name.replace(".nii.gz", "").replace("amos_", "")))
    return ids


def clean_training_cases(modalities: list[str]) -> None:
    """Remove any files in test dirs that belong to training cases (old policy)."""
    train_ids = _training_case_ids()
    if not train_ids:
        print("  clean: no imagesTr found, nothing to remove")
        return
    for mod in modalities:
        for subdir in (f"imagesTs_{mod}", f"labelsTs_{mod}"):
            d = NNUNET_RAW / subdir
            if not d.exists():
                continue
            removed = 0
            for f in list(d.glob("amos_*.nii.gz")):
                stem = f.name.replace(".nii.gz", "").replace("_0000", "")
                try:
                    cid = int(stem.replace("amos_", ""))
                except ValueError:
                    continue
                if cid in train_ids:
                    f.unlink()
                    removed += 1
            if removed:
                print(f"  clean {subdir}: removed {removed} training-set files")
            else:
                print(f"  clean {subdir}: no training-set files found")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modalities", nargs="+", default=list(ALL_MODALITIES),
                    choices=list(ALL_MODALITIES))
    ap.add_argument("--clean", action="store_true",
                    help="Remove training-set cases from test dirs before rebuilding "
                         "(fixes dirs built by the old train+val policy)")
    args = ap.parse_args()

    if args.clean:
        print("Removing training cases from test dirs...")
        clean_training_cases(args.modalities)

    counts: dict = {m: 0 for m in args.modalities}

    for case_id, mod, img, lab in _val_cases_from_bids():
        if mod not in args.modalities:
            continue
        case    = f"amos_{case_id:04d}"
        img_dir = NNUNET_RAW / f"imagesTs_{mod}"
        lab_dir = NNUNET_RAW / f"labelsTs_{mod}"
        _link(img, img_dir / f"{case}_0000.nii.gz")
        _link(lab, lab_dir / f"{case}.nii.gz")
        counts[mod] += 1

    for mod in args.modalities:
        print(f"  {mod:4s}: {counts[mod]} val cases → imagesTs_{mod}/ (+labelsTs_{mod}/)")


if __name__ == "__main__":
    main()
