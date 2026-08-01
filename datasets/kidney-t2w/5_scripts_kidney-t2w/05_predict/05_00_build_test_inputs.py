#!/usr/bin/env python3
"""
Build the nnUNet test-input dir for the KIDNEY-T2W evaluation set.

KIDNEY-T2W is T2-only (the single contrast available). Chaos models are
single-channel, so each volume is fed as channel _0000 under item "t2".

Reads:  ../../1_BIDS_kidney-t2w/kidney-t2w/sub-*/anat/  (+ derivatives masks)
Writes: ../../2_nnUNet_kidney-t2w/raw/imagesTs_t2/{case}_0000.nii.gz
                                      /labelsTs_t2/{case}.nii.gz

    python 05_00_build_test_inputs.py
"""
import gzip
import shutil
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_kidney-t2w" / "kidney-t2w"
DERIV_DIR    = BIDS_ROOT / "derivatives" / "manual_masks"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_kidney-t2w" / "raw"

MODALITY = "t2"


def gzip_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def main() -> None:
    subs = sorted(p.name for p in BIDS_ROOT.glob("sub-*") if p.is_dir())
    if not subs:
        raise SystemExit(f"No sub-* under {BIDS_ROOT} -- run 00_00_download.py + 00_01_bidsify.py first.")

    img_dir = NNUNET_RAW / f"imagesTs_{MODALITY}"
    lab_dir = NNUNET_RAW / f"labelsTs_{MODALITY}"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    n_ok, missing = 0, []
    for sub in subs:
        case = sub[len("sub-"):]
        img = BIDS_ROOT / sub / "anat" / f"{sub}_T2w.nii.gz"
        seg = DERIV_DIR / sub / "anat" / f"{sub}_T2w_dseg.nii.gz"
        if not img.exists() or not seg.exists():
            missing.append(case)
            continue
        gzip_copy(img, img_dir / f"{case}_0000.nii.gz")
        gzip_copy(seg, lab_dir / f"{case}.nii.gz")
        n_ok += 1

    status = f"{n_ok}/{len(subs)} -> {img_dir.name} (+labels)"
    if missing:
        status += f"  [MISSING {len(missing)}: {missing[:5]}]"
    print(f"  {MODALITY:7s}: {status}")


if __name__ == "__main__":
    main()
