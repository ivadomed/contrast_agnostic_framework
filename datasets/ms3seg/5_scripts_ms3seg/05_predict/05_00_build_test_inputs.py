#!/usr/bin/env python3
"""
Build nnUNet test inputs for MS3SEG (FLAIR/T1w/T2w). Mirrors mslesseg's build script.
Run AFTER 03_preprocess/03_00_reorient_to_lps.py.
"""
import os
import shutil
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_ms3seg" / "ms3seg-brain"
DERIV_DIR    = BIDS_ROOT / "derivatives" / "manual_masks"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_ms3seg" / "raw"

CONTRASTS = {"flair": "FLAIR", "t1w": "T1w", "t2w": "T2w"}


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main() -> None:
    subs = sorted(p.name for p in BIDS_ROOT.glob("sub-*") if p.is_dir())
    if not subs:
        raise SystemExit(f"No sub-* under {BIDS_ROOT} -- run 00_00_ingest_and_bidsify.py first.")

    for contrast, bids_suffix in CONTRASTS.items():
        img_dir = NNUNET_RAW / f"imagesTs_{contrast}"
        lab_dir = NNUNET_RAW / f"labelsTs_{contrast}"
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)

        n_ok, missing = 0, []
        for sub in subs:
            case_id = sub[len("sub-"):]
            img_src = BIDS_ROOT / sub / "anat" / f"{sub}_{bids_suffix}.nii.gz"
            mask_src = DERIV_DIR / sub / "anat" / f"{sub}_FLAIR_dseg.nii.gz"
            if not img_src.exists() or not mask_src.exists():
                missing.append(case_id)
                continue
            link_or_copy(img_src, img_dir / f"{case_id}_0000.nii.gz")
            link_or_copy(mask_src, lab_dir / f"{case_id}.nii.gz")
            n_ok += 1

        status = f"{n_ok}/{len(subs)} -> {img_dir.name} (native hardlink, +labels)"
        if missing:
            status += f"  [MISSING {len(missing)}: {missing[:5]}]"
        print(f"  {contrast:5s}: {status}")


if __name__ == "__main__":
    main()
