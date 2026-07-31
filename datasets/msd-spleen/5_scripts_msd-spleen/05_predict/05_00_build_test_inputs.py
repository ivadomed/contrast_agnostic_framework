#!/usr/bin/env python3
"""
Build the nnUNet test-input dir for the MSD-SPLEEN evaluation set.

MSD-SPLEEN is CT-only (the single "contrast" available). The chaos models are
single-channel, so each CT volume is fed as channel _0000. This materialises an
image dir + matching GT label dir so 06_evaluate can score predictions.

Reads:  ../../1_BIDS_msd-spleen/msd-spleen/sub-SP##/anat/  (+ derivatives masks)
Writes: ../../2_nnUNet_msd-spleen/raw/imagesTs_ct/{case}_0000.nii.gz
                                      /labelsTs_ct/{case}.nii.gz

Case id = participant label without the sub- prefix (e.g. SP02). The "_ct" suffix
mirrors chaos's per-modality test dirs so 05_predict / 06_evaluate stay parallel.

    python 05_00_build_test_inputs.py
"""
import gzip
import shutil
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_msd-spleen" / "msd-spleen"
DERIV_DIR    = BIDS_ROOT / "derivatives" / "manual_masks"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_msd-spleen" / "raw"

MODALITY = "ct"   # the only contrast in MSD-SPLEEN


def gzip_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f_in, gzip.open(dst, "wb", compresslevel=1) as f_out:
        shutil.copyfileobj(f_in, f_out)


def main() -> None:
    subs = sorted(p.name for p in BIDS_ROOT.glob("sub-SP*") if p.is_dir())
    if not subs:
        raise SystemExit(f"No sub-SP* under {BIDS_ROOT} — run 00_00_download_and_bidsify.py first.")

    img_dir = NNUNET_RAW / f"imagesTs_{MODALITY}"
    lab_dir = NNUNET_RAW / f"labelsTs_{MODALITY}"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    n_ok, missing = 0, []
    for sub in subs:
        case = sub[len("sub-"):]                       # SP02
        img = BIDS_ROOT / sub / "anat" / f"{sub}_CT.nii"
        seg = DERIV_DIR / sub / "anat" / f"{sub}_CT_dseg.nii"
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
