#!/usr/bin/env python3
"""
Build the nnUNet test-input dirs for the LLD-MMRI-HCC evaluation set.

Two items -- T2w and dwi (the phases genuinely different in contrast mechanism from
atlas-liver-hcc's CE-T1w training contrast). atlas-liver-hcc's models are
single-channel, so each volume is fed as channel _0000.

Reads:  ../../1_BIDS_lld-mmri-hcc/lld-mmri-hcc/sub-lldhccNNN/anat/  (+ derivatives masks)
Writes: ../../2_nnUNet_lld-mmri-hcc/raw/imagesTs_t2wi/{case}_0000.nii.gz
                                        /labelsTs_t2wi/{case}.nii.gz
                                        /imagesTs_dwi/{case}_0000.nii.gz
                                        /labelsTs_dwi/{case}.nii.gz

Case id = participant label without the sub- prefix (e.g. lldhcc000). The "_t2wi"/
"_dwi" suffixes mirror atlas-liver-hcc's own "_t1w" per-modality test dir naming so
05_predict / 06_evaluate stay parallel across datasets.

    python 05_00_build_test_inputs.py
"""
import gzip
import shutil
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_lld-mmri-hcc" / "lld-mmri-hcc"
DERIV_DIR    = BIDS_ROOT / "derivatives" / "manual_masks"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_lld-mmri-hcc" / "raw"

PHASES = {"t2wi": "T2w", "dwi": "dwi"}   # item name -> BIDS suffix


def gzip_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".gz":
        shutil.copyfile(src, dst)
    else:
        with open(src, "rb") as f_in, gzip.open(dst, "wb", compresslevel=1) as f_out:
            shutil.copyfileobj(f_in, f_out)


def main() -> None:
    subs = sorted(p.name for p in BIDS_ROOT.glob("sub-lldhcc*") if p.is_dir())
    if not subs:
        raise SystemExit(f"No sub-lldhcc* under {BIDS_ROOT} — run 00_utils/00_01_bidsify.py first.")

    for item, suffix in PHASES.items():
        img_dir = NNUNET_RAW / f"imagesTs_{item}"
        lab_dir = NNUNET_RAW / f"labelsTs_{item}"
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)

        n_ok, missing = 0, []
        for sub in subs:
            case = sub[len("sub-"):]                       # lldhcc000
            img = BIDS_ROOT / sub / "anat" / f"{sub}_{suffix}.nii.gz"
            seg = DERIV_DIR / sub / "anat" / f"{sub}_{suffix}_dseg.nii.gz"
            if not img.exists() or not seg.exists():
                missing.append(case)
                continue
            gzip_copy(img, img_dir / f"{case}_0000.nii.gz")
            gzip_copy(seg, lab_dir / f"{case}.nii.gz")
            n_ok += 1

        status = f"{n_ok}/{len(subs)} -> {img_dir.name} (+labels)"
        if missing:
            status += f"  MISSING: {missing}"
        print(status)


if __name__ == "__main__":
    main()
