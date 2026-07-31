#!/usr/bin/env python3
"""
Build the nnUNet test-input dirs for the CIRRMRI-LIVER evaluation set.

CIRRMRI-LIVER has two contrasts (T1w, T2w) -- the chaos models are single-channel,
so each volume is fed as channel _0000 under its own modality item (t1/t2),
mirroring how trusted's ct/us items or chaos's own t1in/t1out/t2spir items work.

Reads:  ../../1_BIDS_cirrmri-liver/cirrmri-liver/sub-CR###/anat/  (+ derivatives masks)
Writes: ../../2_nnUNet_cirrmri-liver/raw/imagesTs_{t1,t2}/{case}_0000.nii.gz
                                        /labelsTs_{t1,t2}/{case}.nii.gz

Case id = participant label without the sub- prefix (e.g. CR010). Not every subject
has both contrasts (291/337 have both, 19 T1-only, 27 T2-only) -- each item's dir
just gets whichever subjects actually have that contrast.

    python 05_00_build_test_inputs.py
"""
import gzip
import shutil
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_cirrmri-liver" / "cirrmri-liver"
DERIV_DIR    = BIDS_ROOT / "derivatives" / "manual_masks"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_cirrmri-liver" / "raw"

ITEMS = {"t1": "T1w", "t2": "T2w"}


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
    subs = sorted(p.name for p in BIDS_ROOT.glob("sub-CR*") if p.is_dir())
    if not subs:
        raise SystemExit(f"No sub-CR* under {BIDS_ROOT} -- run 00_00_download.py + 00_01_bidsify.py first.")

    for item, suffix in ITEMS.items():
        img_dir = NNUNET_RAW / f"imagesTs_{item}"
        lab_dir = NNUNET_RAW / f"labelsTs_{item}"
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)

        n_ok, missing = 0, []
        for sub in subs:
            case = sub[len("sub-"):]
            img = BIDS_ROOT / sub / "anat" / f"{sub}_{suffix}.nii.gz"
            seg = DERIV_DIR / sub / "anat" / f"{sub}_{suffix}_dseg.nii.gz"
            if not img.exists() or not seg.exists():
                missing.append(case)
                continue
            gzip_copy(img, img_dir / f"{case}_0000.nii.gz")
            gzip_copy(seg, lab_dir / f"{case}.nii.gz")
            n_ok += 1

        status = f"{n_ok}/{len(subs)} -> {img_dir.name} (+labels)"
        print(f"  {item:7s}: {status}  [missing {len(missing)}: contrast not present for those subjects]")


if __name__ == "__main__":
    main()
