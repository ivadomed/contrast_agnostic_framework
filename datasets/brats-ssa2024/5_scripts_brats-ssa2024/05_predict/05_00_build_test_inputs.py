#!/usr/bin/env python3
"""
Build the nnUNet test-input dirs for the BraTS-SSA 2024 evaluation set (T1n/T1c/T2w/T2f).

brats2024-glioma's models are single-channel (trained on one contrast at a time), so
each volume is fed as channel _0000 -- same as its own cross-contrast test
(PREDICT_ITEMS_DEFAULT="t1n t1c t2w t2f"). BraTS-SSA volumes are standard BraTS size
(240x240x155, 1mm iso) so no resampling is needed; images are hard-linked native. Run
this AFTER 03_preprocess/03_00_reorient_to_lps.py (BIDS must already be LPS).

Reads:  ../../1_BIDS_brats-ssa2024/ssa-brain-brats2024/sub-<id>/anat/  (+ derivatives masks)
Writes: ../../2_nnUNet_brats-ssa2024/raw/imagesTs_t1n/{case}_0000.nii.gz  labelsTs_t1n/{case}.nii.gz
                                          /imagesTs_t1c/...                labelsTs_t1c/...
                                          /imagesTs_t2w/...                labelsTs_t2w/...
                                          /imagesTs_t2f/...                labelsTs_t2f/...
        (all four contrasts share the SAME GT mask per case -- co-registered space)

Case ids match the BIDS sub id (e.g. "00002000" from sub-00002000).

Idempotent: re-run skips existing hardlinks. Light (pure filesystem ops) but
dispatched via .sh/run_job for consistency.

    bash 05_00_build_test_inputs.sh
    python 05_00_build_test_inputs.py   # direct
"""
import os
import shutil
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_brats-ssa2024" / "ssa-brain-brats2024"
DERIV_DIR    = BIDS_ROOT / "derivatives" / "manual_masks"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_brats-ssa2024" / "raw"

CONTRASTS = {"t1n": "T1n", "t1c": "T1c", "t2w": "T2w", "t2f": "T2f"}


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
            mask_src = DERIV_DIR / sub / "anat" / f"{sub}_dseg.nii.gz"
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
