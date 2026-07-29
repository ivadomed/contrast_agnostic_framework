#!/usr/bin/env python3
"""
Build the nnUNet test-input dirs for the MSLesSeg evaluation set (FLAIR/T1w/T2w).

open-ms's model is single-channel (trained on FLAIR alone), so each volume is fed
as channel _0000 — same as its own cross-contrast test (FLAIR/T2W/T1W). MSLesSeg
volumes are small (182x218x182, ~7.2M voxels, already at ~1mm iso) so — unlike
TRUSTED's ultrasound case — no pre-resampling is needed; images are hard-linked
native. Run this AFTER 03_preprocess/03_00_reorient_to_lps.py (BIDS must already
be LPS) so the nnUNet tree inherits the corrected orientation.

Reads:  ../../1_BIDS_mslesseg/mslesseg-brain/sub-<id>[/ses-<k>]/anat/  (+ derivatives masks)
Writes: ../../2_nnUNet_mslesseg/raw/imagesTs_flair/{case}_0000.nii.gz  labelsTs_flair/{case}.nii.gz
                                    /imagesTs_t1w/{case}_0000.nii.gz   labelsTs_t1w/{case}.nii.gz
                                    /imagesTs_t2w/{case}_0000.nii.gz   labelsTs_t2w/{case}.nii.gz
        (all three contrasts share the SAME lesion mask per case — co-registered space)

Case ids (match the BIDS sub[/ses] pair, flattened): "P54" (test, no session) or
"P1_T1" (train split's timepoint 1, from sub-P1/ses-T1).

Idempotent: re-run skips existing hardlinks. Light (pure filesystem ops) but
dispatched via .sh/run_job for consistency.

    bash 05_00_build_test_inputs.sh
    python 05_00_build_test_inputs.py   # direct
"""
import os
import shutil
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_mslesseg" / "mslesseg-brain"
DERIV_DIR    = BIDS_ROOT / "derivatives" / "manual_masks"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_mslesseg" / "raw"

CONTRASTS = {"flair": "FLAIR", "t1w": "T1w", "t2w": "T2w"}


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _find_cases():
    """Yield (case_id, anat_dir, fname_stub) for every sub[/ses] pair in the BIDS tree."""
    for sub_dir in sorted(BIDS_ROOT.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        pid = sub_dir.name[len("sub-"):]
        ses_dirs = sorted(sub_dir.glob("ses-*"))
        if ses_dirs:
            for ses_dir in ses_dirs:
                tp = ses_dir.name[len("ses-"):]
                yield f"{pid}_{tp}", ses_dir / "anat", f"{sub_dir.name}_{ses_dir.name}"
        else:
            yield pid, sub_dir / "anat", sub_dir.name


def main() -> None:
    cases = list(_find_cases())
    if not cases:
        raise SystemExit(f"No sub-* under {BIDS_ROOT} — run 00_00_ingest_and_bidsify.py first.")

    for contrast, bids_suffix in CONTRASTS.items():
        img_dir = NNUNET_RAW / f"imagesTs_{contrast}"
        lab_dir = NNUNET_RAW / f"labelsTs_{contrast}"
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)

        n_ok, missing = 0, []
        for case_id, anat_dir, fname_stub in cases:
            img_src = anat_dir / f"{fname_stub}_{bids_suffix}.nii.gz"
            mask_src = DERIV_DIR / anat_dir.relative_to(BIDS_ROOT) / f"{fname_stub}_FLAIR_dseg.nii.gz"
            if not img_src.exists() or not mask_src.exists():
                missing.append(case_id)
                continue
            link_or_copy(img_src, img_dir / f"{case_id}_0000.nii.gz")
            link_or_copy(mask_src, lab_dir / f"{case_id}.nii.gz")
            n_ok += 1

        status = f"{n_ok}/{len(cases)} -> {img_dir.name} (native hardlink, +labels)"
        if missing:
            status += f"  [MISSING {len(missing)}: {missing[:5]}]"
        print(f"  {contrast:5s}: {status}")


if __name__ == "__main__":
    main()
