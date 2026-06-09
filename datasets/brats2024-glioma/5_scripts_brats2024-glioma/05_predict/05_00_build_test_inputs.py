#!/usr/bin/env python3
"""
Build per-contrast nnUNet test-input dirs for the BraTS 2024 held-out test set.

The segmentation models are single-channel (trained on T1n as channel _0000).
To evaluate cross-contrast generalisation we feed each available contrast in turn
as channel _0000.  This script materialises one input dir per contrast, each
containing `{caseID}_0000.nii.gz` for every held-out test case.

These dirs are DATASET-level (shared by every experiment) — build them once:

    python 05_00_build_test_inputs.py            # builds all 4 contrasts
    python 05_00_build_test_inputs.py --contrasts t1n t2f

Reads:  ../../1_BIDS_brats2024-glioma/glioma-brain-brats2024/sub-<case>/anat/
Writes: ../../2_nnUNet_brats2024-glioma/raw/<DS_NAME>/imagesTs_<contrast>/
Test cases come from ../../4_splits_brats2024-glioma/test_cases.json
"""
import argparse
import gzip
import json
import shutil
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]      # …/datasets/brats2024-glioma/
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_brats2024-glioma" / "glioma-brain-brats2024"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_brats2024-glioma" / "raw"
TEST_CASES   = DATASET_ROOT / "4_splits_brats2024-glioma" / "test_cases.json"

# contrast tag → BIDS anat filename suffix (matches 02_00_convert.py channel map)
CONTRAST_SUFFIX = {
    "t1n": "T1w",                 # T1 native           (in-domain, training contrast)
    "t1c": "ce-gadolinium_T1w",   # T1 contrast-enhanced
    "t2w": "T2w",                 # T2-weighted
    "t2f": "FLAIR",               # T2-FLAIR
}


def gzip_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    with open(src, "rb") as f_in, gzip.open(dst, "wb", compresslevel=1) as f_out:
        shutil.copyfileobj(f_in, f_out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", type=int, default=51)
    ap.add_argument("--contrasts", nargs="+", default=list(CONTRAST_SUFFIX),
                    choices=list(CONTRAST_SUFFIX))
    args = ap.parse_args()

    ds_name = f"Dataset{args.dataset_id:03d}_BraTS2024GliomaT1n"
    out_root = NNUNET_RAW / ds_name
    cases = json.loads(TEST_CASES.read_text())
    print(f"{len(cases)} test cases → {ds_name}, contrasts: {args.contrasts}")

    for contrast in args.contrasts:
        suffix  = CONTRAST_SUFFIX[contrast]
        out_dir = out_root / f"imagesTs_{contrast}"
        out_dir.mkdir(parents=True, exist_ok=True)
        n_ok, missing = 0, []
        for case_id in cases:
            src = BIDS_ROOT / f"sub-{case_id}" / "anat" / f"sub-{case_id}_{suffix}.nii"
            if not src.exists():
                missing.append(case_id)
                continue
            gzip_copy(src, out_dir / f"{case_id}_0000.nii.gz")
            n_ok += 1
        status = f"{n_ok}/{len(cases)} written → {out_dir}"
        if missing:
            status += f"  [MISSING {len(missing)}: {missing[:5]}{'…' if len(missing) > 5 else ''}]"
        print(f"  {contrast:4s} ({suffix:>20s}): {status}")


if __name__ == "__main__":
    main()
