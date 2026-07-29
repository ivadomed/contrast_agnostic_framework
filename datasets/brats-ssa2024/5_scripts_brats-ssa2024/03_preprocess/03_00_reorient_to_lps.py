#!/usr/bin/env python3
"""
Normalise BraTS-SSA 2024 image/mask orientation to LPS (project canonical convention).

WHY
---
BraTS-SSA ships in RAS voxel-array convention (verified via nib.aff2axcodes on the raw
seg.nii files), not the project's LPS (the chaos/sliver07/open-ms/brats2024-glioma
convention). Same situation as AMOS/MSLesSeg. Reorientation is a pure axis permute/flip
(nibabel `as_reoriented`): NO interpolation, NO resampling -- lossless and label-safe.

Per-file logic lives in the shared core datasets/00_commun_scripts/00_00_utils/orient.py;
this script supplies only BraTS-SSA's BIDS file list.

WHAT IT TOUCHES
---------------
  1_BIDS_brats-ssa2024/ssa-brain-brats2024/sub-*/anat/*.nii.gz                     (images)
  1_BIDS_brats-ssa2024/ssa-brain-brats2024/derivatives/manual_masks/sub-*/anat/*.nii.gz (masks)
  2_nnUNet_brats-ssa2024/raw/{imagesTs,labelsTs}_{t1n,t1c,t2w,t2f}/*.nii.gz        (nnUNet inputs,
      once 05_00_build_test_inputs.py has been run)

0_raw_brats-ssa2024 is left pristine (BIDS files are freshly-written gzip copies from
00_00_ingest_and_bidsify.py, not hardlinks, so there's nothing to break there -- but the
BIDS/nnUNet trees are still the only trees this script touches).

Usage:
    python 03_00_reorient_to_lps.py              # fix BIDS (+ nnUNet if built)
    python 03_00_reorient_to_lps.py --dry-run    # report only, change nothing
    python 03_00_reorient_to_lps.py --trees bids # restrict to one tree
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "00_commun_scripts" / "00_00_utils"))
import orient  # noqa: E402

DATASET_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_brats-ssa2024" / "ssa-brain-brats2024"
DERIV_DIR    = BIDS_ROOT / "derivatives" / "manual_masks"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_brats-ssa2024" / "raw"


def _iter_files(trees: list[str]):
    if "bids" in trees:
        yield from sorted(BIDS_ROOT.glob("sub-*/anat/*.nii.gz"))
        yield from sorted(DERIV_DIR.glob("sub-*/anat/*.nii.gz"))
    if "nnunet" in trees:
        for sub in ("imagesTs_t1n", "imagesTs_t1c", "imagesTs_t2w", "imagesTs_t2f",
                    "labelsTs_t1n", "labelsTs_t1c", "labelsTs_t2w", "labelsTs_t2f"):
            yield from sorted((NNUNET_RAW / sub).glob("*.nii.gz"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change, write nothing")
    ap.add_argument("--trees", nargs="+", default=["bids", "nnunet"],
                    choices=["bids", "nnunet"],
                    help="which tree(s) to process (default: both)")
    args = ap.parse_args()

    orient.run(list(_iter_files(args.trees)), dry_run=args.dry_run,
               title="BraTS-SSA 2024 orientation", rel_to=DATASET_ROOT)


if __name__ == "__main__":
    main()
