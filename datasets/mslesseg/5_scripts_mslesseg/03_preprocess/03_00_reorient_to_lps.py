#!/usr/bin/env python3
"""
Normalise MSLesSeg image/mask orientation to LPS (project canonical convention).

WHY
---
MSLesSeg ships already co-registered/skull-stripped/resampled to 1mm-iso MNI152
space (182,218,182), but in LAS voxel-array convention (verified via
nib.aff2axcodes), not the project's LPS (the chaos/sliver07/open-ms convention).
The affine is anatomically correct either way — this is purely a storage-convention
difference, same situation as AMOS (LAS/RAS -> LPS). Reorientation is a pure axis
permute/flip (nibabel `as_reoriented`): NO interpolation, NO resampling — lossless
and label-safe, so image and mask reorient identically and stay aligned.

Per-file logic lives in the shared core
datasets/00_commun_scripts/00_00_utils/orient.py; this script supplies only
MSLesSeg's BIDS file list (subject-only for the archive's test-split cases,
subject+session for its train-split cases with up to 3 timepoints — see
00_utils/00_00_ingest_and_bidsify.py).

WHAT IT TOUCHES
---------------
  1_BIDS_mslesseg/mslesseg-brain/sub-*/[ses-*/]anat/*.nii.gz                     (images)
  1_BIDS_mslesseg/mslesseg-brain/derivatives/manual_masks/sub-*/[ses-*/]anat/*.nii.gz (masks)
  2_nnUNet_mslesseg/raw/{imagesTs,labelsTs}_{flair,t1w,t2w}/*.nii.gz            (nnUNet inputs,
      once 05_00_build_test_inputs.py has been run — usually run AFTER this step, so the
      --trees nnunet pass is for idempotent re-checks / a build-then-reorder situation)

0_raw_mslesseg is left pristine (BIDS files are hard-links to it; writes go to a temp
file + atomic os.replace, which breaks the hard-link and leaves the raw inode untouched).

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
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_mslesseg" / "mslesseg-brain"
DERIV_DIR    = BIDS_ROOT / "derivatives" / "manual_masks"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_mslesseg" / "raw"


def _iter_files(trees: list[str]):
    if "bids" in trees:
        yield from sorted(BIDS_ROOT.glob("sub-*/anat/*.nii.gz"))
        yield from sorted(BIDS_ROOT.glob("sub-*/ses-*/anat/*.nii.gz"))
        yield from sorted(DERIV_DIR.glob("sub-*/anat/*.nii.gz"))
        yield from sorted(DERIV_DIR.glob("sub-*/ses-*/anat/*.nii.gz"))
    if "nnunet" in trees:
        for sub in ("imagesTs_flair", "imagesTs_t1w", "imagesTs_t2w",
                    "labelsTs_flair", "labelsTs_t1w", "labelsTs_t2w"):
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
               title="MSLesSeg orientation", rel_to=DATASET_ROOT)


if __name__ == "__main__":
    main()
