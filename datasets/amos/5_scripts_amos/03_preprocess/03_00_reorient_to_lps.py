#!/usr/bin/env python3
"""
Normalise AMOS image/mask orientation to LPS (match CHAOS / SLIVER07).

WHY
---
AMOS raw NIfTI files are stored with a different voxel-array convention than
the project's other abdominal datasets:

    CHAOS  CT  → LPS        SLIVER07 CT → LPS     (canonical here)
    AMOS   CT  → LAS        (anterior/posterior, i.e. Y, flipped)
    AMOS   MRI → RAS        (left/right AND anterior/posterior flipped)

The AMOS affines are anatomically *correct* (the spine sits posterior once the
affine is honoured), so this is purely a storage-convention difference — any
viewer/tool that respects the affine shows AMOS right-side-up. But tooling that
works in raw voxel-array space (ignoring the affine) sees AMOS flipped on Y
relative to CHAOS/SLIVER. This step removes that discrepancy by reorienting the
voxel array to LPS while keeping the affine consistent.

The reorientation is a pure axis permute/flip (nibabel `as_reoriented`): NO
interpolation, NO resampling. It is therefore lossless and label-safe — image
and segmentation reorient identically and stay perfectly aligned.

The per-file logic now lives in the shared core
datasets/00_commun_scripts/00_00_utils/orient.py (TRUSTED's idempotent
orientation check uses the same code path); this script supplies only AMOS's
BIDS + nnUNet file lists.

WHAT IT TOUCHES
---------------
Both trees the data flows through (idempotent — already-LPS files are skipped):
  1_BIDS_amos/amos-abdominal/sub-*/anat/*.nii.gz                      (images)
  1_BIDS_amos/amos-abdominal/derivatives/manual_masks/sub-*/anat/*.nii.gz  (masks)
  2_nnUNet_amos/raw/{imagesTs,labelsTs}_{ct,mri}/*.nii.gz             (nnUNet inputs)

0_raw_amos is left pristine. The BIDS files are hard-links to 0_raw; writes go
to a temp file + atomic `os.replace`, which breaks the hard-link and leaves the
raw inode untouched. (The nnUNet tree is processed independently because it was
hard-linked from the *old* BIDS files.)

Usage:
    python 03_00_reorient_to_lps.py              # fix BIDS + nnUNet trees
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
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_amos" / "amos-abdominal"
DERIV_DIR    = BIDS_ROOT / "derivatives" / "manual_masks"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_amos" / "raw"


def _iter_files(trees: list[str]):
    if "bids" in trees:
        yield from sorted(BIDS_ROOT.glob("sub-*/anat/*.nii.gz"))
        yield from sorted(DERIV_DIR.glob("sub-*/anat/*.nii.gz"))
    if "nnunet" in trees:
        for sub in ("imagesTs_ct", "imagesTs_mri", "labelsTs_ct", "labelsTs_mri"):
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
               title="AMOS orientation", rel_to=DATASET_ROOT)


if __name__ == "__main__":
    main()
