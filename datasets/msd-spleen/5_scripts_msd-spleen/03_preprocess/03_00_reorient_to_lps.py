#!/usr/bin/env python3
"""
Normalise MSD-SPLEEN image/mask orientation to LPS (match CHAOS / SLIVER07).

WHY
---
MSD-SPLEEN (Medical Segmentation Decathlon Task09_Spleen) raw NIfTI files are
stored as RAS — the same storage-convention mismatch AMOS's MRI half had:

    CHAOS  CT  → LPS        SLIVER07 CT → LPS     (canonical here)
    MSD-SPLEEN CT → RAS     (left/right AND anterior/posterior flipped)

The affines are anatomically correct (any viewer/tool that respects the affine
shows the volume right-side-up); this is purely a storage-convention difference.
Tooling that works in raw voxel-array space (ignoring the affine) — including
visual QA and any array-space comparison — sees MSD-SPLEEN flipped on X and Y
relative to CHAOS/SLIVER07. This step removes that discrepancy.

The reorientation is a pure axis permute/flip (nibabel `as_reoriented`): NO
interpolation, NO resampling. It is therefore lossless and label-safe — image
and segmentation reorient identically and stay perfectly aligned.

The per-file logic lives in the shared core
datasets/00_commun_scripts/00_00_utils/orient.py (same code path AMOS/TRUSTED
use); this script supplies only MSD-SPLEEN's BIDS + nnUNet file lists.

WHAT IT TOUCHES
---------------
Both trees the data flows through (idempotent — already-LPS files are skipped):
  1_BIDS_msd-spleen/msd-spleen/sub-SP*/anat/*.nii                      (images)
  1_BIDS_msd-spleen/msd-spleen/derivatives/manual_masks/sub-SP*/anat/*.nii  (masks)
  2_nnUNet_msd-spleen/raw/{imagesTs,labelsTs}_ct/*.nii.gz              (nnUNet inputs)

0_raw_msd-spleen is left pristine. The BIDS files are hard-link-equivalent copies
of 0_raw (written by 00_00_download_and_bidsify.py, not hard-linked, so no
break-the-hard-link concern here, but writes still go temp-file + atomic replace
for consistency with the shared core's contract).

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
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_msd-spleen" / "msd-spleen"
DERIV_DIR    = BIDS_ROOT / "derivatives" / "manual_masks"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_msd-spleen" / "raw"


def _iter_files(trees: list[str]):
    if "bids" in trees:
        yield from sorted(BIDS_ROOT.glob("sub-SP*/anat/*.nii"))
        yield from sorted(DERIV_DIR.glob("sub-SP*/anat/*.nii"))
    if "nnunet" in trees:
        for sub in ("imagesTs_ct", "labelsTs_ct"):
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
               title="MSD-SPLEEN orientation", rel_to=DATASET_ROOT)


if __name__ == "__main__":
    main()
