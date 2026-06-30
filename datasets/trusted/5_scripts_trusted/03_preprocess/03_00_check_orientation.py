#!/usr/bin/env python3
"""
Verify (and, if ever needed, fix) TRUSTED image/mask orientation to LPS.

TRUSTED volumes already ship as ('L','P','S') — the project's canonical convention
(matches CHAOS / SLIVER07), confirmed at ingest — so this is an idempotent CHECK:
every file should report `ok` and nothing is rewritten. It exists for §1b
compliance and as a guard against a future re-ingest that changes orientation; the
shared core (datasets/00_commun_scripts/00_00_utils/orient.py) is the same code
AMOS uses to actually reorient, so if a TRUSTED file ever differs it gets the same
lossless axis permute/flip (no resampling), leaving 0_raw pristine.

Touches the BIDS + nnUNet trees (0_raw_trusted is left pristine):
  1_BIDS_trusted/trusted-kidney/sub-*/anat/*.nii.gz                       (images)
  1_BIDS_trusted/trusted-kidney/derivatives/manual_masks/sub-*/anat/*.nii.gz (masks)
  2_nnUNet_trusted/raw/{imagesTs,labelsTs}_{ct,us}/*.nii.gz               (nnUNet inputs)

Usage:
    python 03_00_check_orientation.py              # check (and fix if differing)
    python 03_00_check_orientation.py --dry-run    # report only, write nothing
    python 03_00_check_orientation.py --trees bids # restrict to one tree
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "00_commun_scripts" / "00_00_utils"))
import orient  # noqa: E402

DATASET_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_trusted" / "trusted-kidney"
DERIV_DIR    = BIDS_ROOT / "derivatives" / "manual_masks"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_trusted" / "raw"


def _iter_files(trees: list[str]):
    if "bids" in trees:
        yield from sorted(BIDS_ROOT.glob("sub-*/anat/*.nii.gz"))
        yield from sorted(DERIV_DIR.glob("sub-*/anat/*.nii.gz"))
    if "nnunet" in trees:
        for sub in ("imagesTs_ct", "imagesTs_us", "labelsTs_ct", "labelsTs_us"):
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

    counts = orient.run(list(_iter_files(args.trees)), dry_run=args.dry_run,
                        title="TRUSTED orientation", rel_to=DATASET_ROOT)
    if counts["fixed"] or counts["would-fix"]:
        print("  NOTE: TRUSTED was expected to already be LPS — investigate the re-ingest.")


if __name__ == "__main__":
    main()
