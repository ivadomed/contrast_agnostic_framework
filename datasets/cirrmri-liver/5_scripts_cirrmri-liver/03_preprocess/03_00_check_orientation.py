#!/usr/bin/env python3
"""
Idempotent orientation CHECK for CIRRMRI-LIVER (matches TRUSTED's pattern: this
dataset was confirmed already LPS at BIDSify time, unlike AMOS/msd-spleen which
needed real reorientation -- this script exists to keep the check a standing,
re-runnable pipeline step per the standardization checklist SS1b, not a one-off).
Delegates to the shared core (00_commun_scripts/00_00_utils/orient.py) in dry-run
mode by default; every file should report 'ok'.

Usage:
    python 03_00_check_orientation.py              # dry-run report only (default)
    python 03_00_check_orientation.py --fix        # actually reorient anything not LPS
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "00_commun_scripts" / "00_00_utils"))
import orient  # noqa: E402

DATASET_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT = DATASET_ROOT / "1_BIDS_cirrmri-liver" / "cirrmri-liver"
DERIV_DIR = BIDS_ROOT / "derivatives" / "manual_masks"
NNUNET_RAW = DATASET_ROOT / "2_nnUNet_cirrmri-liver" / "raw"


def _iter_files(trees):
    if "bids" in trees:
        yield from sorted(BIDS_ROOT.glob("sub-CR*/anat/*.nii.gz"))
        yield from sorted(DERIV_DIR.glob("sub-CR*/anat/*.nii.gz"))
    if "nnunet" in trees:
        for sub in ("imagesTs_t1", "imagesTs_t2", "labelsTs_t1", "labelsTs_t2"):
            yield from sorted((NNUNET_RAW / sub).glob("*.nii.gz"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fix", action="store_true",
                    help="actually reorient (default: dry-run report only)")
    ap.add_argument("--trees", nargs="+", default=["bids", "nnunet"],
                    choices=["bids", "nnunet"])
    args = ap.parse_args()

    orient.run(list(_iter_files(args.trees)), dry_run=not args.fix,
               title="CIRRMRI-LIVER orientation", rel_to=DATASET_ROOT)


if __name__ == "__main__":
    main()
