#!/usr/bin/env python3
"""
Normalise KIDNEY-T2W image/mask orientation to LPS (match CHAOS / SLIVER07).

Raw NIfTI is stored as RSP (right/superior/posterior increasing) -- a coronal
acquisition (thin axis is A-P, not S-I), unlike most other datasets in this
project. The reorientation is a pure axis permute/flip (nibabel `as_reoriented`):
NO interpolation, NO resampling -- lossless and label-safe.

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
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_kidney-t2w" / "kidney-t2w"
DERIV_DIR    = BIDS_ROOT / "derivatives" / "manual_masks"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_kidney-t2w" / "raw"


def _iter_files(trees: list[str]):
    if "bids" in trees:
        yield from sorted(BIDS_ROOT.glob("sub-*/anat/*.nii.gz"))
        yield from sorted(DERIV_DIR.glob("sub-*/anat/*.nii.gz"))
    if "nnunet" in trees:
        for sub in ("imagesTs_t2", "labelsTs_t2"):
            yield from sorted((NNUNET_RAW / sub).glob("*.nii.gz"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--trees", nargs="+", default=["bids", "nnunet"],
                    choices=["bids", "nnunet"])
    args = ap.parse_args()

    orient.run(list(_iter_files(args.trees)), dry_run=args.dry_run,
               title="KIDNEY-T2W orientation", rel_to=DATASET_ROOT)


if __name__ == "__main__":
    main()
