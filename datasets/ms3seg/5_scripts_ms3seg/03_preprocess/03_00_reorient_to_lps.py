#!/usr/bin/env python3
"""
Normalise MS3SEG image/mask orientation to LPS. Source images are LAS, the combined
mask is RAS (verified empirically -- image/mask orientation MISMATCH within the same
archive, unlike every other dataset onboarded so far) -- both get reoriented to the
project's canonical LPS here, which also resolves that mismatch since both land on
the same target axcodes. Lossless axis permute/flip (nibabel `as_reoriented`), no
interpolation. Per-file logic lives in the shared
datasets/00_commun_scripts/00_00_utils/orient.py.

Usage:
    python 03_00_reorient_to_lps.py [--dry-run] [--trees bids|nnunet]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "00_commun_scripts" / "00_00_utils"))
import orient  # noqa: E402

DATASET_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_ms3seg" / "ms3seg-brain"
DERIV_DIR    = BIDS_ROOT / "derivatives" / "manual_masks"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_ms3seg" / "raw"


def _iter_files(trees: list[str]):
    if "bids" in trees:
        yield from sorted(BIDS_ROOT.glob("sub-*/anat/*.nii.gz"))
        yield from sorted(DERIV_DIR.glob("sub-*/anat/*.nii.gz"))
    if "nnunet" in trees:
        for sub in ("imagesTs_flair", "imagesTs_t1w", "imagesTs_t2w",
                    "labelsTs_flair", "labelsTs_t1w", "labelsTs_t2w"):
            yield from sorted((NNUNET_RAW / sub).glob("*.nii.gz"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--trees", nargs="+", default=["bids", "nnunet"], choices=["bids", "nnunet"])
    args = ap.parse_args()
    orient.run(list(_iter_files(args.trees)), dry_run=args.dry_run,
               title="MS3SEG orientation", rel_to=DATASET_ROOT)


if __name__ == "__main__":
    main()
