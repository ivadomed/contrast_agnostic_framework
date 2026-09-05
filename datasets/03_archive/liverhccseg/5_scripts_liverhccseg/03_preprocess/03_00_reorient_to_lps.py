#!/usr/bin/env python3
"""
Reorient LiverHccSeg to the project's canonical LPS voxel-array convention.

Found 2026-08-29 (user caught it visually comparing rendered slices across datasets):
13/14 cases are (R,A,S) and 1/14 is (L,A,S), vs atlas-liver-hcc's and lld-mmri-hcc's
100%-consistent (L,P,S) -- a genuine two-axis flip for most cases, one-axis for the
rest. This was never checked during onboarding (only shape/spacing were verified) --
the ONLY dataset of the three not already in LPS. Reorients BOTH the BIDS tree (source
of truth) and the already-built nnUNet raw test-input copies (what predict/evaluate
actually consume right now) -- these are independent hardlinks to the same pre-fix
0_raw inode, so both need fixing; reorient_file's atomic replace breaks each hardlink
independently and is lossless/label-safe (pure axis permute, no interpolation).

Usage:  python 03_00_reorient_to_lps.py [--dry-run]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "00_commun_scripts" / "00_00_utils"))
from orient import run  # noqa: E402

DATASET_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT = DATASET_ROOT / "1_BIDS_liverhccseg" / "liverhccseg"
NNUNET_RAW = DATASET_ROOT / "2_nnUNet_liverhccseg" / "raw"


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    bids_files = sorted(BIDS_ROOT.glob("sub-*/anat/*.nii.gz")) + \
                 sorted((BIDS_ROOT / "derivatives" / "manual_masks").glob("sub-*/anat/*.nii.gz"))
    run(bids_files, dry_run=dry_run, title="BIDS tree", rel_to=DATASET_ROOT)

    nnunet_files = sorted(NNUNET_RAW.glob("imagesTs_*/*.nii.gz")) + \
                   sorted(NNUNET_RAW.glob("labelsTs_*/*.nii.gz"))
    run(nnunet_files, dry_run=dry_run, title="nnUNet raw test inputs", rel_to=DATASET_ROOT)


if __name__ == "__main__":
    main()
