#!/usr/bin/env python3
"""
Deterministic train / val / test split for BraTS 2024 Glioma.

Strategy
--------
- 700 cases (one per tumour, no multi-session).
- Hold out 10 % (70 cases) as a sealed test set.
- Remaining 630 split into 4-fold cross-validation (subject-level):
    fold k val  : cases 630//4 * k  …  630//4 * (k+1)   (157 or 158 each)
    fold k train: the other ~472-473 cases.
- All randomness is seeded (SEED=12345) → fully deterministic.

Outputs
-------
4_splits_brats2024-glioma/
    splits_final.json    -- 4-fold CV splits in nnUNet format
    test_cases.json      -- 70 held-out case IDs

2_nnUNet_brats2024-glioma/preprocessed/Dataset050_BraTS2024Glioma/
    splits_final.json    -- symlinked/copied; nnUNet reads it from here

Usage
-----
Run AFTER 02_00_convert.py (the nnUNet case IDs must exist in imagesTr).
    .venv/bin/python 02_01_create_splits.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]          # …/brats2024-glioma/
sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_00_utils"))
from splits_lib import kfold_splits, write_splits_final      # noqa: E402
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_brats2024-glioma" / "glioma-brain-brats2024"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_brats2024-glioma" / "raw"
NNUNET_PRE   = DATASET_ROOT / "2_nnUNet_brats2024-glioma" / "preprocessed"
SPLITS_DIR   = DATASET_ROOT / "4_splits_brats2024-glioma"

DATASET_NAME = "Dataset050_BraTS2024Glioma"
N_FOLDS      = 4
TEST_FRAC    = 0.10   # 70 / 700
SEED         = 12345


def discover_cases() -> list[str]:
    """
    Collect case IDs from the BIDS participants.tsv.
    Case ID = BIDS label without 'sub-' prefix, matching what 02_00_convert.py writes.
    E.g. 'sub-BraTSGLI00005100' → 'BraTSGLI00005100'
    """
    tsv = BIDS_ROOT / "participants.tsv"
    if not tsv.exists():
        raise FileNotFoundError(f"participants.tsv not found: {tsv}")
    lines = tsv.read_text().strip().split("\n")
    # header: participant_id  brats_id
    cases = [line.split("\t")[0].removeprefix("sub-") for line in lines[1:] if line.strip()]
    return sorted(cases)


def main() -> None:
    cases = discover_cases()
    print(f"Found {len(cases)} cases")

    rng = random.Random(SEED)
    shuffled = cases.copy()
    rng.shuffle(shuffled)

    n_test = round(len(shuffled) * TEST_FRAC)
    test_cases  = shuffled[:n_test]
    trainval    = shuffled[n_test:]
    print(f"Test : {len(test_cases)} cases")
    print(f"Train+val: {len(trainval)} cases → {N_FOLDS}-fold CV")

    # 4-fold subject-level CV on trainval (shared chunking; see splits_lib).
    splits = kfold_splits(trainval, N_FOLDS)
    for k, s in enumerate(splits):
        print(f"  Fold {k}: {len(s['train'])} train, {len(s['val'])} val")

    # Sanity checks
    for k, split in enumerate(splits):
        overlap = set(split["train"]) & set(split["val"])
        assert not overlap, f"Fold {k} train/val overlap: {overlap}"
        test_contam = set(split["train"] + split["val"]) & set(test_cases)
        assert not test_contam, f"Fold {k} test contamination: {test_contam}"
    print("Sanity checks passed (no overlap, no test contamination).")

    # Write splits_final.json to 4_splits/ and copy into the nnUNet preprocessed dir.
    splits_path = write_splits_final(splits, SPLITS_DIR, NNUNET_PRE, DATASET_NAME)
    print(f"Written: {splits_path}  (+ copied into {NNUNET_PRE / DATASET_NAME})")

    test_path = SPLITS_DIR / "test_cases.json"
    test_path.write_text(json.dumps(sorted(test_cases), indent=2))
    print(f"Written: {test_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
