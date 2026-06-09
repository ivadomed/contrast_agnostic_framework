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
import shutil
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]          # …/brats2024-glioma/
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

    # 4-fold CV on trainval
    chunk = len(trainval) // N_FOLDS
    # Distribute remainder across first folds (at most +1 case per fold)
    fold_sizes = [chunk + (1 if i < len(trainval) % N_FOLDS else 0) for i in range(N_FOLDS)]
    fold_val_cases: list[list[str]] = []
    offset = 0
    for sz in fold_sizes:
        fold_val_cases.append(trainval[offset:offset + sz])
        offset += sz

    splits = []
    for k in range(N_FOLDS):
        val_ids   = sorted(fold_val_cases[k])
        train_ids = sorted(c for i, fold in enumerate(fold_val_cases) if i != k for c in fold)
        splits.append({"train": train_ids, "val": val_ids})
        print(f"  Fold {k}: {len(train_ids)} train, {len(val_ids)} val")

    # Sanity checks
    for k, split in enumerate(splits):
        overlap = set(split["train"]) & set(split["val"])
        assert not overlap, f"Fold {k} train/val overlap: {overlap}"
        test_contam = set(split["train"] + split["val"]) & set(test_cases)
        assert not test_contam, f"Fold {k} test contamination: {test_contam}"
    print("Sanity checks passed (no overlap, no test contamination).")

    # Write to 4_splits_brats2024-glioma/
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    splits_path = SPLITS_DIR / "splits_final.json"
    splits_path.write_text(json.dumps(splits, indent=2))
    print(f"Written: {splits_path}")

    test_path = SPLITS_DIR / "test_cases.json"
    test_path.write_text(json.dumps(sorted(test_cases), indent=2))
    print(f"Written: {test_path}")

    # Copy splits_final.json to nnUNet preprocessed dir (where nnUNet reads it)
    pre_ds = NNUNET_PRE / DATASET_NAME
    pre_ds.mkdir(parents=True, exist_ok=True)
    shutil.copy2(splits_path, pre_ds / "splits_final.json")
    print(f"Copied splits_final.json → {pre_ds / 'splits_final.json'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
