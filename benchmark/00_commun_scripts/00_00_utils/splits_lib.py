#!/usr/bin/env python3
"""
Shared helpers for the 01_create_splits stage.

The per-dataset split scripts differ in case discovery and hold-out strategy (BraTS:
fraction-based sealed test; CHAOS: N held-out MR patients + CT as a pure cross-modality
test). What they share — the subject-level k-fold CV chunking and writing the nnUNet
splits_final.json — lives here so the fiddly, correctness-critical part is defined once.
"""
import json
import shutil
from pathlib import Path


def kfold_splits(pool: list, n_folds: int) -> list:
    """Subject-level k-fold CV over an already-ordered (e.g. seed-shuffled) pool.

    The remainder is distributed one case per fold to the first ``len(pool) % n_folds``
    folds (so fold sizes differ by at most 1). Returns nnUNet-format splits:
    ``[{"train": [...sorted...], "val": [...sorted...]}, ...]``.
    """
    chunk = len(pool) // n_folds
    sizes = [chunk + (1 if i < len(pool) % n_folds else 0) for i in range(n_folds)]
    folds, off = [], 0
    for sz in sizes:
        folds.append(pool[off:off + sz]); off += sz
    splits = []
    for k in range(n_folds):
        val = sorted(folds[k])
        train = sorted(c for i, f in enumerate(folds) if i != k for c in f)
        splits.append({"train": train, "val": val})
    return splits


def write_splits_final(splits: list, splits_dir, nnunet_pre, dataset_name: str) -> Path:
    """Write splits_final.json to splits_dir AND copy it into the nnUNet preprocessed
    dataset dir (where nnU-Net reads it). Returns the path in splits_dir."""
    splits_dir = Path(splits_dir); splits_dir.mkdir(parents=True, exist_ok=True)
    splits_path = splits_dir / "splits_final.json"
    splits_path.write_text(json.dumps(splits, indent=2))
    pre_ds = Path(nnunet_pre) / dataset_name
    pre_ds.mkdir(parents=True, exist_ok=True)
    shutil.copy2(splits_path, pre_ds / "splits_final.json")
    return splits_path
