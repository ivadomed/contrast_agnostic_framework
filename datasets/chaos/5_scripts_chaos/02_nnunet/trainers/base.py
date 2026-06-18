"""
nnUNetTrainerCHAOSBase — CHAOS dataset base trainer.

Adds CHAOS-specific split validation + internal-test contamination guard on top of
nnUNetTrainerFast (seed, epochs, WandB hooks). Place this FIRST in the MRO of every
concrete CHAOS trainer so its do_split() wins.

CHAOS test_cases.json is a dict (not a flat list like BraTS):
    {"mr_internal_test": [...], "ct_test": [...], ...}
Training cases are MR T1-in only; the holdout MR patients must never leak into
train/val. (CT is never in training by construction.)

MRO: nnUNetTrainerCHAOSBaseline
  → nnUNetTrainerCHAOSBase     (do_split — anti-contamination guard)
  → nnUNetTrainerFast          (seed, epochs, WandB hooks)
  → nnUNetTrainer
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from src.nnunet.trainers.fast import nnUNetTrainerFast

N_EXPECTED_FOLDS = 4


class nnUNetTrainerCHAOSBase(nnUNetTrainerFast):

    def do_split(self):
        dataset_base = getattr(
            self, "preprocessed_dataset_folder_base",
            Path(self.preprocessed_dataset_folder).parent,
        )
        splits_path = Path(dataset_base) / "splits_final.json"
        if not splits_path.exists():
            raise FileNotFoundError(
                f"splits_final.json not found at {splits_path}. "
                "Run 01_01_create_splits.py first. Never let nnUNet auto-generate the split."
            )

        with open(splits_path) as f:
            splits = json.load(f)
        if len(splits) != N_EXPECTED_FOLDS:
            raise ValueError(f"Expected {N_EXPECTED_FOLDS} folds, got {len(splits)}.")
        if self.fold >= N_EXPECTED_FOLDS:
            raise ValueError(
                f"fold={self.fold} out of range for {N_EXPECTED_FOLDS}-fold file — "
                "nnUNet would silently fall back to a random split. Blocked."
            )

        tr_keys, val_keys = splits[self.fold]["train"], splits[self.fold]["val"]
        overlap = set(tr_keys) & set(val_keys)
        if overlap:
            raise ValueError(f"Train/val overlap in fold {self.fold}: {overlap}")

        # Internal-test contamination guard (dict-shaped test_cases.json).
        splits_dir = Path(os.environ.get(
            "SPLITS_DIR",
            str(Path(__file__).resolve().parents[4] / "4_splits_chaos"),
        ))
        test_path = splits_dir / "test_cases.json"
        if test_path.exists():
            meta = json.loads(test_path.read_text())
            test_cases = set(meta.get("mr_internal_test", [])) | set(meta.get("ct_test", []))
            contamination = [c for c in tr_keys + val_keys if c in test_cases]
            if contamination:
                raise ValueError(
                    f"Internal-test contamination in fold {self.fold}: {contamination}"
                )

        audit = Path(self.output_folder) / "split_verification.json"
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text(json.dumps(
            {"fold": self.fold, "n_train": len(tr_keys), "n_val": len(val_keys),
             "train_keys": sorted(tr_keys), "val_keys": sorted(val_keys)}, indent=2))
        self.print_to_log_file(
            f"[Split] Fold {self.fold}: {len(tr_keys)} train, {len(val_keys)} val → {audit}")
        return super().do_split()
