"""
nnUNetTrainerOnHarmonyBase — ON-Harmony dataset base.

Adds ON-Harmony-specific split validation to nnUNetTrainerFast.
All concrete ON-Harmony trainers use multiple inheritance:

    class nnUNetTrainerOnHarmonyV26_6(nnUNetTrainerOnHarmonyBase, nnUNetTrainerV26_6):
        ...

MRO: nnUNetTrainerOnHarmonyV26_6
  → nnUNetTrainerOnHarmonyBase  (do_split override)
  → nnUNetTrainerV26_6          (train_step / validation_step)
  → nnUNetTrainerFast            (seed, epochs, WandB hooks)
  → nnUNetTrainer                (nnunetv2 upstream)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from src.nnunet.trainers.fast import nnUNetTrainerFast

N_EXPECTED_FOLDS = 4


class nnUNetTrainerOnHarmonyBase(nnUNetTrainerFast):
    """
    ON-Harmony dataset mixin — split validation + contamination guard.

    Place this as the FIRST parent in every ON-Harmony concrete trainer so
    that do_split() from this class takes priority in the MRO.
    """

    def do_split(self):
        """
        Load splits_final.json and validate rigorously before handing off to nnUNet.

        Hard-fails on: missing file, wrong fold count, fold out-of-range,
        train/val overlap, test-subject contamination.
        """
        dataset_base = getattr(
            self, "preprocessed_dataset_folder_base",
            Path(self.preprocessed_dataset_folder).parent,
        )
        splits_path = Path(dataset_base) / "splits_final.json"

        if not splits_path.exists():
            raise FileNotFoundError(
                f"splits_final.json not found at {splits_path}. "
                "Run 01_create_splits.py to create it. "
                "NEVER let nnUNet auto-generate the split."
            )

        with open(splits_path) as f:
            splits = json.load(f)

        if len(splits) != N_EXPECTED_FOLDS:
            raise ValueError(
                f"Expected {N_EXPECTED_FOLDS} folds, got {len(splits)} in splits_final.json."
            )
        if self.fold >= N_EXPECTED_FOLDS:
            raise ValueError(
                f"fold={self.fold} out of range for {N_EXPECTED_FOLDS}-fold file — "
                "nnUNet would silently fall back to a random split. Blocked."
            )

        tr_keys = splits[self.fold]["train"]
        val_keys = splits[self.fold]["val"]

        overlap = set(tr_keys) & set(val_keys)
        if overlap:
            raise ValueError(f"Train/val overlap in fold {self.fold}: {overlap}")

        # Test-subject contamination check
        splits_dir = Path(os.environ.get(
            "SPLITS_DIR",
            str(Path(__file__).resolve().parents[5] / "4_splits_on-harmony"),
        ))
        test_cases_path = splits_dir / "test_cases.json"
        if test_cases_path.exists():
            with open(test_cases_path) as f:
                test_cases = json.load(f)
            test_subjects = {c["subject"] for c in test_cases}

            def subject_of(cid: str) -> str:
                return cid.split("_ses-")[0]

            contamination = [
                cid for cid in tr_keys + val_keys
                if subject_of(cid) in test_subjects
            ]
            if contamination:
                raise ValueError(
                    f"Test-subject contamination in fold {self.fold}: {contamination}"
                )

        audit_path = Path(self.output_folder) / "split_verification.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_path, "w") as f:
            json.dump(
                {"fold": self.fold, "n_train": len(tr_keys), "n_val": len(val_keys),
                 "train_keys": sorted(tr_keys), "val_keys": sorted(val_keys)},
                f, indent=2,
            )
        print(f"[Split] Fold {self.fold}: {len(tr_keys)} train, {len(val_keys)} val. "
              f"Audit → {audit_path}")

        return super().do_split()
