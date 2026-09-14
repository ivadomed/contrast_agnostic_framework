"""
nnUNetTrainerAutoPETBase — AutoPET (whole-body FDG/PSMA PET-CT tumor lesion) dataset base.

Adds split validation + held-out-test contamination guard on top of nnUNetTrainerFast.
Place FIRST in every concrete AutoPET trainer's MRO so its do_split() wins (same
mechanism as chaos / brats2024-glioma / open-ms / toothfairy2's own base.py).

nnU-Net case ids are the archive's own study ids, e.g. "fdg_<patient>_<study>" — see
01_create_splits/01_01_create_splits.py for exactly how patient-level (not study-level)
splitting avoids leaking a multi-timepoint patient's second study across train/val/test.

Splits cover ONLY the 501 lesion-positive FDG patients (negative controls and the whole
PSMA cohort are excluded from 4_splits_autopet/splits_final.json — PSMA is eval-only, see
05_predict; negative controls are a separate false-positive-rate side-check, never in the
headline train/val/test pool — see CLAUDE.md's pre-flight "claimed-N vs usable-N" rule and
00_utils/env.sh's header for the exact counts).

N_EXPECTED_FOLDS = 3 — this project's permanent fold policy (folds 0/1/2 only, see
CLAUDE.md "FOLD POLICY"), not a copy of any other dataset's file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from src.nnunet.trainers.fast import nnUNetTrainerFast

N_EXPECTED_FOLDS = 3


class nnUNetTrainerAutoPETBase(nnUNetTrainerFast):
    """AutoPET split validation mixin (place first in the MRO)."""

    def do_split(self):
        # splits_final.json lives in THIS PROJECT's own 4_splits_autopet/ (via SPLITS_DIR,
        # exported by common_env.sh), NOT in nnU-Net's preprocessed_dataset_folder_base —
        # unlike stock nnU-Net, which expects/auto-generates its split next to the
        # preprocessed data. Reading from SPLITS_DIR is what makes
        # 01_create_splits/01_01_create_splits.py's patient-level 3-fold split (over the
        # 501 positive FDG patients, NOT the archive's own reference split) actually take
        # effect — a real bug caught during the 2026-09-14 TamIA sizing probe (this method
        # originally read the wrong path, present-but-empty on TamIA's
        # nnUNet_preprocessed, and crashed loudly rather than falling back to nnU-Net's own
        # auto-generated split — the FileNotFoundError below did its job).
        splits_dir = Path(os.environ.get(
            "SPLITS_DIR",
            str(Path(__file__).resolve().parents[4] / "4_splits_autopet"),
        ))
        splits_path = splits_dir / "splits_final.json"
        if not splits_path.exists():
            raise FileNotFoundError(
                f"splits_final.json not found at {splits_path}. "
                "Run 01_create_splits/01_01_create_splits.py first (this project's OWN "
                "patient-level 3-fold split over the 501 positive FDG patients — NOT the "
                "archive's own reference splits_final.json, which mixes negative controls "
                "and has no sealed test set). NEVER let nnUNet auto-generate the split."
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

        # Held-out-test contamination guard (same splits_dir as above).
        test_path = splits_dir / "test_cases.json"
        if test_path.exists():
            test_cases = set(json.loads(test_path.read_text()).get("test", []))
            contamination = [c for c in tr_keys + val_keys if c in test_cases]
            if contamination:
                raise ValueError(
                    f"Held-out-test contamination in fold {self.fold}: {contamination}"
                )

        audit = Path(self.output_folder) / "split_verification.json"
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text(json.dumps(
            {"fold": self.fold, "n_train": len(tr_keys), "n_val": len(val_keys),
             "train_keys": sorted(tr_keys), "val_keys": sorted(val_keys)}, indent=2))
        self.print_to_log_file(
            f"[Split] Fold {self.fold}: {len(tr_keys)} train, {len(val_keys)} val -> {audit}")

        # ⚠️ Return OUR OWN tr_keys/val_keys directly — do NOT call super().do_split().
        # A second real bug caught in the same 2026-09-14 probe pass: stock nnU-Net's
        # do_split() reads/creates its OWN splits_final.json at
        # preprocessed_dataset_folder_base (a completely different path from our
        # SPLITS_DIR) and returns THAT file's keys, discarding everything validated above.
        # Calling it here would have silently handed every fold nnU-Net's own
        # auto-generated 80:20 (or 5-fold) split instead of this project's real one — the
        # exact failure mode this class's docstring and error message warn against,
        # reintroduced by the previous `return super().do_split()` line. This class must
        # be the sole source of truth for tr_keys/val_keys.
        return tr_keys, val_keys
