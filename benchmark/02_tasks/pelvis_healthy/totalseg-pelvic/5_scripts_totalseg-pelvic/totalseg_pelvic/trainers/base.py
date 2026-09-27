"""
nnUNetTrainerTotalsegPelvicBase — totalseg-pelvic (TotalSegmentator CT+MRI, pelvic/hip
musculoskeletal) dataset base.

Adds split validation + held-out-test contamination guard on top of nnUNetTrainerFast.
Place FIRST in every concrete totalseg-pelvic trainer's MRO so its do_split() wins (same
mechanism as chaos / autopet / toothfairy2's own base.py).

CT and MRI are UNPAIRED cohorts (different patients) with independent splits under
4_splits_totalseg-pelvic/{ct,mri}/ — SPLITS_DIR (exported per-contrast by 00_utils/env.sh
/ env_mri.sh) already resolves to the right subdir, so this class does not need to know
which modality it's training on.

N_EXPECTED_FOLDS = 3 — this project's permanent fold policy (folds 0/1/2 only, see
CLAUDE.md "FOLD POLICY"), not a copy of any other dataset's file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from src.nnunet.trainers.fast import nnUNetTrainerFast

N_EXPECTED_FOLDS = 3


class nnUNetTrainerTotalsegPelvicBase(nnUNetTrainerFast):
    """totalseg-pelvic split validation mixin (place first in the MRO)."""

    def do_split(self):
        splits_dir = Path(os.environ.get("SPLITS_DIR", ""))
        if not splits_dir or not str(splits_dir):
            raise RuntimeError(
                "SPLITS_DIR not set — source 00_utils/env.sh or env_mri.sh before "
                "training (it resolves the per-contrast split subdir)."
            )
        splits_path = splits_dir / "splits_final.json"
        if not splits_path.exists():
            raise FileNotFoundError(
                f"splits_final.json not found at {splits_path}. Run "
                "01_create_splits/01_01_create_splits.py first (this project's OWN "
                "case-level 3-fold split, built separately for CT and MRI since they are "
                "unpaired cohorts — NOT the archives' own bundled meta.csv split). NEVER "
                "let nnUNet auto-generate the split."
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

        # Do NOT call super().do_split() — see autopet/chaos's identical comment: stock
        # nnU-Net reads/creates its own splits_final.json at a different path and would
        # silently discard everything validated above.
        return tr_keys, val_keys
