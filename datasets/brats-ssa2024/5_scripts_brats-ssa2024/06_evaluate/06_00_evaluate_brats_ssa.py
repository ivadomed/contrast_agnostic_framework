#!/usr/bin/env python3
"""
brats-ssa2024 evaluator -- thin shim over the shared, method-agnostic evaluator at
datasets/00_commun_scripts/00_03_evaluate/evaluate.py (Dice / HD95 per case per label).
Same CLI; see the commun evaluate.py header.

Label note: brats2024-glioma's dataset.json has 4 foreground labels (NCR, SNFH, ET, RC).
BraTS-SSA 2024 is pre-treatment -- its GT never has RC (resection cavity, verified
empirically on the raw seg.nii files). Call with --labels NCR SNFH ET to score only
those three (pred_id == gt_id, same numbering both sides) -- RC predictions (if any)
are simply excluded from scoring, not counted as false positives against a label that
structurally cannot exist in this cohort.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "00_commun_scripts" / "00_03_evaluate"))
from evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
