#!/usr/bin/env python3
"""
atlas-liver-hcc evaluator — thin shim over the shared, method-agnostic evaluator at
datasets/00_commun_scripts/00_03_evaluate/evaluate.py (Dice / HD95 per case per label).
Same CLI; see the commun evaluate.py header. atlas-liver-hcc scores `liver` and `tumour`
(--dataset_json points at Dataset080's dataset.json; pred_id == gt_id) — `tumour` is the
paper-relevant, texture-defined target; `liver` is the surrounding-organ boundary.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "00_commun_scripts" / "00_03_evaluate"))
from evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
