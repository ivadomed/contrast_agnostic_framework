#!/usr/bin/env python3
"""
ispy1 evaluator -- thin shim over the shared, method-agnostic evaluator at
benchmark/00_commun_scripts/00_03_evaluate/evaluate.py (Dice / HD95 per case per
label). Same CLI; see that header.

ispy1 scores `tumour` only (background=0 / tumour=1, identical numbering to
ispy2's Dataset100 dataset.json -- no --label_map needed; pred_id == gt_id ==
ispy1_XXXX for both test items).
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise SystemExit("PROJECT_ROOT not set -- source 00_utils/env.sh before running this script")
sys.path.insert(0, str(Path(PROJECT_ROOT) / "benchmark" / "00_commun_scripts" / "00_03_evaluate"))
from evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
