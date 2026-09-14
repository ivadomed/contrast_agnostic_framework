#!/usr/bin/env python3
"""
ispy2 evaluator — thin shim over the shared, method-agnostic evaluator at
datasets/00_commun_scripts/00_03_evaluate/evaluate.py (Dice / HD95 per case per
label). Same CLI; see the commun evaluate.py header. ispy2 scores `tumour` only
(single foreground label, same 0/1 numbering as ambl's own dataset.json — no
--label_map remap needed, unlike lld-mmri-hcc -> atlas-liver-hcc).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "00_commun_scripts" / "00_03_evaluate"))
from evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
