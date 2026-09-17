#!/usr/bin/env python3
"""
pddca evaluator — thin shim over the shared evaluator
datasets/00_commun_scripts/00_03_evaluate/evaluate.py. Same CLI.

pddca scores ONE label, `mandible`, MANDIBLE-ONLY: toothfairy2's predicted label 1 vs
PDDCA's Mandible mask, one-to-one via --label_map '{"mandible": [1, 1]}' on the RAW
3-class prediction (labels 2/3 score as background). There is NO union merge step in
this dataset — PDDCA's mandible EXCLUDES the teeth, per its own protocol doc ("Only the
bone is segmented, while the teeth are excluded") and confirmed empirically.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "00_commun_scripts" / "00_03_evaluate"))
from evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
