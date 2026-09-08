#!/usr/bin/env python3
"""
hanseg evaluator — thin shim over the shared evaluator
datasets/00_commun_scripts/00_03_evaluate/evaluate.py. Same CLI.

hanseg scores ONE label, `mandible`. Predictions must already have been collapsed to
the mandible union by 05_predict/05_20_merge_mandible_union.py (see that file: HaN-Seg's
Bone_Mandible = toothfairy2's mandible ∪ lower_teeth), so by the time this runs both
sides are binary {0,1} and the map is the identity [1, 1].
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "00_commun_scripts" / "00_03_evaluate"))
from evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
