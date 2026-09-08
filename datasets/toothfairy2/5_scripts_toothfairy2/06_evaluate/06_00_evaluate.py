#!/usr/bin/env python3
"""
toothfairy2 evaluator — thin shim over the shared, method-agnostic evaluator at
datasets/00_commun_scripts/00_03_evaluate/evaluate.py (Dice / HD95 per case per
label). Same CLI; see that header.

toothfairy2 scores THREE foreground labels — mandible, lower_teeth, pharynx — all
in the same 0..3 id space as its own dataset.json, so no --label_map remap is needed
for own-model evaluation. (Cross-dataset evaluation against hanseg DOES need one:
hanseg annotates a single Bone_Mandible that corresponds to the UNION of this
dataset's mandible + lower_teeth — see 00_utils/toothfairy2_labels.py's
MANDIBLE_UNION_* and the hanseg scripts.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "00_commun_scripts" / "00_03_evaluate"))
from evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
