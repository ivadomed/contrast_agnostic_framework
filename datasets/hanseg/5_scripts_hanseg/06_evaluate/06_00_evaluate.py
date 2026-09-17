#!/usr/bin/env python3
"""
hanseg evaluator — thin shim over the shared evaluator
datasets/00_commun_scripts/00_03_evaluate/evaluate.py. Same CLI.

hanseg scores ONE label, `mandible`.

⚠️ CORRECTED 2026-09-17: HaN-Seg's Bone_Mandible **EXCLUDES the teeth**, so it does NOT
equal toothfairy2's mandible ∪ lower_teeth. Current scoring is MANDIBLE-ONLY — the RAW
3-class prediction is passed straight in with --label_map '{"mandible": [1, 1]}' (labels
2/3 score as background) by 06_03_eval_mandible_only.sh. The older union path
(05_predict/05_20_merge_mandible_union.py, which collapses predictions to binary first)
is kept only to reproduce the superseded union view; re-scoring changed no conclusion,
but the union is wrong on the facts. See datasets/toothfairy2/RESULTS_NOTES.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "00_commun_scripts" / "00_03_evaluate"))
from evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
