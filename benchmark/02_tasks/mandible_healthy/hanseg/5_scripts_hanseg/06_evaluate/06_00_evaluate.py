#!/usr/bin/env python3
"""
hanseg evaluator — thin shim over the shared evaluator
benchmark/00_commun_scripts/00_03_evaluate/evaluate.py. Same CLI.

hanseg scores ONE label, `mandible`.

⚠️ CORRECTED 2026-09-17: HaN-Seg's Bone_Mandible **EXCLUDES the teeth**, so it does NOT
equal toothfairy2's mandible ∪ lower_teeth. Current scoring is MANDIBLE-ONLY — the RAW
3-class prediction is passed straight in with --label_map '{"mandible": [1, 1]}' (labels
2/3 score as background) by 06_01_evaluate_run.sh and 06_03_eval_mandible_only.sh. The older
union path (merge predictions to mandible+teeth first) is retired; its merge script is archived
in benchmark/03_archive/toothfairy2_mandible_superseded_eval_20260918/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]
                       / "00_commun_scripts" / "00_03_evaluate"))
from evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
