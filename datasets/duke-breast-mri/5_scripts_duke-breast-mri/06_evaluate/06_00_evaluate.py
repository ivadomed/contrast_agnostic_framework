#!/usr/bin/env python3
"""
duke-breast-mri evaluator — thin shim over the shared, method-agnostic
evaluator at datasets/00_commun_scripts/00_03_evaluate/evaluate.py (Dice /
HD95 per case per label). Same CLI; see the commun evaluate.py header.
duke-breast-mri scores `tumour` only (single foreground label, background=0 /
tumour=1 — see 02_nnunet/02_01_convert_test_t1wce.py's manifest; pred_id ==
gt_id == duke_XXX).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "00_commun_scripts" / "00_03_evaluate"))
from evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
