#!/usr/bin/env python3
"""
BraTS 2024 Glioma evaluator — thin shim over the shared, method-agnostic
evaluator at datasets/00_commun_scripts/00_03_evaluate/evaluate.py.

The logic that used to live here (Dice/HD95, the per-case loop) was identical
across every dataset and now lives in the commun scripts. This file is kept
(same path, same CLI) so 06_01_evaluate_run.sh is unchanged. See the commun
evaluate.py header for the full CLI.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "00_commun_scripts" / "00_03_evaluate"))
from evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
