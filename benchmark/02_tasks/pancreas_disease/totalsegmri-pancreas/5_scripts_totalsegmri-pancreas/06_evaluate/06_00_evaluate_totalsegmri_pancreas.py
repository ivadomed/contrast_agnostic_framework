#!/usr/bin/env python3
"""totalsegmri-pancreas evaluator: thin shim over the shared evaluator benchmark/00_commun_scripts/00_03_evaluate/evaluate.py (Dice / HD95 per case per label); same CLI.
Labels: the SOURCE's dataset.json label set; GT numbering must match (else pass --label_map)."""
import os, sys
from pathlib import Path
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise SystemExit("PROJECT_ROOT not set -- source 00_utils/env.sh first")
sys.path.insert(0, str(Path(PROJECT_ROOT) / "benchmark" / "00_commun_scripts" / "00_03_evaluate"))
from evaluate import main  # noqa: E402
if __name__ == "__main__":
    main()
