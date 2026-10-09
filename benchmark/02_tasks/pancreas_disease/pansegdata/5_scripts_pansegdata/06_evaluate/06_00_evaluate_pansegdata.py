#!/usr/bin/env python3
"""pansegdata evaluator -- thin shim over the shared, method-agnostic evaluator at
benchmark/00_commun_scripts/00_03_evaluate/evaluate.py (Dice / HD95 per case per label). Same CLI; see that header.
Label set (dataset.json of Dataset150/151): background 0, pancreas 1 -> score `--labels pancreas`; pred ids == GT ids, so no
--label_map is needed. The shared evaluate_run_common.sh calls the commun evaluate.py directly; this shim exists for
ad-hoc use and for the canonical 06_00 slot."""
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
