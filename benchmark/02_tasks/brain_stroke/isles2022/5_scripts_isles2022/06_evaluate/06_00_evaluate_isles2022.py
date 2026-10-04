#!/usr/bin/env python3
"""isles2022 evaluator -- thin shim over the shared, method-agnostic evaluator at
benchmark/00_commun_scripts/00_03_evaluate/evaluate.py (Dice / HD95 per case per label). Same CLI; see that header.
Label set (dataset.json of Dataset140/141): background 0, lesion 1 -> score `--labels lesion`; pred ids == GT ids, so no
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
