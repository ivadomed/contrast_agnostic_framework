#!/usr/bin/env python3
"""
isles2022 evaluator -- thin shim over the shared, method-agnostic evaluator at
benchmark/00_commun_scripts/00_03_evaluate/evaluate.py (Dice / HD95 per case per
label). Same CLI; see that header.

TODO: describe this dataset's label set here (ids, any --label_map remap needed
for cross-dataset evaluation, etc.) -- see toothfairy2's or totalseg-pelvic's
06_00_evaluate.py for real examples of this docstring.
"""
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
