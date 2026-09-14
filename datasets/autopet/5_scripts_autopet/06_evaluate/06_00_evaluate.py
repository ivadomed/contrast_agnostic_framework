#!/usr/bin/env python3
"""
autopet evaluator — thin shim over the shared, method-agnostic evaluator at
datasets/00_commun_scripts/00_03_evaluate/evaluate.py (Dice / HD95 per case per label).
Same CLI; see that header.

autopet scores ONE foreground label — lesion (binary tumor-lesion mask) — same id space
across ct/pet/psma_ct/psma_pet (the archive's dataset.json labels {"background": 0,
"lesion": 1} apply uniformly; PSMA is scored with the SAME label id, since it's the same
annotation protocol, just a different cohort/tracer — no --label_map remap needed).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "00_commun_scripts" / "00_03_evaluate"))
from evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
