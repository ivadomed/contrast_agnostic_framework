#!/usr/bin/env python3
"""
totalseg-pelvic evaluator — thin shim over the shared, method-agnostic evaluator at
datasets/00_commun_scripts/00_03_evaluate/evaluate.py (Dice / HD95 per case per label).
Same CLI; see that header.

totalseg-pelvic scores the 10 pelvic/hip labels (see 02_nnunet/02_00_convert.py's
LABEL_MAP), same id space across ct/mri (both modalities' label maps were built by this
project's own conversion script using the identical LABEL_MAP, not the archives' own
per-structure files directly) — no --label_map remap needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "00_commun_scripts" / "00_03_evaluate"))
from evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
