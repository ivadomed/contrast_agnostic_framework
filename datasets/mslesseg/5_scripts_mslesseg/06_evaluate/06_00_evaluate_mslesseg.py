#!/usr/bin/env python3
"""
mslesseg evaluator — thin shim over the shared, method-agnostic evaluator at
datasets/00_commun_scripts/00_03_evaluate/evaluate.py (Dice / HD95 per case per label).
Same CLI; see the commun evaluate.py header.

Unlike TRUSTED (kidney: cross-label-space merge needed), MSLesSeg's ground truth is
a direct single binary lesion label (0/1) that matches open-ms's own dataset.json
{background: 0, lesion: 1} exactly — pred_id == gt_id, no --label_map needed. Call
with --dataset_json pointing at open-ms's Dataset070 dataset.json (OPENMS_DATASET_JSON,
set by 00_utils/env.sh).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "00_commun_scripts" / "00_03_evaluate"))
from evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
