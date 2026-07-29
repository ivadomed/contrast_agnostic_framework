#!/usr/bin/env python3
"""
ms3seg evaluator -- thin shim over the shared evaluate.py. Same CLI.

Unlike mslesseg/open-ms (direct 1:1 label match), MS3SEG's GT lesion label is a
DIFFERENT numeric id (255, an artifact of the source's uint8-grayscale mask
encoding) than open-ms's predicted lesion id (1). Callers pass
--label_map '{"lesion": [1, 255]}' (pred_id, gt_id) -- verified empirically by
spatial-overlap comparison against the archive's separate binary masks, see
00_utils/00_00_ingest_and_bidsify.py. Ventricles (64) and normal-WMH (191) are
excluded from scoring entirely (open-ms was never trained to predict them).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "00_commun_scripts" / "00_03_evaluate"))
from evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
