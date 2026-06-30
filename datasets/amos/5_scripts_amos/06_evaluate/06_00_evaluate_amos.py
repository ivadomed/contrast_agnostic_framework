#!/usr/bin/env python3
"""
AMOS evaluator for chaos-trained model predictions — thin shim over the shared
metric core at datasets/00_commun_scripts/00_00_utils/eval_metrics.py.

The Dice/HD95 + per-case loop now lives in the commun scripts. What stays here is
the AMOS-specific knowledge that makes this a *cross-dataset* evaluation: the
prediction label space (chaos) differs from the GT label space (AMOS), so each
organ is scored by name against its correct id in EACH space.

LABEL REMAP — chaos pred-id → AMOS gt-id, per organ:
  organ          chaos_id  amos_id
  liver             1         6
  right_kidney      2         2
  left_kidney       3         3
  spleen            4         1
Direct id comparison would be WRONG (chaos pred=1 is liver but amos gt=1 is spleen).

This file keeps the same path + CLI so 06_01_evaluate_run.sh is unchanged.

Usage:
  python 06_00_evaluate_amos.py --pred_dir <dir> --gt_dir <dir> --out_csv <path>
                                [--name <tag>] [--workers N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "00_commun_scripts" / "00_00_utils"))
import eval_metrics  # noqa: E402

# organ name → (chaos_pred_id, amos_gt_id). The only organs chaos can predict
# that AMOS also annotates. Generated from chaos dataset.json + AMOS dataset.json.
ORGAN_MAP = {
    "liver":        (1, 6),
    "right_kidney": (2, 2),
    "left_kidney":  (3, 3),
    "spleen":       (4, 1),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True,
                    help="Directory of chaos-model predictions (chaos label IDs)")
    ap.add_argument("--gt_dir", required=True,
                    help="Directory of AMOS GT labels (AMOS label IDs)")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--name", default="",
                    help="Tag stored in the 'group' column (e.g. 'ct' or 'mri')")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    triples = [(organ, chaos_id, amos_id)
               for organ, (chaos_id, amos_id) in ORGAN_MAP.items()]
    note = "chaos pred-ids remapped to AMOS GT-ids per organ (see ORGAN_MAP)"
    eval_metrics.run_evaluation(
        pred_dir=args.pred_dir, gt_dir=args.gt_dir, out_csv=args.out_csv,
        name=args.name, triples=triples, workers=args.workers, summary_note=note)


if __name__ == "__main__":
    main()
