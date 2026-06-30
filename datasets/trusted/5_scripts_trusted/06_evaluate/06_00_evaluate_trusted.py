#!/usr/bin/env python3
"""
TRUSTED evaluator for chaos-trained model predictions — thin shim over the shared
metric core at datasets/00_commun_scripts/00_00_utils/eval_metrics.py.

What stays here is the TRUSTED-specific knowledge that makes this a *cross-dataset,
cross-label-space* evaluation:

  * TRUSTED ground truth is a SINGLE binary kidney label (gt_id = 1). CT volumes
    merge both kidneys into that one label; US volumes contain a single kidney.
  * chaos predicts the two kidneys SEPARATELY: right_kidney(2) + left_kidney(3).

So we score one organ, "kidney", whose prediction foreground is the UNION of chaos
labels {2, 3} against GT label 1. The merge is expressed as a triple with a list
pred_id — eval_metrics handles list pred_ids via np.isin (their union). This does
NOT penalize left/right confusion: in a cropped US field-of-view the model cannot
reliably tell side, and the question here is "is the kidney segmented at all" under
MR→{CT,US} domain shift.

This file keeps the same CLI shape as sliver07/amos 06_00 so 06_01_evaluate_run.sh
is a near-identical shim.

Usage:
  python 06_00_evaluate_trusted.py --pred_dir <dir> --gt_dir <dir> --out_csv <path>
                                   [--name <tag>] [--workers N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "00_commun_scripts" / "00_00_utils"))
import eval_metrics  # noqa: E402

# organ name → (pred_id, gt_id). pred_id is a LIST → union of chaos right_kidney(2)
# and left_kidney(3); gt_id is TRUSTED's single binary kidney label (1).
ORGAN_MAP = {
    "kidney": ([2, 3], 1),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True,
                    help="Directory of chaos-model predictions (chaos label IDs)")
    ap.add_argument("--gt_dir", required=True,
                    help="Directory of TRUSTED GT labels (binary kidney = 1)")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--name", default="",
                    help="Tag stored in the 'group' column (e.g. 'ct' or 'us')")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    triples = [(organ, pred_id, gt_id) for organ, (pred_id, gt_id) in ORGAN_MAP.items()]
    note = "chaos right_kidney(2)+left_kidney(3) MERGED → 'kidney' vs TRUSTED binary GT(1)"
    eval_metrics.run_evaluation(
        pred_dir=args.pred_dir, gt_dir=args.gt_dir, out_csv=args.out_csv,
        name=args.name, triples=triples, workers=args.workers, summary_note=note)


if __name__ == "__main__":
    main()
