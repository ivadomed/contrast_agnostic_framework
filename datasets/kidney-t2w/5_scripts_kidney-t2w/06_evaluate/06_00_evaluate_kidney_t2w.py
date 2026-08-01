#!/usr/bin/env python3
"""
KIDNEY-T2W evaluator for chaos-trained model predictions — thin shim over the
shared metric core, mirroring TRUSTED's evaluator exactly (same cross-label-space
situation: chaos predicts right_kidney(2)/left_kidney(3) SEPARATELY, KIDNEY-T2W
GT is a single binary kidney label (gt_id=1) merging both kidneys).

Usage:
  python 06_00_evaluate_kidney_t2w.py --pred_dir <dir> --gt_dir <dir> --out_csv <path>
                                      [--name <tag>] [--workers N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "00_commun_scripts" / "00_00_utils"))
import eval_metrics  # noqa: E402

ORGAN_MAP = {
    "kidney": ([2, 3], 1),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--workers", type=int, default=8)
    eval_metrics.add_fov_args(ap)
    args = ap.parse_args()

    triples = [(organ, pred_id, gt_id) for organ, (pred_id, gt_id) in ORGAN_MAP.items()]
    note = "chaos right_kidney(2)+left_kidney(3) MERGED -> 'kidney' vs KIDNEY-T2W binary GT(1)"
    eval_metrics.run_evaluation(
        pred_dir=args.pred_dir, gt_dir=args.gt_dir, out_csv=args.out_csv,
        name=args.name, triples=triples, workers=args.workers, summary_note=note,
        fov=eval_metrics.fov_from_args(args))


if __name__ == "__main__":
    main()
