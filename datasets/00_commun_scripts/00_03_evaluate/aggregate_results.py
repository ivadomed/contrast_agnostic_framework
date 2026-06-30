#!/usr/bin/env python3
"""
Common cross-experiment / cross-fold aggregator (shared across datasets).

Reads each run's fold{0..3}/eval_all.csv under --metrics_dir, computes cross-fold
mean±std Dice and HD95 per contrast per label, and writes:
  {metrics_dir}/02_00_aggregated_metrics.md     full per-contrast/per-label tables
  {metrics_dir}/02_01_summary_by_modality.md     compact experiment × modality grid
  {metrics_dir}/02_01_heatmap_{dice,hd95}.png    heatmaps for the grid
  {metrics_dir}/02_02_heatmap_fold0_*.png         (only with --fold0-heatmap)

This replaces the per-dataset copies of `06_02_aggregate_results.py`; those files
now delegate here and supply only their --title (and --fold0-heatmap for chaos).
All logic lives in 00_00_utils/eval_aggregate.py.

Usage:
  python aggregate_results.py --metrics_dir <DIR> --title "CHAOS" [--run_keys K...] [--fold0-heatmap]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "00_00_utils"))
import eval_aggregate as agg  # noqa: E402


def main(default_title="Aggregated", default_fold0=False):
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", required=True,
                    help="METRICS_ROOT dir containing {category}_{run_id}/ subdirs")
    ap.add_argument("--title", default=default_title,
                    help="dataset title used in report headings")
    ap.add_argument("--run_keys", nargs="*",
                    help="Specific {category}_{run_id} keys to include (default: all with data)")
    ap.add_argument("--fold0-heatmap", action="store_true", default=default_fold0,
                    help="also emit fold-0 heatmaps (02_02_heatmap_fold0_*.png)")
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_path = metrics_dir / "02_00_aggregated_metrics.md"

    if args.run_keys:
        candidates = [metrics_dir / k for k in args.run_keys]
    else:
        candidates = sorted(p for p in metrics_dir.iterdir()
                            if p.is_dir() and not p.name.startswith("."))

    runs = {}
    for run_dir in candidates:
        data = agg.load_run(run_dir)
        if data:
            runs[run_dir.name] = data
        else:
            print(f"  skip {run_dir.name}: no eval_all.csv found in any fold", file=sys.stderr)

    if not runs:
        print("No runs with evaluation data found.", file=sys.stderr)
        sys.exit(1)

    agg.build_report(runs, out_path, args.title)
    agg.build_modality_summary(runs, metrics_dir / "02_01_summary_by_modality.md",
                               metrics_dir, args.title)
    if args.fold0_heatmap:
        agg.build_fold0_heatmap(metrics_dir)


if __name__ == "__main__":
    main()
