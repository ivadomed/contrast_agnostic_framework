#!/usr/bin/env python3
"""
LEGACY -- renamed off the canonical 06_02 slot (2026-09-27) now that a
config-driven equivalent exists: 06_02_aggregate_per_organ_from_config.sh ->
00_commun_scripts/00_03_evaluate/aggregate_per_organ_from_config.py, driven by
configs/sliver07_{t1in,t2spir}_00_comparison.yaml. Kept only for an ad-hoc
"show me everything under metrics_root" check; use the config-driven wrapper
for the real headline report.

LATENT BUG found and NOT carried into the replacement: this file's own hand-rolled
`load_run` below never imported eval_folds' EVAL_FOLDS/filter_fold_dirs cap, so it
silently included leftover fold3 data from before the 2026-07-09 3-fold policy
(4 folds' worth of data instead of the correct 3) -- confirmed via direct
side-by-side comparison during the 2026-09-27 migration (same fold0-2 values,
this script's mean shifted slightly by including a 4th, off-policy fold). The
config-driven replacement uses the shared eval_aggregate.load_run, which is
correctly fold-capped like every other aggregator in this project.

Aggregate SLIVER07 evaluation results across all chaos-trained methods.

Reads METRICS_ROOT/chaos_model/<contrast>/{CATEGORY}_{RUN_ID}/fold{k}/eval_all.csv
for every run found under the metrics root, computes cross-fold Dice and HD95
(liver label only), and writes a comparison Markdown table to the metrics root's
00_comparison.md.

This keeps its own liver-only report layout, but the cross-fold statistic comes
from the shared commun aggregation core. The fold loader here is the liver-only
variant (SLIVER07 GT annotates the liver alone), so it stays local.

Usage:
  python 06_96_aggregate_results_legacy.py
  python 06_96_aggregate_results_legacy.py --metrics_root <path>
"""
import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]
                       / "00_commun_scripts" / "00_00_utils"))
from eval_aggregate import cross_fold_stats  # noqa: E402

DATASET_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_ROOT = DATASET_ROOT / "8_results_sliver07" / "02_metrics"


def load_run(run_dir: Path) -> dict:
    """Return {metric: {fold: [values]}} for liver label across all folds."""
    data: dict = defaultdict(lambda: defaultdict(list))
    for fold_dir in sorted(run_dir.glob("fold*")):
        csv_path = fold_dir / "eval_all.csv"
        if not csv_path.exists():
            continue
        fold = fold_dir.name
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                if row.get("label") != "liver":
                    continue
                for key in ("dice", "hd95"):
                    v = float(row[key])
                    data[key][fold].append(v)
    return dict(data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics_root", type=Path, default=DEFAULT_METRICS_ROOT)
    args = ap.parse_args()

    root = args.metrics_root
    if not root.exists():
        raise SystemExit(f"METRICS_ROOT not found: {root}")

    runs: dict = {}
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        data = load_run(run_dir)
        if data:
            runs[run_dir.name] = data
        else:
            print(f"  skip {run_dir.name}: no eval_all.csv with liver rows found")

    if not runs:
        raise SystemExit("No evaluation data found. Run 06_03_evaluate_all_chaos.sh first.")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# SLIVER07 — MR→CT Generalization Results",
        "",
        f"Generated: {now}  |  Experiments: {len(runs)}  |  Modality: CT  |  Label: liver",
        "",
        "Values are **cross-fold mean ± std** (per-fold case means averaged across 4 folds).",
        "All models are trained on **chaos MR T1-DUAL in-phase** (cross-dataset inference).",
        "baseline is the MR-only control — expected to fail on CT.",
        "",
    ]

    for metric, title, fmt in (
        ("dice", "## Dice (↑, higher is better)", lambda m, s: f"{m*100:.1f}±{s*100:.1f}"),
        ("hd95", "## HD95 mm (↓, lower is better)", lambda m, s: f"{m:.1f}±{s:.1f}"),
    ):
        lines += [title, "", "| experiment | liver | n_folds |", "|---|---|---|"]
        for run_id, data in sorted(runs.items()):
            m, s, n = cross_fold_stats(data.get(metric, {}))
            val = fmt(m, s) if np.isfinite(m) else "—"
            lines.append(f"| {run_id} | {val} | {n} |")
        lines.append("")

    out = root / "00_comparison.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
