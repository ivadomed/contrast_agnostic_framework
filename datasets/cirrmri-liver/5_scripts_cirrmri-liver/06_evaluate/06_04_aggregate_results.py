#!/usr/bin/env python3
"""
Aggregate CIRRMRI-LIVER evaluation results across all chaos-trained methods.

Reads METRICS_ROOT/chaos_model/<contrast>/{CATEGORY}_{RUN_ID}/fold{k}/eval_all.csv
for every run found under the metrics root, computes cross-fold Dice and HD95
(liver label only) PER MODALITY GROUP (t1, t2 -- CIRRMRI-LIVER has two contrasts,
unlike msd-spleen/sliver07's single CT item), and writes a comparison Markdown
table to the metrics root's 00_comparison.md.

Usage:
  python 06_04_aggregate_results.py
  python 06_04_aggregate_results.py --metrics_root <path>
"""
import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "00_commun_scripts" / "00_00_utils"))
from eval_aggregate import cross_fold_stats  # noqa: E402

DATASET_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_ROOT = DATASET_ROOT / "8_results_cirrmri-liver" / "02_metrics"
GROUPS = ("t1", "t2")


def load_run(run_dir: Path) -> dict:
    """Return {group: {metric: {fold: [values]}}} for liver label across all folds."""
    data = {g: defaultdict(lambda: defaultdict(list)) for g in GROUPS}
    for fold_dir in sorted(run_dir.glob("fold*")):
        csv_path = fold_dir / "eval_all.csv"
        if not csv_path.exists():
            continue
        fold = fold_dir.name
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                if row.get("label") != "liver":
                    continue
                group = row.get("group")
                if group not in GROUPS:
                    continue
                for key in ("dice", "hd95"):
                    v = float(row[key])
                    data[group][key][fold].append(v)
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics_root", type=Path, default=DEFAULT_METRICS_ROOT)
    args = ap.parse_args()

    root = args.metrics_root
    if not root.exists():
        raise SystemExit(f"METRICS_ROOT not found: {root}")

    runs = {}
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        data = load_run(run_dir)
        if any(data[g]["dice"] for g in GROUPS):
            runs[run_dir.name] = data
        else:
            print(f"  skip {run_dir.name}: no eval_all.csv with liver rows found")

    if not runs:
        raise SystemExit("No evaluation data found. Run 06_02/06_03_evaluate_all_*.sh first.")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# CIRRMRI-LIVER — MRI->MRI Generalization Results",
        "",
        f"Generated: {now}  |  Experiments: {len(runs)}  |  Modalities: T1w, T2w  |  Label: liver",
        "",
        "Values are **cross-fold mean ± std** (per-fold case means averaged across folds 0-2).",
        "All models are trained on **chaos MR** (cross-dataset MRI->MRI inference — different "
        "scanner/protocol/pathology cohort than CHAOS's own training data).",
        "baseline is the plain nnU-Net control (no domain randomization).",
        "",
    ]

    for metric, title, fmt in (
        ("dice", "## Dice (↑, higher is better)", lambda m, s: f"{m*100:.1f}±{s*100:.1f}"),
        ("hd95", "## HD95 mm (↓, lower is better)", lambda m, s: f"{m:.1f}±{s:.1f}"),
    ):
        lines += [title, "", "| experiment | t1 (T1w) | t2 (T2w) | n_folds |", "|---|---|---|---|"]
        for run_id, data in sorted(runs.items()):
            vals = []
            n_folds = 0
            for g in GROUPS:
                m, s, n = cross_fold_stats(data[g].get(metric, {}))
                vals.append(fmt(m, s) if np.isfinite(m) else "—")
                n_folds = max(n_folds, n)
            lines.append(f"| {run_id} | {vals[0]} | {vals[1]} | {n_folds} |")
        lines.append("")

    out = root / "00_comparison.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
