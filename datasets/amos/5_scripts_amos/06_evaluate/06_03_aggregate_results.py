#!/usr/bin/env python3
"""
Aggregate AMOS evaluation results across all chaos-trained methods.

Reads METRICS_ROOT/chaos_models_{CATEGORY}_{RUN_ID}/fold{k}/eval_all.csv,
computes cross-fold Dice and HD95 per organ per modality, and writes a
comparison table to METRICS_ROOT/00_comparison.md.

Usage:
  python 06_03_aggregate_results.py
  python 06_03_aggregate_results.py --metrics_root <path>
"""
import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

DATASET_ROOT         = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_ROOT = DATASET_ROOT / "8_results_amos" / "02_metrics"

ORGAN_MAP = {
    "liver":        (1, 6),
    "right_kidney": (2, 2),
    "left_kidney":  (3, 3),
    "spleen":       (4, 1),
}
ORGANS = list(ORGAN_MAP)


def load_run(run_dir: Path) -> dict:
    """Return {metric: {modality: {organ: {fold: [values]}}}}."""
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for fold_dir in sorted(run_dir.glob("fold*")):
        csv_path = fold_dir / "eval_all.csv"
        if not csv_path.exists():
            continue
        fold = fold_dir.name
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                mod = row["group"]
                organ = row["label"]
                for key in ("dice", "hd95"):
                    data[key][mod][organ][fold].append(float(row[key]))
    return dict(data)


def cross_fold_stats(per_fold: dict) -> tuple:
    fold_means = [np.nanmean(np.array(vs, float))
                  for vs in per_fold.values()
                  if np.isfinite(np.array(vs, float)).any()]
    if not fold_means:
        return float("nan"), float("nan"), 0
    fm = np.array(fold_means)
    return float(np.nanmean(fm)), float(np.nanstd(fm)), len(fm)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics_root", type=Path, default=DEFAULT_METRICS_ROOT)
    args = ap.parse_args()

    root = args.metrics_root
    if not root.exists():
        raise SystemExit(f"METRICS_ROOT not found: {root}")

    runs: dict[str, dict] = {}
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        data = load_run(run_dir)
        if data:
            runs[run_dir.name] = data
        else:
            print(f"  skip {run_dir.name}: no eval_all.csv found")

    if not runs:
        raise SystemExit("No evaluation data. Run 06_02_evaluate_all_chaos.sh first.")

    all_mods = sorted({m for d in runs.values() for m in d.get("dice", {})})
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# AMOS — MR→CT/MRI Generalization Results (chaos-trained models)",
        "",
        f"Generated: {now}  |  Experiments: {len(runs)}",
        f"Modalities: {', '.join(all_mods)}  |  Organs: {', '.join(ORGANS)}",
        "",
        "Models trained on **chaos MR T1-DUAL in-phase** (cross-dataset inference).",
        "Predictions use chaos label IDs; GT uses AMOS label IDs — remapped per organ.",
        "See 06_00_evaluate_amos.py ORGAN_MAP for the explicit id→id mapping.",
        "baseline = MR-only control (expected to struggle on CT).",
        "",
    ]

    for metric, title, fmt in (
        ("dice", "## Dice (↑, higher is better)", "{:.4f}±{:.4f}"),
        ("hd95", "## HD95 mm (↓, lower is better)", "{:.2f}±{:.2f}"),
    ):
        lines += [title, ""]
        for mod in all_mods:
            lines += [f"### {mod.upper()}", "",
                      "| experiment | " + " | ".join(ORGANS) + " | n_folds |",
                      "|" + "---|" * (len(ORGANS) + 2)]
            for run_id, data in sorted(runs.items()):
                cells, n_folds = [], 0
                for organ in ORGANS:
                    per_fold = data.get(metric, {}).get(mod, {}).get(organ, {})
                    m, s, n = cross_fold_stats(per_fold)
                    cells.append(fmt.format(m, s) if np.isfinite(m) else "—")
                    n_folds = max(n_folds, n)
                lines.append(f"| {run_id} | " + " | ".join(cells) + f" | {n_folds} |")
            lines.append("")

    out = root / "00_comparison.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
