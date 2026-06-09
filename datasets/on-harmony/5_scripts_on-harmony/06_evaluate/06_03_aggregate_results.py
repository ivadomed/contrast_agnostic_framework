#!/usr/bin/env python
"""
Aggregate evaluation results across contrasts and produce comparison tables.

Usage
-----
  .venv/bin/python scripts/nnunet_onharmony/05_aggregate_results.py <RUN_ID> [<RUN_ID> ...]
  .venv/bin/python scripts/nnunet_onharmony/05_aggregate_results.py --all

Reads
-----
  eval/onharmony/{RUN_ID}/{contrast}/metrics.json

Writes
------
  eval/onharmony/aggregate/eval_long.csv
  eval/onharmony/aggregate/eval_wide.csv
  eval/onharmony/aggregate/eval_summary.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT    = PROJECT_ROOT / "eval" / "onharmony"

CLASS_NAMES = {
    1: "cortical_gm",
    2: "white_matter",
    3: "csf_ventricles",
    4: "subcortical_gm",
    5: "brainstem",
    6: "cerebellum",
}

CONTRASTS = ["T1w", "T2w", "bold", "dwi_ap", "epi_ap", "gre_echo1_mag"]


def parse_metrics(metrics_json: Path, run_id: str, contrast: str) -> list[dict]:
    """Extract per-case per-class Dice and HD95 from nnUNetv2_evaluate_folder output."""
    if not metrics_json.exists():
        return []
    with open(metrics_json) as f:
        data = json.load(f)

    rows = []
    # nnUNet evaluate_folder format: {metric: {class: {case_id: value}}}
    # or newer format: {case_id: {metric_class: value}}
    # Handle both gracefully
    for case_id, case_metrics in data.get("metric_per_case", {}).items():
        for metric_name, class_vals in case_metrics.items():
            if metric_name not in ("Dice", "HD95"):
                continue
            for cls_str, val in class_vals.items():
                try:
                    cls = int(cls_str)
                except ValueError:
                    continue
                if cls == 0 or cls not in CLASS_NAMES:
                    continue
                rows.append({
                    "run_id":    run_id,
                    "method":    run_id.split("_")[0],
                    "contrast":  contrast,
                    "case_id":   case_id,
                    "class":     cls,
                    "class_name": CLASS_NAMES[cls],
                    "metric":    metric_name,
                    "value":     float(val) if val is not None else float("nan"),
                })
    return rows


def aggregate(run_ids: list[str]) -> None:
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    agg_dir = EVAL_ROOT / "aggregate"
    agg_dir.mkdir(exist_ok=True)

    all_rows = []
    for run_id in run_ids:
        for contrast in CONTRASTS:
            metrics_json = EVAL_ROOT / run_id / contrast / "metrics.json"
            rows = parse_metrics(metrics_json, run_id, contrast)
            all_rows.extend(rows)
            if rows:
                print(f"  {run_id} / {contrast}: {len(rows)} rows")
            else:
                print(f"  {run_id} / {contrast}: no data")

    if not all_rows:
        print("No results found.")
        return

    df = pd.DataFrame(all_rows)

    # ── eval_long.csv: one row per (run_id, contrast, case, class, metric) ──
    df.to_csv(agg_dir / "eval_long.csv", index=False)
    print(f"\nWritten: {agg_dir}/eval_long.csv ({len(df)} rows)")

    # ── eval_wide.csv: mean Dice per (run_id, contrast, class) ──────────────
    dice_df = df[df["metric"] == "Dice"]
    wide = (
        dice_df.groupby(["run_id", "method", "contrast", "class_name"])["value"]
        .mean()
        .reset_index()
        .pivot_table(index=["run_id", "method"], columns=["contrast", "class_name"], values="value")
    )
    wide.columns = ["_".join(c) for c in wide.columns]
    wide.to_csv(agg_dir / "eval_wide.csv")
    print(f"Written: {agg_dir}/eval_wide.csv")

    # ── eval_summary.md: mean Dice across classes per (method, contrast) ─────
    summary = (
        dice_df.groupby(["run_id", "method", "contrast"])["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "mean_dice"})
        .pivot_table(index=["run_id", "method"], columns="contrast", values="mean_dice")
    )
    # Add "overall" column
    summary["overall"] = summary.mean(axis=1)
    summary = summary.round(3)

    md_lines = ["# Segmentation Benchmark — Mean Dice Summary", ""]
    md_lines.append(summary.to_markdown())
    md_lines.append("")
    md_lines.append("_Mean Dice (excluding background class 0) across all test cases._")
    md_lines.append("_Contrasts: T1w, T2w, bold, dwi_ap, epi_ap, gre_echo1_mag._")

    with open(agg_dir / "eval_summary.md", "w") as f:
        f.write("\n".join(md_lines))
    print(f"Written: {agg_dir}/eval_summary.md")

    # Print quick overview
    print("\n── Mean Dice Summary ──")
    print(summary.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate segmentation benchmark results")
    parser.add_argument("run_ids", nargs="*", help="RUN_IDs to aggregate")
    parser.add_argument("--all", action="store_true", help="Aggregate all RUN_IDs found in eval/onharmony/")
    args = parser.parse_args()

    if args.all:
        run_ids = sorted(
            d.name for d in EVAL_ROOT.iterdir()
            if d.is_dir() and d.name != "aggregate"
        )
        print(f"Found {len(run_ids)} RUN_IDs: {run_ids}")
    elif args.run_ids:
        run_ids = args.run_ids
    else:
        parser.print_help()
        return

    aggregate(run_ids)


if __name__ == "__main__":
    main()
