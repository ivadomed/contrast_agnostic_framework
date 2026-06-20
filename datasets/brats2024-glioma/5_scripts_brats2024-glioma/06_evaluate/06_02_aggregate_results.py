#!/usr/bin/env python3
"""
Cross-experiment, cross-fold aggregation of BraTS 2024 Glioma evaluation results.

Reads predictions/fold{0..3}/eval/eval_all.csv for each run under --results_dir,
computes cross-fold mean±std Dice and HD95 per contrast per label, and writes a
Markdown report to --out_dir.

Usage:
  python 06_aggregate_results.py --results_dir <nnUNet_results> --out_dir <03_aggregated>
  python 06_02_aggregate_results.py --metrics_dir <METRICS_ROOT> \
      --run_keys nnUNet_brats2024-glioma_t1n_v26_6_2_train090_val000_20260608_003445 nnUNet_brats2024-glioma_t1n_baseline_20260606_162001
"""
import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def load_run(metrics_dir: Path) -> dict:
    """Return {metric: {contrast: {label: {fold: [values]}}}} for all fold CSVs.

    metrics_dir is the run's directory under METRICS_ROOT, e.g.
    02_metrics/nnUNet_brats2024-glioma_t1n_v26_6_2_train090_val000_20260608_003445/.  It contains fold0/, fold1/, …
    each with an eval_all.csv.
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for fold_dir in sorted(metrics_dir.glob("fold*")):
        csv_path = fold_dir / "eval_all.csv"
        if not csv_path.exists():
            continue
        fold = fold_dir.name  # "fold0", "fold1", …
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                contrast = row["group"]
                label = row["label"]
                for key in ("dice", "hd95"):
                    v = float(row[key])
                    data[key][contrast][label][fold].append(v)
    return data


def cross_fold_stats(per_fold: dict) -> tuple:
    """Given {fold: [values]}, return (mean, std, n_folds) across fold means."""
    fold_means = []
    for vs in per_fold.values():
        arr = np.array(vs, float)
        if np.isfinite(arr).any():
            fold_means.append(np.nanmean(arr))
    if not fold_means:
        return float("nan"), float("nan"), 0
    fm = np.array(fold_means)
    return float(np.nanmean(fm)), float(np.nanstd(fm)), len(fm)


def fmt_cell(mean, std, precision):
    if not np.isfinite(mean):
        return "—"
    if precision == 4:
        return f"{mean*100:.1f}±{std*100:.1f}"
    return f"{mean:.1f}±{std:.1f}"


def cross_fold_class_mean(run_data: dict, metric: str, contrast: str) -> float:
    """Cross-fold AND cross-class average for one (experiment, metric, contrast).

    For each fold, pool all per-case values across every label (nan-aware mean),
    then average those per-fold means across folds. Returns NaN if no finite data.
    """
    by_contrast = run_data.get(metric, {}).get(contrast, {})
    per_fold = defaultdict(list)
    for lab_folds in by_contrast.values():        # {label: {fold: [values]}}
        for fold, vs in lab_folds.items():
            per_fold[fold].extend(vs)
    m, _, _ = cross_fold_stats(per_fold)
    return m


def build_modality_summary(runs: dict, md_path: Path, heatmap_dir: Path) -> None:
    """Compact summary: rows = experiments, columns = modalities, value = cross-fold
    cross-class average. One table (+ heatmap) per metric. Adds a trailing 'all'
    column averaging across modalities."""
    all_contrasts = sorted({c for d in runs.values() for c in d.get("dice", {})})
    run_ids = sorted(runs)
    if not all_contrasts:
        print("No eval data — skipping modality summary.", file=sys.stderr)
        return

    cols = all_contrasts + ["all"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# BraTS 2024 Glioma — Summary by Modality",
        "",
        f"Generated: {now}  |  Experiments: {len(run_ids)}  |  Modalities: {', '.join(all_contrasts)}",
        "",
        "Each cell is the **cross-fold, cross-class average** (mean over all labels and "
        "folds). The `all` column averages across modalities. — = no finite data.",
        "",
    ]

    # matrices[metric] = (np.array [n_runs x n_cols], used for heatmap)
    matrices = {}
    for metric, title, prec in (("dice", "Dice (↑)", 4), ("hd95", "HD95 mm (↓)", 2)):
        mat = np.full((len(run_ids), len(cols)), np.nan)
        for i, run_id in enumerate(run_ids):
            per_mod = [cross_fold_class_mean(runs[run_id], metric, c) for c in all_contrasts]
            for j, v in enumerate(per_mod):
                mat[i, j] = v
            finite = [v for v in per_mod if np.isfinite(v)]
            mat[i, -1] = float(np.mean(finite)) if finite else np.nan
        matrices[metric] = mat

        fmt = (lambda v: "—" if not np.isfinite(v) else f"{v*100:.1f}") if prec == 4 \
            else (lambda v: "—" if not np.isfinite(v) else f"{v:.1f}")
        lines += [f"## {title}", "",
                  "| experiment | " + " | ".join(cols) + " |",
                  "|" + "---|" * (len(cols) + 1)]
        for i, run_id in enumerate(run_ids):
            lines.append(f"| {run_id} | " + " | ".join(fmt(mat[i, j]) for j in range(len(cols))) + " |")
        lines.append("")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n→ {md_path}")

    _save_heatmaps(matrices, run_ids, cols, heatmap_dir)


def _save_heatmaps(matrices: dict, run_ids: list, cols: list, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"  (matplotlib unavailable, skipping heatmaps: {e})", file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    # Red/blue diverging: blue = good, red = bad for BOTH metrics.
    #   Dice (higher better): RdBu    → high→blue (good), low→red (bad)
    #   HD95 (lower better):  RdBu_r  → low→blue (good), high→red (bad)
    specs = (("dice", "Dice (↑)", 4, "RdBu"),
             ("hd95", "HD95 mm (↓)", 2, "RdBu_r"))
    for metric, title, prec, cmap in specs:
        mat = matrices.get(metric)
        if mat is None:
            continue
        fig_w = max(6, 1.1 * len(cols) + 3)
        fig_h = max(2.5, 0.5 * len(run_ids) + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(cols)), cols)
        ax.set_yticks(range(len(run_ids)), run_ids)
        ax.tick_params(axis='y', labelsize=7)
        ax.set_title(f"{title} — cross-fold, cross-class mean")
        # annotate each cell; pick text colour by the cell's actual luminance
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isfinite(v):
                    txt = f"{v*100:.1f}" if prec == 4 else f"{v:.1f}"
                    r, g, b, _ = im.cmap(im.norm(v))
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                            color="white" if lum < 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("modality"); ax.set_ylabel("experiment")
        fig.tight_layout()
        p = out_dir / f"02_01_heatmap_{metric}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"→ {p}")


def build_report(runs: dict, out_path: Path) -> None:
    """
    runs: {run_id: data} where data = load_run() output
    """
    # Collect all contrasts and labels across all runs
    all_contrasts = sorted({c for d in runs.values() for c in d.get("dice", {})})
    all_labels = sorted({
        lab
        for d in runs.values()
        for c_data in d.get("dice", {}).values()
        for lab in c_data
    })

    if not all_contrasts or not all_labels:
        print("No eval data found — nothing to aggregate.", file=sys.stderr)
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    run_ids = sorted(runs)

    lines = [
        f"# BraTS 2024 Glioma — Aggregated Results",
        f"",
        f"Generated: {now}  |  Experiments: {len(run_ids)}  |  "
        f"Contrasts: {', '.join(all_contrasts)}  |  Labels: {', '.join(all_labels)}",
        f"",
        f"Values are **cross-fold mean±std** (fold-level means averaged across folds 0-3).",
        f"— = no finite values available.",
        f"",
    ]

    for metric, title, prec in (("dice", "Dice (↑)", 4), ("hd95", "HD95 mm (↓)", 2)):
        lines += [f"## {title}", ""]

        for contrast in all_contrasts:
            header = f"| experiment | " + " | ".join(all_labels) + " | folds |"
            sep = "|" + "---|" * (len(all_labels) + 2)
            lines += [f"### {contrast}", "", header, sep]
            for run_id in run_ids:
                data = runs[run_id].get(metric, {}).get(contrast, {})
                cells = []
                n_folds = 0
                for lab in all_labels:
                    per_fold = data.get(lab, {})
                    m, s, n = cross_fold_stats(per_fold)
                    cells.append(fmt_cell(m, s, prec))
                    n_folds = max(n_folds, n)
                lines.append(f"| {run_id} | " + " | ".join(cells) + f" | {n_folds} |")
            lines.append("")

    # Summary: mean across all contrasts per label per experiment
    lines += ["## Summary — mean across all contrasts", ""]
    header = "| experiment | " + " | ".join(f"Dice {l}" for l in all_labels) + " | folds |"
    sep = "|" + "---|" * (len(all_labels) + 2)
    lines += [header, sep]
    for run_id in run_ids:
        data = runs[run_id].get("dice", {})
        cells = []
        n_folds = 0
        for lab in all_labels:
            all_fold_vals = defaultdict(list)
            for contrast in all_contrasts:
                for fold, vs in data.get(contrast, {}).get(lab, {}).items():
                    all_fold_vals[fold].extend(vs)
            m, s, n = cross_fold_stats(all_fold_vals)
            cells.append(fmt_cell(m, s, 4))
            n_folds = max(n_folds, n)
        lines.append("| " + run_id + " | " + " | ".join(cells) + f" | {n_folds} |")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n→ {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", required=True,
                    help="METRICS_ROOT dir containing {category}_{run_id}/ subdirs")
    ap.add_argument("--run_keys", nargs="*",
                    help="Specific {category}_{run_id} keys to include (default: all with data)")
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
        data = load_run(run_dir)
        if data:
            runs[run_dir.name] = data
        else:
            print(f"  skip {run_dir.name}: no eval_all.csv found in any fold", file=sys.stderr)

    if not runs:
        print("No runs with evaluation data found.", file=sys.stderr)
        sys.exit(1)

    build_report(runs, out_path)
    build_modality_summary(runs, metrics_dir / "02_01_summary_by_modality.md", metrics_dir)


if __name__ == "__main__":
    main()
