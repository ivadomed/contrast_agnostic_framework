#!/usr/bin/env python3
"""
Shared cross-experiment / cross-fold aggregation core for evaluation results.

Single source of truth for the aggregation primitives and report/heatmap builders
that were copy-pasted into brats2024-glioma and chaos `06_02_aggregate_results.py`
(and, in part, into amos/sliver07 `06_03_aggregate_results.py`). The per-dataset
scripts now import from here and supply only their title + which outputs to build.

Data shape produced by load_run():
    {metric: {contrast: {label: {fold: [values]}}}}
where metric ∈ {"dice", "hd95"}, contrast is the CSV `group` column (a modality),
label is the CSV `label` column, fold is "fold0".."fold3".

This module reads only `<run_dir>/fold*/eval_all.csv` (columns group,case,label,
dice,hd95) — the same contract every dataset's 06_01_evaluate_run.sh writes.
"""
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


# ── loaders ──────────────────────────────────────────────────────────────────

def load_run(metrics_dir: Path) -> dict:
    """Return {metric: {contrast: {label: {fold: [values]}}}} for all fold CSVs.

    metrics_dir is a single run's directory containing fold0/, fold1/, … each
    with an eval_all.csv.
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
                    data[key][contrast][label][fold].append(float(row[key]))
    return data


def load_fold0(metrics_dir: Path) -> dict:
    """Like load_run() but restricted to fold0 only.

    Returns {metric: {contrast: {label: [values]}}} — no fold dimension since
    it's a single fold; values are pooled across cases.
    """
    csv_path = metrics_dir / "fold0" / "eval_all.csv"
    if not csv_path.exists():
        return {}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            contrast, label = row["group"], row["label"]
            for key in ("dice", "hd95"):
                data[key][contrast][label].append(float(row[key]))
    return data


# ── stats ────────────────────────────────────────────────────────────────────

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


def fmt_cell(mean, std, precision):
    if not np.isfinite(mean):
        return "—"
    if precision == 4:
        return f"{mean*100:.1f}±{std*100:.1f}"
    return f"{mean:.1f}±{std:.1f}"


# ── reports ──────────────────────────────────────────────────────────────────

def build_report(runs: dict, out_path: Path, title: str) -> None:
    """Full per-contrast, per-label cross-fold mean±std tables + a cross-contrast
    summary. runs: {run_id: load_run() output}."""
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
        f"# {title} — Aggregated Results",
        f"",
        f"Generated: {now}  |  Experiments: {len(run_ids)}  |  "
        f"Contrasts: {', '.join(all_contrasts)}  |  Labels: {', '.join(all_labels)}",
        f"",
        f"Values are **cross-fold mean±std** (fold-level means averaged across folds 0-3).",
        f"— = no finite values available.",
        f"",
    ]

    for metric, mtitle, prec in (("dice", "Dice (↑)", 4), ("hd95", "HD95 mm (↓)", 2)):
        lines += [f"## {mtitle}", ""]

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


def build_modality_summary(runs: dict, md_path: Path, heatmap_dir: Path, title: str) -> None:
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
        f"# {title} — Summary by Modality",
        "",
        f"Generated: {now}  |  Experiments: {len(run_ids)}  |  Modalities: {', '.join(all_contrasts)}",
        "",
        "Each cell is the **cross-fold, cross-class average** (mean over all labels and "
        "folds). The `all` column averages across modalities. — = no finite data.",
        "",
    ]

    # matrices[metric] = (np.array [n_runs x n_cols], used for heatmap)
    matrices = {}
    for metric, mtitle, prec in (("dice", "Dice (↑)", 4), ("hd95", "HD95 mm (↓)", 2)):
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
        lines += [f"## {mtitle}", "",
                  "| experiment | " + " | ".join(cols) + " |",
                  "|" + "---|" * (len(cols) + 1)]
        for i, run_id in enumerate(run_ids):
            lines.append(f"| {run_id} | " + " | ".join(fmt(mat[i, j]) for j in range(len(cols))) + " |")
        lines.append("")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n→ {md_path}")

    save_heatmaps(matrices, run_ids, cols, heatmap_dir,
                  filename="02_01_heatmap_{metric}.png", subtitle="cross-fold, cross-class mean")


def save_heatmaps(matrices: dict, run_ids: list, cols: list, out_dir: Path,
                  filename: str = "heatmap_{metric}.png",
                  subtitle: str = "cross-fold, cross-class mean") -> None:
    """Render Dice/HD95 heatmaps (experiments × columns). filename is a template
    taking {metric}. Blue = good, red = bad for both metrics."""
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
    for metric, mtitle, prec, cmap in specs:
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
        ax.set_title(f"{mtitle} — {subtitle}")
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
        p = out_dir / filename.format(metric=metric)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"→ {p}")


def build_fold0_heatmap(metrics_dir: Path) -> None:
    """Heatmap of fold-0 Dice/HD95 per experiment × modality, saved alongside the
    cross-fold heatmaps. Reads fold0/eval_all.csv for every run that has one."""
    candidates = sorted(p for p in metrics_dir.iterdir()
                        if p.is_dir() and not p.name.startswith("."))
    fold0_data = {}
    for run_dir in candidates:
        d = load_fold0(run_dir)
        if d:
            fold0_data[run_dir.name] = d

    if not fold0_data:
        return

    all_contrasts = sorted({c for d in fold0_data.values() for c in d.get("dice", {})})
    run_ids = sorted(fold0_data)
    cols = all_contrasts + ["avg"]

    matrices = {}
    for metric in ("dice", "hd95"):
        mat = np.full((len(run_ids), len(cols)), np.nan)
        for i, run_id in enumerate(run_ids):
            by_contrast = fold0_data[run_id].get(metric, {})
            per_mod = []
            for j, c in enumerate(all_contrasts):
                vals = [v for lab_vals in by_contrast.get(c, {}).values()
                        for v in lab_vals if np.isfinite(v)]
                v = float(np.nanmean(vals)) if vals else np.nan
                mat[i, j] = v
                per_mod.append(v)
            finite = [v for v in per_mod if np.isfinite(v)]
            mat[i, -1] = float(np.mean(finite)) if finite else np.nan
        matrices[metric] = mat

    save_heatmaps(matrices, run_ids, cols, metrics_dir,
                  filename="02_02_heatmap_fold0_{metric}.png", subtitle="fold 0, cross-class mean")
