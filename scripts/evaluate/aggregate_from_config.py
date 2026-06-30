#!/usr/bin/env python3
"""
Config-driven cross-experiment aggregation of segmentation evaluation results.

Reads a YAML config listing short run IDs, resolves each to its directory under
metrics_dir (trying nnUNet_ and auglab_ prefixes automatically), computes
cross-fold mean±std Dice and HD95, and writes:
  {metrics_dir}/{output_prefix}_summary.md
  {metrics_dir}/{output_prefix}_heatmap_dice.png
  {metrics_dir}/{output_prefix}_heatmap_hd95.png

Config YAML format:
  title: "CHAOS T1in — results"
  metrics_dir: "${METRICS_ROOT}/chaos_model/t1in"   # env vars expanded
  output_prefix: "03_03_results"
  runs:
    - chaos_t1in_baseline_20260614_153230
    - chaos_t1in_v26_6_2_train050_val100_20260615_213615
    - ...

Optional config fields:
  in_domain_contrast: t1in   # highlights that column in red on the heatmap

Run key resolution order:
  1. nnUNet_{key}
  2. auglab_{key}
  3. {key}  (exact)

Run names containing "v26_6_2" are labelled "(Ours)" in all outputs.

Usage:
  python aggregate_from_config.py <config.yaml>
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import re

import numpy as np
import yaml

_PREFIXES = ("nnUNet_", "auglab_", "")

# ── multi-source helpers ─────────────────────────────────────────────────────

def load_run_from_sources(sources: list, key: str) -> tuple:
    """Load eval_all.csv data across multiple metrics dirs with column prefixing.

    Each source dict has:
      metrics_dir    Path
      column_prefix  str   (prepended to group name, e.g. "chaos_")
      column_rename  dict  (group → full col name, overrides prefix, e.g. {"ct": "chaos_ct_liver"})

    Returns (data_dict, max_fold_count).  data_dict has the same shape as
    load_run() so all downstream code (cross_fold_class_mean, build_summary…) works unchanged.
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    max_folds = 0
    for src in sources:
        metrics_dir = src["metrics_dir"]
        prefix = src.get("column_prefix", "")
        rename = src.get("column_rename", {})
        run_dir = resolve_run_dir(metrics_dir, key)
        if run_dir is None:
            continue
        n_folds = 0
        for fold_dir in sorted(run_dir.glob("fold*")):
            csv_path = fold_dir / "eval_all.csv"
            if not csv_path.exists():
                continue
            n_folds += 1
            with csv_path.open() as f:
                for row in csv.DictReader(f):
                    group = row["group"]
                    col = rename.get(group, f"{prefix}{group}")
                    label = row["label"]
                    fold = fold_dir.name
                    for k in ("dice", "hd95"):
                        data[k][col][label][fold].append(float(row[k]))
        max_folds = max(max_folds, n_folds)
    return data, max_folds

# Substrings to highlight in bold within run-name labels.
# Listed longest-first so overlapping patterns don't shadow each other.
_BOLD_RE = re.compile(
    r'synthseg_noEM|synthseg_EM|auglab_default|auglabAug|v26_6_2|\(Ours\)|baseline'
)


def _segment_label(s: str) -> list:
    """Split label string into [(text, is_bold), …] segments."""
    segs, last = [], 0
    for m in _BOLD_RE.finditer(s):
        if m.start() > last:
            segs.append((s[last:m.start()], False))
        segs.append((m.group(), True))
        last = m.end()
    if last < len(s):
        segs.append((s[last:], False))
    return segs


def _format_label_md(s: str) -> str:
    return "".join(f"**{t}**" if bold else t for t, bold in _segment_label(s))


def _format_label_mathtext(s: str) -> str:
    parts = []
    for text, bold in _segment_label(s):
        if bold:
            parts.append(r"$\mathbf{" + text.replace("_", r"\_") + r"}$")
        else:
            parts.append(text)
    return "".join(parts)


def _fmt_val(v: float, prec: int) -> str:
    if not np.isfinite(v):
        return "—"
    return f"{v * 100:.1f}" if prec == 4 else f"{v:.1f}"


def resolve_run_dir(metrics_dir: Path, key: str) -> Path | None:
    for prefix in _PREFIXES:
        p = metrics_dir / f"{prefix}{key}"
        if p.is_dir():
            return p
    return None


def load_run(run_dir: Path) -> dict:
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for fold_dir in sorted(run_dir.glob("fold*")):
        csv_path = fold_dir / "eval_all.csv"
        if not csv_path.exists():
            continue
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                contrast = row["group"]
                label = row["label"]
                fold = fold_dir.name
                for k in ("dice", "hd95"):
                    data[k][contrast][label][fold].append(float(row[k]))
    return data


def count_eval_folds(run_dir: Path) -> int:
    return sum(1 for fd in sorted(run_dir.glob("fold*")) if (fd / "eval_all.csv").exists())


def cross_fold_stats(per_fold: dict) -> tuple:
    fold_means = []
    for vs in per_fold.values():
        arr = np.array(vs, float)
        if np.isfinite(arr).any():
            fold_means.append(float(np.nanmean(arr)))
    if not fold_means:
        return float("nan"), float("nan"), 0
    fm = np.array(fold_means)
    return float(np.nanmean(fm)), float(np.nanstd(fm)), len(fm)


def cross_fold_class_mean(run_data: dict, metric: str, contrast: str) -> float:
    by_contrast = run_data.get(metric, {}).get(contrast, {})
    per_fold: dict = defaultdict(list)
    for lab_folds in by_contrast.values():
        for fold, vs in lab_folds.items():
            per_fold[fold].extend(vs)
    m, _, _ = cross_fold_stats(per_fold)
    return m


def _best_per_col(mat: np.ndarray, metric: str) -> list:
    """Return the best-row index per column (-1 if column is all NaN)."""
    best = []
    for j in range(mat.shape[1]):
        col = mat[:, j]
        if np.any(np.isfinite(col)):
            idx = int(np.nanargmax(col)) if metric == "dice" else int(np.nanargmin(col))
        else:
            idx = -1
        best.append(idx)
    return best


def _display_key(key: str) -> str:
    return f"{key} (Ours)" if "v26_6_2" in key else key


def build_summary(runs_ordered, runs_data, fold_counts, title, out_dir: Path, prefix: str,
                  in_domain_contrast: str = None, column_order: list = None):
    all_contrasts_set = {c for d in runs_data.values() for c in d.get("dice", {})}
    if column_order:
        ordered = [c for c in column_order if c in all_contrasts_set]
        remaining = sorted(c for c in all_contrasts_set if c not in ordered)
        all_contrasts = ordered + remaining
    else:
        all_contrasts = sorted(all_contrasts_set)
    if not all_contrasts:
        print("No eval data — nothing to aggregate.", file=sys.stderr)
        return

    cols = all_contrasts + ["all"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Column headers: bold the in-domain contrast
    col_headers = [f"**{c}**" if c == in_domain_contrast else c for c in cols]

    lines = [
        f"# {title}",
        "",
        f"Generated: {now}  |  Experiments: {len(runs_ordered)}  |  "
        f"Modalities: {', '.join(all_contrasts)}",
        "",
        "Each cell is the **cross-fold, cross-class average** (mean over all labels and folds). "
        "`all` = average across modalities. **Bold** = best per column. "
        + (f"**{in_domain_contrast}** = in-domain contrast. " if in_domain_contrast else "")
        + "— = no data.",
        "",
    ]

    matrices = {}
    for metric, heading, prec in (("dice", "Dice ↑", 4), ("hd95", "HD95 mm ↓", 2)):
        mat = np.full((len(runs_ordered), len(cols)), np.nan)
        for i, key in enumerate(runs_ordered):
            per_mod = [cross_fold_class_mean(runs_data[key], metric, c) for c in all_contrasts]
            for j, v in enumerate(per_mod):
                mat[i, j] = v
            finite = [v for v in per_mod if np.isfinite(v)]
            mat[i, -1] = float(np.mean(finite)) if finite else np.nan
        matrices[metric] = mat

        best = _best_per_col(mat, metric)

        lines += [
            f"## {heading}",
            "",
            "| experiment | folds | " + " | ".join(col_headers) + " |",
            "|" + "---|" * (len(cols) + 2),
        ]
        for i, key in enumerate(runs_ordered):
            nf = fold_counts.get(key, 0)
            cells = []
            for j in range(len(cols)):
                s = _fmt_val(mat[i, j], prec)
                if s != "—" and best[j] == i:
                    s = f"**{s}**"
                cells.append(s)
            lines.append(f"| {_format_label_md(_display_key(key))} | {nf} | " + " | ".join(cells) + " |")
        lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{prefix}_summary.md"
    md_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n→ {md_path}")

    _save_heatmaps(matrices, runs_ordered, fold_counts, cols, out_dir, prefix, title,
                   in_domain_contrast=in_domain_contrast)


def _save_heatmaps(matrices, runs_ordered, fold_counts, cols, out_dir: Path, prefix: str,
                   title: str, in_domain_contrast: str = None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (matplotlib unavailable, skipping heatmaps: {e})", file=sys.stderr)
        return

    ylabels = [
        _format_label_mathtext(f"{_display_key(k)}  ({fold_counts.get(k, 0)}f)")
        for k in runs_ordered
    ]
    in_domain_col = cols.index(in_domain_contrast) if in_domain_contrast in cols else None

    specs = (
        ("dice", "Dice ↑", 4, "RdBu"),
        ("hd95", "HD95 mm ↓", 2, "RdBu_r"),
    )
    for metric, heading, prec, cmap in specs:
        mat = matrices.get(metric)
        if mat is None:
            continue

        best = _best_per_col(mat, metric)

        n_runs, n_cols = mat.shape
        fig_w = max(8, 1.4 * n_cols + 5)
        fig_h = max(3, 0.55 * n_runs + 2)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, aspect="auto", cmap=cmap)

        # In-domain column: subtle background band drawn behind the image
        if in_domain_col is not None:
            ax.axvspan(in_domain_col - 0.5, in_domain_col + 0.5,
                       color="crimson", alpha=0.08, zorder=0)

        ax.set_xticks(range(n_cols))
        xlabels = ax.set_xticklabels(cols, rotation=30, ha="right")
        if in_domain_col is not None:
            xlabels[in_domain_col].set_color("crimson")
            xlabels[in_domain_col].set_fontweight("bold")
            xlabels[in_domain_col].set_fontsize(10)

        ax.set_yticks(range(n_runs))
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_title(f"{title} — {heading}", pad=10)

        for i in range(n_runs):
            for j in range(n_cols):
                v = mat[i, j]
                if np.isfinite(v):
                    txt = f"{v * 100:.1f}" if prec == 4 else f"{v:.1f}"
                    r, g, b, _ = im.cmap(im.norm(v))
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    fw = "bold" if best[j] == i else "normal"
                    ax.text(
                        j, i, txt,
                        ha="center", va="center", fontsize=8,
                        color="white" if lum < 0.5 else "black",
                        fontweight=fw,
                    )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("modality")
        ax.set_ylabel("experiment")
        fig.tight_layout()

        p = out_dir / f"{prefix}_heatmap_{metric}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"→ {p}")


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate eval results from a YAML config listing run IDs."
    )
    ap.add_argument("config", help="Path to YAML config file")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    output_prefix = cfg.get("output_prefix", "03_aggregated")
    title = cfg.get("title", "Aggregated Results")
    run_keys = cfg.get("runs", [])
    in_domain_contrast = cfg.get("in_domain_contrast", None)
    column_order = cfg.get("column_order", None)

    # Multi-source mode: config has a `sources` list.
    # Single-source mode (backward compat): config has `metrics_dir`.
    if "sources" in cfg:
        sources = [
            {
                "metrics_dir": Path(os.path.expandvars(s["metrics_dir"])),
                "column_prefix": s.get("column_prefix", ""),
                "column_rename": s.get("column_rename", {}),
            }
            for s in cfg["sources"]
        ]
        # output_dir: explicit key, or fall back to first source's metrics_dir
        if "output_dir" in cfg:
            out_dir = Path(os.path.expandvars(cfg["output_dir"]))
        else:
            out_dir = sources[0]["metrics_dir"]
        for src in sources:
            if not src["metrics_dir"].is_dir():
                print(f"  note: source metrics_dir does not exist yet: {src['metrics_dir']}", file=sys.stderr)
        multi_source = True
    else:
        metrics_dir = Path(os.path.expandvars(cfg["metrics_dir"]))
        sources = [{"metrics_dir": metrics_dir, "column_prefix": "", "column_rename": {}}]
        out_dir = metrics_dir
        if not metrics_dir.is_dir():
            print(f"  note: metrics_dir does not exist yet: {metrics_dir}", file=sys.stderr)
        multi_source = False

    runs_data = {}
    fold_counts = {}
    runs_ordered = []

    for key in run_keys:
        if multi_source:
            data, n_folds = load_run_from_sources(sources, key)
        else:
            run_dir = resolve_run_dir(sources[0]["metrics_dir"], key)
            if run_dir is None:
                print(f"  skip {key}: directory not found in {sources[0]['metrics_dir']}", file=sys.stderr)
                continue
            data = load_run(run_dir)
            n_folds = count_eval_folds(run_dir)
        if not data:
            print(f"  skip {key}: no eval_all.csv found in any source", file=sys.stderr)
            continue
        runs_data[key] = data
        fold_counts[key] = n_folds
        runs_ordered.append(key)

    if not runs_data:
        print("No runs with evaluation data found.", file=sys.stderr)
        sys.exit(1)

    build_summary(runs_ordered, runs_data, fold_counts, title, out_dir, output_prefix,
                  in_domain_contrast=in_domain_contrast, column_order=column_order)


if __name__ == "__main__":
    main()
