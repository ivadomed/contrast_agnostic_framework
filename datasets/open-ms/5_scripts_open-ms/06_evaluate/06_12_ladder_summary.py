#!/usr/bin/env python3
"""
Clear, self-explanatory summary of the PALETTE causal-ablation ladder(s) — one
markdown table + one bar chart per ladder, isolating just the relevant rungs
(rather than the generic all-9-methods table/heatmap from 06_10).

A "ladder" is a sequence of runs where each step adds exactly one ingredient on
top of the previous step, so the Dice delta between consecutive rungs is
attributable to that one ingredient. Two ladders are supported out of the box:

  auglab  (floor = auglab_default, the full AugLab intensity-augmentation
           recipe with no synthesis): +K-means -> +Voronoi -> +real texture (PALETTE)
  baseline (floor = baseline, no augmentation at all): +K-means -> +Voronoi ->
           +real texture (PALETTE alone, i.e. v26_6_2-alone)

Reads the same fold*/eval_all.csv files (folds capped via EVAL_FOLD_INDICES, see
eval_folds.py) as 06_10/06_11 — no new evaluation, just a focused presentation.

Usage:
  python 06_12_ladder_summary.py --ladder auglab
  python 06_12_ladder_summary.py --ladder baseline
  python 06_12_ladder_summary.py --ladder both
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DATASET_ROOT = Path(__file__).resolve().parents[2]                 # datasets/open-ms
METRICS_ROOT = DATASET_ROOT / "8_results_open-ms/02_metrics/open_ms_model/flair"
ABLATIONS_ROOT = METRICS_ROOT / "ablations"   # exclusive-to-this-ladder run dirs + outputs live here

sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_00_utils"))
from eval_folds import EVAL_FOLDS  # noqa: E402

CONTRASTS = ["flair", "t2w", "t1w"]


def _resolve(prefix: str) -> str:
    """Return the ablations/ dir name starting with `prefix` (the timestamp suffix
    isn't known until the run has been launched/evaluated) — picks the most
    recently modified match, or returns `prefix` unchanged (and lets the caller
    report it as missing) if nothing matches yet."""
    matches = sorted(ABLATIONS_ROOT.glob(f"{prefix}*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return f"ablations/{matches[0].name}" if matches else prefix


def _rungs_auglab():
    return [
        ("auglab_default (floor)", "— (full AugLab recipe, no synthesis)",
         "auglab_open-ms_flair_auglab_default_20260706_061243"),
        ("no_voronoi", "+ K-means intensity clustering",
         "ablations/auglab_open-ms_flair_auglab_kmeans_label_remap_train025_val100_20260707_065414"),
        ("+voronoi", "+ Voronoi spatial sub-parcellation",
         "ablations/auglab_open-ms_flair_auglab_kmeans_label_remap_voronoi_train025_val100_20260707_065409"),
        ("PALETTE", "+ real-texture fill (vs. noise)",
         "ablations/auglab_open-ms_flair_auglabAug_v26_6_2_train025_val100_20260706_061243"),
    ]


def _rungs_baseline():
    return [
        ("baseline (floor)", "— (no augmentation at all)",
         "nnUNet_open-ms_flair_baseline_20260706_061243"),
        ("K-means+noise, alone", "+ K-means intensity clustering (SynthSeg-EM analog)",
         _resolve("nnUNet_open-ms_flair_baseline_kmeans_label_remap_train050_val100_")),
        ("+Voronoi, alone", "+ Voronoi spatial sub-parcellation",
         _resolve("nnUNet_open-ms_flair_baseline_kmeans_label_remap_voronoi_train050_val100_")),
        ("PALETTE alone", "+ real-texture fill (vs. noise)",
         "nnUNet_open-ms_flair_v26_6_2_train050_val100_20260706_061243"),
    ]


def load_case_means(run_dir: Path) -> dict:
    """{contrast: [per-case dice, mean over labels and EVAL_FOLDS]}."""
    per = defaultdict(lambda: defaultdict(list))
    for fold_name in EVAL_FOLDS:
        csv_path = run_dir / fold_name / "eval_all.csv"
        if not csv_path.exists():
            continue
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                try:
                    v = float(row["dice"])
                except (KeyError, ValueError):
                    continue
                if np.isfinite(v):
                    per[row["group"]][row["case"]].append(v)
    return {c: {k: float(np.mean(v)) for k, v in cases.items()} for c, cases in per.items()}


def render_ladder(name: str, rungs: list) -> tuple[str, dict]:
    """Returns (markdown, {contrast: [dice per rung]}) for plotting."""
    lines = [f"## {name}-anchored ladder", "",
             "Each rung adds exactly one ingredient on top of the previous rung — the "
             "Dice delta is attributable to that ingredient alone (every other AugLab "
             f"setting held fixed). Folds: {', '.join(EVAL_FOLDS)}.", "",
             "| rung | adds | " + " | ".join(f"{c} dice" for c in CONTRASTS) + " | Δ vs. prev (t1w/t2w) |",
             "|---|---|" + "---|" * len(CONTRASTS) + "---|"]

    series = {c: [] for c in CONTRASTS}
    prev = None
    missing = []
    for label, ingredient, run_key in rungs:
        run_dir = METRICS_ROOT / run_key
        if not run_dir.is_dir():
            missing.append(run_key)
            cells = ["—"] * len(CONTRASTS)
            for c in CONTRASTS:
                series[c].append(float("nan"))
        else:
            data = load_case_means(run_dir)
            means = {c: float(np.mean(list(data.get(c, {}).values()))) * 100
                     if data.get(c) else float("nan") for c in CONTRASTS}
            cells = [f"{means[c]:.1f}" for c in CONTRASTS]
            for c in CONTRASTS:
                series[c].append(means[c])
        delta = ""
        if prev is not None:
            dt1 = series["t1w"][-1] - prev["t1w"]
            dt2 = series["t2w"][-1] - prev["t2w"]
            delta = f"{dt1:+.1f} / {dt2:+.1f}"
        lines.append(f"| **{label}** | {ingredient} | " + " | ".join(cells) + f" | {delta} |")
        prev = {c: series[c][-1] for c in CONTRASTS}

    if missing:
        lines += ["", f"_Not yet available: {', '.join(missing)}_"]

    # Decomposition: % of total (floor -> top) gain contributed by each rung, per contrast.
    lines += ["", "**Decomposition of the total floor→top gain:**", ""]
    for c in ("t1w", "t2w"):
        vals = series[c]
        if any(np.isnan(v) for v in vals):
            continue
        total = vals[-1] - vals[0]
        if abs(total) < 1e-9:
            continue
        parts = []
        for i in range(1, len(vals)):
            step = vals[i] - vals[i - 1]
            parts.append(f"{rungs[i][1]}: {step/total*100:.0f}%")
        lines.append(f"- **{c.upper()}** (total +{total:.1f} pts): " + ", ".join(parts))

    return "\n".join(lines), series


def plot_ladder(name: str, rungs: list, series: dict, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (matplotlib unavailable, skipping plot: {e})", file=sys.stderr)
        return

    labels = [r[0] for r in rungs]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c, marker in zip(CONTRASTS, ("o", "s", "^")):
        ax.plot(x, series[c], marker=marker, label=c.upper(), linewidth=2)
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Dice (%)")
    ax.set_title(f"{name}-anchored causal ladder — open-ms FLAIR-trained, cross-contrast")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"→ {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ladder", choices=["auglab", "baseline", "both"], default="both")
    args = ap.parse_args()

    ladders = []
    if args.ladder in ("auglab", "both"):
        ladders.append(("auglab", _rungs_auglab()))
    if args.ladder in ("baseline", "both"):
        ladders.append(("baseline", _rungs_baseline()))

    all_md = ["# open-ms — PALETTE causal-ablation ladders", ""]
    for name, rungs in ladders:
        md, series = render_ladder(name, rungs)
        all_md.append(md)
        all_md.append("")
        plot_ladder(name, rungs, series, ABLATIONS_ROOT / f"ladder_{name}.png")

    out_path = ABLATIONS_ROOT / "ladders.md"
    out_path.write_text("\n".join(all_md) + "\n")
    print("\n".join(all_md))
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
