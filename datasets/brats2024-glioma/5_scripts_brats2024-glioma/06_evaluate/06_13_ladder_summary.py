#!/usr/bin/env python3
"""
BraTS T1n causal-ablation ladder — baseline-anchored: each rung adds exactly one
ingredient on top of the previous rung (K-means -> +label-remap -> +Voronoi
sub-parcellation [noise fill] -> +real-texture fill [PALETTE alone]), so the OOD
Dice/HD95 delta between consecutive rungs is attributable to that one ingredient
(everything else held fixed). Mirrors open-ms's 06_12_ladder_summary.py pattern;
non-headline run dirs live under ablations/ (see CLAUDE.md's 02_metrics subdir
convention) — reads them from there, writes its own outputs there too.

Reads the same fold*/eval_all.csv files used by significance_from_config.py (no new
evaluation, just a focused presentation). OOD = mean over held-out contrasts
(t1c, t2w, t2f), excluding the training contrast (t1n).

Usage:
  .venv/bin/python 06_13_ladder_summary.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

DATASET_ROOT = Path(__file__).resolve().parents[2]   # datasets/brats2024-glioma
METRICS_ROOT = DATASET_ROOT / "8_results_brats2024-glioma/02_metrics/brats2024_glioma_model/t1n"
ABLATIONS_ROOT = METRICS_ROOT / "ablations"

IN_DOMAIN = "t1n"
OOD_CONTRASTS = ["t1c", "t2w", "t2f"]

RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     "brats2024-glioma_t1n_baseline_20260622_044535"),
    ("+kmeans", "+ K-means intensity clustering",
     "ablations/auglab_brats2024-glioma_t1n_baseline_kmeans_20260730_200711"),
    ("+label_remap", "+ label remap",
     "ablations/auglab_brats2024-glioma_t1n_baseline_kmeans_label_remap_20260730_200711"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     "ablations/auglab_brats2024-glioma_t1n_baseline_kmeans_label_remap_voronoi_20260730_200711"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     "ablations/nnUNet_brats2024-glioma_t1n_v26_6_2_train050_val100_20260730_200711"),
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     "auglab_brats2024-glioma_t1n_auglabAug_v26_6_2_train050_val000_20260710_040303"),
    ("+AugLab (val100)", "+ 100%-synth validation",
     "auglab_brats2024-glioma_t1n_auglabAug_v26_6_2_train050_val100_20260725_113540"),
]


def load_case_means(run_dir: Path, metric: str) -> dict:
    """{contrast: {case: mean metric over labels and folds}}."""
    per = {}
    for fold_dir in sorted(run_dir.glob("fold*")):
        csv_path = fold_dir / "eval_all.csv"
        if not csv_path.exists():
            continue
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                c = row["contrast"] if "contrast" in row else row.get("group")
                try:
                    v = float(row[metric])
                except (KeyError, ValueError):
                    continue
                if not np.isfinite(v):
                    continue
                per.setdefault(c, {}).setdefault(row["case"], []).append(v)
    return {c: {k: float(np.mean(v)) for k, v in cases.items()} for c, cases in per.items()}


def _resolve_run_dir(run_key: str) -> Path:
    """run_key may already carry a category prefix (nnUNet_/auglab_) or an
    ablations/ prefix; if the bare path doesn't exist, try both category prefixes
    (mirrors significance_from_config.py's resolve_run_dir)."""
    p = METRICS_ROOT / run_key
    if p.is_dir():
        return p
    parent, name = p.parent, p.name
    for prefix in ("nnUNet_", "auglab_"):
        cand = parent / f"{prefix}{name}"
        if cand.is_dir():
            return cand
    return p


def rung_means(run_key: str, metric: str):
    run_dir = _resolve_run_dir(run_key)
    if not run_dir.is_dir():
        return None, None
    data = load_case_means(run_dir, metric)
    scale = 100 if metric == "dice" else 1
    ood_vals = [float(np.mean(list(data[c].values()))) for c in OOD_CONTRASTS if data.get(c)]
    ood = float(np.mean(ood_vals)) * scale if ood_vals else float("nan")
    ind = float(np.mean(list(data[IN_DOMAIN].values()))) * scale if data.get(IN_DOMAIN) else float("nan")
    return ood, ind


def main():
    series = {"dice": [], "hd95": []}
    lines = ["# BraTS T1n — causal ablation ladder", "",
             "Each rung adds exactly one ingredient on top of the previous rung — the "
             "OOD Dice/HD95 delta is attributable to that ingredient alone. "
             f"OOD = mean over held-out contrasts ({', '.join(OOD_CONTRASTS)}), "
             f"training contrast ({IN_DOMAIN}) excluded.", "",
             "| rung | adds | OOD Dice | OOD HD95 | Δ Dice vs. prev | Δ HD95 vs. prev |",
             "|---|---|---|---|---|---|"]
    prev_dice = prev_hd95 = None
    missing = []
    for label, ingredient, run_key in RUNGS:
        ood_dice, _ = rung_means(run_key, "dice")
        ood_hd95, _ = rung_means(run_key, "hd95")
        if ood_dice is None:
            missing.append(run_key)
            ood_dice = ood_hd95 = float("nan")
        series["dice"].append(ood_dice)
        series["hd95"].append(ood_hd95)
        d_dice = f"{ood_dice - prev_dice:+.2f}" if prev_dice is not None and np.isfinite(ood_dice) else ""
        d_hd95 = f"{ood_hd95 - prev_hd95:+.2f}" if prev_hd95 is not None and np.isfinite(ood_hd95) else ""
        lines.append(f"| **{label}** | {ingredient} | {ood_dice:.2f} | {ood_hd95:.2f} | {d_dice} | {d_hd95} |")
        prev_dice, prev_hd95 = ood_dice, ood_hd95

    if missing:
        lines += ["", f"_Not yet available: {', '.join(missing)}_"]

    ABLATIONS_ROOT.mkdir(parents=True, exist_ok=True)
    md_path = ABLATIONS_ROOT / "ladder_summary.md"
    md_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n→ {md_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(matplotlib unavailable, skipping plot: {e})", file=sys.stderr)
        return

    labels = [r[0] for r in RUNGS]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(x, series["dice"], "o-", color="#4aa564")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    axes[0].set_ylabel("OOD Dice (%)"); axes[0].set_title("BraTS T1n ladder — Dice")
    axes[0].grid(alpha=0.3)
    axes[1].plot(x, series["hd95"], "o-", color="#4aa564")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    axes[1].set_ylabel("OOD HD95 mm (lower better)"); axes[1].set_title("BraTS T1n ladder — HD95")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    png_path = ABLATIONS_ROOT / "ladder_t1n.png"
    fig.savefig(png_path, dpi=150)
    print(f"→ {png_path}")


if __name__ == "__main__":
    main()
