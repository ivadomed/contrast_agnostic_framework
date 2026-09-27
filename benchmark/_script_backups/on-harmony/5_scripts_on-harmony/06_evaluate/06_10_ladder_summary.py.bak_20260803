#!/usr/bin/env python3
"""
on-harmony T1w causal-ablation ladder — baseline-anchored: each rung adds exactly one
ingredient on top of the previous rung (K-means -> +label-remap -> +Voronoi
sub-parcellation [noise fill] -> +real-texture fill [PALETTE alone]), so the OOD
Dice/HD95 delta between consecutive rungs is attributable to that one ingredient
(everything else held fixed). Mirrors brats2024-glioma's 06_13_ladder_summary.py and
chaos's 06_33_ladder_summary_t2spir.py; non-headline run dirs live under ablations/
(see CLAUDE.md's 02_metrics subdir convention) — reads them from there, writes its own
outputs there too.

Reads the same fold*/eval_all.csv files used by significance_from_config.py (no new
evaluation, just a focused presentation). OOD = mean over held-out contrasts
(T2w, bold, dwi_ap, epi_ap, gre_echo1_mag), excluding the training contrast (T1w).

Usage:
  .venv/bin/python 06_10_ladder_summary.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

DATASET_ROOT = Path(__file__).resolve().parents[2]   # datasets/on-harmony
METRICS_ROOT = DATASET_ROOT / "8_results_on-harmony/02_metrics/on_harmony_model/T1w"
ABLATIONS_ROOT = METRICS_ROOT / "ablations"

IN_DOMAIN = "T1w"
OOD_CONTRASTS = ["T2w", "bold", "dwi_ap", "epi_ap", "gre_echo1_mag"]

RUNGS = [
    ("baseline (floor)", "— (no augmentation at all)",
     "on-harmony_T1w_baseline_20260623_192811"),
    ("+kmeans", "+ K-means intensity clustering",
     "ablations/on-harmony_T1w_baseline_kmeans_20260801_191042"),
    ("+label_remap", "+ label remap",
     "ablations/on-harmony_T1w_baseline_kmeans_label_remap_20260801_191042"),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
     "ablations/on-harmony_T1w_baseline_kmeans_label_remap_voronoi_20260801_191042"),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
     "on-harmony_T1w_v26_6_2_train050_val100_20260623_192811"),
    ("+AugLab (val000)", "+ full AugLab recipe on top",
     "on-harmony_T1w_auglabAug_v26_6_2_train050_val000_20260727_075205"),
    ("+AugLab (val100)", "+ 100%-synth validation",
     "on-harmony_T1w_auglabAug_v26_6_2_train050_val100_20260727_075205"),
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
                c = row.get("group")
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
    lines = ["# on-harmony T1w — causal ablation ladder", "",
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
    axes[0].plot(x, series["dice"], "o-", color="#2a78d6")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    axes[0].set_ylabel("OOD Dice (%)"); axes[0].set_title("on-harmony T1w ladder — Dice")
    axes[0].grid(alpha=0.3)
    axes[1].plot(x, series["hd95"], "o-", color="#2a78d6")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    axes[1].set_ylabel("OOD HD95 mm (lower better)"); axes[1].set_title("on-harmony T1w ladder — HD95")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    png_path = ABLATIONS_ROOT / "ladder_t1w.png"
    fig.savefig(png_path, dpi=150)
    print(f"→ {png_path}")


if __name__ == "__main__":
    main()
