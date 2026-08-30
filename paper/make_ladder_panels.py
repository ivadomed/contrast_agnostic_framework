#!/usr/bin/env python3
"""
Reads the 8 canonical ladder_series.json files (written by the shared
datasets/00_commun_scripts/00_03_evaluate/ladder_ood_common.py engine) and
emits 6 grouped PDF panels -- 3 boundary-type groups (tissue interface:
CHAOS T1in+T2spir; no tissue interface: BraTS T1n/T2w + Open-MS FLAIR/T1w;
dense label map: ON-Harmony T1w+T2w) x 2 metrics (Dice for the main-text
figure, HD95 for the supplementary one). Each multi-task panel overlays its
tasks as same-marker-per-task lines, with linestyle tracking contrast
FAMILY consistently across tasks (solid = T1-weighted, dashed =
T2-weighted/FLAIR) rather than primary/secondary training modality, so a
solid line always means "T1-family" and a dashed one always means
"T2-family" no matter which task's panel you're looking at -- sharing one
legend per panel, so the figure reads as "one story per mechanism category"
rather than one panel per dataset.

Usage:
  .venv/bin/python make_ladder_panels.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "cvpr_format_latex" / "figures" / "ladder_panels"
OUT.mkdir(parents=True, exist_ok=True)

# Remove the old per-task (5 dataset x 2 metric = 10) panels this script used
# to emit -- superseded by the 3-group x 2-metric = 6 panels below.
for stale in OUT.glob("*.pdf"):
    stale.unlink()

# group_key -> (display title, colour, [(short label, json path, marker, linestyle), ...])
GROUPS = {
    "interface": (
        "Tissue interface (CHAOS)", "#2471a3",
        [
            ("T1in", "datasets/chaos/8_results_chaos/02_metrics/chaos_model/t1in/ablations/ladder_series.json", "o", "-"),
            ("T2spir", "datasets/chaos/8_results_chaos/02_metrics/chaos_model/t2spir/ablations/ladder_series.json", "s", "--"),
        ],
    ),
    "no_interface": (
        "No tissue interface (BraTS, Open-MS)", "#c0392b",
        [
            ("BraTS T1n", "datasets/brats2024-glioma/8_results_brats2024-glioma/02_metrics/brats2024_glioma_model/t1n/ablations/ladder_series.json", "o", "-"),
            ("BraTS T2w", "datasets/brats2024-glioma/8_results_brats2024-glioma/02_metrics/brats2024_glioma_model/t2w/ablations/ladder_series.json", "o", "--"),
            ("Open-MS FLAIR", "datasets/open-ms/8_results_open-ms/02_metrics/open_ms_model/flair/ablations/ladder_series.json", "s", "--"),
            ("Open-MS T1w", "datasets/open-ms/8_results_open-ms/02_metrics/open_ms_model/t1w/ablations/ladder_series.json", "s", "-"),
        ],
    ),
    "dense": (
        "Dense label map (ON-Harmony)", "#7d3c98",
        [
            ("T1w", "datasets/on-harmony/8_results_on-harmony/02_metrics/on_harmony_model/T1w/ablations/ladder_series.json", "o", "-"),
            ("T2w", "datasets/on-harmony/8_results_on-harmony/02_metrics/on_harmony_model/T2w/ablations/ladder_series.json", "o", "--"),
        ],
    ),
}
GROUP_ORDER = ["interface", "no_interface", "dense"]

# Short rung labels for tiny subfigure x-axes. val100 is dropped from this
# figure entirely (reported only in the validation-time-randomization
# supplementary section) so val000 no longer needs disambiguating -- plain
# "+AugLab".
SHORT = {
    "baseline (floor)": "base", "+kmeans": "+km", "+label_remap": "+lbl",
    "+voronoi (noise fill)": "+vor\n(noise)", "v26_6_2 (real fill)": "+real\nfill",
    "+AugLab (val000)": "+AugLab",
}
N_RUNGS = 6  # drop the trailing "+AugLab (val100)" rung (index 6)
FILL_SWAP_IDX = 4  # "v26_6_2 (real fill)" -- the rung this whole ablation isolates

for metric, ylab in (("dice", "OOD Dice (%)"), ("hd95", "OOD HD95 (mm)")):
    for gkey in GROUP_ORDER:
        title, col, series = GROUPS[gkey]
        fig, ax = plt.subplots(figsize=(2.55, 1.95))
        x = None
        for short_label, rel, marker, ls in series:
            data = json.loads((REPO / rel).read_text())
            labels = [SHORT.get(l, l) for l in data["labels"][:N_RUNGS]]
            x = list(range(len(labels)))
            y = data[metric][:N_RUNGS]
            ax.plot(x, y, marker=marker, linestyle=ls, color=col, linewidth=1.5,
                     markersize=4, zorder=2, label=short_label if len(series) > 1 else None)
            ax.scatter([x[FILL_SWAP_IDX]], [y[FILL_SWAP_IDX]], color=col, s=50,
                       edgecolor="black", linewidth=0.8, zorder=3)
        # Shade the fill-swap STEP itself -- the edge from "+voronoi (noise)"
        # (rung 3) to "real fill" (rung 4) -- rather than centering the band
        # on rung 4, which previously bled half-way into both neighbours.
        ax.axvspan(FILL_SWAP_IDX - 1, FILL_SWAP_IDX, color=col, alpha=0.10, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=5.4, rotation=0)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(title, fontsize=7.8, pad=2)
        ax.set_ylabel(ylab, fontsize=7, labelpad=1)
        ax.grid(alpha=0.25, linewidth=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if len(series) > 1:
            ax.legend(fontsize=5.8, loc="lower right", frameon=False, handlelength=1.6,
                       borderaxespad=0.2, labelspacing=0.25)
        fig.tight_layout(pad=0.3)
        out_path = OUT / f"group_{gkey}_{metric}.pdf"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"-> {out_path}")

print(f"\n{len(GROUP_ORDER)*2} grouped panels written to {OUT}")
