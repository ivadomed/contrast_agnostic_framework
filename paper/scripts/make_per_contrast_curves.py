#!/usr/bin/env python3
"""
Causal-ablation figures: one panel per DATASET (CHAOS, ON-Harmony, Brats-GLI,
Open-MS), grouped under two boundary-type headers ("Tissue interface": CHAOS
+ ON-Harmony; "No tissue interface": Brats-GLI + Open-MS -- see
paper/NARRATIVE.md and the on-harmony label-composition check in
sec/4_experiments.tex for why ON-Harmony now groups with CHAOS rather than
standing as a separate "dense label map" category). Produces two versions:
Dice (main-paper fig:ladder) and HD95 (supplementary fig:ladder-hd95).

Each panel overlays every held-out EVAL CONTRAST as its own curve (not pooled
OOD) across all 7 rungs, coloured by the fill-swap step's OWN per-contrast
significance (Holm-corrected within that ladder's own contrast family): teal
= significant improvement, red = significant worsening, grey = not
significant. Line style = which of the dataset's two training modalities
produced that curve (solid = trained on a T1-weighted contrast, dashed =
T2-weighted/FLAIR); marker shape = which eval contrast, one consistent
mapping reused across every panel and both metrics.

Usage:
  .venv/bin/python make_per_contrast_curves.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "datasets/00_commun_scripts/00_00_utils"))
sys.path.insert(0, str(REPO / "datasets/00_commun_scripts/00_03_evaluate"))
from ladder_ood_common import load_case_means, resolve_run_dir, rung_means  # noqa: E402
from stat_tests import holm, wilcoxon_p  # noqa: E402

OUT = REPO / "paper" / "cvpr_format_latex" / "figures" / "per_contrast_curves"
OUT.mkdir(parents=True, exist_ok=True)


def _load_wrapper(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


WRAPPERS = [
    ("CHAOS T1in", "datasets/chaos/5_scripts_chaos/06_evaluate/06_34_ladder_summary_t1in.py"),
    ("ON-Harmony T1w", "datasets/on-harmony/5_scripts_on-harmony/06_evaluate/06_10_ladder_summary.py"),
    ("Brats-GLI T1n", "datasets/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/06_13_ladder_summary.py"),
    ("Open-MS FLAIR", "datasets/open-ms/5_scripts_open-ms/06_evaluate/06_14_ladder_summary_ood.py"),
    ("CHAOS T2spir", "datasets/chaos/5_scripts_chaos/06_evaluate/06_33_ladder_summary_t2spir.py"),
    ("ON-Harmony T2w", "datasets/on-harmony/5_scripts_on-harmony/06_evaluate/06_11_ladder_summary_t2w.py"),
    ("Brats-GLI T2w", "datasets/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/06_14_ladder_summary_t2w.py"),
    ("Open-MS T1w", "datasets/open-ms/5_scripts_open-ms/06_evaluate/06_18_ladder_summary_t1w.py"),
]

FILL_SWAP_IDX = 4
RUNG_SHORT = ["base", "+km", "+lbl", "+vor", "+real", "+AL(v000)", "+AL(v100)"]
IMPROVE, WORSEN, FLAT = "#2f7d6b", "#c0392b", "#8a8a8a"
FILLSWAP_BAND = "#f0c96b"
T1_FAMILY = {"Brats-GLI T1n", "Open-MS T1w", "CHAOS T1in", "ON-Harmony T1w"}
INTERFACE_TASKS = {"CHAOS T1in", "CHAOS T2spir", "ON-Harmony T1w", "ON-Harmony T2w"}

MARKER = {
    "t1in": "o", "t1out": "s", "t2spir": "^", "ct": "D",
    "t1n": "v", "t1c": "P", "t2w": "X", "t2f": "*",
    "flair": "h", "t1w": "<", "bold": ">", "dwi_ap": "p",
    "epi_ap": "8", "gre_echo1_mag": "d",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.9,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})


def norm(c):
    return c.lower()


def gather(metric: str):
    task_data = {}
    for name, rel in WRAPPERS:
        mod = _load_wrapper(REPO / rel)
        per_contrast = {}
        pvals_this_task, contrasts_this_task = [], []
        for contrast in mod.OOD_CONTRASTS:
            series = []
            for label, ingredient, run_key in mod.RUNGS:
                v, _ = rung_means(mod.METRICS_ROOT, run_key, metric, mod.IN_DOMAIN, [contrast])
                series.append(v)
            run3 = resolve_run_dir(mod.METRICS_ROOT, mod.RUNGS[3][2])
            run4 = resolve_run_dir(mod.METRICS_ROOT, mod.RUNGS[4][2])
            c3 = load_case_means(run3, metric).get(contrast, {})
            c4 = load_case_means(run4, metric).get(contrast, {})
            common = sorted(set(c3) & set(c4))
            x = np.array([c3[k] for k in common])
            y = np.array([c4[k] for k in common])
            p = wilcoxon_p(x, y) if len(x) else float("nan")
            pvals_this_task.append(p)
            contrasts_this_task.append(contrast)
            per_contrast[contrast] = series
        adj = holm(pvals_this_task)
        sig = dict(zip(contrasts_this_task, adj))
        task_data[name] = {"series": per_contrast, "sig": sig, "n": len(common)}
    return task_data


def build_figure(metric: str, ylabel: str, out_name: str, higher_is_better: bool):
    task_data = gather(metric)

    def color_for(task, contrast):
        s = np.array(task_data[task]["series"][contrast])
        delta = s[FILL_SWAP_IDX] - s[FILL_SWAP_IDX - 1]
        if not higher_is_better:
            delta = -delta
        p = task_data[task]["sig"][contrast]
        if not np.isfinite(p) or p >= 0.05:
            return FLAT
        return IMPROVE if delta >= 0 else WORSEN

    def plot_panel(ax, task, dataset_title):
        d = task_data[task]
        ls = "-" if task in T1_FAMILY else "--"
        for contrast, series in d["series"].items():
            s = np.array(series)
            color = color_for(task, contrast)
            marker = MARKER[norm(contrast)]
            z = 4 if color != FLAT else 2
            lw = 2.0 if color != FLAT else 1.3
            alpha = 0.95 if color != FLAT else 0.62
            ax.plot(range(len(s)), s, color=color, linestyle=ls, linewidth=lw, alpha=alpha,
                    marker=marker, markersize=6.5, markeredgecolor="white", markeredgewidth=0.6, zorder=z)
        ax.axvspan(FILL_SWAP_IDX - 1, FILL_SWAP_IDX, color=FILLSWAP_BAND, alpha=0.28, zorder=0, linewidth=0)
        ax.set_xticks(range(7))
        ax.set_xticklabels(RUNG_SHORT, fontsize=8, rotation=32, ha="right")
        ax.set_title(dataset_title, fontsize=12, pad=8, fontweight="medium")
        ax.grid(alpha=0.15, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=9, length=3)

    fig = plt.figure(figsize=(16.5, 6.3))
    gs = fig.add_gridspec(1, 4, wspace=0.38, left=0.05, right=0.985, top=0.80, bottom=0.30)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    plot_panel(axes[0], "CHAOS T1in", "CHAOS")
    plot_panel(axes[1], "ON-Harmony T1w", "ON-Harmony")
    plot_panel(axes[2], "Brats-GLI T1n", "BraTS-GLI")
    plot_panel(axes[3], "Open-MS FLAIR", "Open-MS")
    axes[0].set_ylabel(ylabel, fontsize=10.5)

    plot_panel(axes[0], "CHAOS T2spir", "CHAOS")
    plot_panel(axes[1], "ON-Harmony T2w", "ON-Harmony")
    plot_panel(axes[2], "Brats-GLI T2w", "BraTS-GLI")
    plot_panel(axes[3], "Open-MS T1w", "Open-MS")

    fig.canvas.draw()

    def group_header(ax_left, ax_right, text):
        p0 = ax_left.get_position(); p1 = ax_right.get_position()
        xmid = (p0.x0 + p1.x1) / 2
        y_line = 0.885
        y_text = y_line + 0.012
        fig.text(xmid, y_text, text, ha="center", va="bottom", fontsize=14, fontweight="semibold", color="#2a2a2a")
        fig.add_artist(plt.Line2D([p0.x0, p1.x1], [y_line, y_line], transform=fig.transFigure,
                                   color="#2a2a2a", linewidth=1.3, solid_capstyle="butt"))
        for x in (p0.x0, p1.x1):
            fig.add_artist(plt.Line2D([x, x], [y_line - 0.012, y_line], transform=fig.transFigure,
                                       color="#2a2a2a", linewidth=1.3))

    group_header(axes[0], axes[1], "Tissue interface")
    group_header(axes[2], axes[3], "No tissue interface")

    ls_handles = [Line2D([0], [0], color="#2a2a2a", linestyle="-", label="trained on T1-weighted", linewidth=1.8),
                  Line2D([0], [0], color="#2a2a2a", linestyle="--", label="trained on T2-weighted/FLAIR", linewidth=1.8)]
    sig_handles = [Line2D([0], [0], color=IMPROVE, marker="o", label="significant improvement", linewidth=2.0, markersize=7),
                   Line2D([0], [0], color=WORSEN, marker="o", label="significant worsening", linewidth=2.0, markersize=7),
                   Line2D([0], [0], color=FLAT, marker="o", label="not significant", linewidth=1.3, alpha=0.7, markersize=6)]
    leg1 = fig.legend(handles=ls_handles + sig_handles, loc="center", ncol=5, fontsize=9.5, frameon=False,
                       bbox_to_anchor=(0.5, 0.175), handlelength=2.4, columnspacing=1.8)
    fig.add_artist(leg1)

    used = set()
    for t in task_data:
        used |= set(norm(c) for c in task_data[t]["series"])
    marker_handles = [Line2D([0], [0], color="#666666", marker=MARKER[c], linestyle="none", markersize=7.5, label=c)
                       for c in sorted(used)]
    fig.legend(handles=marker_handles, loc="center", ncol=7, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.055), title="eval contrast (marker)", title_fontsize=9,
               handletextpad=0.5, columnspacing=1.6)

    out_pdf = OUT / out_name
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)
    print("wrote", out_pdf)

    tally = {"sig+": 0, "sig-": 0, "ns": 0}
    tally_interface = {"sig+": 0, "sig-": 0, "ns": 0}
    tally_no_interface = {"sig+": 0, "sig-": 0, "ns": 0}
    for t in task_data:
        for c in task_data[t]["series"]:
            col = color_for(t, c)
            key = "sig+" if col == IMPROVE else "sig-" if col == WORSEN else "ns"
            tally[key] += 1
            (tally_interface if t in INTERFACE_TASKS else tally_no_interface)[key] += 1
    print(f"  [{metric}] TALLY all:", tally)
    print(f"  [{metric}] TALLY tissue-interface (CHAOS+ON-Harmony):", tally_interface)
    print(f"  [{metric}] TALLY no-tissue-interface (BraTS+Open-MS):", tally_no_interface)


build_figure("dice", "Eval-contrast Dice (%)", "fill_swap_per_contrast.pdf", higher_is_better=True)
build_figure("hd95", "Eval-contrast HD95 (mm)", "fill_swap_per_contrast_hd95.pdf", higher_is_better=False)
