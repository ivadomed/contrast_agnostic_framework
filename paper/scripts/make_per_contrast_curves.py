#!/usr/bin/env python3
"""
Causal-ablation figures: one panel per DATASET (CHAOS, ON-Harmony, Brats-GLI,
Open-MS, ATLAS-Liver-HCC), grouped under two boundary-type headers ("Tissue
interface": CHAOS + ON-Harmony; "No tissue interface": Brats-GLI, Open-MS,
ATLAS-Liver-HCC -- see paper/NARRATIVE.md and the on-harmony
label-composition check in sec/4_experiments.tex for why ON-Harmony now
groups with CHAOS rather than standing as a separate "dense label map"
category). Produces two versions: Dice (main-paper fig:ladder) and HD95
(supplementary fig:ladder-hd95).

Each panel overlays every held-out EVAL CONTRAST as its own curve across all
7 rungs. As of 2026-08-31 (requested by the user's supervisor, after an
exploratory pass in make_per_contrast_curves_panelpooled.py showed it read
more clearly than the original all-rungs-colored scheme): only the REAL-FILL
step (rung 3 "+vor" -> rung 4 "+real") is colored by that curve's own
per-contrast significance (Holm-corrected within that ladder's own contrast
family) -- light teal = significant improvement, light red = significant
worsening, grey = not significant; every other rung-to-rung segment is a
uniform light grey, since only the real-fill step is actually being tested.
A bold black "panel average" line (pooled mean at every rung across every
curve/modality/source in that panel) runs the full ladder, with its own
real-fill segment colored to match the panel-pooled header annotation
(below) exactly -- same statistic, same color, so the header's claim and the
one bold curve backing it stay visually paired. Each panel is additionally
annotated with ONE p-value pooling BOTH training modalities into a single
paired Wilcoxon test (Holm-corrected across the 5 panels) -- a coarser,
dataset-level reading than tab:dissociation's finer per-(task,modality) rows,
directional (a low p does not by itself mean "helps": ON-Harmony's pooled
effect is a significant WORSENING, labeled and colored as such). Line style
= which of the dataset's two training modalities produced that curve (solid
= trained on a T1-weighted contrast, dashed = T2-weighted/FLAIR); marker
shape = which eval contrast, one consistent mapping reused across every
panel and both metrics.

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
from ladder_ood_common import (  # noqa: E402
    load_case_means, resolve_run_dir, rung_means,
    _cross_dataset_contrast_labels, _per_contrast_rung_means_cross_dataset, _dataset_name,
)
from stat_tests import holm, wilcoxon_p, fmt_p  # noqa: E402

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
    ("ATLAS-Liver-HCC", "datasets/atlas-liver-hcc/5_scripts_atlas-liver-hcc/06_evaluate/06_13_ladder_summary.py"),
]

FILL_SWAP_IDX = 4
RUNG_SHORT = ["base", "+km", "+lbl", "+vor", "+real", "+AL(v000)", "+AL(v100)"]
IMPROVE, WORSEN, FLAT = "#2f7d6b", "#c0392b", "#8a8a8a"
# Lighter tints for the many thin per-curve real-fill segments, so they read
# as secondary detail; the bold panel-average curve's own real-fill segment
# uses the full-saturation IMPROVE/WORSEN above instead, matching the
# panel-pooled header text exactly.
IMPROVE_LIGHT, WORSEN_LIGHT = "#8fc4b4", "#e2988c"
NEUTRAL = "#b5b5b5"
FILLSWAP_BAND = "#f0c96b"
PANELS = [
    ("CHAOS", ["CHAOS T1in", "CHAOS T2spir"]),
    ("ON-Harmony", ["ON-Harmony T1w", "ON-Harmony T2w"]),
    ("BraTS-GLI", ["Brats-GLI T1n", "Brats-GLI T2w"]),
    ("Open-MS", ["Open-MS FLAIR", "Open-MS T1w"]),
    ("ATLAS-Liver-HCC", ["ATLAS-Liver-HCC"]),
]
# ATLAS-Liver-HCC trains on one modality (T1w) permanently -- it has no second
# training-modality sibling, so it gets exactly one plot_panel() call (below)
# and lands in T1_FAMILY (solid line) since it IS T1-weighted.
T1_FAMILY = {"Brats-GLI T1n", "Open-MS T1w", "CHAOS T1in", "ON-Harmony T1w", "ATLAS-Liver-HCC"}
CROSS_DATASET_TASKS = {"ATLAS-Liver-HCC"}

MARKER = {
    "t1in": "o", "t1out": "s", "t2spir": "^", "ct": "D",
    "t1n": "v", "t1c": "P", "t2w": "X", "t2f": "*",
    "flair": "h", "t1w": "<", "bold": ">", "dwi_ap": "p",
    "epi_ap": "8", "gre_echo1_mag": "d",
    # ATLAS-Liver-HCC's cross-dataset OOD streams ('<dataset>/<item>' labels,
    # see ladder_ood_common._cross_dataset_contrast_labels).
    "lld-mmri-hcc/t2wi": "^", "lld-mmri-hcc/dwi": "v",
    "liverhccseg/ce-pre_t1w": "o", "liverhccseg/ce-art_t1w": "s",
    "liverhccseg/ce-pv_t1w": "D", "liverhccseg/ce-del_t1w": "P",
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


def _cross_dataset_item_source(ood_sources, label):
    """label is '<dataset>/<item>' (see _cross_dataset_contrast_labels) -- find
    which of ood_sources produced it, and the bare item name within it."""
    ds_name, item = label.split("/", 1)
    for metrics_root in ood_sources:
        if _dataset_name(metrics_root) == ds_name:
            return metrics_root, item
    raise KeyError(f"no source for {label!r}")


def gather(metric: str):
    task_data = {}
    for name, rel in WRAPPERS:
        mod = _load_wrapper(REPO / rel)
        per_contrast = {}
        pvals_this_task, contrasts_this_task = [], []
        if name in CROSS_DATASET_TASKS:
            # Cross-dataset ladder (single training modality): OOD "contrasts" are
            # '<dataset>/<item>' streams pooled from multiple evaluator datasets,
            # not columns within one dataset's own metrics tree (see
            # ladder_ood_common.run_ladder_cross_dataset).
            labels = []
            for _, _, run_key in mod.RUNGS:
                found = _cross_dataset_contrast_labels(mod.OOD_SOURCES, run_key, metric)
                if len(found) > len(labels):
                    labels = found
            for label in labels:
                series = [_per_contrast_rung_means_cross_dataset(mod.OOD_SOURCES, run_key, metric, [label])[label]
                          for _, _, run_key in mod.RUNGS]
                src, item = _cross_dataset_item_source(mod.OOD_SOURCES, label)
                run3 = resolve_run_dir(src, mod.RUNGS[3][2])
                run4 = resolve_run_dir(src, mod.RUNGS[4][2])
                c3 = load_case_means(run3, metric).get(item, {}) if run3.is_dir() else {}
                c4 = load_case_means(run4, metric).get(item, {}) if run4.is_dir() else {}
                common = sorted(set(c3) & set(c4))
                x = np.array([c3[k] for k in common])
                y = np.array([c4[k] for k in common])
                p = wilcoxon_p(x, y) if len(x) else float("nan")
                pvals_this_task.append(p)
                contrasts_this_task.append(label)
                per_contrast[label] = series
        else:
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


def _task_pairs(rel: str, metric: str):
    """Every case-level (x, y) pair at the real-fill step for one task,
    cross-dataset-aware (mirrors gather()'s own branch, but returns raw
    pairs instead of per-contrast series/significance)."""
    mod = _load_wrapper(REPO / rel)
    rung3_key, rung4_key = mod.RUNGS[3][2], mod.RUNGS[4][2]
    if hasattr(mod, "OOD_SOURCES"):
        x, y = [], []
        for metrics_root in mod.OOD_SOURCES:
            dir3 = resolve_run_dir(metrics_root, rung3_key)
            dir4 = resolve_run_dir(metrics_root, rung4_key)
            c3 = load_case_means(dir3, metric) if dir3.is_dir() else {}
            c4 = load_case_means(dir4, metric) if dir4.is_dir() else {}
            for item in set(c3) | set(c4):
                i3, i4 = c3.get(item, {}), c4.get(item, {})
                for cid in sorted(set(i3) & set(i4)):
                    x.append(i3[cid]); y.append(i4[cid])
        return np.array(x), np.array(y)
    dir3 = resolve_run_dir(mod.METRICS_ROOT, rung3_key)
    dir4 = resolve_run_dir(mod.METRICS_ROOT, rung4_key)
    cases3 = load_case_means(dir3, metric)
    cases4 = load_case_means(dir4, metric)
    x, y = [], []
    for contrast in mod.OOD_CONTRASTS:
        c3, c4 = cases3.get(contrast, {}), cases4.get(contrast, {})
        for cid in sorted(set(c3) & set(c4)):
            x.append(c3[cid]); y.append(c4[cid])
    return np.array(x), np.array(y)


def panel_rung_means(tasks, metric: str):
    """Pooled mean at EACH of the 7 rungs, across every case/contrast/source
    in a panel (raw case values pooled, same convention as _task_pairs/the
    panel-pooled significance test below -- not a per-contrast mean-of-means)."""
    scale = 100 if metric == "dice" else 1
    out = []
    for rung_idx in range(7):
        vals = []
        for name in tasks:
            rel = dict(WRAPPERS)[name]
            mod = _load_wrapper(REPO / rel)
            run_key = mod.RUNGS[rung_idx][2]
            if hasattr(mod, "OOD_SOURCES"):
                for metrics_root in mod.OOD_SOURCES:
                    run_dir = resolve_run_dir(metrics_root, run_key)
                    if not run_dir.is_dir():
                        continue
                    data = load_case_means(run_dir, metric)
                    for cases in data.values():
                        vals.extend(cases.values())
            else:
                run_dir = resolve_run_dir(mod.METRICS_ROOT, run_key)
                if not run_dir.is_dir():
                    continue
                data = load_case_means(run_dir, metric)
                for contrast in mod.OOD_CONTRASTS:
                    vals.extend(data.get(contrast, {}).values())
        out.append(float(np.mean(vals)) * scale if vals else float("nan"))
    return np.array(out)


def panel_pvalues(metric: str, higher_is_better: bool):
    """One pooled p-value + direction per panel (both training modalities'
    cases combined into a single test), Holm-corrected across the 5 panels
    -- a coarser, dataset-level question than tab:dissociation's per-
    (task,modality) rows. wilcoxon_p is two-sided, so a low p alone does not
    say WHICH way the fill swap moved things -- ON-Harmony's pooled effect
    is a significant WORSENING, not an improvement like the other 4 panels,
    and callers must label/color it as such rather than defaulting every
    low p to read as "helps"."""
    raw, deltas = [], []
    rels = dict(WRAPPERS)
    for _, tasks in PANELS:
        xs, ys = [], []
        for name in tasks:
            x, y = _task_pairs(rels[name], metric)
            xs.append(x); ys.append(y)
        x, y = np.concatenate(xs), np.concatenate(ys)
        raw.append(wilcoxon_p(x, y))
        delta = float(np.mean(y) - np.mean(x))
        deltas.append(delta if higher_is_better else -delta)
    adj = holm(raw)
    titles = [title for title, _ in PANELS]
    return {t: (p, d) for t, p, d in zip(titles, adj, deltas)}


def build_figure(metric: str, ylabel: str, out_name: str, higher_is_better: bool):
    task_data = gather(metric)
    pvals = panel_pvalues(metric, higher_is_better)
    I0, I1 = FILL_SWAP_IDX - 1, FILL_SWAP_IDX

    def header_color(title):
        p, delta = pvals[title]
        if not (np.isfinite(p) and p < 0.05):
            return "#666666"
        return IMPROVE if delta >= 0 else WORSEN

    def draw(ax, task, dataset_title):
        d = task_data[task]
        ls = "-" if task in T1_FAMILY else "--"
        for contrast, series in d["series"].items():
            s = np.array(series)
            marker = MARKER[norm(contrast)]
            # Base curve: every rung, neutral -- the red/teal/grey scheme
            # tests only the real-fill step, so only that step should look
            # tested; the rest of the ladder is context, not evidence.
            ax.plot(range(len(s)), s, color=NEUTRAL, linestyle=ls, linewidth=1.1, alpha=0.55,
                    marker=marker, markersize=6, markeredgecolor="white", markeredgewidth=0.5, zorder=1)
            # Real-fill segment only: colored (lightly) by this curve's own
            # per-curve significance, drawn on top of the neutral base.
            sig_p = d["sig"][contrast]
            delta = (s[I1] - s[I0]) * (1 if higher_is_better else -1)
            if not np.isfinite(sig_p) or sig_p >= 0.05:
                color = FLAT
            else:
                color = IMPROVE_LIGHT if delta >= 0 else WORSEN_LIGHT
            z = 4 if color != FLAT else 3
            lw = 2.2 if color != FLAT else 1.5
            alpha = 0.95 if color != FLAT else 0.75
            ax.plot([I0, I1], [s[I0], s[I1]], color=color, linestyle=ls, linewidth=lw, alpha=alpha,
                    marker=marker, markersize=7.5, markeredgecolor="white", markeredgewidth=0.7, zorder=z)
        ax.axvspan(I0, I1, color=FILLSWAP_BAND, alpha=0.28, zorder=0, linewidth=0)
        ax.set_xticks(range(7))
        ax.set_xticklabels(RUNG_SHORT, fontsize=8, rotation=32, ha="right")
        ax.set_title(dataset_title, fontsize=12, pad=8, fontweight="medium")
        ax.grid(alpha=0.15, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=9, length=3)

    fig = plt.figure(figsize=(20.0, 6.3))
    gs = fig.add_gridspec(1, 5, wspace=0.38, left=0.04, right=0.988, top=0.80, bottom=0.30)
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]

    draw(axes[0], "CHAOS T1in", "CHAOS")
    draw(axes[1], "ON-Harmony T1w", "ON-Harmony")
    draw(axes[2], "Brats-GLI T1n", "BraTS-GLI")
    draw(axes[3], "Open-MS FLAIR", "Open-MS")
    draw(axes[4], "ATLAS-Liver-HCC", "ATLAS-Liver-HCC")
    axes[0].set_ylabel(ylabel, fontsize=10.5)

    draw(axes[0], "CHAOS T2spir", "CHAOS")
    draw(axes[1], "ON-Harmony T2w", "ON-Harmony")
    draw(axes[2], "Brats-GLI T2w", "BraTS-GLI")
    draw(axes[3], "Open-MS T1w", "Open-MS")
    # ATLAS-Liver-HCC trains on T1w only -- no second-modality sibling call.

    # Bold black panel average: pooled mean at every rung across all curves/
    # modalities/sources in that panel. Its real-fill segment is colored to
    # match the panel-pooled header text exactly (same statistic, same
    # color) -- the header's claim and the one bold curve backing it stay
    # visually paired; everywhere else the average line is plain black.
    for ax, (title, tasks) in zip(axes, PANELS):
        avg = panel_rung_means(tasks, metric)
        seg_color = header_color(title)
        ax.plot(range(0, I0 + 1), avg[:I0 + 1], color="black", linestyle="-", linewidth=2.6,
                marker="o", markersize=5.5, markerfacecolor="black", markeredgecolor="white",
                markeredgewidth=0.8, zorder=5, solid_capstyle="round")
        ax.plot(range(I1, 7), avg[I1:], color="black", linestyle="-", linewidth=2.6,
                marker="o", markersize=5.5, markerfacecolor="black", markeredgecolor="white",
                markeredgewidth=0.8, zorder=5, solid_capstyle="round")
        ax.plot([I0, I1], [avg[I0], avg[I1]], color=seg_color, linestyle="-", linewidth=3.2,
                marker="o", markersize=6.5, markerfacecolor=seg_color, markeredgecolor="white",
                markeredgewidth=0.8, zorder=6, solid_capstyle="round")

    # One pooled-p annotation per panel, placed above the fill-swap band.
    for ax, (title, _) in zip(axes, PANELS):
        p, delta = pvals[title]
        sig = np.isfinite(p) and p < 0.05
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.14 * (ymax - ymin))
        color = header_color(title)
        word = "ns" if not sig else ("improves *" if delta >= 0 else "worsens *")
        label = f"panel-pooled: {word}\np={fmt_p(p)}"
        ax.text(FILL_SWAP_IDX - 0.5, ax.get_ylim()[1] - 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                label, ha="center", va="top", fontsize=7.6, fontweight="bold", color=color)

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
    group_header(axes[2], axes[4], "No tissue interface")

    ls_handles = [Line2D([0], [0], color="#2a2a2a", linestyle="-", label="trained on T1-weighted", linewidth=1.8),
                  Line2D([0], [0], color="#2a2a2a", linestyle="--", label="trained on T2-weighted/FLAIR", linewidth=1.8),
                  Line2D([0], [0], color="black", label="panel average, other rungs", linewidth=2.6),
                  Line2D([0], [0], color=NEUTRAL, label="per-curve, other rungs", linewidth=1.1)]
    sig_handles = [Line2D([0], [0], color=IMPROVE_LIGHT, marker="o", label="per curve: significant improvement", linewidth=2.0, markersize=7),
                   Line2D([0], [0], color=WORSEN_LIGHT, marker="o", label="per curve: significant worsening", linewidth=2.0, markersize=7),
                   Line2D([0], [0], color=FLAT, marker="o", label="per curve: not significant", linewidth=1.3, alpha=0.7, markersize=6),
                   Line2D([0], [0], color=IMPROVE, marker="o", label="panel average: matches header", linewidth=3.0, markersize=7.5)]
    leg1 = fig.legend(handles=ls_handles + sig_handles, loc="center", ncol=4, fontsize=9, frameon=False,
                       bbox_to_anchor=(0.5, 0.175), handlelength=2.2, columnspacing=1.6)
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
    for title, _ in PANELS:
        p, delta = pvals[title]
        print(f"  [{metric}] {title}: panel-pooled p={fmt_p(p)} delta={delta:+.3f}")


if __name__ == "__main__":
    # Guarded so the sibling make_per_contrast_curves_panelpooled.py can
    # import this module's gather()/constants for reuse without re-triggering
    # (and silently re-dating) the live paper figures as a side effect.
    build_figure("dice", "Eval-contrast Dice (%)", "fill_swap_per_contrast.pdf", higher_is_better=True)
    build_figure("hd95", "Eval-contrast HD95 (mm)", "fill_swap_per_contrast_hd95.pdf", higher_is_better=False)
