#!/usr/bin/env python3
"""
IND-inclusive sibling of make_per_contrast_curves.py, built to answer a direct
question from the user (2026-09-01): "we aim for something that's working on
all contrasts, including IND. Do the results change a lot if we include IND?"
That question was already answered for the per-dataset ladder_ood_common.py
plots (each now reports an "all contrasts" figure alongside the OOD-only one,
see that module's run_ladder() docstring) -- this script does the SAME thing
for the actual paper figure (fig:ladder / fill_swap_per_contrast.pdf), so the
two can be compared directly.

For all 4 datasets (CHAOS, ON-Harmony, BraTS-GLI, Open-MS), each with a
genuine held-out training contrast, each panel gets a second bold pooled
line ("ALL, incl. in-domain") alongside the existing OOD-only bold line,
plus the training contrast itself plotted as a thin dashed/starred curve
among the per-contrast lines -- exactly mirroring ladder_ood_common's
series_all treatment.

This is a NEW, separate script/output -- it does not overwrite
make_per_contrast_curves.py or its fill_swap_per_contrast.pdf.

ATLAS-Liver-HCC was a 5th panel here (deliberately left OOD-only/unchanged,
since it had no single in-domain contrast) until 2026-09-02, when the whole
atlas-liver-hcc extension was excluded from the paper -- see CLAUDE.md
"Atlas-Liver-HCC exclusion (2026-09-02)".

Usage:
  .venv/bin/python make_per_contrast_curves_with_ind.py
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
    load_case_means, resolve_run_dir, rung_means, _per_contrast_rung_means,
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
    # ATLAS-Liver-HCC REMOVED 2026-09-02 -- dataset excluded from the paper
    # entirely, see CLAUDE.md "Atlas-Liver-HCC exclusion (2026-09-02)".
]

FILL_SWAP_IDX = 4
RUNG_SHORT = ["base", "+km", "+lbl", "+vor", "+real", "+AL(v000)", "+AL(v100)"]
IMPROVE, WORSEN, FLAT = "#2f7d6b", "#c0392b", "#8a8a8a"
IMPROVE_LIGHT, WORSEN_LIGHT = "#8fc4b4", "#e2988c"
NEUTRAL = "#b5b5b5"
FILLSWAP_BAND = "#f0c96b"
ALL_LINE = "#1a5276"  # bold navy for the new "ALL, incl. in-domain" panel-average line
PANELS = [
    ("CHAOS", ["CHAOS T1in", "CHAOS T2spir"]),
    ("ON-Harmony", ["ON-Harmony T1w", "ON-Harmony T2w"]),
    ("BraTS-GLI", ["Brats-GLI T1n", "Brats-GLI T2w"]),
    ("Open-MS", ["Open-MS FLAIR", "Open-MS T1w"]),
]
T1_FAMILY = {"Brats-GLI T1n", "Open-MS T1w", "CHAOS T1in", "ON-Harmony T1w"}
# Dead since 2026-09-02 (ATLAS-Liver-HCC, the only cross-dataset-ladder task
# with no single in-domain contrast, was excluded from the paper -- see
# CLAUDE.md "Atlas-Liver-HCC exclusion"). Left empty rather than removing the
# cross-dataset branches below wholesale, since every remaining task uses
# the within-dataset path unconditionally.
CROSS_DATASET_TASKS = set()

MARKER = {
    "t1in": "o", "t1out": "s", "t2spir": "^", "ct": "D",
    "t1n": "v", "t1c": "P", "t2w": "X", "t2f": "*",
    "flair": "h", "t1w": "<", "bold": ">", "dwi_ap": "p",
    "epi_ap": "8", "gre_echo1_mag": "d",
    "lld-mmri-hcc/t2wi": "^", "lld-mmri-hcc/dwi": "v",
    "liverhccseg/ce-pre_t1w": "o", "liverhccseg/ce-art_t1w": "s",
    "liverhccseg/ce-pv_t1w": "D", "liverhccseg/ce-del_t1w": "P",
}
IN_DOMAIN_MARKER = "*"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.9,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})


def norm(c):
    return c.lower()


def _in_domain_label(mod):
    return f"{mod.IN_DOMAIN} (in-domain)"


def gather(metric: str):
    """Same as make_per_contrast_curves.gather(), plus one extra curve per
    non-cross-dataset task: the training/in-domain contrast itself, Holm-
    corrected together with that task's OOD contrasts (same family of tests,
    same real-fill step)."""
    task_data = {}
    for name, rel in WRAPPERS:
        mod = _load_wrapper(REPO / rel)
        per_contrast = {}
        pvals_this_task, contrasts_this_task = [], []
        if name in CROSS_DATASET_TASKS:
            labels = []
            for _, _, run_key in mod.RUNGS:
                found = _cross_dataset_contrast_labels(mod.OOD_SOURCES, run_key, metric)
                if len(found) > len(labels):
                    labels = found
            for label in labels:
                series = [_per_contrast_rung_means_cross_dataset(mod.OOD_SOURCES, run_key, metric, [label])[label]
                          for _, _, run_key in mod.RUNGS]
                ds_name, item = label.split("/", 1)
                src = next(s for s in mod.OOD_SOURCES if _dataset_name(s) == ds_name)
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
            contrasts_here = list(mod.OOD_CONTRASTS) + [mod.IN_DOMAIN]
            for contrast in contrasts_here:
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
                key = _in_domain_label(mod) if contrast == mod.IN_DOMAIN else contrast
                contrasts_this_task.append(key)
                per_contrast[key] = series
        adj = holm(pvals_this_task)
        sig = dict(zip(contrasts_this_task, adj))
        task_data[name] = {"series": per_contrast, "sig": sig, "n": len(common)}
    return task_data


def _task_pairs(rel: str, metric: str, include_in_domain: bool):
    """Every case-level (x, y) pair at the real-fill step for one task. When
    include_in_domain, also concatenates the training contrast's own pairs
    (no-op for cross-dataset tasks, which have no single in-domain contrast)."""
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
    contrasts = list(mod.OOD_CONTRASTS) + ([mod.IN_DOMAIN] if include_in_domain else [])
    x, y = [], []
    for contrast in contrasts:
        c3, c4 = cases3.get(contrast, {}), cases4.get(contrast, {})
        for cid in sorted(set(c3) & set(c4)):
            x.append(c3[cid]); y.append(c4[cid])
    return np.array(x), np.array(y)


def panel_rung_means(tasks, metric: str, include_in_domain: bool):
    """Pooled mean at EACH of the 7 rungs. include_in_domain adds the training
    contrast's own cases for non-cross-dataset tasks (no-op for ATLAS)."""
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
                contrasts = list(mod.OOD_CONTRASTS) + ([mod.IN_DOMAIN] if include_in_domain else [])
                for contrast in contrasts:
                    vals.extend(data.get(contrast, {}).values())
        out.append(float(np.mean(vals)) * scale if vals else float("nan"))
    return np.array(out)


def panel_pvalues(metric: str, higher_is_better: bool, include_in_domain: bool):
    raw, deltas = [], []
    rels = dict(WRAPPERS)
    for _, tasks in PANELS:
        xs, ys = [], []
        for name in tasks:
            x, y = _task_pairs(rels[name], metric, include_in_domain)
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
    pvals_ood = panel_pvalues(metric, higher_is_better, include_in_domain=False)
    pvals_all = panel_pvalues(metric, higher_is_better, include_in_domain=True)
    I0, I1 = FILL_SWAP_IDX - 1, FILL_SWAP_IDX

    def header_color(pvals, title):
        p, delta = pvals[title]
        if not (np.isfinite(p) and p < 0.05):
            return "#666666"
        return IMPROVE if delta >= 0 else WORSEN

    def draw(ax, task, dataset_title):
        d = task_data[task]
        mod = _load_wrapper(REPO / dict(WRAPPERS)[task])
        ls = "-" if task in T1_FAMILY else "--"
        in_dom_key = None if task in CROSS_DATASET_TASKS else _in_domain_label(mod)
        for contrast, series in d["series"].items():
            s = np.array(series)
            is_ind = contrast == in_dom_key
            marker = IN_DOMAIN_MARKER if is_ind else MARKER.get(norm(contrast), "x")
            msize = 8 if is_ind else 6
            ax.plot(range(len(s)), s, color=NEUTRAL, linestyle=ls, linewidth=1.1, alpha=0.55,
                    marker=marker, markersize=msize, markeredgecolor="white", markeredgewidth=0.5, zorder=1)
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
                    marker=marker, markersize=msize + 1.5, markeredgecolor="white", markeredgewidth=0.7, zorder=z)
        ax.axvspan(I0, I1, color=FILLSWAP_BAND, alpha=0.28, zorder=0, linewidth=0)
        ax.set_xticks(range(7))
        ax.set_xticklabels(RUNG_SHORT, fontsize=8, rotation=32, ha="right")
        ax.set_title(dataset_title, fontsize=12, pad=8, fontweight="medium")
        ax.grid(alpha=0.15, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=9, length=3)

    fig = plt.figure(figsize=(16.0, 6.9))
    gs = fig.add_gridspec(1, 4, wspace=0.38, left=0.04, right=0.988, top=0.78, bottom=0.30)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    draw(axes[0], "CHAOS T1in", "CHAOS")
    draw(axes[1], "ON-Harmony T1w", "ON-Harmony")
    draw(axes[2], "Brats-GLI T1n", "BraTS-GLI")
    draw(axes[3], "Open-MS FLAIR", "Open-MS")
    axes[0].set_ylabel(ylabel, fontsize=10.5)

    draw(axes[0], "CHAOS T2spir", "CHAOS")
    draw(axes[1], "ON-Harmony T2w", "ON-Harmony")
    draw(axes[2], "Brats-GLI T2w", "BraTS-GLI")
    draw(axes[3], "Open-MS T1w", "Open-MS")

    for ax, (title, tasks) in zip(axes, PANELS):
        avg = panel_rung_means(tasks, metric, include_in_domain=False)
        seg_color = header_color(pvals_ood, title)
        ax.plot(range(0, I0 + 1), avg[:I0 + 1], color="black", linestyle="-", linewidth=2.6,
                marker="o", markersize=5.5, markerfacecolor="black", markeredgecolor="white",
                markeredgewidth=0.8, zorder=5, solid_capstyle="round")
        ax.plot(range(I1, 7), avg[I1:], color="black", linestyle="-", linewidth=2.6,
                marker="o", markersize=5.5, markerfacecolor="black", markeredgecolor="white",
                markeredgewidth=0.8, zorder=5, solid_capstyle="round")
        ax.plot([I0, I1], [avg[I0], avg[I1]], color=seg_color, linestyle="-", linewidth=3.2,
                marker="o", markersize=6.5, markerfacecolor=seg_color, markeredgecolor="white",
                markeredgewidth=0.8, zorder=6, solid_capstyle="round")

        avg_all = panel_rung_means(tasks, metric, include_in_domain=True)
        seg_color_all = header_color(pvals_all, title)
        ax.plot(range(0, I0 + 1), avg_all[:I0 + 1], color=ALL_LINE, linestyle=":", linewidth=2.4,
                marker="D", markersize=5, markerfacecolor=ALL_LINE, markeredgecolor="white",
                markeredgewidth=0.7, zorder=5, alpha=0.9)
        ax.plot(range(I1, 7), avg_all[I1:], color=ALL_LINE, linestyle=":", linewidth=2.4,
                marker="D", markersize=5, markerfacecolor=ALL_LINE, markeredgecolor="white",
                markeredgewidth=0.7, zorder=5, alpha=0.9)
        ax.plot([I0, I1], [avg_all[I0], avg_all[I1]], color=seg_color_all, linestyle=":", linewidth=3.4,
                marker="D", markersize=6, markerfacecolor=seg_color_all, markeredgecolor="white",
                markeredgewidth=0.9, zorder=6)

    for ax, (title, _) in zip(axes, PANELS):
        p_ood, delta_ood = pvals_ood[title]
        color_ood = header_color(pvals_ood, title)
        sig_ood = np.isfinite(p_ood) and p_ood < 0.05
        word_ood = "ns" if not sig_ood else ("improves *" if delta_ood >= 0 else "worsens *")
        ymin, ymax = ax.get_ylim()
        p_all, delta_all = pvals_all[title]
        color_all = header_color(pvals_all, title)
        sig_all = np.isfinite(p_all) and p_all < 0.05
        word_all = "ns" if not sig_all else ("improves *" if delta_all >= 0 else "worsens *")
        ax.set_ylim(ymin, ymax + 0.22 * (ymax - ymin))
        top = ax.get_ylim()[1] - 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.text(FILL_SWAP_IDX - 0.5, top,
                f"OOD: {word_ood}  p={fmt_p(p_ood)}", ha="center", va="top",
                fontsize=7.2, fontweight="bold", color=color_ood)
        ax.text(FILL_SWAP_IDX - 0.5, top - 0.10 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"ALL: {word_all}  p={fmt_p(p_all)}", ha="center", va="top",
                fontsize=7.2, fontweight="bold", color=color_all)

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
                  Line2D([0], [0], color="#2a2a2a", linestyle="--", label="trained on T2-weighted/FLAIR", linewidth=1.8),
                  Line2D([0], [0], color="black", label="panel average, OOD only", linewidth=2.6),
                  Line2D([0], [0], color=ALL_LINE, linestyle=":", marker="D", markersize=5,
                         label="panel average, ALL (incl. in-domain)", linewidth=2.4),
                  Line2D([0], [0], color=NEUTRAL, label="per-curve, other rungs", linewidth=1.1),
                  Line2D([0], [0], color="#666666", marker=IN_DOMAIN_MARKER, linestyle="none", markersize=9,
                         label="training contrast (in-domain)")]
    sig_handles = [Line2D([0], [0], color=IMPROVE_LIGHT, marker="o", label="per curve: significant improvement", linewidth=2.0, markersize=7),
                   Line2D([0], [0], color=WORSEN_LIGHT, marker="o", label="per curve: significant worsening", linewidth=2.0, markersize=7),
                   Line2D([0], [0], color=FLAT, marker="o", label="per curve: not significant", linewidth=1.3, alpha=0.7, markersize=6)]
    leg1 = fig.legend(handles=ls_handles + sig_handles, loc="center", ncol=3, fontsize=8.6, frameon=False,
                       bbox_to_anchor=(0.5, 0.175), handlelength=2.2, columnspacing=1.4)
    fig.add_artist(leg1)

    used = set()
    for t in task_data:
        used |= set(c for c in task_data[t]["series"] if not c.endswith("(in-domain)"))
    used = set(norm(c) for c in used)
    marker_handles = [Line2D([0], [0], color="#666666", marker=MARKER.get(c, "x"), linestyle="none", markersize=7.5, label=c)
                       for c in sorted(used)]
    fig.legend(handles=marker_handles, loc="center", ncol=7, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.045), title="eval contrast (marker)", title_fontsize=9,
               handletextpad=0.5, columnspacing=1.6)

    out_pdf = OUT / out_name
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)
    print("wrote", out_pdf)
    for title, _ in PANELS:
        p_ood, d_ood = pvals_ood[title]
        p_all, d_all = pvals_all[title]
        print(f"  [{metric}] {title}: OOD p={fmt_p(p_ood)} delta={d_ood:+.3f}   "
              f"ALL p={fmt_p(p_all)} delta={d_all:+.3f}")


if __name__ == "__main__":
    build_figure("dice", "Eval-contrast Dice (%)", "fill_swap_per_contrast_with_ind.pdf", higher_is_better=True)
    build_figure("hd95", "Eval-contrast HD95 (mm)", "fill_swap_per_contrast_with_ind_hd95.pdf", higher_is_better=False)
