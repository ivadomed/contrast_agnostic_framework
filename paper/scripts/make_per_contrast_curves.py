#!/usr/bin/env python3
"""
Causal-ablation figure (main-text fig:ladder + supplementary fig:ladder-hd95):
one panel per DATASET, grouped under two boundary-type headers.

  Tissue interface     : CHAOS, ON-Harmony, ToothFairy2
  No tissue interface  : BraTS-GLI, Open-MS, I-SPY2

Each panel overlays every held-out EVAL CONTRAST as its own thin curve. Only
the REAL-FILL step (+voronoi noise fill -> v26_6_2 real fill) is coloured, by
that curve's own per-contrast significance -- light teal = significant
improvement, light red = significant worsening, grey = not significant. Every
other rung-to-rung segment is uniform light grey, because only that one step is
being tested. A bold black "panel average" line runs the full ladder with its
real-fill segment in full saturation, matching the panel-pooled p-value
annotated above the shaded step. The annotation is direction-labelled: a
two-sided p does not say which way the step moved things, and on ON-Harmony it
is a significant WORSENING.

DATA SOURCE (rewritten 2026-09-09): everything is read from each ladder's
`ladder_series.json`, written by the shared engine
(00_commun_scripts/00_03_evaluate/ladder_ood_common.py). This replaces the
previous approach of importing each dataset's wrapper module and reaching into
its METRICS_ROOT/OOD_CONTRASTS globals, which could not reach a wrapper that
builds its rungs inside main() (toothfairy2's does) and duplicated significance
logic the engine already runs. Adding a task is now a one-line PANELS entry.

Only the first 6 rungs are drawn: the trailing "+AugLab (val100)" rung is
reported in the validation-time-randomization supplementary section instead,
and not every ladder has it (I-SPY2's stops at 6), so 6 keeps every panel on
the same x-axis.

Usage:
  .venv/bin/python make_per_contrast_curves.py
"""
from __future__ import annotations

import json
import sys
from itertools import cycle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "datasets/00_commun_scripts/00_00_utils"))
sys.path.insert(0, str(REPO / "datasets/00_commun_scripts/00_03_evaluate"))
from ladder_ood_common import load_case_means, resolve_run_dir  # noqa: E402
from stat_tests import holm, wilcoxon_p, fmt_p  # noqa: E402

OUT = REPO / "paper" / "cvpr_format_latex" / "figures" / "per_contrast_curves"
OUT.mkdir(parents=True, exist_ok=True)

M = "datasets/{ds}/8_results_{ds}/02_metrics/{model}/{contrast}/ablations/ladder_series.json"


def _j(ds, model, contrast):
    return M.format(ds=ds, model=model, contrast=contrast)


# panel title -> list of (line label, ladder json path). A panel with two
# entries is a dataset trained twice, once per training modality.
PANELS = [
    ("CHAOS", "interface", [
        ("T1in",  _j("chaos", "chaos_model", "t1in")),
        ("T2spir", _j("chaos", "chaos_model", "t2spir")),
    ]),
    ("ON-Harmony", "interface", [
        ("T1w", _j("on-harmony", "on_harmony_model", "T1w")),
        ("T2w", _j("on-harmony", "on_harmony_model", "T2w")),
    ]),
    ("ToothFairy2", "interface", [
        ("CBCT", _j("toothfairy2", "toothfairy2_model", "cbct")),
    ]),
    ("BraTS-GLI", "no_interface", [
        ("T1n", _j("brats2024-glioma", "brats2024_glioma_model", "t1n")),
        ("T2w", _j("brats2024-glioma", "brats2024_glioma_model", "t2w")),
    ]),
    ("Open-MS", "no_interface", [
        ("FLAIR", _j("open-ms", "open_ms_model", "flair")),
        ("T1w",   _j("open-ms", "open_ms_model", "t1w")),
    ]),
    ("I-SPY2", "no_interface", [
        ("T1WCE", _j("ispy2", "ispy2_model", "t1wce")),
        ("T2w",   _j("ispy2", "ispy2_model", "t2w")),
    ]),
]

N_RUNGS = 6
FILL = 4                      # index of "v26_6_2 (real fill)"; step is FILL-1 -> FILL
RUNG_SHORT = ["base", "+km", "+lbl", "+vor", "+real", "+AugLab"]
IMPROVE, WORSEN, FLAT = "#2f7d6b", "#c0392b", "#8a8a8a"
IMPROVE_LIGHT, WORSEN_LIGHT = "#8fc4b4", "#e2988c"
NEUTRAL = "#b5b5b5"
FILLSWAP_BAND = "#f0c96b"
# Solid = trained on a T1-weighted contrast, dashed = T2-weighted/FLAIR, so a
# line style means the same thing in every panel regardless of which of a
# dataset's two training modalities it came from.
T1_FAMILY = {"T1in", "T1w", "T1n", "T1WCE", "CBCT"}
MARKER_CYCLE = ["o", "s", "^", "v", "D", "P", "X", "*", "h", "<", ">", "p", "8", "d"]

plt.rcParams.update({
    "font.family": "sans-serif", "axes.edgecolor": "#555555", "axes.linewidth": 0.9,
    "xtick.color": "#333333", "ytick.color": "#333333",
})


def load(rel: str):
    p = REPO / rel
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if len(d.get("dice", [])) < N_RUNGS:
        return None
    d["_path"] = p
    return d


def ood_contrasts(d) -> list:
    """Held-out contrasts only. run_ladder also tracks the training contrast in
    per_contrast (keyed '<name> (in-domain)') for the markdown/JSON; this figure
    reports the OOD estimand, so those are dropped."""
    return [c for c in d["per_contrast"]["dice"] if "(in-domain)" not in c]


def metrics_roots(d) -> list:
    """Where this ladder's per-case CSVs live, for panel-level pooling.
    Within-dataset: the ablations dir's parent. Cross-dataset: the evaluator
    roots the engine recorded."""
    if d.get("ood_sources"):
        return [Path(s) for s in d["ood_sources"]]
    return [d["_path"].parent.parent]


def panel_pooled(ladders, metric, higher_is_better):
    """One (p, delta) per panel, pooling BOTH training modalities' case-level
    pairs into a single paired test -- a coarser, dataset-level question than
    tab:dissociation's per-(task,modality) rows.

    A single-ladder panel needs no pooling: the engine already stored exactly
    that test, so use its value rather than recomputing it."""
    if len(ladders) == 1:
        d = ladders[0]
        sig = (d.get("fill_swap_significance") or {}).get(metric) or {}
        s = np.asarray(d[metric][:N_RUNGS], float)
        delta = (s[FILL] - s[FILL - 1]) * (1 if higher_is_better else -1)
        return sig.get("pooled_p", float("nan")), delta

    xs, ys = [], []
    for d in ladders:
        keys = d.get("run_keys")
        if not keys or len(keys) <= FILL:
            continue
        prev_key, cur_key = keys[FILL - 1], keys[FILL]
        oods = set(ood_contrasts(d))
        for root in metrics_roots(d):
            prev = load_case_means(resolve_run_dir(root, prev_key), metric)
            cur = load_case_means(resolve_run_dir(root, cur_key), metric)
            for item in set(prev) | set(cur):
                # cross-dataset per_contrast keys are '<dataset>/<item>'
                if oods and item not in oods and not any(o.endswith("/" + item) for o in oods):
                    continue
                a, b = prev.get(item, {}), cur.get(item, {})
                common = sorted(set(a) & set(b))
                if common:
                    xs.append(np.array([a[k] for k in common]))
                    ys.append(np.array([b[k] for k in common]))
    if not xs:
        return float("nan"), float("nan")
    x, y = np.concatenate(xs), np.concatenate(ys)
    p = wilcoxon_p(x, y)
    # Case means are stored 0-1 for dice; x100 so this delta is in Dice POINTS,
    # comparable with the single-ladder branch above (which reads the already
    # -scaled series) and with the summary tables.
    scale = 100.0 if metric == "dice" else 1.0
    delta = float(np.mean(y) - np.mean(x)) * scale * (1 if higher_is_better else -1)
    return p, delta


def sig_color(p, delta, light=False):
    if not (np.isfinite(p) and p < 0.05):
        return FLAT
    if delta >= 0:
        return IMPROVE_LIGHT if light else IMPROVE
    return WORSEN_LIGHT if light else WORSEN


def build(metric, ylabel, out_name, higher_is_better):
    loaded = [(title, grp, [(lab, load(rel)) for lab, rel in items])
              for title, grp, items in PANELS]
    loaded = [(t, g, [(l, d) for l, d in items if d is not None]) for t, g, items in loaded]
    missing = [t for t, _, items in loaded if not items]
    if missing:
        print(f"  WARNING: no ladder data for panel(s): {', '.join(missing)}", file=sys.stderr)
    loaded = [(t, g, items) for t, g, items in loaded if items]

    # One marker per eval contrast, shared across every panel and both metrics.
    all_c = sorted({c for _, _, items in loaded for _, d in items for c in ood_contrasts(d)})
    marker_of = dict(zip(all_c, cycle(MARKER_CYCLE)))

    # Panel-level pooled test, Holm-corrected across the panels.
    pooled = [panel_pooled([d for _, d in items], metric, higher_is_better)
              for _, _, items in loaded]
    p_adj = holm([p for p, _ in pooled])
    stats = {t: (pa, dl) for (t, _, _), pa, (_, dl) in zip(loaded, p_adj, pooled)}

    # 2 rows x 3 cols, ONE BOUNDARY TYPE PER ROW. Six panels in a single row
    # would be ~1.2in each once scaled to a CVPR full-width figure -- too small
    # to read -- and this layout also makes the grouping structural rather than
    # something the reader has to track from a header span.
    rows = [[(t, g, it) for t, g, it in loaded if g == grp_key]
            for grp_key in ("interface", "no_interface")]
    rows = [r for r in rows if r]
    ncol = max(len(r) for r in rows)
    nrow = len(rows)
    fig = plt.figure(figsize=(4.3 * ncol, 4.15 * nrow))
    gs = fig.add_gridspec(nrow, ncol, wspace=0.30, hspace=0.62,
                          left=0.055, right=0.99, top=0.90, bottom=0.205)
    axes, row_axes = [], []
    for ri, row in enumerate(rows):
        this = [fig.add_subplot(gs[ri, ci]) for ci in range(len(row))]
        row_axes.append(this)
        axes.extend(this)
    loaded = [panel for row in rows for panel in row]

    for ax, (title, _grp, items) in zip(axes, loaded):
        for lab, d in items:
            ls = "-" if lab in T1_FAMILY else "--"
            per_p = ((d.get("fill_swap_significance") or {}).get(metric) or {}).get("per_contrast", {})
            for c in ood_contrasts(d):
                s = np.asarray(d["per_contrast"][metric][c][:N_RUNGS], float)
                if not np.isfinite(s).any():
                    continue
                mk = marker_of[c]
                ax.plot(range(len(s)), s, color=NEUTRAL, linestyle=ls, linewidth=1.1,
                        alpha=0.55, marker=mk, markersize=6, markeredgecolor="white",
                        markeredgewidth=0.5, zorder=1, label=c)
                if np.isfinite(s[FILL - 1:FILL + 1]).all():
                    dl = (s[FILL] - s[FILL - 1]) * (1 if higher_is_better else -1)
                    col = sig_color(per_p.get(c, float("nan")), dl, light=True)
                    ax.plot([FILL - 1, FILL], s[FILL - 1:FILL + 1], color=col, linestyle=ls,
                            linewidth=2.2 if col != FLAT else 1.5,
                            alpha=0.95 if col != FLAT else 0.75, marker=mk, markersize=7.5,
                            markeredgecolor="white", markeredgewidth=0.7,
                            zorder=4 if col != FLAT else 3)

        # Bold panel average: mean over this panel's ladders' pooled OOD series.
        avg = np.nanmean([np.asarray(d[metric][:N_RUNGS], float) for _, d in items], axis=0)
        p, dl = stats[title]
        seg = sig_color(p, dl) if np.isfinite(p) and p < 0.05 else "#666666"
        ax.plot(range(0, FILL), avg[:FILL], color="black", linewidth=2.6, marker="o",
                markersize=5.5, markerfacecolor="black", markeredgecolor="white",
                markeredgewidth=0.8, zorder=5)
        ax.plot(range(FILL, len(avg)), avg[FILL:], color="black", linewidth=2.6, marker="o",
                markersize=5.5, markerfacecolor="black", markeredgecolor="white",
                markeredgewidth=0.8, zorder=5)
        ax.plot([FILL - 1, FILL], avg[FILL - 1:FILL + 1], color=seg, linewidth=3.2, marker="o",
                markersize=6.5, markerfacecolor=seg, markeredgecolor="white",
                markeredgewidth=0.8, zorder=6)

        ax.axvspan(FILL - 1, FILL, color=FILLSWAP_BAND, alpha=0.28, zorder=0, linewidth=0)
        ax.set_xticks(range(N_RUNGS))
        ax.set_xticklabels(RUNG_SHORT, fontsize=8, rotation=32, ha="right")
        ax.set_title(title, fontsize=12, pad=8, fontweight="medium")
        ax.grid(alpha=0.15, linewidth=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="both", labelsize=9, length=3)

        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.16 * (hi - lo))
        top = ax.get_ylim()[1]
        word = "ns" if not (np.isfinite(p) and p < 0.05) else \
            ("improves *" if dl >= 0 else "worsens *")
        ax.text(FILL - 0.5, top - 0.02 * (top - ax.get_ylim()[0]),
                f"panel-pooled: {word}\np={fmt_p(p)}", ha="center", va="top",
                fontsize=7.6, fontweight="bold", color=seg)

    for this in row_axes:
        this[0].set_ylabel(ylabel, fontsize=10.5)
    fig.canvas.draw()

    # One header per ROW, spanning that row's panels -- the row IS the group.
    GROUP_TITLE = {"interface": "Tissue interface", "no_interface": "No tissue interface"}
    for row, this in zip(rows, row_axes):
        p0, p1 = this[0].get_position(), this[-1].get_position()
        y = p0.y1 + 0.038
        fig.text((p0.x0 + p1.x1) / 2, y + 0.008, GROUP_TITLE[row[0][1]], ha="center",
                 va="bottom", fontsize=13.5, fontweight="semibold", color="#2a2a2a")
        fig.add_artist(plt.Line2D([p0.x0, p1.x1], [y, y], transform=fig.transFigure,
                                  color="#2a2a2a", linewidth=1.3, solid_capstyle="butt"))
        for x in (p0.x0, p1.x1):
            fig.add_artist(plt.Line2D([x, x], [y - 0.009, y], transform=fig.transFigure,
                                      color="#2a2a2a", linewidth=1.3))

    style_h = [
        Line2D([0], [0], color="#2a2a2a", linestyle="-", label="trained on T1-weighted", linewidth=1.8),
        Line2D([0], [0], color="#2a2a2a", linestyle="--", label="trained on T2-weighted/FLAIR", linewidth=1.8),
        Line2D([0], [0], color="black", label="panel average, other rungs", linewidth=2.6),
        Line2D([0], [0], color=NEUTRAL, label="per-curve, other rungs", linewidth=1.1),
    ]
    sig_h = [
        Line2D([0], [0], color=IMPROVE_LIGHT, marker="o", label="per curve: significant improvement", linewidth=2.0, markersize=7),
        Line2D([0], [0], color=WORSEN_LIGHT, marker="o", label="per curve: significant worsening", linewidth=2.0, markersize=7),
        Line2D([0], [0], color=FLAT, marker="o", label="per curve: not significant", linewidth=1.3, alpha=0.7, markersize=6),
        Line2D([0], [0], color=IMPROVE, marker="o", label="panel average: matches header", linewidth=3.0, markersize=7.5),
    ]
    fig.add_artist(fig.legend(handles=style_h + sig_h, loc="center", ncol=4, fontsize=9,
                              frameon=False, bbox_to_anchor=(0.5, 0.098),
                              handlelength=2.2, columnspacing=1.6))
    fig.legend(handles=[Line2D([0], [0], color="#666666", marker=marker_of[c], linestyle="none",
                               markersize=7.5, label=c) for c in all_c],
               loc="center", ncol=8, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.034),
               title="eval contrast (marker)", title_fontsize=9,
               handletextpad=0.5, columnspacing=1.4)

    out = OUT / out_name
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)
    for t, _, _ in loaded:
        p, dl = stats[t]
        print(f"  [{metric}] {t}: panel-pooled p={fmt_p(p)} delta={dl:+.3f}")


if __name__ == "__main__":
    build("dice", "Eval-contrast Dice (%)", "fill_swap_per_contrast.pdf", True)
    build("hd95", "Eval-contrast HD95 (mm)", "fill_swap_per_contrast_hd95.pdf", False)
