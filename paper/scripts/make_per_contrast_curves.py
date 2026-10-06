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
import os
import sys
from itertools import cycle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_00_utils"))
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_03_evaluate"))
from ladder_ood_common import (_merge_patient_pairs, _dataset_name, load_case_means, resolve_run_dir, _src_key as _engine_src_key,  # noqa: E402
                               _labelled_case_values, _patient_key, _merge_by)
from stat_tests import holm, wilcoxon_p, fmt_p  # noqa: E402

OUT = REPO / "paper" / "cvpr_format_latex" / "figures" / "per_contrast_curves"
OUT.mkdir(parents=True, exist_ok=True)

# Same figure, PNG, in the repo's cross-dataset roll-up dir -- the home
# CLAUDE.md gives to summaries that span every task at once (it already holds
# meta_task_heatmap_*). The ablation ladders are exactly that: one panel per
# task, and the only place all of them are visible side by side. Keeping a copy
# here means reading the cross-task result does not require building the paper.
COMMUN = REPO / "benchmark" / "01_commun_results"   # was datasets/... before the 2026-09-27 restructuring

M = "benchmark/02_tasks/{task}/{ds}/8_results_{ds}/02_metrics/{model}/{contrast}/ablations/ladder_series.json"
TASK_OF = {"totalseg-pelvic": "pelvis_healthy", "chaos": "abdomen_healthy", "on-harmony": "brain_healthy", "toothfairy2": "mandible_healthy",
           "brats2024-glioma": "brain_tumor", "open-ms": "brain_ms", "ispy2": "breast_cancer"}


def _j(ds, model, contrast):
    return M.format(task=TASK_OF[ds], ds=ds, model=model, contrast=contrast)


# panel title -> list of (line label, ladder json path). A panel with two
# entries is a dataset trained twice, once per training modality.
PANELS = [
    ("Abdomen", "interface", [
        ("T1in",  _j("chaos", "chaos_model", "t1in")),
        ("T2spir", _j("chaos", "chaos_model", "t2spir")),
    ]),
    ("Brain", "interface", [
        ("T1w", _j("on-harmony", "on_harmony_model", "T1w")),
        ("T2w", _j("on-harmony", "on_harmony_model", "T2w")),
        ("DWI", _j("on-harmony", "on_harmony_model", "dwi_ap")),
    ]),
    ("Mandible", "interface", [
        ("CBCT", _j("toothfairy2", "toothfairy2_model", "cbct")),
    ]),
    ("Pelvis", "interface", [
        ("CT",  _j("totalseg-pelvic", "totalseg_pelvic_model", "ct")),
        ("MRI", _j("totalseg-pelvic", "totalseg_pelvic_model", "mri")),
    ]),
    ("Glioma", "no_interface", [
        ("T1n", _j("brats2024-glioma", "brats2024_glioma_model", "t1n")),
        ("T1c", _j("brats2024-glioma", "brats2024_glioma_model", "t1c")),
        ("T2w", _j("brats2024-glioma", "brats2024_glioma_model", "t2w")),
        ("FLAIR", _j("brats2024-glioma", "brats2024_glioma_model", "t2f")),
    ]),
    ("MS", "no_interface", [
        ("FLAIR", _j("open-ms", "open_ms_model", "flair")),
        ("T1w",   _j("open-ms", "open_ms_model", "t1w")),
    ]),
    ("Breast", "no_interface", [
        ("T1WCE", _j("ispy2", "ispy2_model", "t1wce")),
        ("T2w",   _j("ispy2", "ispy2_model", "t2w")),
    ]),
]

N_RUNGS = 6
FILL = 4                      # index of "v26_6_2 (real fill)"; step is FILL-1 -> FILL
RUNG_SHORT = ["base", "+km", "+lbl", "+vor", "+real", "+Auglab"]
IMPROVE, WORSEN, FLAT = "#2f7d6b", "#c0392b", "#8a8a8a"
IMPROVE_LIGHT, WORSEN_LIGHT = "#8fc4b4", "#e2988c"
NEUTRAL = "#b5b5b5"
FILLSWAP_BAND = "#f0c96b"
# Solid = trained on a T1-weighted contrast, dashed = T2-weighted/FLAIR, so a
# line style means the same thing in every panel regardless of which of a
# dataset's two training modalities it came from.
T1_FAMILY = {"T1in", "T1w", "T1n", "T1c", "T1WCE", "CBCT"}
MARKER_CYCLE = ["o", "s", "^", "v", "D", "P", "X", "*", "h", "<", ">", "p", "8", "d"]

plt.rcParams.update({
    "font.family": "sans-serif", "axes.edgecolor": "#555555", "axes.linewidth": 0.9,
    "xtick.color": "#333333", "ytick.color": "#333333",
})


def _src_key(run_subdir, run_key: str) -> str:
    """Engine's per-source run_key rewrite, keyed by a bare run_subdir."""
    return _engine_src_key({"run_subdir": run_subdir} if run_subdir else None, run_key)


# LADDER_JSON_SUFFIX (optional): read <ladder_series.json><suffix> where that file exists -- e.g. the
# pre-retrain copies, to keep the figure consistent with the text until every ladder is regenerated.
JSON_SUFFIX = os.environ.get("LADDER_JSON_SUFFIX", "")


def load(rel: str):
    p = REPO / rel
    if JSON_SUFFIX and (REPO / (rel + JSON_SUFFIX)).exists():
        p = REPO / (rel + JSON_SUFFIX)
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
    """(root, run_subdir) pairs naming every place this ladder's per-case CSVs
    live, for panel-level pooling.

      * cross-dataset ladder  -> the evaluator roots the engine recorded
      * within-dataset ladder -> the ablations dir's parent, PLUS any
        extra_ood_sources the wrapper pooled in (I-SPY2 pools the external Duke
        cohort). Missing that second part would re-derive the panel test from a
        narrower pool than the ladder's own reported numbers.

    run_subdir is the segment a source needs inserted before the run id (see
    _src_key in the engine); None for the ordinary case."""
    if d.get("ood_sources"):
        return [(Path(s), None) for s in d["ood_sources"]]
    roots = [(d["_path"].parent.parent, None)]
    roots += [(Path(e["metrics_root"]), e.get("run_subdir"))
              for e in d.get("extra_ood_sources", [])]
    return roots


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

    if any(d.get("mode") == "grouped_by_contrast" for d in ladders):
        return _panel_pooled_grouped(ladders, metric, higher_is_better)

    # {"<training>/<item>": {patient-namespaced case: (prev, cur)}}. Merged by
    # the ENGINE's own _merge_patient_pairs, so a physical patient counts once
    # across held-out contrasts AND across this panel's training modalities --
    # the same unit the combined tables test on. This used to concatenate raw
    # case pairs instead, which counted every (patient, contrast, model) row as
    # independent: Open-MS's 8 test patients became 48 "samples" and its panel
    # read p=0.0014 while its own ladders cannot go below 0.0078 (2026-10-02).
    pairs_by_contrast: dict = {}
    for d in ladders:
        keys = d.get("run_keys")
        if not keys or len(keys) <= FILL:
            continue
        prev_key, cur_key = keys[FILL - 1], keys[FILL]
        oods = set(ood_contrasts(d))
        train = d.get("contrast_label", "?")
        for root, subdir in metrics_roots(d):
            ns = _dataset_name(root)
            pk, ck = _src_key(subdir, prev_key), _src_key(subdir, cur_key)
            prev = load_case_means(resolve_run_dir(root, pk), metric)
            cur = load_case_means(resolve_run_dir(root, ck), metric)
            for item in set(prev) | set(cur):
                # cross-dataset per_contrast keys are '<dataset>/<item>'
                if oods and item not in oods and not any(o.endswith("/" + item) for o in oods):
                    continue
                a, b = prev.get(item, {}), cur.get(item, {})
                bucket = pairs_by_contrast.setdefault(f"{train}/{item}", {})
                for k in set(a) & set(b):
                    bucket[f"{ns}|{k}"] = (a[k], b[k])
    x, y = _merge_patient_pairs(pairs_by_contrast)
    if not len(x):
        return float("nan"), float("nan")
    p = wilcoxon_p(x, y)
    # Case means are stored 0-1 for dice; x100 so this delta is in Dice POINTS,
    # comparable with the single-ladder branch above (which reads the already
    # -scaled series) and with the summary tables.
    scale = 100.0 if metric == "dice" else 1.0
    delta = float(np.mean(y) - np.mean(x)) * scale * (1 if higher_is_better else -1)
    return p, delta


def _panel_pooled_grouped(ladders, metric, higher_is_better):
    """Panel test for ladders written by the engine's GROUPED mode (I-SPY2, 2026-10-01):
    every case of every contrast group of every ladder in the panel, paired at the
    fill-swap step, same-patient pairs merged (an I-SPY2 patient's _uni/_bil FOV variants,
    and the same test patient scored by both training-modality models) -- the same unit
    the engine's own grouped tests use. Re-derived from the per-case CSVs through the
    engine's own loader so the panel pool cannot drift from the ladders' pools."""
    pairs = {}
    for li, d in enumerate(ladders):
        keys = d["run_keys"]
        root = d["_path"].parent.parent
        extra = [{"metrics_root": Path(e["metrics_root"]), "run_subdir": e.get("run_subdir")}
                 for e in d.get("extra_ood_sources", [])]
        prev = _labelled_case_values(root, d["ood_contrasts"], extra, keys[FILL - 1], metric)
        cur = _labelled_case_values(root, d["ood_contrasts"], extra, keys[FILL], metric)
        members = {l for m in d["ood_groups"].values() for l in m}
        for lbl in members:
            a, b = prev.get(lbl, {}), cur.get(lbl, {})
            for k in sorted(set(a) & set(b)):
                if np.isfinite(a[k]) and np.isfinite(b[k]):
                    pairs[f"{k}§{li}§{lbl}"] = (float(a[k]), float(b[k]))
    if not pairs:
        return float("nan"), float("nan")
    x, y = _merge_by(pairs, lambda k: _patient_key(k.split("§", 1)[0]))
    scale = 100.0 if metric == "dice" else 1.0
    delta = float(np.mean([np.mean(np.asarray(d[metric][:N_RUNGS], float)[FILL] -
                                   np.asarray(d[metric][:N_RUNGS], float)[FILL - 1]) for d in ladders]))
    return wilcoxon_p(x, y), delta * (1 if higher_is_better else -1)


THUMBS = REPO / "paper" / "cvpr_format_latex" / "figures" / "task_thumbs"   # make_task_thumbnails.py


def _place_thumbnail(ax, title, width=0.26):
    """Draw the task's example image (THUMBS/<title>.png) in the emptiest corner of
    the panel: candidate boxes in axes coordinates are scored by how many plotted
    points (line vertices plus segment samples) fall inside them; the top-centre
    band is never used (the panel annotation lives there)."""
    import matplotlib.image as mpimg
    f = THUMBS / f"{title}.png"
    if not f.exists():
        return
    img = mpimg.imread(f)
    bb = ax.get_window_extent()
    h = width * (img.shape[0] / img.shape[1]) * (bb.width / bb.height)
    h = min(h, 0.36)
    w = h / ((img.shape[0] / img.shape[1]) * (bb.width / bb.height))
    to_axes = ax.transData + ax.transAxes.inverted()
    pts = []
    for ln in ax.get_lines():
        xy = np.column_stack([ln.get_xdata(), ln.get_ydata()]).astype(float)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) == 0:
            continue
        for a0, a1 in zip(xy[:-1], xy[1:]):
            pts.extend(a0 + (a1 - a0) * t for t in np.linspace(0, 1, 12))
        pts.extend(xy)
    pts = to_axes.transform(np.asarray(pts)) if pts else np.zeros((0, 2))
    # Grid search over the whole panel; the annotation block (top ~22%, middle
    # third) is off limits. Score = plotted points inside the box (+ small margin);
    # ties go to the lowest, then rightmost box.
    def score(x0, y0):
        pad = 0.015
        inside = ((pts[:, 0] > x0 - pad) & (pts[:, 0] < x0 + w + pad) &
                  (pts[:, 1] > y0 - pad) & (pts[:, 1] < y0 + h + pad))
        return int(inside.sum())
    cands = []
    for x0 in np.arange(0.01, 1 - w - 0.005, 0.02):
        for y0 in np.arange(0.01, 1 - h - 0.005, 0.02):
            if y0 + h > 0.78 and x0 < 0.80 and x0 + w > 0.35:
                continue
            cands.append((score(x0, y0), round(y0, 3), -round(x0, 3), x0, y0))
    _, _, _, x0, y0 = min(cands)
    iax = ax.inset_axes([x0, y0, w, h])
    iax.imshow(img, interpolation="lanczos")
    iax.set_xticks([]); iax.set_yticks([])
    for sp in iax.spines.values():
        sp.set_edgecolor("#9a9a9a"); sp.set_linewidth(0.6)


def sig_color(p, delta, light=False):
    if not (np.isfinite(p) and p < 0.05):
        return FLAT
    if delta >= 0:
        return IMPROVE_LIGHT if light else IMPROVE
    return WORSEN_LIGHT if light else WORSEN


SHOW_THUMBS = False   # example-slice insets (make_task_thumbnails.py); off for now (Paul, 2026-10-05)


def build(metric, ylabel, out_name, higher_is_better, layout="wide"):
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
    groups = [[(t, g, it) for t, g, it in loaded if g == grp_key]
              for grp_key in ("no_interface", "interface")]
    groups = [r for r in groups if r]
    if layout == "column":
        # One paper column: each group wraps onto rows of 2 panels.
        rows = [grp[i:i + 2] for grp in groups for i in range(0, len(grp), 2)]
        ncol, nrow = 2, len(rows)
        odd = any(len(r) == 1 for r in rows)
        fig = plt.figure(figsize=(6.6, 2.75 * nrow + (0.6 if odd else 1.3)))
        gs = fig.add_gridspec(nrow, ncol, wspace=0.28, hspace=0.75,
                              left=0.11, right=0.99, top=0.94, bottom=0.05 if odd else 0.10)
    else:
        rows = groups
        ncol = max(len(r) for r in rows)
        nrow = len(rows)
        fig = plt.figure(figsize=(4.3 * ncol, 4.15 * nrow))
        gs = fig.add_gridspec(nrow, ncol, wspace=0.30, hspace=0.62,
                              left=0.055, right=0.99, top=0.90, bottom=0.225)
    axes, row_axes = [], []
    for ri, row in enumerate(rows):
        this = [fig.add_subplot(gs[ri, ci]) for ci in range(len(row))]
        row_axes.append(this)
        axes.extend(this)
    loaded = [panel for row in rows for panel in row]

    for ax, (title, _grp, items) in zip(axes, loaded):
        for lab, d in items:
            ls = "-"   # one style for every pair (no train-contrast family coding)
            per_p = ((d.get("fill_swap_significance") or {}).get(metric) or {}).get("per_contrast", {})
            for c in ood_contrasts(d):
                s = np.asarray(d["per_contrast"][metric][c][:N_RUNGS], float)
                if not np.isfinite(s).any():
                    continue
                ax.plot(range(len(s)), s, color=NEUTRAL, linestyle=ls, linewidth=1.1,
                        alpha=0.55, zorder=1, label=c)
                if np.isfinite(s[FILL - 1:FILL + 1]).all():
                    dl = (s[FILL] - s[FILL - 1]) * (1 if higher_is_better else -1)
                    col = sig_color(per_p.get(c, float("nan")), dl, light=True)
                    ax.plot([FILL - 1, FILL], s[FILL - 1:FILL + 1], color=col, linestyle=ls,
                            linewidth=2.2 if col != FLAT else 1.5,
                            alpha=0.95 if col != FLAT else 0.75,
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
        ax.set_xticklabels(RUNG_SHORT, fontsize=11 if layout == "column" else 8, rotation=32, ha="right")
        ax.set_title(title, fontsize=14 if layout == "column" else 12, pad=8, fontweight="medium")
        ax.grid(alpha=0.15, linewidth=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="both", labelsize=11 if layout == "column" else 9, length=3)

        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.12 * (hi - lo))
        top = ax.get_ylim()[1]
        star = "*" if (np.isfinite(p) and p < 0.05) else ""
        if metric == "dice":
            # Relative gain at the fill swap (Paul, 2026-10-03): panel delta divided by
            # the panel-average noise-fill Dice; * = panel-pooled test significant.
            before = avg[FILL - 1]
            ann = (f"{dl / before * 100.0:+.1f}%{star}" if np.isfinite(before) and before > 1.0
                   else f"{dl:+.1f}{star}")
        else:
            ann = f"{-dl:+.1f} mm{star}"   # dl is the improvement; print the HD95 change itself
        if SHOW_THUMBS:
            _place_thumbnail(ax, title)
        ax.text(FILL - 0.5, top - 0.01 * (top - ax.get_ylim()[0]),
                ann, ha="center", va="top",
                fontsize=14 if layout == "column" else 10, fontweight="bold", color=seg)

    for this in row_axes:
        this[0].set_ylabel(ylabel, fontsize=13 if layout == "column" else 10.5)
    fig.canvas.draw()

    # One header per ROW, spanning that row's panels -- the row IS the group.
    GROUP_TITLE = {"interface": "Interface-bounded", "no_interface": "Appearance-defined"}
    seen = set()
    for row, this in zip(rows, row_axes):
        if row[0][1] in seen:
            continue
        seen.add(row[0][1])
        p0, p1 = this[0].get_position(), this[-1].get_position()
        if layout == "column":
            p1 = this[0].get_position() if len(this) == 1 else p1
            p1 = type(p1)([[p1.x0, p1.y0], [row_axes[0][-1].get_position().x1, p1.y1]])
        y = p0.y1 + (0.030 if layout == "column" else 0.038)
        fig.text((p0.x0 + p1.x1) / 2, y + 0.008, GROUP_TITLE[row[0][1]], ha="center",
                 va="bottom", fontsize=13.5, fontweight="semibold", color="#2a2a2a")
        fig.add_artist(plt.Line2D([p0.x0, p1.x1], [y, y], transform=fig.transFigure,
                                  color="#2a2a2a", linewidth=1.3, solid_capstyle="butt"))
        for x in (p0.x0, p1.x1):
            fig.add_artist(plt.Line2D([x, x], [y - 0.009, y], transform=fig.transFigure,
                                      color="#2a2a2a", linewidth=1.3))

    if layout == "column":
        style_h = [
            Line2D([0], [0], color="black", label="task average", linewidth=2.6),
            Line2D([0], [0], color=NEUTRAL, label="pair of train–test contrasts\n(e.g. trained on T1n,\ntested on T2w)", linewidth=1.3),
            Line2D([], [], linestyle="none", label="values per pair:\nsupplementary tables"),
            Line2D([0], [0], color=IMPROVE_LIGHT, label="that pair at the swap:\nsignificant gain", linewidth=2.2),
            Line2D([0], [0], color=WORSEN_LIGHT, label="significant loss", linewidth=2.2),
            Line2D([0], [0], color=FLAT, label="not significant", linewidth=1.5),
        ]
        single = [i for i, r in enumerate(rows) if len(r) == 1]
        if single:   # a row with one panel: the legend fills its empty cell
            cell = gs[single[0], 1].get_position(fig)
            fig.legend(handles=style_h, loc="center", ncol=1, fontsize=11.5, frameon=False,
                       bbox_to_anchor=((cell.x0 + cell.x1) / 2, (cell.y0 + cell.y1) / 2),
                       handlelength=2.0)
        else:
            fig.legend(handles=style_h, loc="center", ncol=2, fontsize=9.5, frameon=False,
                       bbox_to_anchor=(0.53, 0.045), handlelength=2.0, columnspacing=1.2)
    else:
        style_h = [
            Line2D([0], [0], color="black", label="task average", linewidth=2.6),
            Line2D([0], [0], color=NEUTRAL, label="pair of train–test contrasts (e.g. trained on T1n, tested on T2w)", linewidth=1.3),
            Line2D([0], [0], color=IMPROVE_LIGHT, label="that pair at the swap: significant gain", linewidth=2.2),
            Line2D([0], [0], color=WORSEN_LIGHT, label="significant loss", linewidth=2.2),
            Line2D([0], [0], color=FLAT, label="not significant", linewidth=1.5),
        ]
        fig.legend(handles=style_h, loc="center", ncol=5, fontsize=9, frameon=False,
                   bbox_to_anchor=(0.5, 0.10), handlelength=2.2, columnspacing=1.6)
        fig.text(0.5, 0.05, "values per training → test contrast pair: supplementary tables",
                 ha="center", va="center", fontsize=9, color="#333333")
        if metric == "dice":
            fig.text(0.5, 0.008,
                      "relative = Δ Dice at the real-fill step ÷ the panel's noise-fill Dice "
                      "(the level the step starts from)",
                      ha="center", va="bottom", fontsize=8, color="#555555", style="italic")

    out = OUT / out_name
    fig.savefig(out, dpi=300)
    written = [out]
    if COMMUN.is_dir():
        png = COMMUN / f"ablation_ladders_{Path(out_name).stem}.png"
        fig.savefig(png, dpi=200)
        written.append(png)
    plt.close(fig)
    for w in written:
        print("wrote", w)
    for t, _, _ in loaded:
        p, dl = stats[t]
        print(f"  [{metric}] {t}: panel-pooled p={fmt_p(p)} delta={dl:+.3f}")


if __name__ == "__main__":
    build("dice", "Eval-contrast Dice (%)", "fill_swap_per_contrast.pdf", True)
    build("hd95", "Eval-contrast HD95 (mm)", "fill_swap_per_contrast_hd95.pdf", False)
    build("dice", "OOD Dice (%)", "fill_swap_per_contrast_column.pdf", True, layout="column")
    build("hd95", "OOD HD95 (mm)", "fill_swap_per_contrast_hd95_column.pdf", False, layout="column")
