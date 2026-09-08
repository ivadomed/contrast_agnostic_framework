#!/usr/bin/env python3
"""
Shared engine for the causal-ablation "ladder" tables/plots used across datasets
(BraTS T1n/T2w, CHAOS T1in/T2spir, Open-MS FLAIR/T1w, ON-Harmony T1w/T2w,
atlas-liver-hcc, ...): each rung adds exactly one ingredient on top of the previous
rung (K-means -> +label-remap -> +Voronoi sub-parcellation [noise fill] -> +real-
texture fill [PALETTE alone] -> +AugLab), so the OOD Dice/HD95 delta between
consecutive rungs is attributable to that one ingredient (everything else held
fixed).

Extracted 2026-08-03 from four independently-copy-pasted per-dataset scripts
(BraTS 06_13, CHAOS-T2spir 06_33, Open-MS 06_12, ON-Harmony 06_10) that had
started to drift -- Open-MS's copy in particular had fallen out of date with a
different rung set, no HD95, and no OOD/in-domain split, which is what produced
a transcription error in the paper's causal-ablation table (an HD95 delta for
one task was accidentally copied from a different task's Dice column). Per
CLAUDE.md's shared-layer rule ("if you catch yourself writing eval/aggregate
logic inline, stop -- it belongs in 00_commun_scripts"), this is that shared
layer; per-dataset scripts should now be thin wrappers that only declare
METRICS_ROOT / IN_DOMAIN / OOD_CONTRASTS / RUNGS and call run_ladder() (or
ood_sources / RUNGS and call run_ladder_cross_dataset() for single-training-
modality datasets like atlas-liver-hcc).

2026-08-29: added the per-eval-contrast "sub-ladder" breakdown (one Dice/HD95
curve per held-out contrast, not just the pooled OOD mean) to BOTH run_ladder()
and run_ladder_cross_dataset(), via one shared plotting helper. Because every
per-dataset script is a thin wrapper that just calls into here, this landed for
every existing ladder (BraTS T1n/T2w, CHAOS T1in/T2spir, Open-MS FLAIR/T1w,
ON-Harmony T1w/T2w, atlas-liver-hcc) by re-running those scripts unchanged --
no per-dataset code was touched.

Reads the same fold*/eval_all.csv files used by significance_from_config.py --
no new evaluation, just a focused, code-shared presentation.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "00_00_utils"))
from stat_tests import holm, wilcoxon_p, fmt_p  # noqa: E402 — shared math, see stat_tests.py

# ── plot styling, matched to the paper's fig:ladder ───────────────────────────
# (paper/scripts/make_per_contrast_curves.py). The per-dataset plots below and
# the paper figure now read the same way: only the real-fill step is coloured by
# significance, everything else is neutral context, and the pooled mean carries
# the ladder's own fill-swap p-value. Keep the two in sync when either changes.
IMPROVE, WORSEN, FLAT = "#2f7d6b", "#c0392b", "#8a8a8a"
IMPROVE_LIGHT, WORSEN_LIGHT = "#8fc4b4", "#e2988c"   # thin per-contrast curves
NEUTRAL = "#b5b5b5"                                   # untested rung-to-rung segments
FILLSWAP_BAND = "#f0c96b"
# Rung label marking the one-variable texture test (noise fill -> real fill).
# Located by label rather than a hardcoded index so a ladder with a different
# rung set still plots -- it just skips the fill-swap decoration.
FILL_SWAP_LABEL_HINT = "real fill"


def load_case_means(run_dir: Path, metric: str) -> dict:
    """{contrast: {case: mean metric over labels and folds}}."""
    per: dict = {}
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


def resolve_run_dir(metrics_root: Path, run_key: str) -> Path:
    """run_key may already carry a category prefix (nnUNet_/auglab_) or an
    ablations/ prefix; if the bare path doesn't exist, try both category
    prefixes (mirrors significance_from_config.py's resolve_run_dir)."""
    p = metrics_root / run_key
    if p.is_dir():
        return p
    parent, name = p.parent, p.name
    for prefix in ("nnUNet_", "auglab_"):
        cand = parent / f"{prefix}{name}"
        if cand.is_dir():
            return cand
    return p


def rung_means(metrics_root: Path, run_key: str, metric: str,
               in_domain: str, ood_contrasts):
    run_dir = resolve_run_dir(metrics_root, run_key)
    if not run_dir.is_dir():
        return None, None
    data = load_case_means(run_dir, metric)
    scale = 100 if metric == "dice" else 1
    ood_vals = [float(np.mean(list(data[c].values()))) for c in ood_contrasts if data.get(c)]
    ood = float(np.mean(ood_vals)) * scale if ood_vals else float("nan")
    ind = float(np.mean(list(data[in_domain].values()))) * scale if data.get(in_domain) else float("nan")
    return ood, ind


def _per_contrast_rung_means(metrics_root: Path, run_key: str, metric: str, ood_contrasts):
    """{contrast: mean} for ONE rung, each ood_contrast scored independently (not
    pooled) -- the per-contrast counterpart to rung_means' pooled OOD figure."""
    run_dir = resolve_run_dir(metrics_root, run_key)
    scale = 100 if metric == "dice" else 1
    out = {}
    if not run_dir.is_dir():
        return out
    data = load_case_means(run_dir, metric)
    for c in ood_contrasts:
        cases = data.get(c)
        out[c] = float(np.mean(list(cases.values()))) * scale if cases else float("nan")
    return out


def _find_fill_swap_idx(rungs):
    """Index of the real-fill rung -- the step the whole ablation isolates.
    Returns None if this ladder has no such rung (or it is first, leaving no
    preceding rung to compare against), in which case callers skip the
    fill-swap decoration rather than guessing a position."""
    for i, (label, _, _) in enumerate(rungs):
        if FILL_SWAP_LABEL_HINT in label.lower():
            return i if i > 0 else None
    return None


def _paired_cases(cases_prev: dict, cases_cur: dict):
    """Case-level paired arrays for one contrast (cases present in both rungs)."""
    common = sorted(set(cases_prev) & set(cases_cur))
    return (np.array([cases_prev[k] for k in common], dtype=float),
            np.array([cases_cur[k] for k in common], dtype=float))


def _fill_swap_pairs(metrics_root, rungs, fill_idx, metric, ood_contrasts,
                     extra_ood_sources=None):
    """{contrast: (x, y)} case-level pairs at the fill-swap step, for a
    within-dataset ladder (plus any extra cross-dataset item columns pooled
    into its OOD set)."""
    prev_key, cur_key = rungs[fill_idx - 1][2], rungs[fill_idx][2]
    prev = load_case_means(resolve_run_dir(metrics_root, prev_key), metric)
    cur = load_case_means(resolve_run_dir(metrics_root, cur_key), metric)
    pairs = {c: _paired_cases(prev.get(c, {}), cur.get(c, {})) for c in ood_contrasts}
    for src in (extra_ood_sources or []):
        ds = _dataset_name(src)
        p_src = load_case_means(resolve_run_dir(src, prev_key), metric)
        c_src = load_case_means(resolve_run_dir(src, cur_key), metric)
        for item in set(p_src) | set(c_src):
            pairs[f"{ds}/{item}"] = _paired_cases(p_src.get(item, {}), c_src.get(item, {}))
    return pairs


def _fill_swap_pairs_cross_dataset(ood_sources, rungs, fill_idx, metric):
    """Cross-dataset sibling of _fill_swap_pairs, keyed '<dataset>/<item>'."""
    prev_key, cur_key = rungs[fill_idx - 1][2], rungs[fill_idx][2]
    pairs = {}
    for src in ood_sources:
        ds = _dataset_name(src)
        prev = load_case_means(resolve_run_dir(src, prev_key), metric)
        cur = load_case_means(resolve_run_dir(src, cur_key), metric)
        for item in set(prev) | set(cur):
            pairs[f"{ds}/{item}"] = _paired_cases(prev.get(item, {}), cur.get(item, {}))
    return pairs


def _fill_swap_significance(pairs_by_contrast):
    """Paired Wilcoxon significance of the fill-swap step: one test per contrast
    (Holm-corrected within this ladder's own contrast family, which is what
    colours the thin per-contrast curves) plus one pooled test over every
    contrast's cases (which is what annotates the bold pooled line).

    Same statistic and same two granularities the paper reports -- the pooled
    test matches tab:dissociation's p, the per-contrast tests match fig:ladder's
    per-curve colouring. Note the pooled test is case-weighted while the pooled
    MEAN plotted alongside it is equal-weight-per-contrast; that combination is
    deliberate and matches the paper, so don't "fix" one to the other.

    The pooled p here is RAW (uncorrected): one ladder has no visibility of the
    others, so it cannot apply the across-ladder Holm correction the paper's
    tab:dissociation does over its 8 rows. Expect this figure to read as more
    significant than the paper's for the same task (e.g. CHAOS T1in: 1.7e-4
    here vs 6.9e-4 corrected) -- that is the correction, not a disagreement.
    paper/scripts/compute_dissociation_pvalues.py owns the corrected numbers."""
    labels = list(pairs_by_contrast)
    raw = [wilcoxon_p(*pairs_by_contrast[c]) if len(pairs_by_contrast[c][0]) else float("nan")
           for c in labels]
    per_contrast = dict(zip(labels, holm(raw)))
    xs = [pairs_by_contrast[c][0] for c in labels if len(pairs_by_contrast[c][0])]
    ys = [pairs_by_contrast[c][1] for c in labels if len(pairs_by_contrast[c][1])]
    pooled_p = wilcoxon_p(np.concatenate(xs), np.concatenate(ys)) if xs else float("nan")
    return {"per_contrast": per_contrast, "pooled_p": pooled_p,
            "n_cases": int(sum(len(a) for a in xs))}


def _sig_color(p, delta, light=False):
    """Paper's colour convention: teal = significant improvement, red =
    significant worsening, grey = not significant. `delta` must already be
    sign-adjusted so positive means "better" for this metric."""
    if not (np.isfinite(p) and p < 0.05):
        return FLAT
    if delta >= 0:
        return IMPROVE_LIGHT if light else IMPROVE
    return WORSEN_LIGHT if light else WORSEN


def _decorate_fill_swap(ax, fill_idx, pooled_p, pooled_delta):
    """Shade the fill-swap step and annotate it with the ladder's pooled
    p-value, labelled by direction -- a two-sided p alone doesn't say which way
    the step moved things, and on some tasks it is a significant WORSENING."""
    ax.axvspan(fill_idx - 1, fill_idx, color=FILLSWAP_BAND, alpha=0.28, zorder=0, linewidth=0)
    sig = np.isfinite(pooled_p) and pooled_p < 0.05
    word = "ns" if not sig else ("improves *" if pooled_delta >= 0 else "worsens *")
    color = _sig_color(pooled_p, pooled_delta)
    if not sig:
        color = "#666666"
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.16 * (ymax - ymin))
    top = ax.get_ylim()[1]
    ax.text(fill_idx - 0.5, top - 0.02 * (top - ax.get_ylim()[0]),
            f"fill swap: {word}\np={fmt_p(pooled_p)}", ha="center", va="top",
            fontsize=8, fontweight="bold", color=color)


def _per_contrast_table_md(rungs, contrast_labels, per_contrast) -> list[str]:
    """One markdown table per metric, rows=rung, columns=eval contrast -- the
    "sub-ladder" numbers underlying the per-contrast PNG."""
    lines = ["## Per-contrast breakdown (each held-out contrast scored on its own, "
             "not pooled into the OOD mean above)", ""]
    for metric, title, fmt in (("dice", "Dice (%)", "{:.2f}"), ("hd95", "HD95 mm", "{:.2f}")):
        lines += [f"### {title}", "",
                  "| rung | " + " | ".join(contrast_labels) + " |",
                  "|---|" + "---|" * len(contrast_labels)]
        for i, (label, _, _) in enumerate(rungs):
            row = [fmt.format(per_contrast[metric][c][i]) if np.isfinite(per_contrast[metric][c][i]) else "—"
                   for c in contrast_labels]
            lines.append(f"| **{label}** | " + " | ".join(row) + " |")
        lines.append("")
    return lines


def run_ladder(*, task_name, contrast_label, metrics_root, ablations_root,
              in_domain, ood_contrasts, rungs, combined_png=True, extra_ood_sources=None):
    """Compute the ladder, write ladder_summary.md + a JSON dump (for the shared
    multi-task plotting script) into ablations_root, plus the dataset-local pooled
    2-panel PNG and a per-eval-contrast "sub-ladder" 2-panel PNG. Returns the dump dict.

    extra_ood_sources (optional): list of cross-dataset metrics-root Paths (e.g.
    amos/sliver07 for chaos) whose item columns are pooled INTO the same OOD
    average as ood_contrasts, with equal weight per item — matching this
    function's existing equal-weight-per-contrast convention (deliberately NOT
    the case-count-weighted pooling run_ladder_cross_dataset uses, so adding a
    source here doesn't silently reintroduce the vote-imbalance problem
    significance_from_config.py's contrast_groups was built to fix). Added
    2026-09-01 to increase held-out case counts for underpowered ablation
    comparisons using cross-dataset sources already validated for this dataset's
    headline table — each rung's run_key must resolve under every source given
    here for that source to contribute (missing sources are silently skipped
    per rung, not treated as an error, so a partially-predicted extension still
    reports the sources that ARE ready).

    2026-09-01: reports BOTH the OOD-only figure (unchanged) and an "all contrasts"
    figure — equal weight per contrast, in_domain included alongside ood_contrasts —
    matching aggregate_from_config.py's existing "all" column convention (not a new
    concept: the rest of the project already reports both an OOD-excluding-training
    view and an all-contrasts-including-training view, e.g. significance_from_config.
    py's "OOD cross-contrast generalization" vs "ALL contrasts combined" sections).
    User request (2026-09-01): the project targets contrast-agnostic performance
    including the training contrast, so the ladder's causal read on rung 4->5 (does
    texture preservation help) shouldn't only be judged on held-out contrasts.
    Both figures are kept side by side (not one replacing the other) since OOD-only
    is still the project's headline generalization claim elsewhere — this lets a
    reader see directly whether including in_domain changes the causal conclusion."""
    extra_ood_sources = extra_ood_sources or []
    extra_labels: list[str] = []
    for _, _, run_key in rungs:
        found = _cross_dataset_contrast_labels(extra_ood_sources, run_key, "dice")
        if len(found) > len(extra_labels):
            extra_labels = found
    ood_labels = list(ood_contrasts) + extra_labels
    in_domain_label = f"{in_domain} (in-domain)"
    full_labels = ood_labels + [in_domain_label]

    series = {"dice": [], "hd95": []}          # OOD only (unchanged from before)
    series_all = {"dice": [], "hd95": []}      # NEW: OOD + in-domain, equal weight per contrast
    per_contrast = {"dice": {c: [] for c in full_labels}, "hd95": {c: [] for c in full_labels}}
    src_note = (f" plus {len(extra_labels)} cross-dataset item(s) pooled in with equal weight "
               f"(sources: {', '.join(_dataset_name(p) for p in extra_ood_sources)})"
               if extra_ood_sources else "")
    lines = [f"# {task_name} — causal ablation ladder", "",
             "Each rung adds exactly one ingredient on top of the previous rung — the "
             "Dice/HD95 delta is attributable to that ingredient alone. Two pooled figures "
             "are reported: **OOD** (mean over held-out contrasts only, training contrast "
             "excluded — this project's headline cross-contrast-generalization claim "
             "elsewhere) and **all** (OOD contrasts + the training contrast, equal weight "
             "per contrast — since the project targets contrast-agnostic performance "
             "including the training contrast, not just held-out ones; matches "
             "aggregate_from_config.py's existing 'all' column convention). "
             f"OOD = mean over ({', '.join(ood_contrasts)}){src_note}; "
             f"all additionally includes the training contrast ({in_domain}).", "",
             "| rung | adds | OOD Dice | OOD HD95 | Δ Dice vs. prev | Δ HD95 vs. prev "
             "| all Dice | all HD95 | Δ Dice vs. prev | Δ HD95 vs. prev |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    prev_dice = prev_hd95 = None
    prev_all_dice = prev_all_hd95 = None
    missing = []
    for label, ingredient, run_key in rungs:
        vals = {}
        vals_all = {}
        for metric in ("dice", "hd95"):
            own_pc = _per_contrast_rung_means(metrics_root, run_key, metric, ood_contrasts)
            extra_pc = _per_contrast_rung_means_cross_dataset(extra_ood_sources, run_key, metric, extra_labels)
            in_domain_pc = _per_contrast_rung_means(metrics_root, run_key, metric, [in_domain])
            pooled = {**own_pc, **extra_pc, in_domain_label: in_domain_pc.get(in_domain, float("nan"))}
            for c in full_labels:
                per_contrast[metric][c].append(pooled.get(c, float("nan")))
            # own_pc/extra_pc/in_domain_pc values are already scaled (dice *100) by their own helpers.
            ood_finite = [pooled[c] for c in ood_labels if np.isfinite(pooled.get(c, float("nan")))]
            all_finite = [pooled[c] for c in full_labels if np.isfinite(pooled.get(c, float("nan")))]
            vals[metric] = float(np.mean(ood_finite)) if ood_finite else float("nan")
            vals_all[metric] = float(np.mean(all_finite)) if all_finite else float("nan")
        ood_dice, ood_hd95 = vals["dice"], vals["hd95"]
        all_dice, all_hd95 = vals_all["dice"], vals_all["hd95"]
        if not np.isfinite(ood_dice):
            missing.append(run_key)
        series["dice"].append(ood_dice)
        series["hd95"].append(ood_hd95)
        series_all["dice"].append(all_dice)
        series_all["hd95"].append(all_hd95)
        d_dice = f"{ood_dice - prev_dice:+.2f}" if prev_dice is not None and np.isfinite(ood_dice) else ""
        d_hd95 = f"{ood_hd95 - prev_hd95:+.2f}" if prev_hd95 is not None and np.isfinite(ood_hd95) else ""
        d_all_dice = f"{all_dice - prev_all_dice:+.2f}" if prev_all_dice is not None and np.isfinite(all_dice) else ""
        d_all_hd95 = f"{all_hd95 - prev_all_hd95:+.2f}" if prev_all_hd95 is not None and np.isfinite(all_hd95) else ""
        lines.append(f"| **{label}** | {ingredient} | {ood_dice:.2f} | {ood_hd95:.2f} | {d_dice} | {d_hd95} "
                     f"| {all_dice:.2f} | {all_hd95:.2f} | {d_all_dice} | {d_all_hd95} |")
        prev_dice, prev_hd95 = ood_dice, ood_hd95
        prev_all_dice, prev_all_hd95 = all_dice, all_hd95

    if missing:
        lines += ["", f"_Not yet available: {', '.join(missing)}_"]
    lines += [""] + _per_contrast_table_md(rungs, full_labels, per_contrast)

    ablations_root.mkdir(parents=True, exist_ok=True)
    md_path = ablations_root / "ladder_summary.md"
    md_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n→ {md_path}")

    # Fill-swap significance (per OOD contrast + pooled), same test the paper's
    # tab:dissociation/fig:ladder report -- computed here so each dataset's own
    # ladder plot carries it rather than only the paper figure.
    fill_idx = _find_fill_swap_idx(rungs)
    sig = {}
    if fill_idx is not None:
        for metric in ("dice", "hd95"):
            sig[metric] = _fill_swap_significance(
                _fill_swap_pairs(metrics_root, rungs, fill_idx, metric, ood_contrasts,
                                 extra_ood_sources=extra_ood_sources))

    dump = {
        "task_name": task_name, "contrast_label": contrast_label,
        "labels": [r[0] for r in rungs], "ingredients": [r[1] for r in rungs],
        "dice": series["dice"], "hd95": series["hd95"],
        "all_dice": series_all["dice"], "all_hd95": series_all["hd95"],
        "in_domain": in_domain, "ood_contrasts": ood_contrasts,
        "per_contrast": per_contrast,
        "fill_swap_significance": sig,
    }
    (ablations_root / "ladder_series.json").write_text(json.dumps(dump, indent=2))
    print(f"→ {ablations_root / 'ladder_series.json'}")

    if combined_png:
        _write_combined_png(ablations_root, contrast_label, task_name, rungs, series, sig=sig)
        _write_per_contrast_png(ablations_root, contrast_label, task_name, rungs, series, per_contrast,
                                 plot_labels=ood_labels, sig=sig)
    return dump


def rung_means_cross_dataset(ood_sources, in_domain_source, run_key, metric):
    """Like rung_means, but for single-contrast datasets (e.g. atlas-liver-hcc) that
    have no second training modality to hold out -- OOD is pooled across MULTIPLE
    metrics roots (separate cross-dataset evaluators, e.g. lld-mmri-hcc + liverhccseg,
    each contributing its own item/contrast columns) instead of held-out contrast
    columns within one dataset's own metrics tree. `run_key` is the SAME run id
    evaluated by every source (it's the same checkpoint, just scored by each
    cross-dataset's own eval script into that dataset's own metrics tree).
    in_domain_source may be None (no in-domain figure available/needed)."""
    scale = 100 if metric == "dice" else 1
    all_vals = []
    for metrics_root in ood_sources:
        run_dir = resolve_run_dir(metrics_root, run_key)
        if not run_dir.is_dir():
            continue
        data = load_case_means(run_dir, metric)
        for cases in data.values():
            all_vals.extend(cases.values())
    ood = float(np.mean(all_vals)) * scale if all_vals else float("nan")

    ind = float("nan")
    if in_domain_source is not None:
        run_dir = resolve_run_dir(in_domain_source, run_key)
        if run_dir.is_dir():
            data = load_case_means(run_dir, metric)
            ind_vals = [v for cases in data.values() for v in cases.values()]
            ind = float(np.mean(ind_vals)) * scale if ind_vals else float("nan")
    return ood, ind


def _leaf_case_values(node, item_to_cases: dict) -> list:
    """Flatten a contrast_groups leaf/list node to a pooled list of raw case
    values, reading from an already-item-collapsed {item: {case: value}} map
    (load_case_means's shape) rather than aggregate_from_config.py's raw
    fold/label tree -- the ladder's OOD figure has always been case-level, this
    keeps that convention while adopting the same grouping semantics."""
    if isinstance(node, str):
        return list(item_to_cases.get(node, {}).values())
    if isinstance(node, list):
        out = []
        for n in node:
            out += _leaf_case_values(n, item_to_cases)
        return out
    raise ValueError(f"bad contrast_groups node for ladder pooling: {node!r}")


def resolve_grouped_ood_value(node, item_to_cases: dict) -> float:
    """Case-level counterpart of aggregate_from_config.resolve_group_value: a
    dict averages its children with equal weight (recursing to any depth); a
    list/leaf pools its cases together and takes the mean. Same contrast_groups
    tree shape/semantics as the significance/aggregate machinery -- reused here
    (not reimplemented) so the ladder's OOD figure can't silently disagree with
    the headline cross-dataset table's weighting scheme."""
    if isinstance(node, dict):
        vals = [resolve_grouped_ood_value(child, item_to_cases) for child in node.values()]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")
    vals = _leaf_case_values(node, item_to_cases)
    return float(np.mean(vals)) if vals else float("nan")


def rung_value_grouped(prefixed_sources, run_key, metric, contrast_groups):
    """Grouped-pooling counterpart of rung_means_cross_dataset. prefixed_sources
    is a list of (metrics_root, column_prefix) pairs (mirrors aggregate_from_
    config.py's sources: schema) -- unlike rung_means_cross_dataset, this
    INCLUDES the in-domain source (e.g. atlas's own t1w) as just another
    prefixed source, since a contrast_groups leaf (e.g. "atlas_t1w") may need
    to reference it directly. Returns the scaled OOD-equivalent scalar (equal-
    weight-per-group, per contrast_groups) for this one rung."""
    scale = 100 if metric == "dice" else 1
    item_to_cases: dict = {}
    for metrics_root, prefix in prefixed_sources:
        run_dir = resolve_run_dir(metrics_root, run_key)
        if not run_dir.is_dir():
            continue
        data = load_case_means(run_dir, metric)
        for item, cases in data.items():
            item_to_cases[f"{prefix}{item}"] = cases
    val = resolve_grouped_ood_value(contrast_groups, item_to_cases)
    return val * scale if np.isfinite(val) else val


def _dataset_name(p: Path) -> str:
    for part in p.parts:
        if part.startswith("8_results_"):
            return part[len("8_results_"):]
    return p.parent.parent.name


def _cross_dataset_contrast_labels(ood_sources, run_key, metric):
    """Discover the (source, item) columns actually present for a rung, labeled
    '<dataset>/<item>' (e.g. 'lld-mmri-hcc/t2wi') -- the cross-dataset counterpart
    of a plain contrast name, since each source evaluator has its own item set."""
    labels = []
    for metrics_root in ood_sources:
        run_dir = resolve_run_dir(metrics_root, run_key)
        if not run_dir.is_dir():
            continue
        data = load_case_means(run_dir, metric)
        ds = _dataset_name(metrics_root)
        labels += [f"{ds}/{item}" for item in data if data[item]]
    return labels


def _per_contrast_rung_means_cross_dataset(ood_sources, run_key, metric, contrast_labels):
    """{'<dataset>/<item>': mean} for ONE rung -- the per-source-item counterpart
    of rung_means_cross_dataset's pooled figure."""
    scale = 100 if metric == "dice" else 1
    per_source = {}
    for metrics_root in ood_sources:
        run_dir = resolve_run_dir(metrics_root, run_key)
        ds = _dataset_name(metrics_root)
        data = load_case_means(run_dir, metric) if run_dir.is_dir() else {}
        for item, cases in data.items():
            per_source[f"{ds}/{item}"] = float(np.mean(list(cases.values()))) * scale if cases else float("nan")
    return {c: per_source.get(c, float("nan")) for c in contrast_labels}


def run_ladder_cross_dataset(*, task_name, contrast_label, ood_sources, ablations_root,
                              rungs, in_domain_source=None, combined_png=True,
                              contrast_groups=None, prefixed_sources=None, in_domain_group=None,
                              exclude_contrast_labels=None):
    """Cross-DATASET sibling of run_ladder(), for datasets with only one training
    modality (no cross-contrast OOD possible). `ood_sources` is a list of metrics-root
    Paths (one per cross-dataset evaluator); `rungs` entries are (label, ingredient,
    run_key) exactly as in run_ladder, where run_key must resolve under every source
    for that rung to score. Writes the same ladder_summary.md / ladder_series.json /
    ladder_<contrast_label>.png trio into ablations_root as run_ladder(), plus a
    per-source-item "sub-ladder" PNG (the cross-dataset counterpart of run_ladder's
    per-contrast breakdown -- one line per '<dataset>/<item>' evaluator stream).

    contrast_groups + prefixed_sources (optional, both required together): the same
    hierarchical-pooling tree used by significance_from_config.py/aggregate_from_
    config.py (see resolve_group_value's block comment there) -- when given, the
    pooled OOD Dice/HD95 figure is computed via rung_value_grouped (equal weight per
    contrast/phase group) instead of rung_means_cross_dataset's flat, case-count-
    weighted mean. prefixed_sources is [(metrics_root, column_prefix), ...],
    INCLUDING the in-domain source, matching a config's `sources:` list -- pass the
    config's own contrast_groups dict directly (e.g. loaded via yaml.safe_load) so
    this can never silently disagree with the headline cross-dataset table's
    weighting. By default the per-contrast breakdown table/PNG shows raw, unpooled
    '<dataset>/<item>' figures for every item found under ood_sources, regardless
    of grouping -- see exclude_contrast_labels below for when that's wrong.

    in_domain_group (optional): name of a top-level contrast_groups key holding
    in-domain/same-broad-contrast evidence (e.g. a group pooling everything from
    the same contrast family as training) -- excluded from this function's pooled
    OOD figure, mirroring significance_from_config.py's in_domain_group exclusion.
    This ladder has no separate "IND" figure the way significance_from_config.py
    does -- the excluded group just isn't part of the OOD Dice/HD95 pooling.

    exclude_contrast_labels (optional): raw '<dataset>/<item>' labels (as they
    appear in the per-contrast breakdown, e.g. "liverhccseg/ce-art_T1w") to drop
    from BOTH the per-contrast markdown table and its PNG -- unlike
    significance_from_config.py's bottom transparency table (which always shows
    every raw column, in-domain included, since that's a deliberate "nothing
    hidden" design), this ladder's per-contrast PNG exists specifically to show
    the sub-ladders underlying the pooled OOD mean line; leaving now-in-domain
    items plotted alongside a mean that no longer includes them is misleading,
    not merely incomplete. Pass the raw labels for whatever contrast_groups
    subtree in_domain_group points at."""
    series = {"dice": [], "hd95": []}
    if contrast_groups and in_domain_group:
        contrast_groups = {k: v for k, v in contrast_groups.items() if k != in_domain_group}
    exclude_contrast_labels = set(exclude_contrast_labels or [])

    # Contrast labels ('<dataset>/<item>') are discovered from whichever rung has the
    # most complete data, so a not-yet-scored rung doesn't shrink the column set.
    contrast_labels: list[str] = []
    for _, _, run_key in rungs:
        found = _cross_dataset_contrast_labels(ood_sources, run_key, "dice")
        if len(found) > len(contrast_labels):
            contrast_labels = found
    contrast_labels = [c for c in contrast_labels if c not in exclude_contrast_labels]
    per_contrast = {"dice": {c: [] for c in contrast_labels}, "hd95": {c: [] for c in contrast_labels}}

    src_names = ", ".join(_dataset_name(p) for p in ood_sources)
    if contrast_groups:
        excl_note = f" `{in_domain_group}` excluded as in-domain/same-broad-contrast." if in_domain_group else ""
        pooling_note = (
            " OOD is pooled with EQUAL WEIGHT PER CONTRAST/PHASE GROUP (see contrast_groups "
            f"in this task's cross_dataset config) -- not a flat case-weighted mean.{excl_note}"
        )
    else:
        pooling_note = (
            " OOD is a flat, case-count-weighted mean across every evaluator item (no "
            "contrast_groups given here)."
        )
    lines = [f"# {task_name} — causal ablation ladder (cross-dataset OOD)", "",
             "Each rung adds exactly one ingredient on top of the previous rung — the "
             "OOD Dice/HD95 delta is attributable to that ingredient alone. This dataset "
             "has only one training modality, so OOD here means pooled cross-DATASET "
             f"generalization (evaluators: {src_names}) rather than held-out contrasts "
             f"within the training dataset.{pooling_note}", "",
             "| rung | adds | OOD Dice | OOD HD95 | Δ Dice vs. prev | Δ HD95 vs. prev |",
             "|---|---|---|---|---|---|"]
    prev_dice = prev_hd95 = None
    missing = []
    for label, ingredient, run_key in rungs:
        if contrast_groups:
            ood_dice = rung_value_grouped(prefixed_sources, run_key, "dice", contrast_groups)
            ood_hd95 = rung_value_grouped(prefixed_sources, run_key, "hd95", contrast_groups)
        else:
            ood_dice, _ = rung_means_cross_dataset(ood_sources, in_domain_source, run_key, "dice")
            ood_hd95, _ = rung_means_cross_dataset(ood_sources, in_domain_source, run_key, "hd95")
        if not np.isfinite(ood_dice):
            missing.append(run_key)
        series["dice"].append(ood_dice)
        series["hd95"].append(ood_hd95)
        for metric in ("dice", "hd95"):
            pc = _per_contrast_rung_means_cross_dataset(ood_sources, run_key, metric, contrast_labels)
            for c in contrast_labels:
                per_contrast[metric][c].append(pc[c])
        d_dice = f"{ood_dice - prev_dice:+.2f}" if prev_dice is not None and np.isfinite(ood_dice) else ""
        d_hd95 = f"{ood_hd95 - prev_hd95:+.2f}" if prev_hd95 is not None and np.isfinite(ood_hd95) else ""
        lines.append(f"| **{label}** | {ingredient} | {ood_dice:.2f} | {ood_hd95:.2f} | {d_dice} | {d_hd95} |")
        prev_dice, prev_hd95 = ood_dice, ood_hd95

    if missing:
        lines += ["", f"_Not yet available: {', '.join(missing)}_"]
    if contrast_labels:
        lines += [""] + _per_contrast_table_md(rungs, contrast_labels, per_contrast)

    ablations_root.mkdir(parents=True, exist_ok=True)
    md_path = ablations_root / "ladder_summary.md"
    md_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n→ {md_path}")

    # Fill-swap significance, same as run_ladder's (see _fill_swap_significance).
    fill_idx = _find_fill_swap_idx(rungs)
    sig = {}
    if fill_idx is not None:
        for metric in ("dice", "hd95"):
            sig[metric] = _fill_swap_significance(
                _fill_swap_pairs_cross_dataset(ood_sources, rungs, fill_idx, metric))

    dump = {
        "task_name": task_name, "contrast_label": contrast_label,
        "labels": [r[0] for r in rungs], "ingredients": [r[1] for r in rungs],
        "dice": series["dice"], "hd95": series["hd95"],
        "ood_sources": [str(p) for p in ood_sources],
        "grouped_pooling": bool(contrast_groups),
        "per_contrast": per_contrast,
        "fill_swap_significance": sig,
    }
    (ablations_root / "ladder_series.json").write_text(json.dumps(dump, indent=2))
    print(f"→ {ablations_root / 'ladder_series.json'}")

    if combined_png:
        _write_combined_png(ablations_root, contrast_label, task_name, rungs, series, sig=sig)
        if contrast_labels:
            _write_per_contrast_png(ablations_root, contrast_label, task_name, rungs, series,
                                     per_contrast, sig=sig)
    return dump


def _write_combined_png(ablations_root, contrast_label, task_name, rungs, series, sig=None):
    """The headline 2-panel (Dice, HD95) pooled-OOD ladder plot -- ladder_<contrast>.png.

    OOD-only, matching the paper's fig:ladder: the pooled line is black with its
    real-fill segment coloured by this ladder's own fill-swap significance, the
    step itself is shaded, and the pooled p-value is annotated on it. The
    all-contrasts (in-domain-inclusive) series is still computed and written to
    ladder_summary.md / ladder_series.json, just not drawn here -- the plots
    report the same OOD estimand the paper does."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(matplotlib unavailable, skipping combined plot: {e})", file=sys.stderr)
        return
    labels = [r[0] for r in rungs]
    x = np.arange(len(labels))
    fill_idx = _find_fill_swap_idx(rungs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for metric, ax, ylabel, higher_is_better in (
        ("dice", axes[0], "OOD Dice (%)", True),
        ("hd95", axes[1], "OOD HD95 mm (lower better)", False),
    ):
        s = np.asarray(series[metric], dtype=float)
        ax.plot(x, s, "o-", color="black", linewidth=2.2, markersize=5, zorder=3)
        if fill_idx is not None:
            delta = (s[fill_idx] - s[fill_idx - 1]) * (1 if higher_is_better else -1)
            pooled_p = (sig or {}).get(metric, {}).get("pooled_p", float("nan"))
            ax.plot([fill_idx - 1, fill_idx], s[fill_idx - 1:fill_idx + 1], "o-",
                    color=_sig_color(pooled_p, delta), linewidth=3.0, markersize=6, zorder=4)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel(ylabel); ax.set_title(f"{task_name} ladder — {metric.upper()}")
        ax.grid(alpha=0.3)
        if fill_idx is not None:
            _decorate_fill_swap(ax, fill_idx, (sig or {}).get(metric, {}).get("pooled_p", float("nan")),
                                (s[fill_idx] - s[fill_idx - 1]) * (1 if higher_is_better else -1))
    fig.tight_layout()
    png_path = ablations_root / f"ladder_{contrast_label}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"→ {png_path}")


def _write_per_contrast_png(ablations_root, contrast_label, task_name, rungs, series, per_contrast,
                             plot_labels=None, sig=None):
    """The "sub-ladder" 2-panel (Dice, HD95) plot -- one thin line per held-out
    eval contrast (each scored on its own, not pooled), plus the pooled OOD mean
    drawn bold black on top, so it's obvious at a glance whether a rung-to-rung
    effect holds across every held-out contrast or is driven by just one.
    Saved as ladder_<contrast>_per_contrast.png.

    Matches the paper's fig:ladder scheme: each thin curve is neutral grey
    except at the real-fill step, which is coloured by that contrast's OWN
    fill-swap significance (light teal/red/grey) -- only that step is tested, so
    only that step looks tested. The bold pooled line's real-fill segment takes
    the full-saturation colour of the ladder's pooled test, matching the
    annotation above it.

    plot_labels (optional): which of per_contrast's keys to draw. run_ladder
    passes its OOD labels only, so the training contrast it also tracks for the
    markdown/JSON stays out of these OOD-estimand plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(matplotlib unavailable, skipping per-contrast plot: {e})", file=sys.stderr)
        return
    labels = [r[0] for r in rungs]
    x = np.arange(len(labels))
    contrast_labels = list(plot_labels if plot_labels is not None else per_contrast["dice"].keys())
    fill_idx = _find_fill_swap_idx(rungs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for metric, ax, ylabel, title, higher_is_better in (
        ("dice", axes[0], "OOD Dice (%)", f"{task_name} sub-ladders — Dice", True),
        ("hd95", axes[1], "OOD HD95 mm (lower better)", f"{task_name} sub-ladders — HD95", False),
    ):
        per_c_p = (sig or {}).get(metric, {}).get("per_contrast", {})
        for c in contrast_labels:
            s = np.asarray(per_contrast[metric].get(c, []), dtype=float)
            if not s.size:
                continue
            ax.plot(x, s, "o-", color=NEUTRAL, alpha=0.6, linewidth=1.2, markersize=3,
                    zorder=1, label=c)
            if fill_idx is not None and np.isfinite(s[fill_idx - 1:fill_idx + 1]).all():
                d = (s[fill_idx] - s[fill_idx - 1]) * (1 if higher_is_better else -1)
                ax.plot([fill_idx - 1, fill_idx], s[fill_idx - 1:fill_idx + 1], "o-",
                        color=_sig_color(per_c_p.get(c, float("nan")), d, light=True),
                        linewidth=2.0, markersize=4.5, zorder=2)
        sm = np.asarray(series[metric], dtype=float)
        ax.plot(x, sm, "o-", color="black", linewidth=2.5, markersize=5, zorder=3,
                label="OOD mean (pooled)")
        if fill_idx is not None:
            pooled_p = (sig or {}).get(metric, {}).get("pooled_p", float("nan"))
            d = (sm[fill_idx] - sm[fill_idx - 1]) * (1 if higher_is_better else -1)
            ax.plot([fill_idx - 1, fill_idx], sm[fill_idx - 1:fill_idx + 1], "o-",
                    color=_sig_color(pooled_p, d), linewidth=3.2, markersize=6, zorder=4)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=0.3)
        if fill_idx is not None:
            _decorate_fill_swap(ax, fill_idx, (sig or {}).get(metric, {}).get("pooled_p", float("nan")),
                                (sm[fill_idx] - sm[fill_idx - 1]) * (1 if higher_is_better else -1))

    # One shared legend below both panels (contrast count varies a lot across
    # datasets -- on-harmony has 5, CHAOS has 3 -- so it doesn't fit cleanly in-axes).
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=min(len(legend_labels), 5),
               fontsize=8, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    png_path = ablations_root / f"ladder_{contrast_label}_per_contrast.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {png_path}")
