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
              in_domain, ood_contrasts, rungs, combined_png=True):
    """Compute the ladder, write ladder_summary.md + a JSON dump (for the shared
    multi-task plotting script) into ablations_root, plus the dataset-local pooled
    2-panel PNG and a per-eval-contrast "sub-ladder" 2-panel PNG. Returns the dump dict.
    """
    series = {"dice": [], "hd95": []}
    per_contrast = {"dice": {c: [] for c in ood_contrasts}, "hd95": {c: [] for c in ood_contrasts}}
    lines = [f"# {task_name} — causal ablation ladder", "",
             "Each rung adds exactly one ingredient on top of the previous rung — the "
             "OOD Dice/HD95 delta is attributable to that ingredient alone. "
             f"OOD = mean over held-out contrasts ({', '.join(ood_contrasts)}), "
             f"training contrast ({in_domain}) excluded.", "",
             "| rung | adds | OOD Dice | OOD HD95 | Δ Dice vs. prev | Δ HD95 vs. prev |",
             "|---|---|---|---|---|---|"]
    prev_dice = prev_hd95 = None
    missing = []
    for label, ingredient, run_key in rungs:
        ood_dice, _ = rung_means(metrics_root, run_key, "dice", in_domain, ood_contrasts)
        ood_hd95, _ = rung_means(metrics_root, run_key, "hd95", in_domain, ood_contrasts)
        if ood_dice is None:
            missing.append(run_key)
            ood_dice = ood_hd95 = float("nan")
        series["dice"].append(ood_dice)
        series["hd95"].append(ood_hd95)
        for metric in ("dice", "hd95"):
            pc = _per_contrast_rung_means(metrics_root, run_key, metric, ood_contrasts)
            for c in ood_contrasts:
                per_contrast[metric][c].append(pc.get(c, float("nan")))
        d_dice = f"{ood_dice - prev_dice:+.2f}" if prev_dice is not None and np.isfinite(ood_dice) else ""
        d_hd95 = f"{ood_hd95 - prev_hd95:+.2f}" if prev_hd95 is not None and np.isfinite(ood_hd95) else ""
        lines.append(f"| **{label}** | {ingredient} | {ood_dice:.2f} | {ood_hd95:.2f} | {d_dice} | {d_hd95} |")
        prev_dice, prev_hd95 = ood_dice, ood_hd95

    if missing:
        lines += ["", f"_Not yet available: {', '.join(missing)}_"]
    lines += [""] + _per_contrast_table_md(rungs, ood_contrasts, per_contrast)

    ablations_root.mkdir(parents=True, exist_ok=True)
    md_path = ablations_root / "ladder_summary.md"
    md_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n→ {md_path}")

    dump = {
        "task_name": task_name, "contrast_label": contrast_label,
        "labels": [r[0] for r in rungs], "ingredients": [r[1] for r in rungs],
        "dice": series["dice"], "hd95": series["hd95"],
        "in_domain": in_domain, "ood_contrasts": ood_contrasts,
        "per_contrast": per_contrast,
    }
    (ablations_root / "ladder_series.json").write_text(json.dumps(dump, indent=2))
    print(f"→ {ablations_root / 'ladder_series.json'}")

    if combined_png:
        _write_combined_png(ablations_root, contrast_label, task_name, rungs, series)
        _write_per_contrast_png(ablations_root, contrast_label, task_name, rungs, series, per_contrast)
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
                              rungs, in_domain_source=None, combined_png=True):
    """Cross-DATASET sibling of run_ladder(), for datasets with only one training
    modality (no cross-contrast OOD possible). `ood_sources` is a list of metrics-root
    Paths (one per cross-dataset evaluator); `rungs` entries are (label, ingredient,
    run_key) exactly as in run_ladder, where run_key must resolve under every source
    for that rung to score. Writes the same ladder_summary.md / ladder_series.json /
    ladder_<contrast_label>.png trio into ablations_root as run_ladder(), plus a
    per-source-item "sub-ladder" PNG (the cross-dataset counterpart of run_ladder's
    per-contrast breakdown -- one line per '<dataset>/<item>' evaluator stream)."""
    series = {"dice": [], "hd95": []}

    # Contrast labels ('<dataset>/<item>') are discovered from whichever rung has the
    # most complete data, so a not-yet-scored rung doesn't shrink the column set.
    contrast_labels: list[str] = []
    for _, _, run_key in rungs:
        found = _cross_dataset_contrast_labels(ood_sources, run_key, "dice")
        if len(found) > len(contrast_labels):
            contrast_labels = found
    per_contrast = {"dice": {c: [] for c in contrast_labels}, "hd95": {c: [] for c in contrast_labels}}

    src_names = ", ".join(_dataset_name(p) for p in ood_sources)
    lines = [f"# {task_name} — causal ablation ladder (cross-dataset OOD)", "",
             "Each rung adds exactly one ingredient on top of the previous rung — the "
             "OOD Dice/HD95 delta is attributable to that ingredient alone. This dataset "
             "has only one training modality, so OOD here means pooled cross-DATASET "
             f"generalization (evaluators: {src_names}) rather than held-out contrasts "
             "within the training dataset.", "",
             "| rung | adds | OOD Dice | OOD HD95 | Δ Dice vs. prev | Δ HD95 vs. prev |",
             "|---|---|---|---|---|---|"]
    prev_dice = prev_hd95 = None
    missing = []
    for label, ingredient, run_key in rungs:
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

    dump = {
        "task_name": task_name, "contrast_label": contrast_label,
        "labels": [r[0] for r in rungs], "ingredients": [r[1] for r in rungs],
        "dice": series["dice"], "hd95": series["hd95"],
        "ood_sources": [str(p) for p in ood_sources],
        "per_contrast": per_contrast,
    }
    (ablations_root / "ladder_series.json").write_text(json.dumps(dump, indent=2))
    print(f"→ {ablations_root / 'ladder_series.json'}")

    if combined_png:
        _write_combined_png(ablations_root, contrast_label, task_name, rungs, series)
        if contrast_labels:
            _write_per_contrast_png(ablations_root, contrast_label, task_name, rungs, series, per_contrast)
    return dump


def _write_combined_png(ablations_root, contrast_label, task_name, rungs, series):
    """The headline 2-panel (Dice, HD95) pooled-OOD ladder plot -- ladder_<contrast>.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(matplotlib unavailable, skipping combined plot: {e})", file=sys.stderr)
        return
    labels = [r[0] for r in rungs]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(x, series["dice"], "o-", color="#4aa564")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    axes[0].set_ylabel("OOD Dice (%)"); axes[0].set_title(f"{task_name} ladder — Dice")
    axes[0].grid(alpha=0.3)
    axes[1].plot(x, series["hd95"], "o-", color="#4aa564")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    axes[1].set_ylabel("OOD HD95 mm (lower better)"); axes[1].set_title(f"{task_name} ladder — HD95")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    png_path = ablations_root / f"ladder_{contrast_label}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"→ {png_path}")


def _write_per_contrast_png(ablations_root, contrast_label, task_name, rungs, series, per_contrast):
    """The "sub-ladder" 2-panel (Dice, HD95) plot -- one thin colored line per held-out
    eval contrast (each scored on its own, not pooled), plus the pooled OOD mean
    (series, same as _write_combined_png's line) drawn bold black on top so it's
    obvious at a glance whether the rung-to-rung effect is consistent across every
    held-out contrast or driven by just one. Saved as ladder_<contrast>_per_contrast.png,
    the sibling of the pooled ladder_<contrast>.png written by _write_combined_png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(matplotlib unavailable, skipping per-contrast plot: {e})", file=sys.stderr)
        return
    labels = [r[0] for r in rungs]
    x = np.arange(len(labels))
    contrast_labels = list(per_contrast["dice"].keys())
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for metric, ax, ylabel, title in (
        ("dice", axes[0], "Dice (%)", f"{task_name} sub-ladders — Dice"),
        ("hd95", axes[1], "HD95 mm (lower better)", f"{task_name} sub-ladders — HD95"),
    ):
        for i, c in enumerate(contrast_labels):
            ax.plot(x, per_contrast[metric][c], "o-", color=cmap(i % 10), alpha=0.6,
                    linewidth=1.3, markersize=3, label=c)
        ax.plot(x, series[metric], "o-", color="black", linewidth=2.5, markersize=5, label="OOD mean (pooled)")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=0.3)

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
