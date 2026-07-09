#!/usr/bin/env python3
"""
Config-driven paired significance testing for cross-experiment segmentation results.

Common and shared: statistical primitives (wilcoxon_p, holm) live in the shared
datasets/00_commun_scripts/00_00_utils/stat_tests.py, imported here rather than
reimplemented — every dataset/experiment that wires this up gets the same math.

Companion to aggregate_from_config.py: consumes the SAME YAML configs (same
`sources`/`runs`/`column_order` fields, same run-ID resolution) so the tests line
up exactly with the summary tables. Used by chaos/on-harmony/open-ms (the
datasets on the YAML-config aggregation path); the amos/brats2024-glioma/
sliver07/trusted family uses the older CLI-args `aggregate_results.py`, which
doesn't have a YAML config for this script to consume.

Statistical model (why this is the right test):
  * Unit of analysis = the test CASE, not the fold. Each external test case is
    segmented by every fold's model; we average a case's Dice over its labels and
    over the evaluated folds to get ONE score per (run, contrast, case). This
    matters: treating each fold's re-scoring of the same patient as a separate
    sample is pseudo-replication (the observations aren't independent) and
    inflates apparent significance. Averaging across folds first gives the
    honest, independent sample size — the number of held-out patients.
  * Methods are compared PAIRED on the same cases -> Wilcoxon signed-rank test
    (two-sided). Dice is bounded, skewed, and has floor effects at 0, so a
    non-parametric paired test is the appropriate choice over a paired t-test.
  * Folds are capped to EVAL_FOLD_INDICES (0-2 — see eval_folds.py), the single
    source of truth shared with eval_aggregate.py and aggregate_from_config.py.
    This is the default AND ONLY behavior, not an opt-in: any fold beyond that
    (e.g. fold3 from older 4-fold runs) is silently excluded, so every run is
    compared on the same footing.

For each (reference run) vs (competitor run):
  * per-contrast Wilcoxon on the shared cases,
  * a pooled test over all held-out contrasts (each (case,contrast) a pair),
  * effect size: median paired Dice difference + win/tie/loss counts,
  * Holm-Bonferroni correction across the family of pooled per-competitor tests.

The reference ("Ours") run is auto-selected as the run whose key contains both
"auglabAug" and "v26_6_2" (the headline model) — CAUTION: ablation variants that
extend that same run id (e.g. "..._auglabAug_v26_6_2_noisefill_...") also match
this substring test and will be picked if they sort earlier in `runs:`. Pass
--ref <exact run id> explicitly whenever the run list contains such variants.

Usage:
  python significance_from_config.py <config.yaml> [--ref <substr>] [--metric dice]

Writes {output_dir}/{output_prefix}_significance.md and prints a summary.
"""
import argparse
import csv
import importlib.util
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "00_00_utils"))
from eval_folds import EVAL_FOLD_INDICES  # noqa: E402 — single source of truth, see eval_folds.py
from stat_tests import wilcoxon_p, holm, fmt_p  # noqa: E402 — shared math, see stat_tests.py

# Reuse run-dir resolution from the aggregator so resolution stays identical.
# aggregate_from_config.py is now a sibling in this same dir (00_03_evaluate/).
_AGG = Path(__file__).resolve().parent / "aggregate_from_config.py"
_spec = importlib.util.spec_from_file_location("aggregate_from_config", _AGG)
_agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_agg)
resolve_run_dir = _agg.resolve_run_dir


def load_run_cases(sources, key, metric):
    """col -> {case_id: mean metric over that case's labels and folds in EVAL_FOLD_INDICES}."""
    per = defaultdict(lambda: defaultdict(list))
    for src in sources:
        run_dir = resolve_run_dir(src["metrics_dir"], key)
        if run_dir is None:
            continue
        prefix = src.get("column_prefix", "")
        rename = src.get("column_rename", {})
        for fold_dir in sorted(run_dir.glob("fold*")):
            try:
                fold_idx = int(fold_dir.name.replace("fold", ""))
            except ValueError:
                continue
            if fold_idx not in EVAL_FOLD_INDICES:
                continue
            csv_path = fold_dir / "eval_all.csv"
            if not csv_path.exists():
                continue
            with csv_path.open() as f:
                for row in csv.DictReader(f):
                    col = rename.get(row["group"], f"{prefix}{row['group']}")
                    try:
                        v = float(row[metric])
                    except (KeyError, ValueError):
                        continue
                    if np.isfinite(v):
                        per[col][row["case"]].append(v)
    return {col: {c: float(np.mean(vs)) for c, vs in cases.items() if vs}
            for col, cases in per.items()}


def paired(ref_cases, comp_cases, col):
    """Aligned (x_ref, y_comp) arrays over shared cases for one contrast column."""
    r, c = ref_cases.get(col, {}), comp_cases.get(col, {})
    common = sorted(set(r) & set(c))
    return np.array([r[k] for k in common]), np.array([c[k] for k in common])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--ref", default=None,
                    help="EXACT run id (not just a substring) to use as reference. Default: "
                         "auto-pick the first run containing both 'auglabAug' and 'v26_6_2' — "
                         "pass this explicitly whenever the run list also contains ablation "
                         "variants of that same run (e.g. '..._noisefill_...'), since they "
                         "match the same substring test.")
    ap.add_argument("--metric", default="dice", choices=["dice", "hd95"])
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"Config not found: {cfg_path}")
    import yaml
    cfg = yaml.safe_load(cfg_path.open())

    if "sources" in cfg:
        sources = [{"metrics_dir": Path(os.path.expandvars(s["metrics_dir"])),
                    "column_prefix": s.get("column_prefix", ""),
                    "column_rename": s.get("column_rename", {})} for s in cfg["sources"]]
        out_dir = Path(os.path.expandvars(cfg["output_dir"])) if "output_dir" in cfg \
            else sources[0]["metrics_dir"]
    else:
        md = Path(os.path.expandvars(cfg["metrics_dir"]))
        sources = [{"metrics_dir": md, "column_prefix": "", "column_rename": {}}]
        out_dir = md

    runs = cfg.get("runs", [])
    column_order = cfg.get("column_order", None)
    prefix = cfg.get("output_prefix", "significance")
    title = cfg.get("title", "Results")

    # Pick reference run. --ref matches EXACTLY (not a substring): ablation variants
    # of the reference (e.g. "..._noisefill_...") share the "auglabAug"+"v26_6_2"
    # substring, so a fuzzy match here could silently pick the wrong run.
    if args.ref:
        ref = args.ref if args.ref in runs else None
    else:
        ref = next((r for r in runs if "auglabAug" in r and "v26_6_2" in r), None) \
            or next((r for r in runs if "v26_6_2" in r), None)
    if ref is None:
        sys.exit(f"Could not identify a reference run; pass --ref <exact run id from {runs}>.")
    competitors = [r for r in runs if r != ref]

    # Load per-case scores for every run (EVAL_FOLD_INDICES-capped — see load_run_cases).
    data = {r: load_run_cases(sources, r, args.metric) for r in runs}

    all_cols = sorted({c for d in data.values() for c in d})
    if column_order:
        cols = [c for c in column_order if c in all_cols] + \
               [c for c in all_cols if c not in column_order]
    else:
        cols = all_cols

    higher_better = args.metric == "dice"
    sign = 1.0 if higher_better else -1.0  # positive "diff" = ref is better

    lines = [f"# {title} — paired significance ({args.metric})", "",
             f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
             f"Reference (Ours): `{ref}`  |  Folds: {', '.join(str(i) for i in EVAL_FOLD_INDICES)}", "",
             "Two-sided Wilcoxon signed-rank on per-case scores (a case's score = mean over "
             "its labels and folds 0-2). Unit = case, so the sample size is the number of "
             "held-out cases, **not** the fold count. `Δ` = median paired difference "
             f"(ref − competitor, in {'Dice pts' if higher_better else 'mm HD95'}); "
             "W/T/L = cases where ref wins/ties/loses.", ""]

    # Pooled per-competitor test (the headline claim), Holm-corrected across competitors.
    pooled_p, pooled_rows = [], []
    for comp in competitors:
        dx, wtl, deltas = [], [0, 0, 0], []
        for col in cols:
            xf, yf = paired(data[ref], data[comp], col)
            dx.extend((xf - yf).tolist())
            for a, b in zip(xf, yf):
                d = sign * (a - b)
                deltas.append(a - b)
                wtl[0 if d > 0 else 2 if d < 0 else 1] += 1
        ax = np.array(dx)
        p = wilcoxon_p(ax, np.zeros_like(ax)) if len(ax) else float("nan")
        med = float(np.median(deltas)) * (100 if higher_better else 1) if deltas else float("nan")
        pooled_p.append(p)
        pooled_rows.append((comp, len(ax), med, wtl))

    holm_p = holm(pooled_p)

    lines += ["## Pooled over all held-out contrasts (Ours vs each competitor)", "",
              "| competitor | n pairs | Δ median | W/T/L | p-value | Holm-corrected |",
              "|---|---|---|---|---|---|"]
    for (comp, n, med, wtl), p, h in zip(pooled_rows, pooled_p, holm_p):
        star = "**" if np.isfinite(h) and h < args.alpha else ""
        lines.append(
            f"| {comp} | {n} | {med:+.2f} | {wtl[0]}/{wtl[1]}/{wtl[2]} | "
            f"{fmt_p(p)} | {star}{fmt_p(h)}{star} |")

    # Per-contrast breakdown (raw p, uncorrected — exploratory, not the headline claim).
    lines += ["", "## Per-contrast (raw p, uncorrected)", ""]
    header = "| competitor | " + " | ".join(cols) + " |"
    lines += [header, "|" + "---|" * (len(cols) + 1)]
    for comp in competitors:
        cells = []
        for col in cols:
            x, y = paired(data[ref], data[comp], col)
            cells.append(f"{fmt_p(wilcoxon_p(x, y))} (n={len(x)})")
        lines.append(f"| {comp} | " + " | ".join(cells) + " |")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}_significance.md"
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[written] {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
