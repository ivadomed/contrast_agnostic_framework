#!/usr/bin/env python3
"""
Config-driven paired significance testing for cross-experiment segmentation results.

Companion to aggregate_from_config.py: consumes the SAME YAML configs (same
`sources`/`runs`/`column_order` fields, same run-ID resolution) so the tests line
up exactly with the summary tables.

Statistical model (why this is the right test):
  * Unit of analysis = the test CASE, not the fold. Each external test case is
    segmented by every fold's model; we average a case's Dice over its labels and
    over the evaluated folds to get ONE score per (run, contrast, case).
  * Methods are compared PAIRED on the same cases -> Wilcoxon signed-rank test
    (two-sided). Dice is bounded, skewed, and has floor effects at 0, so a
    non-parametric paired test is the appropriate choice over a paired t-test.
  * Because the sample size is the number of cases (tens–hundreds), NOT the number
    of folds, dropping 4-fold -> 3-fold does not change the test's degrees of
    freedom. It only re-estimates each per-case Dice from 3 models instead of 4,
    which moves the scores negligibly. This script reports p-values under both
    fold sets side-by-side so that claim is verifiable, not asserted.

For each (reference run) vs (competitor run):
  * per-contrast Wilcoxon on the shared cases,
  * a pooled test over all held-out contrasts (each (case,contrast) a pair),
  * effect size: median paired Dice difference + win/tie/loss counts,
  * Holm-Bonferroni correction across the family of pooled per-competitor tests.

The reference ("Ours") run is auto-selected as the run whose key contains both
"auglabAug" and "v26_6_2" (the headline model); override with --ref <substr>.

Usage:
  python significance_from_config.py <config.yaml> [--ref <substr>] \
      [--folds-full 0,1,2,3] [--folds-reduced 0,1,2] [--metric dice]

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
from scipy import stats

# Reuse run-dir resolution from the sibling aggregator so resolution stays identical.
_AGG = Path(__file__).with_name("aggregate_from_config.py")
_spec = importlib.util.spec_from_file_location("aggregate_from_config", _AGG)
_agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_agg)
resolve_run_dir = _agg.resolve_run_dir


def parse_folds(s: str):
    return None if s.strip().lower() in ("", "all") else {int(x) for x in s.split(",")}


def load_run_cases(sources, key, metric, folds=None):
    """col -> {case_id: mean metric over that case's labels and evaluated folds}.

    `folds` is a set of fold indices to include, or None for all.
    """
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
            if folds is not None and fold_idx not in folds:
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


def wilcoxon_p(x, y):
    """Two-sided Wilcoxon signed-rank p-value; nan if undefined."""
    if len(x) < 1:
        return float("nan")
    d = x - y
    if not np.any(d != 0):
        return float("nan")
    try:
        return float(stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, preserving input order."""
    idx = [i for i, p in enumerate(pvals) if np.isfinite(p)]
    out = [float("nan")] * len(pvals)
    m = len(idx)
    order = sorted(idx, key=lambda i: pvals[i])
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i])
        adj = max(adj, prev)  # enforce monotonicity
        out[i] = adj
        prev = adj
    return out


def fmt_p(p):
    if not np.isfinite(p):
        return "—"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--ref", default=None,
                    help="substring picking the reference run (default: auto 'auglabAug'+'v26_6_2')")
    ap.add_argument("--metric", default="dice", choices=["dice", "hd95"])
    ap.add_argument("--folds-full", default="all")
    ap.add_argument("--folds-reduced", default="0,1,2")
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

    # Pick reference run.
    if args.ref:
        ref = next((r for r in runs if args.ref in r), None)
    else:
        ref = next((r for r in runs if "auglabAug" in r and "v26_6_2" in r), None) \
            or next((r for r in runs if "v26_6_2" in r), None)
    if ref is None:
        sys.exit("Could not identify a reference run; pass --ref <substr>.")
    competitors = [r for r in runs if r != ref]

    full = parse_folds(args.folds_full)
    reduced = parse_folds(args.folds_reduced)

    # Load per-case scores for every run, for both fold sets.
    def load_all(folds):
        return {r: load_run_cases(sources, r, args.metric, folds) for r in runs}
    data_full, data_red = load_all(full), load_all(reduced)

    all_cols = sorted({c for d in data_full.values() for c in d})
    if column_order:
        cols = [c for c in column_order if c in all_cols] + \
               [c for c in all_cols if c not in column_order]
    else:
        cols = all_cols

    higher_better = args.metric == "dice"
    sign = 1.0 if higher_better else -1.0  # positive "diff" = ref is better

    lines = [f"# {title} — paired significance ({args.metric})", "",
             f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
             f"Reference (Ours): `{ref}`", "",
             "Two-sided Wilcoxon signed-rank on per-case scores (a case's score = mean over "
             "its labels and evaluated folds). Unit = case, so the sample size is the number of "
             "held-out cases, **not** the fold count. `Δ` = median paired difference "
             f"(ref − competitor, in {'Dice pts' if higher_better else 'mm HD95'}); "
             "W/T/L = cases where ref wins/ties/loses.", ""]

    # Pooled per-competitor test (the headline claim), both fold sets, Holm-corrected.
    pooled_full_p, pooled_red_p, pooled_rows = [], [], []
    for comp in competitors:
        dx_full, dx_red, wtl, deltas = [], [], [0, 0, 0], []
        for col in cols:
            xf, yf = paired(data_full[ref], data_full[comp], col)
            dx_full.extend((xf - yf).tolist())
            xr, yr = paired(data_red[ref], data_red[comp], col)
            dx_red.extend((xr - yr).tolist())
            for a, b in zip(xf, yf):
                d = sign * (a - b)
                deltas.append(a - b)
                wtl[0 if d > 0 else 2 if d < 0 else 1] += 1
        af = np.array(dx_full)
        pf = wilcoxon_p(af, np.zeros_like(af)) if len(af) else float("nan")
        ar = np.array(dx_red)
        pr = wilcoxon_p(ar, np.zeros_like(ar)) if len(ar) else float("nan")
        med = float(np.median(deltas)) * (100 if higher_better else 1) if deltas else float("nan")
        pooled_full_p.append(pf)
        pooled_red_p.append(pr)
        pooled_rows.append((comp, len(af), med, wtl))

    holm_full = holm(pooled_full_p)
    holm_red = holm(pooled_red_p)

    lines += ["## Pooled over all held-out contrasts (Ours vs each competitor)", "",
              "| competitor | n pairs | Δ median | W/T/L | p (4-fold) | Holm (4f) | p (3-fold) | Holm (3f) |",
              "|---|---|---|---|---|---|---|---|"]
    for (comp, n, med, wtl), pf, hf, pr, hr in zip(
            pooled_rows, pooled_full_p, holm_full, pooled_red_p, holm_red):
        star = lambda h: "**" if np.isfinite(h) and h < args.alpha else ""
        lines.append(
            f"| {comp} | {n} | {med:+.2f} | {wtl[0]}/{wtl[1]}/{wtl[2]} | "
            f"{fmt_p(pf)} | {star(hf)}{fmt_p(hf)}{star(hf)} | "
            f"{fmt_p(pr)} | {star(hr)}{fmt_p(hr)}{star(hr)} |")

    # Per-contrast breakdown (raw p, 4-fold vs 3-fold) — no correction, exploratory.
    lines += ["", "## Per-contrast (raw p; 4-fold vs 3-fold, uncorrected)", ""]
    header = "| competitor | " + " | ".join(cols) + " |"
    lines += [header, "|" + "---|" * (len(cols) + 1)]
    for comp in competitors:
        cells = []
        for col in cols:
            xf, yf = paired(data_full[ref], data_full[comp], col)
            xr, yr = paired(data_red[ref], data_red[comp], col)
            pf, pr = wilcoxon_p(xf, yf), wilcoxon_p(xr, yr)
            cells.append(f"{fmt_p(pf)}/{fmt_p(pr)} (n={len(xf)})")
        lines.append(f"| {comp} | " + " | ".join(cells) + " |")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}_significance.md"
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[written] {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
