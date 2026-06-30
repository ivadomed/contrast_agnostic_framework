#!/usr/bin/env python3
"""
Low-data-regime benchmark — build learning curves (Dice / HD95 vs N training
subjects), one line per method, per modality, with 4-fold mean±std error bars.

Reads the standard per-run metric dirs written by 06_01_evaluate_run.sh:
    <METRICS_BASE>/{CATEGORY}_chaos_t1in_<method>_lowdata_n<NN>_<ts>/fold*/eval_all.csv
where METRICS_BASE = 8_results_chaos/02_metrics/chaos_model/t1in.

This benchmark scores the IN-DOMAIN contrast only (= training contrast, t1in): the
low-data axis measures how each technique holds up as N shrinks, not cross-contrast
generalization (that is the main benchmark's job). Pass --in_domain to override.

Writes the benchmark roll-up to the contrast-specific  <METRICS_BASE>/05_01_low_data/ :
    lowdata_curve.csv             tidy (method, N, modality, metric, mean, std, n_folds)
    lowdata_summary.md            method × N grid (in-domain Dice)
    lowdata_dice_<in_domain>.png  Dice vs N, one line per method
    lowdata_hd95_<in_domain>.png  HD95 vs N, one line per method

Reuses the shared aggregation primitives (load_run, cross_fold_stats) so it stays
consistent with every other aggregator in the repo.

Usage:
    .venv/bin/python 06_21_lowdata_curve.py --metrics_base <DIR> [--out_dir <DIR>]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "00_commun_scripts" / "00_00_utils"))
from eval_aggregate import load_run, cross_fold_stats  # noqa: E402

RUN_RE = re.compile(r"^(nnUNet|auglab)_chaos_[a-z0-9]+_(?P<method>.+)_lowdata_n0*(?P<n>\d+)_")

# Prettier, stable legend labels + plotting order.
METHOD_LABEL = {
    "baseline": "baseline",
    "v26_6_2_train050_val100": "v26_6_2",
    "auglab_default": "auglab_default",
    "synthseg_EM": "synthseg_EM",
    "synthseg_noEM": "synthseg_noEM",
    "auglabAug_v26_6_2_train025_val100": "auglabAug_v26_6_2",
}
METHOD_ORDER = list(METHOD_LABEL)


def cross_fold_class_stats(run_data: dict, metric: str, contrast: str) -> tuple:
    """(mean, std, n_folds) pooled over all labels per fold, then across folds."""
    by_contrast = run_data.get(metric, {}).get(contrast, {})
    per_fold = defaultdict(list)
    for lab_folds in by_contrast.values():
        for fold, vs in lab_folds.items():
            per_fold[fold].extend(vs)
    return cross_fold_stats(per_fold)


def discover(metrics_base: Path) -> dict:
    """{method: {N: load_run() data}} for every *_lowdata_n* run dir."""
    out = defaultdict(dict)
    for run_dir in sorted(metrics_base.glob("*_lowdata_n*")):
        if not run_dir.is_dir():
            continue
        m = RUN_RE.match(run_dir.name)
        if not m:
            print(f"  (skip unparseable: {run_dir.name})", file=sys.stderr)
            continue
        method, n = m.group("method"), int(m.group("n"))
        data = load_run(run_dir)
        if data:
            out[method][n] = data
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_base", required=True,
                    help="8_results_chaos/02_metrics/chaos_model/t1in")
    ap.add_argument("--in_domain", default="t1in",
                    help="in-domain contrast to score (= training contrast). The low-data "
                         "curve is in-domain only; default t1in.")
    ap.add_argument("--out_dir", default=None, help="default: <metrics_base>/05_01_low_data")
    args = ap.parse_args()

    metrics_base = Path(args.metrics_base)
    out_dir = Path(args.out_dir) if args.out_dir else metrics_base / "05_01_low_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover(metrics_base)
    if not runs:
        print(f"No *_lowdata_n* runs under {metrics_base}", file=sys.stderr)
        sys.exit(1)

    in_domain = args.in_domain
    all_mods = sorted({c for byN in runs.values() for d in byN.values() for c in d.get("dice", {})})
    if in_domain not in all_mods:
        print(f"ERROR: in-domain contrast '{in_domain}' not among evaluated modalities {all_mods}",
              file=sys.stderr)
        sys.exit(1)
    # In-domain only: every roll-up (CSV, plots, summary) is restricted to this contrast.
    modalities = [in_domain]
    methods = [m for m in METHOD_ORDER if m in runs] + [m for m in runs if m not in METHOD_ORDER]

    # ── tidy CSV ────────────────────────────────────────────────────────────
    rows = []
    for method in methods:
        for n in sorted(runs[method]):
            data = runs[method][n]
            for metric in ("dice", "hd95"):
                for mod in modalities:
                    mean, std, nf = cross_fold_class_stats(data, metric, mod)
                    if np.isfinite(mean):
                        rows.append(dict(method=method, label=METHOD_LABEL.get(method, method),
                                         N=n, modality=mod, metric=metric,
                                         mean=mean, std=std, n_folds=nf))
    csv_path = out_dir / "lowdata_curve.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "label", "N", "modality", "metric", "mean", "std", "n_folds"])
        w.writeheader(); w.writerows(rows)
    print(f"→ {csv_path}  ({len(rows)} rows)")

    # ── plots: Dice/HD95 vs N, per modality ─────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
    except Exception as e:  # pragma: no cover
        print(f"  (matplotlib unavailable, skipping plots: {e})", file=sys.stderr)
        plt = None

    if plt is not None:
        for metric, ylab in (("dice", "Dice (↑)"), ("hd95", "HD95 mm (↓)")):
            for mod in modalities:
                fig, ax = plt.subplots(figsize=(6.5, 4.5))
                for method in methods:
                    xs, ys, es = [], [], []
                    for n in sorted(runs[method]):
                        mean, std, _ = cross_fold_class_stats(runs[method][n], metric, mod)
                        if np.isfinite(mean):
                            xs.append(n); ys.append(mean * (100 if metric == "dice" else 1)); es.append(std * (100 if metric == "dice" else 1))
                    if xs:
                        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=METHOD_LABEL.get(method, method))
                ax.set_xscale("log", base=2)
                ax.set_xticks(sorted({n for m in methods for n in runs[m]}))
                ax.get_xaxis().set_major_formatter(ScalarFormatter())
                ax.set_xlabel("# training subjects (N, per fold)")
                ax.set_ylabel(ylab)
                ax.set_title(f"CHAOS low-data — {ylab} — {mod}")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                fig.tight_layout()
                p = out_dir / f"lowdata_{metric}_{mod}.png"
                fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
                print(f"→ {p}")

    # ── markdown grid: cross-modality Dice, method × N ──────────────────────
    all_N = sorted({n for m in methods for n in runs[m]})
    lines = [f"# CHAOS Low-Data Benchmark — in-domain ({in_domain}) Dice vs N", "",
             f"Cross-fold, cross-class mean Dice (×100) on the **in-domain** contrast "
             f"`{in_domain}` (= training contrast). N = train subjects/fold.", "",
             "| method | " + " | ".join(f"N={n}" for n in all_N) + " |",
             "|" + "---|" * (len(all_N) + 1)]
    for method in methods:
        cells = []
        for n in all_N:
            if n in runs[method]:
                mean = cross_fold_class_stats(runs[method][n], "dice", in_domain)[0]
                cells.append(f"{mean*100:.1f}" if np.isfinite(mean) else "—")
            else:
                cells.append("—")
        lines.append(f"| {METHOD_LABEL.get(method, method)} | " + " | ".join(cells) + " |")
    md_path = out_dir / "lowdata_summary.md"
    md_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n→ {md_path}")


if __name__ == "__main__":
    main()
