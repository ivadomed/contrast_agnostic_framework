#!/usr/bin/env python3
"""
Config-driven cross-TRAINING-MODALITY aggregation + inline significance.

A dataset trains each of the fixed 6/7-method suite TWICE — once per training
modality (e.g. chaos: t1in and t2spir; brats2024-glioma: t1n and t2w). This
script answers "how does method X do on tested contrast Y, averaged across
BOTH training-modality models" — i.e. it pools the two per-modality models of
the same method into one row, the same way aggregate_from_config.py pools
folds into one cross-fold mean, just one level up.

For each method it:
  * pools per-fold eval_all.csv rows from BOTH modalities' run dirs (fold keys
    tagged by modality so a 3-fold + 3-fold pool behaves as 6 equally-weighted
    fold-means — reuses cross_fold_class_mean unchanged);
  * pools per-case scores the same way (case keys tagged by modality) to run
    the SAME macroΔ sign-flip significance test as significance_from_config.py
    (imported from stat_tests.py — one implementation, two callers), comparing
    every method against the configured `ref` (normally
    auglabAug_v26_6_2_train050_val000).

Writes {output_dir}/{output_prefix}_summary.md (+ heatmap_dice.png/heatmap_hd95.png)
with, after the usual "all" column, one extra "sig. vs ref" column: the
Holm-corrected one-sided ("ref better") macroΔ p-value of `ref` vs that row's
method. The `ref` row itself is blank (no self-comparison).

Config YAML format:
  title: "CHAOS — combined training modalities (T1in + T2spir)"
  output_dir: "${METRICS_ROOT}/chaos_model/combined_contrasts"
  output_prefix: "05_01_combined"
  ref: auglabAug_v26_6_2_train050_val000        # exact method key, see `modalities[*].runs`
  column_order: [t1in, t2spir, ct, t1out]        # optional, tested-contrast display order

  modalities:
    - name: t1in
      metrics_dir: "${METRICS_ROOT}/chaos_model/t1in"
      runs:
        baseline: chaos_t1in_baseline_20260614_153230
        synthseg_noEM: chaos_t1in_synthseg_noEM_train100_val000_20260611_120000
        ...
        auglabAug_v26_6_2_train050_val000: chaos_t1in_auglabAug_v26_6_2_train050_val000_20260615_213615
        auglabAug_v26_6_2_train050_val100: chaos_t1in_auglabAug_v26_6_2_train050_val100_20260616_112420
    - name: t2spir
      metrics_dir: "${METRICS_ROOT}/chaos_model/t2spir"
      runs: {...same method keys...}

Method row order = the key order of the FIRST modality's `runs` mapping
(YAML preserves insertion order), unless `method_order` is given explicitly.
A method missing from a modality's `runs` (or with no eval data yet) still
contributes whatever data the OTHER modality has — never dropped from the row
list, consistent with aggregate_from_config.py's "blank row, not a missing
row" policy.

Usage:
  python combined_modality_summary.py <config.yaml>
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
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "00_00_utils"))  # sibling: 00_commun_scripts/00_00_utils
from eval_folds import EVAL_FOLD_INDICES, filter_fold_dirs  # noqa: E402
from stat_tests import holm, macro_perm  # noqa: E402 — shared math, see stat_tests.py

# Reuse run-dir resolution + display/formatting helpers from the per-modality
# aggregator so a method label renders identically in both table families.
_spec = importlib.util.spec_from_file_location(
    "aggregate_from_config", _HERE / "aggregate_from_config.py")
_agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_agg)


def load_combined_run(modalities: list, method_key: str) -> tuple:
    """Pool per-fold eval data for one method across ALL modalities' run dirs.

    Fold keys are tagged "<modality>_<fold>" so cross_fold_class_mean's
    per-fold averaging treats the pooled folds as equally-weighted samples
    (3 folds/modality x 2 modalities = 6 fold-means, not a modality-weighted
    average biased by unequal fold counts). Returns (data, fold_counts) where
    fold_counts maps modality name -> n evaluated folds found for this method.
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    fold_counts = {}
    for mod in modalities:
        run_id = mod["runs"].get(method_key)
        fold_counts[mod["name"]] = 0
        if not run_id:
            continue
        run_dir = _agg.resolve_run_dir(mod["metrics_dir"], run_id)
        if run_dir is None:
            continue
        n = 0
        for fold_dir in filter_fold_dirs(sorted(run_dir.glob("fold*"))):
            csv_path = fold_dir / "eval_all.csv"
            if not csv_path.exists():
                continue
            n += 1
            tag = f'{mod["name"]}_{fold_dir.name}'
            with csv_path.open() as f:
                for row in csv.DictReader(f):
                    contrast, label = row["group"], row["label"]
                    for k in ("dice", "hd95"):
                        data[k][contrast][label][tag].append(float(row[k]))
        fold_counts[mod["name"]] = n
    return data, fold_counts


def load_combined_cases(modalities: list, method_key: str, metric: str) -> dict:
    """col -> {case_tag: mean metric over labels, folds in EVAL_FOLD_INDICES}.

    case_tag = "<modality>::<case>" — this keeps the ref/competitor pairing
    correct (a case is only ever compared against the SAME modality's model of
    another method) while still pooling both modalities into one bigger,
    equally-weighted sample per tested contrast for the sign-flip test.
    """
    per = defaultdict(lambda: defaultdict(list))
    for mod in modalities:
        run_id = mod["runs"].get(method_key)
        if not run_id:
            continue
        run_dir = _agg.resolve_run_dir(mod["metrics_dir"], run_id)
        if run_dir is None:
            continue
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
                    case_tag = f'{mod["name"]}::{row["case"]}'
                    try:
                        v = float(row[metric])
                    except (KeyError, ValueError):
                        continue
                    if np.isfinite(v):
                        per[row["group"]][case_tag].append(v)
    return {col: {c: float(np.mean(vs)) for c, vs in cases.items() if vs}
            for col, cases in per.items()}


def paired(ref_cases: dict, comp_cases: dict, col: str) -> tuple:
    r, c = ref_cases.get(col, {}), comp_cases.get(col, {})
    common = sorted(set(r) & set(c))
    return np.array([r[k] for k in common]), np.array([c[k] for k in common])


def significance_column(runs_ordered: list, ref_key: str, all_contrasts: list, metric: str,
                        modalities: list) -> dict:
    """method_key -> Holm-corrected one-sided ("ref better") macroΔ p-value vs ref.

    Same estimand/test as significance_from_config.py's headline block
    (contrast-stratified paired sign-flip on macroΔ), just fed pooled-modality
    per-case data instead of single-modality data. Holm correction is applied
    across all non-ref competitors at once (one family of tests per metric).
    """
    higher_better = metric == "dice"
    ref_cases = load_combined_cases(modalities, ref_key, metric)
    competitors = [k for k in runs_ordered if k != ref_key]
    p1s = []
    for key in competitors:
        comp_cases = load_combined_cases(modalities, key, metric)
        arrs = []
        for col in all_contrasts:
            x, y = paired(ref_cases, comp_cases, col)
            if len(x):
                arrs.append(x - y)
        _, _, p1 = macro_perm(arrs, higher_better)
        p1s.append(p1)
    hp = holm(p1s)
    out = {ref_key: float("nan")}
    out.update(dict(zip(competitors, hp)))
    return out


def build_combined_summary(runs_ordered, runs_data, fold_counts, sig_by_metric, title,
                            out_dir: Path, prefix: str, ref_key: str, modalities: list,
                            column_order: list = None):
    all_contrasts_set = {c for d in runs_data.values() for c in d.get("dice", {})}
    if column_order:
        ordered = [c for c in column_order if c in all_contrasts_set]
        remaining = sorted(c for c in all_contrasts_set if c not in ordered)
        all_contrasts = ordered + remaining
    else:
        all_contrasts = sorted(all_contrasts_set)
    if not all_contrasts:
        print("No eval data — nothing to aggregate.", file=sys.stderr)
        return

    cols = all_contrasts + ["all"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# {title}", "",
        f"Generated: {now}  |  Methods: {len(runs_ordered)}  |  "
        f"Modalities: {', '.join(all_contrasts)}", "",
        "Each cell is the **cross-fold, cross-class, cross-training-modality average** "
        "(mean over all labels, folds, AND the two training-modality models of the same "
        "method). `all` = average across tested contrasts. **Bold** = best per column. "
        f"`sig. vs ref` = Holm-corrected one-sided (ref better) macroΔ p-value of "
        f"`{_agg._display_key(ref_key)}` vs that row (blank on the ref's own row); "
        "**bold** = p < 0.05. — = no data.", "",
    ]

    matrices = {}
    for metric, heading, prec in (("dice", "Dice ↑", 4), ("hd95", "HD95 mm ↓", 2)):
        mat = np.full((len(runs_ordered), len(cols)), np.nan)
        for i, key in enumerate(runs_ordered):
            per_mod = [_agg.cross_fold_class_mean(runs_data[key], metric, c) for c in all_contrasts]
            for j, v in enumerate(per_mod):
                mat[i, j] = v
            finite = [v for v in per_mod if np.isfinite(v)]
            mat[i, -1] = float(np.mean(finite)) if finite else np.nan
        matrices[metric] = mat

        best = _agg._best_per_col(mat, metric)
        sig = sig_by_metric[metric]

        lines += [
            f"## {heading}", "",
            "| method | folds | " + " | ".join(cols) + " | sig. vs ref |",
            "|" + "---|" * (len(cols) + 3),
        ]
        for i, key in enumerate(runs_ordered):
            nf = "+".join(str(fold_counts[key].get(m["name"], 0)) for m in modalities)
            cells = []
            for j in range(len(cols)):
                s = _agg._fmt_val(mat[i, j], prec)
                if s != "—" and best[j] == i:
                    s = f"**{s}**"
                cells.append(s)
            sig_cell = "—" if key == ref_key else _agg._fmt_sig(sig.get(key, float("nan")))
            lines.append(f"| {_agg._format_label_md(_agg._display_key(key))} | {nf} | "
                         + " | ".join(cells) + f" | {sig_cell} |")
        lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{prefix}_summary.md"
    md_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n→ {md_path}")

    _save_heatmaps_with_sig(matrices, runs_ordered, fold_counts, sig_by_metric, ref_key,
                            cols, out_dir, prefix, title, modalities)


def _save_heatmaps_with_sig(matrices, runs_ordered, fold_counts, sig_by_metric, ref_key,
                            cols, out_dir: Path, prefix: str, title: str, modalities: list):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (matplotlib unavailable, skipping heatmaps: {e})", file=sys.stderr)
        return

    ylabels = [
        _agg._format_label_mathtext(
            f'{_agg._display_key(k)}  '
            f'({"+".join(str(fold_counts[k].get(m["name"], 0)) for m in modalities)}f)')
        for k in runs_ordered
    ]
    plot_cols = cols + ["sig. vs ref"]

    specs = (("dice", "Dice ↑", 4, "RdBu"), ("hd95", "HD95 mm ↓", 2, "RdBu_r"))
    for metric, heading, prec, cmap in specs:
        mat = matrices.get(metric)
        if mat is None:
            continue
        sig = sig_by_metric[metric]
        best = _agg._best_per_col(mat, metric)

        n_runs, n_data_cols = mat.shape
        n_cols = n_data_cols + 1  # + sig column
        full = np.full((n_runs, n_cols), np.nan)
        full[:, :n_data_cols] = mat

        fig_w = max(8, 1.4 * n_cols + 5)
        fig_h = max(3, 0.55 * n_runs + 2)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        masked = np.ma.masked_invalid(full)
        im = ax.imshow(masked, aspect="auto", cmap=cmap)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(plot_cols, rotation=30, ha="right")
        ax.set_yticks(range(n_runs))
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_title(f"{title} — {heading}", pad=10)

        for i in range(n_runs):
            for j in range(n_data_cols):
                v = mat[i, j]
                if np.isfinite(v):
                    txt = f"{v * 100:.1f}" if prec == 4 else f"{v:.1f}"
                    r, g, b, _ = im.cmap(im.norm(v))
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    fw = "bold" if best[j] == i else "normal"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                           color="white" if lum < 0.5 else "black", fontweight=fw)

        # Significance column: no color scale (masked NaN cell), just text —
        # bold red if p < 0.05, gray otherwise; blank on the ref's own row.
        key_for_row = runs_ordered
        for i, key in enumerate(key_for_row):
            if key == ref_key:
                continue
            p = sig.get(key, float("nan"))
            if not np.isfinite(p):
                continue
            txt = f"{p:.1e}" if p < 1e-4 else f"{p:.3f}"
            sig_star = p < 0.05
            ax.text(n_data_cols, i, txt, ha="center", va="center", fontsize=8,
                   color="crimson" if sig_star else "gray",
                   fontweight="bold" if sig_star else "normal")
        ax.axvline(n_data_cols - 0.5, color="black", linewidth=0.8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("modality  |  significance vs ref")
        ax.set_ylabel("method")
        fig.tight_layout()

        p_out = out_dir / f"{prefix}_heatmap_{metric}.png"
        fig.savefig(p_out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"→ {p_out}")


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate eval results pooled across a dataset's two training "
                    "modalities, with an inline significance column vs a reference method.")
    ap.add_argument("config", help="Path to YAML config file")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")
    cfg = yaml.safe_load(config_path.open())

    title = cfg.get("title", "Combined-Modality Results")
    output_prefix = cfg.get("output_prefix", "combined")
    out_dir = Path(os.path.expandvars(cfg["output_dir"]))
    column_order = cfg.get("column_order", None)
    ref_key = cfg["ref"]

    modalities = []
    for m in cfg["modalities"]:
        modalities.append({
            "name": m["name"],
            "metrics_dir": Path(os.path.expandvars(m["metrics_dir"])),
            "runs": m.get("runs", {}),
        })
        if not modalities[-1]["metrics_dir"].is_dir():
            print(f"  note: source metrics_dir does not exist yet: {modalities[-1]['metrics_dir']}",
                 file=sys.stderr)
    if len(modalities) < 2:
        sys.exit("combined_modality_summary.py expects >= 2 entries under `modalities:`.")

    method_order = cfg.get("method_order") or list(modalities[0]["runs"].keys())
    if ref_key not in method_order:
        sys.exit(f"`ref: {ref_key}` is not among the method keys: {method_order}")

    runs_data, fold_counts, runs_ordered = {}, {}, []
    any_data = False
    for key in method_order:
        data, fc = load_combined_run(modalities, key)
        if not data:
            print(f"  blank row {key}: no eval_all.csv found in any modality", file=sys.stderr)
        else:
            any_data = True
        runs_data[key] = data
        fold_counts[key] = fc
        runs_ordered.append(key)

    if not any_data:
        print("No methods with evaluation data found.", file=sys.stderr)
        sys.exit(1)

    all_contrasts = sorted({c for d in runs_data.values() for c in d.get("dice", {})})
    sig_by_metric = {
        metric: significance_column(runs_ordered, ref_key, all_contrasts, metric, modalities)
        for metric in ("dice", "hd95")
    }

    build_combined_summary(runs_ordered, runs_data, fold_counts, sig_by_metric, title,
                           out_dir, output_prefix, ref_key, modalities, column_order=column_order)


if __name__ == "__main__":
    main()
