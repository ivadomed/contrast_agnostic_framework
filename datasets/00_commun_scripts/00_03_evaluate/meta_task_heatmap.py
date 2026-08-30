#!/usr/bin/env python3
"""
Cross-dataset TASK-LEVEL heatmap — one overall conclusion across all 4 datasets.

One level up from combined_modality_summary.py: that script pools a dataset's TWO
training modalities into one row per method (one column per tested contrast, plus an
`all` column). This script pools ALL 4 DATASETS' combined_modality tables into one
row per method, one COLUMN PER DATASET ("task": brats2024-glioma, chaos, on-harmony,
open-ms — each task's cell = that dataset's own `all` value, i.e. already averaged
over its tested contrasts AND its two training modalities), plus an `overall` column
(equal weight per task) and an inline `sig. vs ref` column — the single "which method
wins overall" table.

Each task's config is one of the existing `*_combined_01_results.yaml` files (chaos/
on-harmony/brats2024-glioma/open-ms) — reused as-is, not re-specified, so this script
can never drift from the per-dataset combined tables it's built on top of.

Statistics: for the significance column, each TASK contributes ONE stratum (unlike
significance_from_config.py / combined_modality_summary.py, which stratify by
contrast) — its stratum array is every one of that task's held-out cases' paired
diffs (ref − competitor), pooled across the task's tested contrasts AND its two
training modalities. Feeding these 4 (one per dataset) arrays into the same macroΔ
sign-flip test (stat_tests.macro_perm) as elsewhere gives an "ours better" p-value
with EQUAL WEIGHT PER TASK — matching the `overall` column's equal-weight-per-task
average, so the test and the table finally agree at this level too. Holm-corrected
across all non-ref competitors.

Meta-config YAML:
  title: "Cross-dataset — task-level heatmap (7-method suite)"
  output_dir: "${PROJECT_ROOT}/datasets/01_commun_results"
  output_prefix: "meta_task_heatmap"
  ref: auglabAug_v26_6_2_train050_val000
  tasks:
    - name: brats2024-glioma
      dataset: brats2024-glioma
      config: "${PROJECT_ROOT}/datasets/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/configs/brats_combined_01_results.yaml"
    - name: chaos
      dataset: chaos
      config: "${PROJECT_ROOT}/datasets/chaos/5_scripts_chaos/06_evaluate/configs/chaos_combined_01_results.yaml"
    - ...

Usage:
  python meta_task_heatmap.py <meta_config.yaml>
Writes: {output_dir}/{output_prefix}_summary.md, _heatmap_dice.png, _heatmap_hd95.png
"""
import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "00_00_utils"))
from stat_tests import holm, macro_perm  # noqa: E402

# Reuse the exact per-modality pooling machinery — one implementation, no drift
# from the per-dataset combined_contrasts tables this rolls up.
def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_agg = _load_sibling("aggregate_from_config")
_cms = _load_sibling("combined_modality_summary")


def load_task_modalities(task: dict, project_root: str) -> tuple:
    """Load one task's own *_combined_01_results.yaml `modalities:` list, with
    METRICS_ROOT set to that task's dataset so its ${METRICS_ROOT}-relative
    metrics_dir paths expand correctly (same pattern as meta_significance.py).
    Returns (modalities, contrast_groups) — contrast_groups is that task's own
    opt-in hierarchical-pooling tree (None if not configured / disabled), read
    straight from the same combined config so this task-level rollup can never
    silently disagree with what that dataset's own combined table shows."""
    os.environ["METRICS_ROOT"] = (
        f"{project_root}/datasets/{task['dataset']}/8_results_{task['dataset']}/02_metrics")
    cfg_path = Path(os.path.expandvars(task["config"]))
    if not cfg_path.exists():
        print(f"  skip task {task['name']}: config not found {cfg_path}", file=sys.stderr)
        return [], None
    cfg = yaml.safe_load(cfg_path.read_text())
    modalities = []
    for m in cfg["modalities"]:
        if "sources" in m:
            sources = [
                {**src, "metrics_dir": Path(os.path.expandvars(src["metrics_dir"]))}
                for src in m["sources"]
            ]
        else:
            sources = [{"metrics_dir": Path(os.path.expandvars(m["metrics_dir"]))}]
        modalities.append({
            "name": m["name"],
            "sources": sources,
            "runs": m.get("runs", {}),
        })
    contrast_groups = cfg.get("contrast_groups") if cfg.get("use_contrast_groups", True) else None
    return modalities, contrast_groups


def task_method_value(modalities: list, method_key: str, metric: str,
                      contrast_groups: dict = None) -> float:
    """This task's `all` cell for one method: cross-fold-class mean per tested
    contrast (pooled over both training modalities), then averaged over
    contrasts, equal weight per contrast — IDENTICAL estimand to
    combined_modality_summary.py's `all` column (same contrast_groups tree,
    when that task's own combined config declares one, so this rollup's
    displayed number always matches the per-dataset table it's built on)."""
    data, _ = _cms.load_combined_run(modalities, method_key)
    if contrast_groups:
        grp_vals = [_agg.resolve_group_value(node, data, metric) for node in contrast_groups.values()]
        finite = [v for v in grp_vals if np.isfinite(v)]
        return float(np.mean(finite)) if finite else float("nan")
    contrasts = sorted(data.get(metric, {}))
    vals = [_agg.cross_fold_class_mean(data, metric, c) for c in contrasts]
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def task_pooled_diffs(modalities: list, ref_key: str, comp_key: str, metric: str,
                      contrast_groups: dict = None) -> np.ndarray:
    """One task's stratum array for the top-level sign-flip test: every held-out
    case's paired diff (ref − comp), pooled across the task's tested contrasts
    AND its two training modalities (case tags already carry the modality, see
    combined_modality_summary.load_combined_cases).

    contrast_groups (optional): resolve each top-level group via
    aggregate_from_config.resolve_group (fed by the label-preserving loader so
    a group like chaos's "CT" gets its per-organ-then-cross-organ treatment),
    then concatenate every group's array into one task stratum — preserves
    "one stratum per TASK" (this function's whole point) while fixing the same
    vote-imbalance the per-dataset headline tests already got fixed for."""
    if contrast_groups:
        ref_by_label = _cms.load_combined_cases_by_label(modalities, ref_key, metric)
        comp_by_label = _cms.load_combined_cases_by_label(modalities, comp_key, metric)
        pooled = [_agg.resolve_group(g, ref_by_label, comp_by_label) for g in contrast_groups.values()]
        pooled = [p for p in pooled if len(p)]
        return np.concatenate(pooled) if pooled else np.array([])
    ref_cases = _cms.load_combined_cases(modalities, ref_key, metric)
    comp_cases = _cms.load_combined_cases(modalities, comp_key, metric)
    cols = sorted(set(ref_cases) | set(comp_cases))
    pooled = []
    for col in cols:
        x, y = _agg.paired(ref_cases, comp_cases, col)
        if len(x):
            pooled.append(x - y)
    return np.concatenate(pooled) if pooled else np.array([])


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("meta_config")
    args = ap.parse_args()

    project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
    cfg_path = Path(args.meta_config)
    if not cfg_path.exists():
        sys.exit(f"Config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())

    title = cfg.get("title", "Cross-dataset task-level heatmap")
    output_prefix = cfg.get("output_prefix", "meta_task_heatmap")
    out_dir = Path(os.path.expandvars(cfg["output_dir"]))
    ref_key = cfg["ref"]

    tasks = []
    for t in cfg["tasks"]:
        modalities, contrast_groups = load_task_modalities(t, project_root)
        if modalities:
            tasks.append({"name": t["name"], "modalities": modalities,
                          "contrast_groups": contrast_groups})
    if len(tasks) < 2:
        sys.exit("meta_task_heatmap.py expects >= 2 tasks with resolvable configs.")

    method_order = cfg.get("method_order") or list(tasks[0]["modalities"][0]["runs"].keys())
    if ref_key not in method_order:
        sys.exit(f"`ref: {ref_key}` is not among the method keys: {method_order}")

    task_names = [t["name"] for t in tasks]
    cols = task_names + ["overall"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# {title}", "",
        f"Generated: {now}  |  Methods: {len(method_order)}  |  Tasks: {', '.join(task_names)}",
        "",
        "Each task column = that dataset's own `all` value (already averaged over its "
        "tested contrasts AND its two training modalities — see its "
        "`combined_contrasts/01_results_summary.md`). `overall` = equal-weight average "
        f"across the {len(tasks)} tasks that have data for that method (a task missing "
        "for a given method — e.g. no HD95, or a method not run on that dataset — is "
        "excluded, not counted as 0). **Bold** = best per column. `sig. vs ref` = Holm-corrected "
        f"one-sided (ref better) macroΔ p-value of `{_agg._display_key(ref_key)}` vs that "
        "row, equal weight per TASK (blank on the ref's own row); **bold** = p < 0.05.", "",
    ]

    matrices = {}
    sig_by_metric = {}
    for metric, heading, prec in (("dice", "Dice ↑", 4), ("hd95", "HD95 mm ↓", 2)):
        mat = np.full((len(method_order), len(cols)), np.nan)
        for i, key in enumerate(method_order):
            per_task = [task_method_value(t["modalities"], key, metric,
                                          contrast_groups=t["contrast_groups"]) for t in tasks]
            for j, v in enumerate(per_task):
                mat[i, j] = v
            finite = [v for v in per_task if np.isfinite(v)]
            mat[i, -1] = float(np.mean(finite)) if finite else np.nan
        matrices[metric] = mat

        higher_better = metric == "dice"
        competitors = [k for k in method_order if k != ref_key]
        p1s = []
        for comp in competitors:
            arrs = [task_pooled_diffs(t["modalities"], ref_key, comp, metric,
                                      contrast_groups=t["contrast_groups"]) for t in tasks]
            arrs = [a for a in arrs if a.size]
            _, _, p1 = macro_perm(arrs, higher_better)
            p1s.append(p1)
        hp = holm(p1s)
        sig_by_metric[metric] = {ref_key: float("nan"), **dict(zip(competitors, hp))}

        best = _agg._best_per_col(mat, metric)
        sig = sig_by_metric[metric]
        lines += [
            f"## {heading}", "",
            "| method | " + " | ".join(cols) + " | sig. vs ref |",
            "|" + "---|" * (len(cols) + 2),
        ]
        for i, key in enumerate(method_order):
            cells = []
            for j in range(len(cols)):
                s = _agg._fmt_val(mat[i, j], prec)
                if s != "—" and best[j] == i:
                    s = f"**{s}**"
                cells.append(s)
            sig_cell = "—" if key == ref_key else _agg._fmt_sig(sig.get(key, float("nan")))
            lines.append(f"| {_agg._format_label_md(_agg._display_key(key))} | "
                         + " | ".join(cells) + f" | {sig_cell} |")
        lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{output_prefix}_summary.md"
    md_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n→ {md_path}")

    _save_heatmaps(matrices, method_order, sig_by_metric, ref_key, cols, out_dir,
                   output_prefix, title)


def _save_heatmaps(matrices, method_order, sig_by_metric, ref_key, cols, out_dir: Path,
                   prefix: str, title: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (matplotlib unavailable, skipping heatmaps: {e})", file=sys.stderr)
        return

    ylabels = [_agg._format_label_mathtext(_agg._display_key(k)) for k in method_order]

    specs = (("dice", "Dice ↑", 4, "RdBu"), ("hd95", "HD95 mm ↓", 2, "RdBu_r"))
    for metric, heading, prec, cmap in specs:
        mat = matrices.get(metric)
        if mat is None:
            continue
        sig = sig_by_metric[metric]
        best = _agg._best_per_col(mat, metric)

        n_rows, n_data_cols = mat.shape
        n_cols = n_data_cols + 1
        plot_cols = cols + ["sig. vs ref"]
        full = np.full((n_rows, n_cols), np.nan)
        full[:, :n_data_cols] = mat

        fig_w = max(8, 1.4 * n_cols + 5)
        fig_h = max(3, 0.55 * n_rows + 2)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        masked = np.ma.masked_invalid(full)
        im = ax.imshow(masked, aspect="auto", cmap=cmap)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(plot_cols, rotation=30, ha="right")
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_title(f"{title} — {heading}", pad=10)

        for i in range(n_rows):
            for j in range(n_data_cols):
                v = mat[i, j]
                if np.isfinite(v):
                    txt = f"{v * 100:.1f}" if prec == 4 else f"{v:.1f}"
                    r, g, b, _ = im.cmap(im.norm(v))
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    fw = "bold" if best[j] == i else "normal"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                           color="white" if lum < 0.5 else "black", fontweight=fw)

        for i, key in enumerate(method_order):
            if key == ref_key:
                continue
            p = sig.get(key, float("nan"))
            if not np.isfinite(p):
                continue
            txt = f"{p:.1e}" if p < 1e-4 else f"{p:.3f}"
            starred = p < 0.05
            ax.text(n_data_cols, i, txt, ha="center", va="center", fontsize=9,
                   color="crimson" if starred else "gray",
                   fontweight="bold" if starred else "normal")
        ax.axvline(n_data_cols - 0.5, color="black", linewidth=0.8)
        ax.axvline(n_data_cols - 1.5, color="black", linewidth=0.6, linestyle="--")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("task (dataset)  |  significance vs ref")
        ax.set_ylabel("method")
        fig.tight_layout()

        p_out = out_dir / f"{prefix}_heatmap_{metric}.png"
        fig.savefig(p_out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"→ {p_out}")


if __name__ == "__main__":
    main()
