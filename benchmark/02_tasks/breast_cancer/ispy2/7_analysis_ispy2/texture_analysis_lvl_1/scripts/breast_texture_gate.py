#!/usr/bin/env python
"""
Gate + full-case-set fill-swap Δ, computed directly from the project's own official
per-fold *_metrics.csv (produced by 06_00_evaluate.py) -- NOT from re-running dice on
images. This sidesteps the bug found 2026-09-25 in breast_texture_analysis.py's first
cut (cropping pred to the GT bbox before computing dice/recall/precision silently
dropped false positives outside the crop, inflating dice and in at least one cell
flipping the sign of the fill-swap Δ vs the ladder's own eval_all.csv-based numbers).

For each (train, eval) pair, loads noise (rung 4, "+voronoi") and real (rung 5,
"v26_6_2") fold{0,1,2}/<item>_metrics.csv, joins on case, computes the flat mean Δ dice
(pts) over ALL case×fold rows on the FULL case set (no subsampling) -- this is what the
ladder engine's own per_contrast "dice" values are (mean of case dice, pooled over
folds) -- and asserts it matches the ladder_series.json per_contrast dice Δ within 0.3
pts before anything else is trusted.

Usage: .venv/bin/python breast_texture_gate.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(PROJECT_ROOT / "datasets" / "00_commun_scripts" / "00_00_utils"))
from stat_tests import wilcoxon_p  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from breast_texture_analysis import build_pairs, rung_run_id, category_for, _ladder  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"
DATA_DIR, TABLES_DIR = OUT / "data", OUT / "tables"
FOLDS = ("fold0", "fold1", "fold2")

# ladder_series.json per_contrast keys for each (train, eval_source, eval_item)
PER_CONTRAST_KEY = {
    ("t1wce", "ispy2", "t1wce"): "t1wce (in-domain)",
    ("t1wce", "ispy2", "t2w"): "t2w",
    ("t1wce", "duke-breast-mri", "precontrast_uni"): "precontrast_uni",
    ("t2w", "ispy2", "t2w"): "t2w (in-domain)",
    ("t2w", "ispy2", "t1wce"): "t1wce",
    ("t2w", "duke-breast-mri", "t1wce_uni"): "t1wce_uni",
    ("t2w", "duke-breast-mri", "precontrast_uni"): "precontrast_uni",
}
LADDER_JSON = {
    "t1wce": PROJECT_ROOT / "benchmark/02_tasks/breast_cancer/ispy2/8_results_ispy2/02_metrics/ispy2_model/t1wce/ablations/ladder_series.json",
    "t2w": PROJECT_ROOT / "benchmark/02_tasks/breast_cancer/ispy2/8_results_ispy2/02_metrics/ispy2_model/t2w/ablations/ladder_series.json",
}

# metrics roots mirror the 01_predictions roots used by breast_texture_analysis.py,
# but under 02_metrics/.../ablations/<CATEGORY>_<run_id>/
ISPY2_METRICS = PROJECT_ROOT / "benchmark/02_tasks/breast_cancer/ispy2/8_results_ispy2/02_metrics/ispy2_model"
DUKE_METRICS = PROJECT_ROOT / "benchmark/02_tasks/breast_cancer/duke-breast-mri/8_results_duke-breast-mri/02_metrics/ispy2_model"


def metrics_dirs(p):
    d = _ladder(LADDER_JSON[p["train"]] if p["eval_source"] == "ispy2"
                else (DUKE_METRICS / p["train"] / "ablations" / p["eval_item"] / "ladder_series.json"))
    noise_id = rung_run_id(d["run_keys"], "+voronoi (noise fill)", d["labels"])
    real_id = rung_run_id(d["run_keys"], "v26_6_2 (real fill)", d["labels"])
    if p["eval_source"] == "ispy2":
        base = ISPY2_METRICS / p["train"] / "ablations"
    else:
        base = DUKE_METRICS / p["train"] / "ablations" / p["eval_item"]
    return base / f"{category_for(noise_id)}_{noise_id}", base / f"{category_for(real_id)}_{real_id}"


def load_metric_csv(fold_dir: Path, item: str):
    f = fold_dir / f"{item}_metrics.csv"
    if not f.exists():
        return {}
    out = {}
    with open(f) as fh:
        for row in csv.DictReader(fh):
            if row["label"] == "tumour":
                out[row["case"]] = float(row["dice"])
    return out


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Gate: full-case-set fill-swap Δ (dice, pts) vs ladder_series.json per_contrast", "",
             "Computed from the project's own official per-fold `<item>_metrics.csv` (06_00_evaluate.py "
             "output) -- no image re-processing, no subsampling, no bug surface. PASS = |our Δ - ladder Δ| <= 0.3 pts.",
             "", "| train | eval | n case×fold | our Δ (pts) | ladder Δ (pts) | diff | gate | wilcoxon p (case-mean, all folds) |",
             "|---|---|--:|--:|--:|--:|---|---|"]
    all_pass = True
    rows_out = []
    for p in build_pairs():
        key = (p["train"], p["eval_source"], p["eval_item"])
        try:
            noise_dir, real_dir = metrics_dirs(p)
        except FileNotFoundError:
            lines.append(f"| {p['train']} | {p['eval_source']}/{p['eval_item']} | - | - | - | - | SKIP (ladder json missing) | - |")
            continue
        per_case_fold = []
        noise_by_case, real_by_case = {}, {}
        for fold in FOLDS:
            n = load_metric_csv(noise_dir / fold, p["pred_subdir"])
            r = load_metric_csv(real_dir / fold, p["pred_subdir"])
            common = set(n) & set(r)
            for c in common:
                per_case_fold.append((c, fold, n[c], r[c]))
            for c in common:
                noise_by_case.setdefault(c, []).append(n[c])
                real_by_case.setdefault(c, []).append(r[c])
        if not per_case_fold:
            lines.append(f"| {p['train']} | {p['eval_source']}/{p['eval_item']} | 0 | - | - | - | SKIP (no metrics csv found) | - |")
            continue
        diffs = [100 * (r - n) for _, _, n, r in per_case_fold]
        our_delta = sum(diffs) / len(diffs)
        cases = sorted(set(noise_by_case) & set(real_by_case))
        noise_case_mean = [sum(noise_by_case[c]) / len(noise_by_case[c]) for c in cases]
        real_case_mean = [sum(real_by_case[c]) / len(real_by_case[c]) for c in cases]
        p_val = wilcoxon_p(real_case_mean, noise_case_mean)

        ladder_delta = None
        pc_key = PER_CONTRAST_KEY.get(key)
        if pc_key is not None:
            ref_json = LADDER_JSON[p["train"]] if p["eval_source"] == "ispy2" else \
                (DUKE_METRICS / p["train"] / "ablations" / p["eval_item"] / "ladder_series.json")
            d = json.loads(ref_json.read_text())
            arr = d["per_contrast"]["dice"].get(pc_key)
            if arr is not None:
                labels = d["labels"]
                ladder_delta = arr[labels.index("v26_6_2 (real fill)")] - arr[labels.index("+voronoi (noise fill)")]
        if ladder_delta is not None:
            diff = our_delta - ladder_delta
            gate = "PASS" if abs(diff) <= 0.3 else "FAIL"
            all_pass &= (gate == "PASS")
            lines.append(f"| {p['train']} | {p['eval_source']}/{p['eval_item']} | {len(diffs)} | {our_delta:+.2f} | "
                          f"{ladder_delta:+.2f} | {diff:+.2f} | {gate} | {p_val:.3g} |")
        else:
            lines.append(f"| {p['train']} | {p['eval_source']}/{p['eval_item']} | {len(diffs)} | {our_delta:+.2f} | "
                          f"n/a | n/a | (no ladder ref) | {p_val:.3g} |")
        rows_out.append(dict(train=p["train"], eval_source=p["eval_source"], eval_item=p["eval_item"],
                              n_case_fold=len(diffs), n_cases=len(cases), delta_dice_pts=our_delta,
                              ladder_delta_pts=ladder_delta, wilcoxon_p=p_val,
                              in_domain=p["in_domain"], same_contrast_cross_dataset=p["same_contrast_cross_dataset"]))
    lines.append("")
    lines.append(f"**Overall gate: {'ALL PASS' if all_pass else 'AT LEAST ONE FAIL/SKIP -- see above'}**")
    (TABLES_DIR / "gate_full_case_delta.md").write_text("\n".join(lines))
    pd.DataFrame(rows_out).to_csv(DATA_DIR / "gate_full_case_delta.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
