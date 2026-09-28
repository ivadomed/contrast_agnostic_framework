#!/usr/bin/env python3
"""
Config-driven cross-experiment aggregation of segmentation evaluation results.

Reads a YAML config listing short run IDs, resolves each to its directory under
metrics_dir (trying nnUNet_ and auglab_ prefixes automatically), computes
cross-fold mean±std Dice and HD95, and writes:
  {metrics_dir}/{output_prefix}_summary.md
  {metrics_dir}/{output_prefix}_heatmap_dice.png
  {metrics_dir}/{output_prefix}_heatmap_hd95.png

Config YAML format:
  title: "CHAOS T1in — results"
  metrics_dir: "${METRICS_ROOT}/chaos_model/t1in"   # env vars expanded
  output_prefix: "01_results"   # canonical per-contrast headline prefix (settled 2026-09-27) --
                                 # this used to show "03_03_results" here as the example, which
                                 # is exactly why chaos/brats2024-glioma ended up on that
                                 # nonstandard prefix (copied verbatim from this docstring) while
                                 # open-ms/ispy2/totalseg-pelvic/duke-breast-mri/healthy-spine-tum
                                 # independently converged on "01_results" -- use that.
  runs:
    - chaos_t1in_baseline_20260614_153230
    - chaos_t1in_v26_6_2_train050_val100_20260615_213615
    - ...

Optional config fields:
  in_domain_contrast: t1in   # highlights that column in red on the heatmap

Run key resolution order:
  1. nnUNet_{key}
  2. auglab_{key}
  3. {key}  (exact)

Run names containing "v26_6_2" are labelled "(Ours)" in all outputs.

Usage:
  python aggregate_from_config.py <config.yaml>
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import re

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_00_utils"))  # sibling: 00_commun_scripts/00_00_utils
from eval_folds import EVAL_FOLD_INDICES, filter_fold_dirs  # noqa: E402 — single source of truth, see eval_folds.py
from stat_tests import (holm, macro_perm, macro_perm_design, macro_ci_design,  # noqa: E402 — shared math, see stat_tests.py
                        stratum_arrays)
# Full significance reporting (OOD/IND/per-contrast breakdowns) lives in the
# dedicated companion script benchmark/00_commun_scripts/00_03_evaluate/
# significance_from_config.py (same config, run separately). This module only
# adds ONE inline "sig. vs ref" column to the summary table/heatmap (see
# `significance_column` below) — auto-wired whenever the run list contains the
# headline OURS method (a run id containing REF_SUBSTR); significance_from_config.py
# reuses load_run_cases/paired from here rather than redefining them.

_PREFIXES = ("nnUNet_", "auglab_", "")

# The headline "OURS" method — any run id containing this substring is picked as
# the reference for the auto-wired inline significance column. Deliberately more
# specific than just "v26_6_2" (which also matches the older non-auglab "v26_6_2
# alone" method and its HP-tuning variants) and than "auglabAug_v26_6_2" alone
# (which also matches train025/train090 HP variants) — this exact substring
# matches ONLY the headline train050_val000 run in every config checked
# (2026-08-01 audit: exactly 0 or 1 occurrence in every existing config).
REF_SUBSTR = "auglabAug_v26_6_2_train050_val000"

# Default sign-flip unit for the significance tests (2026-09-21). "patient": all of one
# physical patient's diffs — every organ, every trained model, and every test contrast —
# flip together (CHAOS's 4 MR test patients appear in t1in AND t1out AND t2spir).
# "stratum": as "patient" within one contrast group, but the same patient in two
# different groups stays two units (the weaker fix: models/organs merged, contrasts not).
# Override per config with `unit_scope:`.
DEFAULT_UNIT_SCOPE = "patient"

# ── multi-source helpers ─────────────────────────────────────────────────────

def load_run_from_sources(sources: list, key: str) -> tuple:
    """Load eval_all.csv data across multiple metrics dirs with column prefixing.

    Each source dict has:
      metrics_dir    Path
      column_prefix  str   (prepended to group name, e.g. "chaos_")
      column_rename  dict  (group → full col name, overrides prefix, e.g. {"ct": "chaos_ct_liver"})

    Returns (data_dict, max_fold_count).  data_dict has the same shape as
    load_run() so all downstream code (cross_fold_class_mean, build_summary…) works unchanged.
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    max_folds = 0
    for src in sources:
        metrics_dir = src["metrics_dir"]
        prefix = src.get("column_prefix", "")
        rename = src.get("column_rename", {})
        run_dir = resolve_run_dir(metrics_dir, key)
        if run_dir is None:
            continue
        n_folds = 0
        for fold_dir in filter_fold_dirs(sorted(run_dir.glob("fold*"))):
            csv_path = fold_dir / "eval_all.csv"
            if not csv_path.exists():
                continue
            n_folds += 1
            with csv_path.open() as f:
                for row in csv.DictReader(f):
                    group = row["group"]
                    col = rename.get(group, f"{prefix}{group}")
                    label = row["label"]
                    fold = fold_dir.name
                    for k in ("dice", "hd95"):
                        data[k][col][label][fold].append(float(row[k]))
        max_folds = max(max_folds, n_folds)
    return data, max_folds

# Substrings to highlight in bold within run-name labels.
# Listed longest-first so overlapping patterns don't shadow each other.
_BOLD_RE = re.compile(
    r'synthseg_noEM|synthseg_EM|auglab_default|auglabAug|v26_6_2|\(Ours\)|baseline|srcsm'
)


def _segment_label(s: str) -> list:
    """Split label string into [(text, is_bold), …] segments."""
    segs, last = [], 0
    for m in _BOLD_RE.finditer(s):
        if m.start() > last:
            segs.append((s[last:m.start()], False))
        segs.append((m.group(), True))
        last = m.end()
    if last < len(s):
        segs.append((s[last:], False))
    return segs


def _format_label_md(s: str) -> str:
    return "".join(f"**{t}**" if bold else t for t, bold in _segment_label(s))


def _format_label_mathtext(s: str) -> str:
    parts = []
    for text, bold in _segment_label(s):
        if bold:
            parts.append(r"$\mathbf{" + text.replace("_", r"\_") + r"}$")
        else:
            parts.append(text)
    return "".join(parts)


def _fmt_val(v: float, prec: int) -> str:
    if not np.isfinite(v):
        return "—"
    return f"{v * 100:.1f}" if prec == 4 else f"{v:.1f}"


def resolve_run_dir(metrics_dir: Path, key: str) -> Path | None:
    for prefix in _PREFIXES:
        p = metrics_dir / f"{prefix}{key}"
        if p.is_dir():
            return p
    return None


def load_run(run_dir: Path) -> dict:
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for fold_dir in filter_fold_dirs(sorted(run_dir.glob("fold*"))):
        csv_path = fold_dir / "eval_all.csv"
        if not csv_path.exists():
            continue
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                contrast = row["group"]
                label = row["label"]
                fold = fold_dir.name
                for k in ("dice", "hd95"):
                    data[k][contrast][label][fold].append(float(row[k]))
    return data


def count_eval_folds(run_dir: Path) -> int:
    return sum(1 for fd in filter_fold_dirs(sorted(run_dir.glob("fold*")))
               if (fd / "eval_all.csv").exists())


def load_run_cases(sources: list, key: str, metric: str) -> dict:
    """col -> {case_id: mean metric over that case's labels and folds in
    EVAL_FOLD_INDICES}. Single source of truth for the paired-case unit of
    analysis (see significance_from_config.py's module docstring for why the
    CASE, fold-0-2-capped, is the correct unit) — shared by the inline
    significance column here and by significance_from_config.py's full report.
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


def _dataset_ns(metrics_dir, idx: int) -> str:
    """Patient-identity namespace of a metrics source: the dataset it lives under
    (`.../benchmark/<name>/...`), so two sources of the SAME dataset (e.g. a t1in- and a
    t2spir-trained model's metrics dirs) share one namespace — their case "MR05" is the
    same physical patient — while two different datasets never collide on a bare id like
    "01". A source outside `benchmark/` falls back to its position in the source list
    (the same position across training modalities, distinct across sources)."""
    m = re.search(r"benchmark/([^/]+)/", str(metrics_dir).replace("\\", "/"))
    return m.group(1) if m else f"src{idx}"


def load_run_cases_raw(sources: list, key: str, metric: str) -> dict:
    """col -> case_key -> label -> fold_name -> value. case_key = "<dataset>|<case_id>"
    (see _dataset_ns). Fold-level granularity is kept (nothing averaged yet) — the one
    place values survive long enough to flat-pool across labels EXACTLY the way
    cross_fold_class_mean/resolve_group_value do for the displayed table cell: mean of
    fold-means, each fold-mean itself pooling every applicable label's value in that
    fold. A structure that averages folds per (label, case) first (as this used to)
    can't reproduce that when a case has an unequal number of valid folds per label
    (e.g. a tumor sub-region absent — Dice undefined/dropped — in one fold's prediction
    but present in another's) — see `_case_flat_value` for where this is consumed, and
    `project_significance_labelflat_fix_20260921` for the regression this closed."""
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for si, src in enumerate(sources):
        run_dir = resolve_run_dir(src["metrics_dir"], key)
        if run_dir is None:
            continue
        prefix = src.get("column_prefix", "")
        rename = src.get("column_rename", {})
        ns = _dataset_ns(src["metrics_dir"], si)
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
                        per[col][f"{ns}|{row['case']}"][row["label"]][fold_dir.name] = v
    return {col: {c: {lab: dict(folds) for lab, folds in labs.items()} for c, labs in cases.items()}
            for col, cases in per.items()}


def _case_flat_value(case_data: dict, label: str = None) -> float:
    """One case's cross_fold_class_mean-style descriptive value, from load_run_cases_raw's
    per-(label, fold) data: mean of fold-means, each fold-mean pooling every applicable
    label's value in that fold (label=None — the "all labels" convention every plain-
    string contrast_groups leaf and every flat/ungrouped column uses) or just the one
    named label's value (an explicit `{column, label}` leaf — unaffected by this, one
    value per fold either way). NaN if nothing applies."""
    labels = [label] if label is not None else list(case_data)
    per_fold = defaultdict(list)
    for lab in labels:
        for fold, v in case_data.get(lab, {}).items():
            per_fold[fold].append(v)
    if not per_fold:
        return float("nan")
    return float(np.mean([np.mean(vs) for vs in per_fold.values()]))


def load_run_cases_by_label(sources: list, key: str, metric: str) -> dict:
    """col -> label -> {case_key: mean metric over folds in EVAL_FOLD_INDICES}, where
    case_key = "<dataset>|<case_id>" (see _dataset_ns). Thin per-label-mean view of
    load_run_cases_raw, kept for callers that want one label's own descriptive value
    (e.g. printing a group tree's per-source detail) — NOT used any more to resolve a
    label=None contrast_groups leaf (see _case_flat_value: that needs fold-level
    granularity, which averaging away here would lose)."""
    raw = load_run_cases_raw(sources, key, metric)
    out = {}
    for col, cases in raw.items():
        by_label = defaultdict(dict)
        for case, labs in cases.items():
            for lab, folds in labs.items():
                by_label[lab][case] = float(np.mean(list(folds.values())))
        out[col] = dict(by_label)
    return out


def paired(ref_cases: dict, comp_cases: dict, col: str) -> tuple:
    """Aligned (x_ref, y_comp) arrays over shared cases for one contrast column."""
    r, c = ref_cases.get(col, {}), comp_cases.get(col, {})
    common = sorted(set(r) & set(c))
    return np.array([r[k] for k in common]), np.array([c[k] for k in common])


def find_ref_key(run_keys: list) -> str | None:
    """First run id containing REF_SUBSTR, or None if the config doesn't
    include the headline OURS method (in which case no significance column
    is added — see main())."""
    return next((k for k in run_keys if REF_SUBSTR in k), None)


def significance_column(sources: list, runs_ordered: list, ref_key: str,
                        all_contrasts: list, metric: str, contrast_groups: dict = None,
                        unit_scope: str = DEFAULT_UNIT_SCOPE) -> dict:
    """method_key -> Holm-corrected one-sided ("ref better") macroΔ p-value vs
    ref, across ALL tested contrasts (equal weight per contrast — the same
    estimand as the summary table's `all` column). Same test as
    significance_from_config.py's headline block, imported from stat_tests.py
    so there is one implementation. NaN (blank cell) for the ref's own row.

    contrast_groups (optional): hierarchical pooling tree (see resolve_group's
    block comment) — when given, `all_contrasts` is ignored and the groups'
    top-level keys are used instead, so this matches significance_from_config.py's
    headline test exactly instead of silently disagreeing with it. Without it each raw
    column is its own stratum. Either way the test runs on patient-level sign-flip
    units (`unit_scope`, see build_design / stat_tests.py's design-primitives block)."""
    higher_better = metric == "dice"
    groups = contrast_groups if contrast_groups else {c: c for c in all_contrasts}
    raw_data = {k: load_run_cases_raw(sources, k, metric) for k in runs_ordered}
    competitors = [k for k in runs_ordered if k != ref_key]
    p1s = []
    for key in competitors:
        entries, K = build_design(groups, raw_data[ref_key], raw_data[key], unit_scope)
        p1s.append(macro_perm_design(entries, K, higher_better)[2])
    hp = holm(p1s)
    out = {ref_key: float("nan")}
    out.update(dict(zip(competitors, hp)))
    return out


def _fmt_sig(p: float, alpha: float = 0.05) -> str:
    if not np.isfinite(p):
        return "—"
    s = f"{p:.1e}" if p < 1e-4 else f"{p:.4f}"
    return f"**{s}**" if p < alpha else s


def cross_fold_stats(per_fold: dict) -> tuple:
    fold_means = []
    for vs in per_fold.values():
        arr = np.array(vs, float)
        if np.isfinite(arr).any():
            fold_means.append(float(np.nanmean(arr)))
    if not fold_means:
        return float("nan"), float("nan"), 0
    fm = np.array(fold_means)
    return float(np.nanmean(fm)), float(np.nanstd(fm)), len(fm)


def cross_fold_class_mean(run_data: dict, metric: str, contrast: str) -> float:
    by_contrast = run_data.get(metric, {}).get(contrast, {})
    per_fold: dict = defaultdict(list)
    for lab_folds in by_contrast.values():
        for fold, vs in lab_folds.items():
            per_fold[fold].extend(vs)
    m, _, _ = cross_fold_stats(per_fold)
    return m


# ── contrast_groups: opt-in hierarchical pooling ────────────────────────────
#
# Shared by aggregate_from_config.py (this file — both the displayed "all" column
# and its own inline "sig. vs ref" column), significance_from_config.py (the
# standalone OOD/ALL/IND report), and combined_modality_summary.py /
# meta_task_heatmap.py (one level up again — cross-training-modality and
# cross-dataset task-level tables). One implementation, four callers, so the
# displayed numbers and every significance test built on top of them can never
# silently disagree — this project already cares a lot about keeping the table
# and the test in sync (see significance_from_config.py's module docstring).
#
# Problem this solves: a flat scheme treats every raw column as one equally-
# weighted "contrast". That silently over-weights whichever raw axis happens to
# have the most columns — a modality tested via 3 dataset sources outvotes one
# tested via a single source 3-to-1, and (chaos's CT columns) a multi-organ
# column like "amos_ct" is ALREADY a within-column blend across liver/spleen/
# kidney with no visibility into that blend at all. contrast_groups fixes both
# by letting a config declare, per top-level "contrast" bucket, a small tree of
# how to pool it:
#   - a bare column name (or {column, label}) is a LEAF.
#   - a YAML list POOLS its children (cases/folds concatenated — disjoint case
#     sets: different sources or labels of the same organ/modality) — "average
#     CT-liver across every source that has one."
#   - a YAML dict AVERAGES its children with equal weight per child (mean of
#     each child's own resolved value/array) — "average liver, spleen, kidney
#     into CT" — recursing to arbitrary depth, so "average CT, T1w, T2w
#     together" falls out for free, no special-casing.
# Entirely opt-in: a config with no `contrast_groups` key (or an explicit
# `use_contrast_groups: false`) gets the untouched flat per-column behaviour —
# the toggle back to "current strategy".
#
# Two resolvers share this tree shape:
#   resolve_group_value  — ONE run's own descriptive mean (fold-unit, matches
#                           cross_fold_class_mean's existing convention) — what
#                           the "all" column / heatmap cells show.
#   resolve_group         — a PAIRED diff array (ref − comp, case-unit, matches
#                           significance testing's existing convention) — what
#                           the significance tests consume.
# They intentionally use different units of analysis (fold vs case) because
# the two existing callers already did before contrast_groups existed — this
# preserves that pre-existing distinction, only fixes the WEIGHTING scheme.
def _leaf_refs(node) -> list:
    """Flatten a leaf or a (possibly nested) list of leaves into [(column, label), ...].
    Raises on a dict nested inside a list — not a shape any real config needs;
    dicts must be the outermost node of whatever contains them."""
    if isinstance(node, str):
        return [(node, None)]
    if isinstance(node, dict) and "column" in node:
        return [(node["column"], node.get("label"))]
    if isinstance(node, list):
        out = []
        for n in node:
            out += _leaf_refs(n)
        return out
    raise ValueError(f"contrast_groups: dict not supported nested inside a list: {node!r}")


def resolve_group_value(node, run_data: dict, metric: str) -> float:
    """Recursively resolve a contrast_groups node to ONE descriptive value for
    a single run (cross-fold mean, same unit as cross_fold_class_mean)."""
    if isinstance(node, dict) and "column" not in node:
        vals = [resolve_group_value(child, run_data, metric) for child in node.values()]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")
    refs = _leaf_refs(node)
    per_fold: dict = defaultdict(list)
    for col, label in refs:
        by_contrast = run_data.get(metric, {}).get(col, {})
        labels = [label] if label is not None else list(by_contrast.keys())
        for lab in labels:
            for fold, vs in by_contrast.get(lab, {}).items():
                per_fold[fold].extend(vs)
    m, _, _ = cross_fold_stats(per_fold)
    return m


def unit_of(case_tag: str) -> str:
    """Physical-patient identity of a case tag. Tags look like "<dataset>|<case>"
    (single-modality) or "<training_modality>::<dataset>|<case>" (combined tables —
    the same test patient is scored by every trained arm); drop the training-modality
    prefix so the arms' scores of one patient share a unit."""
    return case_tag.split("::", 1)[-1]


def _leaf_entries(ref_raw: dict, comp_raw: dict, leaf) -> list:
    """[(leaf_id, case_tag, diff)] for one leaf (column, optional label), from
    load_run_cases_raw's fold-level data. label=None flat-pools every applicable label
    into each fold's mean first (via `_case_flat_value` — matches
    cross_fold_class_mean/resolve_group_value's convention for the displayed table
    cell); an explicit `label:` uses only that label. Fixes a 2026-09-21 regression:
    an earlier version of this function averaged each label's own per-case mean
    equally regardless of how many folds backed it (`_collapse_labels` on
    already-fold-averaged data), which silently diverged from the table whenever a
    case had an unequal number of valid folds per label — see
    `project_significance_labelflat_fix_20260921`."""
    col, label = (leaf, None) if isinstance(leaf, str) else (leaf["column"], leaf.get("label"))
    r_col, c_col = ref_raw.get(col, {}), comp_raw.get(col, {})
    out = []
    for t in sorted(set(r_col) & set(c_col)):
        rv, cv = _case_flat_value(r_col[t], label), _case_flat_value(c_col[t], label)
        if np.isfinite(rv) and np.isfinite(cv):
            out.append(((col, label), t, rv - cv))
    return out


def resolve_group_entries(node, ref_raw: dict, comp_raw: dict) -> list:
    """Recursively resolve a contrast_groups node to [(leaf_id, case_tag, weight*diff)],
    the weights summing to 1 — so Σ of the third column IS the group's paired mean diff
    (ref − comp), with exactly the pool-cases-in-a-list / equal-weight-per-child-in-a-dict
    arithmetic the summary tables use. Keeping every case as its own entry (instead of
    collapsing a nested group to one number per organ, as this used to) is what lets the
    significance tests see patients — see stat_tests.py's design-primitives block."""
    if isinstance(node, str) or (isinstance(node, dict) and "column" in node):
        ents = _leaf_entries(ref_raw, comp_raw, node)
        return [(lid, t, d / len(ents)) for lid, t, d in ents]
    if isinstance(node, list):
        # A list POOLS cases across its leaves (concatenate, then mean) — lists only ever
        # hold leaves / lists of leaves (see _leaf_refs), so weight = 1 / total cases.
        raw = _raw_leaf_entries(node, ref_raw, comp_raw)
        return [(lid, t, d / len(raw)) for lid, t, d in raw]
    if isinstance(node, dict):
        kids = [resolve_group_entries(v, ref_raw, comp_raw) for v in node.values()]
        kids = [k for k in kids if k]
        return [(lid, t, c / len(kids)) for k in kids for lid, t, c in k]
    raise ValueError(f"bad contrast_groups node: {node!r}")


def _raw_leaf_entries(node, ref_raw: dict, comp_raw: dict) -> list:
    """Unweighted [(leaf_id, tag, diff)] of every leaf under a leaf/list node."""
    if isinstance(node, list):
        return [e for n in node for e in _raw_leaf_entries(n, ref_raw, comp_raw)]
    if isinstance(node, str) or (isinstance(node, dict) and "column" in node):
        return _leaf_entries(ref_raw, comp_raw, node)
    raise ValueError(f"contrast_groups: dict not supported nested inside a list: {node!r}")


def build_design(groups: dict, ref_raw: dict, comp_raw: dict,
                 unit_scope: str = DEFAULT_UNIT_SCOPE) -> tuple:
    """(entries, K) for the design-based tests in stat_tests.py. One stratum per
    non-empty group in `groups` (name -> contrast_groups node); an entry's unit is the
    physical patient (`unit_scope="patient"`) or the patient within its stratum
    (`"stratum"`). Groups with no paired cases are dropped, as before. `ref_raw`/
    `comp_raw` come from load_run_cases_raw (fold-level, NOT load_run_cases_by_label)."""
    if unit_scope not in ("patient", "stratum"):
        raise ValueError(f"unit_scope must be 'patient' or 'stratum', got {unit_scope!r}")
    entries, K = [], 0
    for node in groups.values():
        ents = resolve_group_entries(node, ref_raw, comp_raw)
        if not ents:
            continue
        for lid, tag, c in ents:
            u = unit_of(tag)
            entries.append((K, lid, u if unit_scope == "patient" else (K, u), c))
        K += 1
    return entries, K


def resolve_group(node, ref_raw: dict, comp_raw: dict) -> np.ndarray:
    """One stratum's paired diff array (ref − comp), ONE ELEMENT PER PATIENT, scaled so
    `arr.mean()` is the group's weighted mean diff and `(arr**2).sum()/arr.size**2` its
    sign-flip variance (see stat_tests.stratum_arrays). Kept for the flat-array callers
    (`macro_perm`/`macro_ci`, meta_task_heatmap, paper scripts); those get patient units
    WITHIN a group but no cross-group patient joining — use `build_design` for that."""
    ents = resolve_group_entries(node, ref_raw, comp_raw)
    arrs = stratum_arrays([(0, lid, unit_of(t), c) for lid, t, c in ents])
    return arrs[0] if arrs else np.array([])


def _describe_group(node, indent: int = 0) -> list:
    pad = "  " * indent
    lines = []
    if isinstance(node, str):
        lines.append(f"{pad}- `{node}` (all labels averaged)")
    elif isinstance(node, dict) and "column" in node:
        lab = node.get("label")
        lines.append(f"{pad}- `{node['column']}`" + (f" — label `{lab}` only" if lab else " (all labels averaged)"))
    elif isinstance(node, list):
        lines.append(f"{pad}- pooled (cases concatenated):")
        for n in node:
            lines += _describe_group(n, indent + 1)
    elif isinstance(node, dict):
        lines.append(f"{pad}- averaged (equal weight per child):")
        for name, child in node.items():
            lines.append(f"{pad}  **{name}**:")
            lines += _describe_group(child, indent + 2)
    return lines


def _best_per_col(mat: np.ndarray, metric: str) -> list:
    """Return the best-row index per column (-1 if column is all NaN)."""
    best = []
    for j in range(mat.shape[1]):
        col = mat[:, j]
        if np.any(np.isfinite(col)):
            idx = int(np.nanargmax(col)) if metric == "dice" else int(np.nanargmin(col))
        else:
            idx = -1
        best.append(idx)
    return best


def _display_key(key: str) -> str:
    return f"{key} (Ours)" if "v26_6_2" in key else key


def build_summary(runs_ordered, runs_data, fold_counts, title, out_dir: Path, prefix: str,
                  in_domain_contrast: str = None, column_order: list = None,
                  sig_by_metric: dict = None, ref_key: str = None, contrast_groups: dict = None):
    # When contrast_groups is set, each displayed column IS a group (one column
    # for CT, one per real contrast) — not one column per raw data source. A
    # group spanning >1 raw source (e.g. CT = ct + amos_ct + sliver07_ct_liver)
    # used to also print those raw columns side by side, with pooling visible
    # only in "all" — inconsistent with what was shown and read as ungrouped.
    # Collapsed 2026-08-31; raw per-source detail still lives in each source's
    # own per-modality results file if it's ever needed.
    if contrast_groups:
        group_names = list(contrast_groups.keys())
        if column_order:
            ordered = [c for c in column_order if c in group_names]
            remaining = [c for c in group_names if c not in ordered]
            all_contrasts = ordered + remaining
        else:
            all_contrasts = group_names
    else:
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

    # Column headers: bold the in-domain contrast
    col_headers = [f"**{c}**" if c == in_domain_contrast else c for c in cols]

    sig_note = ""
    if sig_by_metric is not None:
        sig_note = (f" `sig. vs ref` = Holm-corrected one-sided (ref better) macroΔ p-value of "
                   f"`{_display_key(ref_key)}` vs that row (blank on the ref's own row); "
                   "**bold** = p < 0.05.")

    lines = [
        f"# {title}",
        "",
        f"Generated: {now}  |  Experiments: {len(runs_ordered)}  |  "
        f"Modalities: {', '.join(all_contrasts)}",
        "",
        "Each cell is the **cross-fold, cross-class average** (mean over all labels and folds)"
        + (" — each column is one contrast-group (pooled from its raw sources per the tree "
           "below; a group with a single source is just that source, unchanged)" if contrast_groups else "")
        + f". `all` = average across {'contrast-groups' if contrast_groups else 'modalities'}. "
        + "**Bold** = best per column. "
        + (f"**{in_domain_contrast}** = in-domain contrast. " if in_domain_contrast else "")
        + "— = no data." + sig_note,
        "",
    ]
    if contrast_groups:
        lines.append("Contrast-group pooling (one column per group; raw sources shown for reference):")
        lines.append("")
        for g, node in contrast_groups.items():
            lines.append(f"- **{g}**:")
            lines += _describe_group(node, indent=1)
        lines.append("")

    matrices = {}
    for metric, heading, prec in (("dice", "Dice ↑", 4), ("hd95", "HD95 mm ↓", 2)):
        mat = np.full((len(runs_ordered), len(cols)), np.nan)
        for i, key in enumerate(runs_ordered):
            if contrast_groups:
                per_mod = [resolve_group_value(contrast_groups[c], runs_data[key], metric)
                          for c in all_contrasts]
            else:
                per_mod = [cross_fold_class_mean(runs_data[key], metric, c) for c in all_contrasts]
            for j, v in enumerate(per_mod):
                mat[i, j] = v
            finite = [v for v in per_mod if np.isfinite(v)]
            mat[i, -1] = float(np.mean(finite)) if finite else np.nan
        matrices[metric] = mat

        best = _best_per_col(mat, metric)
        sig = sig_by_metric.get(metric) if sig_by_metric else None
        sig_header = " | sig. vs ref |" if sig is not None else " |"

        lines += [
            f"## {heading}",
            "",
            "| experiment | folds | " + " | ".join(col_headers) + sig_header,
            "|" + "---|" * (len(cols) + 2 + (1 if sig is not None else 0)),
        ]
        for i, key in enumerate(runs_ordered):
            nf = fold_counts.get(key, 0)
            cells = []
            for j in range(len(cols)):
                s = _fmt_val(mat[i, j], prec)
                if s != "—" and best[j] == i:
                    s = f"**{s}**"
                cells.append(s)
            row = f"| {_format_label_md(_display_key(key))} | {nf} | " + " | ".join(cells)
            if sig is not None:
                sig_cell = "—" if key == ref_key else _fmt_sig(sig.get(key, float("nan")))
                row += f" | {sig_cell}"
            lines.append(row + " |")
        lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{prefix}_summary.md"
    md_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n→ {md_path}")

    _save_heatmaps(matrices, runs_ordered, fold_counts, cols, out_dir, prefix, title,
                   in_domain_contrast=in_domain_contrast, sig_by_metric=sig_by_metric,
                   ref_key=ref_key)


def _save_heatmaps(matrices, runs_ordered, fold_counts, cols, out_dir: Path, prefix: str,
                   title: str, in_domain_contrast: str = None,
                   sig_by_metric: dict = None, ref_key: str = None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (matplotlib unavailable, skipping heatmaps: {e})", file=sys.stderr)
        return

    ylabels = [
        _format_label_mathtext(f"{_display_key(k)}  ({fold_counts.get(k, 0)}f)")
        for k in runs_ordered
    ]
    in_domain_col = cols.index(in_domain_contrast) if in_domain_contrast in cols else None

    specs = (
        ("dice", "Dice ↑", 4, "RdBu"),
        ("hd95", "HD95 mm ↓", 2, "RdBu_r"),
    )
    for metric, heading, prec, cmap in specs:
        mat = matrices.get(metric)
        if mat is None:
            continue

        best = _best_per_col(mat, metric)
        sig = sig_by_metric.get(metric) if sig_by_metric else None

        n_runs, n_data_cols = mat.shape
        n_cols = n_data_cols + (1 if sig is not None else 0)
        plot_cols = cols + (["sig. vs ref"] if sig is not None else [])
        full = np.full((n_runs, n_cols), np.nan)
        full[:, :n_data_cols] = mat

        fig_w = max(8, 1.4 * n_cols + 5)
        fig_h = max(3, 0.55 * n_runs + 2)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        masked = np.ma.masked_invalid(full)
        im = ax.imshow(masked, aspect="auto", cmap=cmap)

        # In-domain column: subtle background band drawn behind the image
        if in_domain_col is not None:
            ax.axvspan(in_domain_col - 0.5, in_domain_col + 0.5,
                       color="crimson", alpha=0.08, zorder=0)

        ax.set_xticks(range(n_cols))
        xlabels = ax.set_xticklabels(plot_cols, rotation=30, ha="right")
        if in_domain_col is not None:
            xlabels[in_domain_col].set_color("crimson")
            xlabels[in_domain_col].set_fontweight("bold")
            xlabels[in_domain_col].set_fontsize(10)

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
                    ax.text(
                        j, i, txt,
                        ha="center", va="center", fontsize=8,
                        color="white" if lum < 0.5 else "black",
                        fontweight=fw,
                    )

        # Significance column: no color scale (masked NaN cell), just text —
        # bold red if p < 0.05, gray otherwise; blank on the ref's own row.
        if sig is not None:
            for i, key in enumerate(runs_ordered):
                if key == ref_key:
                    continue
                p_val = sig.get(key, float("nan"))
                if not np.isfinite(p_val):
                    continue
                txt = f"{p_val:.1e}" if p_val < 1e-4 else f"{p_val:.3f}"
                starred = p_val < 0.05
                ax.text(n_data_cols, i, txt, ha="center", va="center", fontsize=8,
                       color="crimson" if starred else "gray",
                       fontweight="bold" if starred else "normal")
            ax.axvline(n_data_cols - 0.5, color="black", linewidth=0.8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("modality" + ("  |  significance vs ref" if sig is not None else ""))
        ax.set_ylabel("experiment")
        fig.tight_layout()

        p = out_dir / f"{prefix}_heatmap_{metric}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"→ {p}")


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate eval results from a YAML config listing run IDs."
    )
    ap.add_argument("config", help="Path to YAML config file")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    output_prefix = cfg.get("output_prefix", "01_results")
    title = cfg.get("title", "Aggregated Results")
    run_keys = cfg.get("runs", [])
    in_domain_contrast = cfg.get("in_domain_contrast", None)
    column_order = cfg.get("column_order", None)

    # Multi-source mode: config has a `sources` list.
    # Single-source mode (backward compat): config has `metrics_dir`.
    if "sources" in cfg:
        sources = [
            {
                "metrics_dir": Path(os.path.expandvars(s["metrics_dir"])),
                "column_prefix": s.get("column_prefix", ""),
                "column_rename": s.get("column_rename", {}),
            }
            for s in cfg["sources"]
        ]
        # output_dir: explicit key, or fall back to first source's metrics_dir
        if "output_dir" in cfg:
            out_dir = Path(os.path.expandvars(cfg["output_dir"]))
        else:
            out_dir = sources[0]["metrics_dir"]
        for src in sources:
            if not src["metrics_dir"].is_dir():
                print(f"  note: source metrics_dir does not exist yet: {src['metrics_dir']}", file=sys.stderr)
        multi_source = True
    else:
        metrics_dir = Path(os.path.expandvars(cfg["metrics_dir"]))
        sources = [{"metrics_dir": metrics_dir, "column_prefix": "", "column_rename": {}}]
        out_dir = metrics_dir
        if not metrics_dir.is_dir():
            print(f"  note: metrics_dir does not exist yet: {metrics_dir}", file=sys.stderr)
        multi_source = False

    runs_data = {}
    fold_counts = {}
    runs_ordered = []
    any_data = False

    # Missing runs are NOT dropped from the table — they still get a row (all "—"
    # cells, 0 folds), keeping the config's declared order/method set intact so a
    # still-training or not-yet-launched experiment is visibly absent rather than
    # silently missing from the comparison.
    for key in run_keys:
        if multi_source:
            data, n_folds = load_run_from_sources(sources, key)
        else:
            run_dir = resolve_run_dir(sources[0]["metrics_dir"], key)
            if run_dir is None:
                print(f"  blank row {key}: directory not found in {sources[0]['metrics_dir']}", file=sys.stderr)
                data, n_folds = {}, 0
            else:
                data = load_run(run_dir)
                n_folds = count_eval_folds(run_dir)
        if not data:
            print(f"  blank row {key}: no eval_all.csv found in any source", file=sys.stderr)
        else:
            any_data = True
        runs_data[key] = data
        fold_counts[key] = n_folds
        runs_ordered.append(key)

    if not any_data:
        print("No runs with evaluation data found.", file=sys.stderr)
        sys.exit(1)

    # contrast_groups (opt-in — see the block comment above resolve_group_value()).
    # Disabled by omitting the key, or by `use_contrast_groups: false`.
    contrast_groups = cfg.get("contrast_groups") if cfg.get("use_contrast_groups", True) else None

    # Inline "sig. vs ref" column: auto-wired whenever the config's run list
    # includes the headline OURS method (REF_SUBSTR) and has data for it, unless
    # explicitly disabled (`no_significance: true`) or overridden (`sig_ref: <exact run id>`).
    sig_by_metric, ref_key = None, None
    if not cfg.get("no_significance", False):
        ref_key = cfg.get("sig_ref") or find_ref_key(runs_ordered)
        if ref_key is not None and ref_key in runs_ordered and runs_data.get(ref_key):
            all_contrasts = sorted({c for d in runs_data.values() for c in d.get("dice", {})})
            sig_by_metric = {
                metric: significance_column(sources, runs_ordered, ref_key, all_contrasts, metric,
                                            contrast_groups=contrast_groups,
                                            unit_scope=cfg.get("unit_scope", DEFAULT_UNIT_SCOPE))
                for metric in ("dice", "hd95")
            }
        else:
            ref_key = None

    build_summary(runs_ordered, runs_data, fold_counts, title, out_dir, output_prefix,
                  in_domain_contrast=in_domain_contrast, column_order=column_order,
                  sig_by_metric=sig_by_metric, ref_key=ref_key, contrast_groups=contrast_groups)


if __name__ == "__main__":
    main()
