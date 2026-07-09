"""
Single source of truth for which folds are included in evaluation/aggregation,
shared across every dataset and experiment (imported by eval_aggregate.py, used
by amos/brats2024-glioma/chaos/sliver07/trusted via aggregate_results.py; and by
datasets/00_commun_scripts/00_03_evaluate/aggregate_from_config.py, used by chaos/on-harmony/open-ms).

Capped at folds 0-2 — this is the default AND ONLY behavior, not an opt-in. Any
fold directory beyond this (e.g. fold3, present in older 4-fold training runs) is
silently excluded from every aggregated report. Rationale: open-ms's texture/
causal-ablation arms train on 3 folds only (see datasets/open-ms/5_scripts_open-ms/
04_train/04_08+); capping every comparison to the same 3 folds keeps every run
(old 4-fold and new 3-fold alike) on equal footing rather than mixing sample
sizes. Decision: 2026-07-07.
"""
EVAL_FOLDS = ("fold0", "fold1", "fold2")

# Plain-int form for callers that index folds numerically (e.g. `for fold in
# EVAL_FOLD_INDICES: pdir = base / f"fold{fold}"`) rather than by directory name.
EVAL_FOLD_INDICES = tuple(int(f[len("fold"):]) for f in EVAL_FOLDS)


def filter_fold_dirs(fold_dirs):
    """Keep only entries whose .name is in EVAL_FOLDS, preserving input order."""
    return [d for d in fold_dirs if d.name in EVAL_FOLDS]
