"""
Shared statistical-test primitives for cross-run segmentation-metric comparisons.

Common and shared: every dataset's significance testing should import these
rather than reimplementing them inline (there was previously exactly one bespoke
Wilcoxon implementation in the whole project, under on-harmony's
texture_analysis_lvl_1 — specific to that analysis, not reusable).

These are intentionally generic (operate on plain paired arrays) — the dataset-
and config-shape-specific part (resolving run dirs, loading eval_all.csv,
choosing the unit of analysis) lives in significance_from_config.py.

Unit-of-analysis note (why this module doesn't do the loading itself): the
correct paired unit for these tests is the held-out CASE, with each case's score
averaged across its labels AND across every evaluated fold — NOT one row per
(fold, case). A given held-out patient is scored by every fold's model, so
treating each fold's re-scoring of the same patient as an independent
observation is pseudo-replication: it inflates the effective sample size and
produces artificially small p-values. Collapsing to one score per patient first
(see significance_from_config.load_run_cases) gives the honest, independent
sample size. See datasets/00_commun_scripts/00_00_utils/eval_folds.py for the
fold cap (0-2) applied when loading.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def wilcoxon_p(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value for paired arrays x, y.

    Returns NaN if undefined (fewer than 1 pair, or all differences are zero).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 1:
        return float("nan")
    d = x - y
    if not np.any(d != 0):
        return float("nan")
    try:
        return float(stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def holm(pvals: list) -> list:
    """Holm-Bonferroni step-down adjusted p-values, preserving input order and
    ignoring (passing through as NaN) any non-finite entries."""
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


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


# ── macroΔ primitives (contrast-stratified paired sign-flip test) ──────────
#
# Shared by significance_from_config.py (per-modality significance blocks) and
# combined_modality_summary.py (the inline significance column on the
# cross-modality tables) — same estimand, same test, two different callers
# instead of two implementations. See significance_from_config.py's module
# docstring for the full statistical rationale.


def macro_stat(arrs: list) -> float:
    """macroΔ = mean over contrasts of each contrast's mean paired diff."""
    return float(np.mean([a.mean() for a in arrs])) if arrs else float("nan")


def macro_perm(arrs: list, higher_better: bool, scale: float = 1.0) -> tuple:
    """Contrast-stratified paired SIGN-FLIP test on macroΔ (closed-form normal
    approximation). `arrs` is a list of per-contrast paired-diff arrays
    (ref − competitor). Returns (macroΔ*scale, p_two_sided, p_one_sided) where
    the one-sided p tests "ref is better" in the direction implied by
    `higher_better` (dice: higher=better; hd95: lower=better)."""
    if not arrs:
        return float("nan"), float("nan"), float("nan")
    obs = macro_stat(arrs)
    K = len(arrs)
    var_T = sum((a ** 2).sum() / (a.size ** 2) for a in arrs) / (K ** 2)
    sd = float(np.sqrt(var_T))
    if sd == 0:
        p_two = 1.0 if obs == 0 else 0.0
        return obs * scale, p_two, (0.0 if (obs > 0) == higher_better and obs != 0 else 1.0)
    z = obs / sd
    p_two = float(2 * stats.norm.sf(abs(z)))
    p_one = float(stats.norm.sf(z) if higher_better else stats.norm.cdf(z))
    return obs * scale, p_two, p_one


def macro_ci(arrs: list, scale: float = 1.0, b_boot: int = 5000, seed: int = 0) -> tuple:
    """Hierarchical bootstrap 95% CI of macroΔ (resample cases within each
    contrast, then average across contrasts)."""
    if not arrs:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    per = np.empty((b_boot, len(arrs)))
    for j, a in enumerate(arrs):
        idx = rng.integers(0, a.size, size=(b_boot, a.size))
        per[:, j] = a[idx].mean(axis=1)
    lo, hi = np.percentile(per.mean(axis=1), [2.5, 97.5])
    return lo * scale, hi * scale
