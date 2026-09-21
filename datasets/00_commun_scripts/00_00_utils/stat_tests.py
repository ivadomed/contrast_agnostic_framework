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
    `higher_better` (dice: higher=better; hd95: lower=better).

    The sign-flip null's mean (0) and variance are exact; the TAIL is a normal
    approximation, not an enumeration or a resampling — so this is not an
    "exact test", and callers should not describe it as one. Checked against a
    200k-draw Monte-Carlo sign-flip null on this project's borderline case
    (0.0237 vs 0.0245, 2026-09-14). It is least accurate for heavy-tailed diffs
    at small n, where it is CONSERVATIVE (a synthetic n=8 array with one large
    outlier gave 0.127 here against a true 0.004), so a borderline NEW result
    should be re-checked by resampling rather than trusted from this alone."""
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


# ── unit-of-independence ("design") primitives — added 2026-09-21 ───────────
#
# WHY THIS EXISTS. `macro_perm`/`macro_ci` above take one flat array per contrast
# and treat EVERY ELEMENT as an independent sign-flip unit. That silently breaks
# in two ways once a contrast is not literally "one independent case per element":
#   1. A hierarchical group (contrast_groups' organ tree: per-organ mean, then mean
#      across organs) used to be collapsed to ONE ELEMENT PER ORGAN before it got
#      here, so CT's ~140 held-out patients became 3 sign-flip units and adding
#      cases could never shrink its variance (MRI groups, per-case, could).
#   2. The same physical patient scored by several models (t1in- AND t2spir-trained
#      arms), or appearing in several test contrasts (CHAOS's T1-in / T1-out / T2-SPIR
#      are the same 4 patients), was counted as several independent units.
#
# THE MODEL. The statistic is linear in the per-case diffs:
#     macroΔ = (1/K) Σ_strata Σ_entries c ,   c = weight × (ref − comp) diff
# where the weights encode the per-organ-then-cross-organ averaging (they sum to 1
# within a stratum) — so the POINT ESTIMATE is exactly what the tables already show;
# only the null distribution changes. An ENTRY is (stratum, leaf, unit, c). A UNIT is
# the thing that flips sign as one block under the sign-flip null: an independent
# patient (all of its entries — every organ, model, and, at unit_scope="patient",
# every contrast — flip together). Null: mean 0, variance Σ_units (Σ_entries c / K)².
#
#   entries: list of (stratum_idx, leaf_id, unit_id, c)      K: number of strata
# See aggregate_from_config.build_design for how entries are built and how
# unit_id is chosen (unit_scope "patient" vs "stratum").

def design_macro(entries: list, K: int) -> float:
    return float(sum(e[3] for e in entries) / K) if entries and K else float("nan")


def _unit_totals(entries: list, K: int) -> dict:
    tot = {}
    for _, _, u, c in entries:
        tot[u] = tot.get(u, 0.0) + c / K
    return tot


# Number of heaviest units whose signs are ENUMERATED exactly by macro_perm_design (2**M
# sign patterns; the remaining, lighter units are handled by a normal approximation).
ENUM_UNITS = 20


def macro_perm_design(entries: list, K: int, higher_better: bool, scale: float = 1.0) -> tuple:
    """Sign-flip test on macroΔ over unit-level `entries` — same return contract as
    `macro_perm`: (macroΔ*scale, p_two_sided, p_one_sided), one-sided = "ref better".

    Tail evaluation (deliberately NOT `macro_perm`'s plain normal approximation): the
    ENUM_UNITS heaviest units (by |contribution|) have their 2**M sign patterns enumerated
    EXACTLY, and the remaining lighter units — the many small CT/AMOS patients — enter as a
    normal term with their exact null variance. With ≤ ENUM_UNITS units this is the exact
    sign-flip test. Why: in this design a handful of units can carry almost all the null
    variance (the 4 CHAOS MR patients each span three contrast strata), and a plain normal
    approximation is then badly conservative — measured 2026-09-21 on the CHAOS combined
    table: p = 0.013 by normal approximation against < 5e-6 by 200k-draw Monte-Carlo for a
    comparison where 140 of 164 units agree in sign. Deterministic (no Monte-Carlo noise),
    and reaches far tails (a many-unit consistent effect is not floored at 1/B)."""
    if not entries or not K:
        return float("nan"), float("nan"), float("nan")
    t = np.array(list(_unit_totals(entries, K).values()))
    obs = float(t.sum())
    order = np.argsort(-np.abs(t))
    top, rest = t[order[:ENUM_UNITS]], t[order[ENUM_UNITS:]]
    T = np.zeros(1)                                   # all 2**M signed sums of the top units
    for v in top:
        T = np.concatenate([T + v, T - v])
    sd_r = float(np.sqrt((rest ** 2).sum()))
    tol = 1e-12 * max(1.0, float(np.abs(t).sum()))
    if sd_r == 0:                                     # exact test (or the rest are all zero)
        p_hi = float((T >= obs - tol).mean())         # P(T ≥ obs)
        p_lo = float((T <= obs + tol).mean())         # P(T ≤ obs)
        p_two = float((np.abs(T) >= abs(obs) - tol).mean())
    else:
        p_hi = float(stats.norm.sf((obs - T) / sd_r).mean())
        p_lo = float(stats.norm.cdf((obs - T) / sd_r).mean())
        p_two = float((stats.norm.sf((abs(obs) - T) / sd_r) + stats.norm.cdf((-abs(obs) - T) / sd_r)).mean())
    return obs * scale, p_two, (p_hi if higher_better else p_lo)


def macro_ci_design(entries: list, K: int, scale: float = 1.0, b_boot: int = 5000,
                    seed: int = 0) -> tuple:
    """95% bootstrap CI of macroΔ over units — the CI counterpart of
    `macro_perm_design`, so a report's CI and p describe the same model. Units are
    resampled WITH replacement inside classes of units that share the same
    (stratum, leaf) footprint (e.g. AMOS patients vs CHAOS-CT patients vs the 4
    CHAOS MR patients, who span three strata and are resampled jointly), which
    keeps each source's / organ's sample size fixed the way the observed design has
    it. Weights are held at their observed values. On flat all-unique-unit data this
    reduces to `macro_ci`'s resample-cases-within-contrast bootstrap."""
    if not entries or not K:
        return float("nan"), float("nan")
    tot, foot = _unit_totals(entries, K), {}
    for g, leaf, u, _ in entries:
        foot.setdefault(u, set()).add((g, leaf))
    classes = {}
    for u, f in foot.items():
        classes.setdefault(frozenset(f), []).append(tot[u])
    rng = np.random.default_rng(seed)
    T = np.zeros(b_boot)
    for vals in classes.values():
        v = np.asarray(vals)
        T += v[rng.integers(0, v.size, size=(b_boot, v.size))].sum(axis=1)
    lo, hi = np.percentile(T, [2.5, 97.5])
    return lo * scale, hi * scale


def stratum_arrays(entries: list, K: int = None) -> list:
    """One array per stratum, ONE ELEMENT PER UNIT WITHIN THAT STRATUM, scaled so that
    `arr.mean()` is the stratum's weighted mean diff and `(arr**2).sum()/arr.size**2`
    is its sign-flip variance. Consequences: (i) the legacy flat-array `macro_perm` /
    `macro_ci` / `macro_stat` keep working unchanged on hierarchical groups and now see
    patients, not organ means; (ii) sign counts (win/loss) are per unit. Strata are
    returned in stratum-index order; empty strata are omitted. (K is unused; kept so
    call sites read like the other design helpers.)"""
    per = {}
    for g, _, u, c in entries:
        per.setdefault(g, {})
        per[g][u] = per[g].get(u, 0.0) + c
    return [np.array(list(per[g].values())) * len(per[g]) for g in sorted(per)]
